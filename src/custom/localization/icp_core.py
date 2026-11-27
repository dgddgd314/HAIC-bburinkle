"""Scan-to-map localization against the fixed-architecture wall cloud (no ROS here).

Pipeline per LiDAR scan (points already in the pelvis frame):
  1. Leveling   - fit the floor plane under the robot -> roll, pitch, pelvis height.
  2. Filtering  - range, height band (no floor / ceiling), voxel downsample.
  3. 3-DoF ICP  - point-to-plane on x, y, yaw against walls_reference.ply,
                  with a shrinking correspondence gate and Tukey weights so that
                  furniture / glass / people (absent from the map) are rejected.
  4. Quality    - inlier ratio + RMSE; a bad match is reported, not trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

MAP_DIR = Path(__file__).resolve().parents[1] / "maps_structural"


# --------------------------------------------------------------------------- utils
def read_ply_xyz(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n = None
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        return np.frombuffer(f.read(n * 12), dtype="<f4").reshape(n, 3).astype(np.float64)


def voxel_down(p: np.ndarray, size: float) -> np.ndarray:
    """One point (centroid) per occupied voxel."""
    if len(p) == 0:
        return p
    key = np.floor(p / size).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv, p)
    return out / cnt[:, None]


def rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Pose:
    x: float
    y: float
    yaw: float
    z: float = 0.79        # pelvis height above floor
    roll: float = 0.0
    pitch: float = 0.0

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = rot_rpy(self.roll, self.pitch, self.yaw)
        T[:3, 3] = (self.x, self.y, self.z)
        return T


@dataclass
class MatchResult:
    pose: Pose
    ok: bool
    inlier_ratio: float     # scan points within 10 cm of a wall
    rmse: float             # of the inliers, metres
    n_points: int
    leveled: bool


# --------------------------------------------------------------------------- map
class WallMap:
    """Wall/door-frame surface samples with normals; only near-vertical surfaces are used."""

    def __init__(self, ply: Path = MAP_DIR / "walls_reference.ply", voxel: float = 0.05,
                 cache: bool = True):
        cache_dir = Path.home() / ".cache" / "haic_bburinkle"
        cache_file = cache_dir / f"{ply.stem}.v{int(voxel * 1000)}.npz"
        if cache and cache_file.exists() and cache_file.stat().st_mtime >= ply.stat().st_mtime:
            d = np.load(cache_file)
            pts, nrm = d["p"], d["n"]
        else:
            pts = voxel_down(read_ply_xyz(ply), voxel)
            nrm = self._normals(pts)
            if cache:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(cache_file, p=pts, n=nrm)
                except OSError:
                    pass
        vertical = np.abs(nrm[:, 2]) < 0.3          # wall faces, not lintel tops/bottoms
        self.p, self.n = pts[vertical], nrm[vertical]
        self.n[:, 2] = 0.0
        self.n /= np.linalg.norm(self.n, axis=1, keepdims=True)
        sign = self._outward_sign(self.p, self.n)
        self.n *= np.where(sign == 0, 1.0, sign)[:, None]
        self.one_sided = sign != 0                  # unknown orientation -> match from both sides
        self.tree = cKDTree(self.p)
        self.zmax = float(pts[:, 2].max())

    @staticmethod
    def _outward_sign(p: np.ndarray, n: np.ndarray) -> np.ndarray:
        """+1 if n points out of the wall solid, -1 if into it, 0 if undecidable.

        Uses the 2 cm wall-section grid at z=1 m (100 = wall, -1 = outside the building): the side of the face
        that lands in a wall cell is the inside."""
        grid = np.load(MAP_DIR / "walls_slice_localization.npy")
        res, ox, oy = 0.02, 5.0, -0.5

        def solid(q):
            c = np.floor((q[:, 0] - ox) / res).astype(int)
            r = np.floor((q[:, 1] - oy) / res).astype(int)
            inb = (r >= 0) & (r < grid.shape[0]) & (c >= 0) & (c < grid.shape[1])
            out = np.ones(len(q), bool)
            out[inb] = grid[r[inb], c[inb]] != 0     # wall (100) or outside the building (-1)
            return out

        sign = np.zeros(len(p))
        for step in (0.03, 0.06, 0.10):             # conservative raster: step out of blocked cells
            front, back = solid(p[:, :2] + step * n[:, :2]), solid(p[:, :2] - step * n[:, :2])
            todo = sign == 0
            sign[todo & front & ~back] = -1.0
            sign[todo & back & ~front] = 1.0
        return sign

    @staticmethod
    def _normals(p: np.ndarray, k: int = 12) -> np.ndarray:
        _, idx = cKDTree(p).query(p, k=k)
        nb = p[idx] - p[idx].mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb)
        _, vec = np.linalg.eigh(cov)
        return vec[:, :, 0]


# --------------------------------------------------------------------------- localizer
class Localizer:
    def __init__(self, wall_map: WallMap | None = None, *,
                 voxel: float = 0.10, min_range: float = 0.4, max_range: float = 25.0,
                 z_band: tuple[float, float] = (0.25, 2.3),
                 gates: tuple[float, ...] = (1.0, 0.6, 0.4, 0.25, 0.15, 0.10, 0.10, 0.10),
                 inlier_dist: float = 0.10, min_inlier_ratio: float = 0.65,
                 max_rmse: float = 0.06, min_points: int = 80):
        self.map = wall_map or WallMap()
        self.voxel, self.min_range, self.max_range = voxel, min_range, max_range
        self.z_band, self.gates = z_band, gates
        self.inlier_dist, self.min_inlier_ratio = inlier_dist, min_inlier_ratio
        self.max_rmse, self.min_points = max_rmse, min_points

    # -- 1. leveling ------------------------------------------------------------
    def level(self, pts: np.ndarray, prior: Pose, tilt_err_deg: float = 5.0,
              iters: int = 200, seed: int = 0,
              height_range: tuple[float, float] = (0.6, 1.0)) -> tuple[Pose, bool]:
        """RANSAC floor fit -> roll, pitch, pelvis height.

        The head LiDAR (lowest beam -22.5 deg) only sees floor from ~2 m out, so the
        floor is the near-horizontal plane at pelvis height below the robot, searched
        within tilt_err_deg of the prior (IMU / last estimate). If that fails the search
        is retried wide (20 deg). height_range (G1 pelvis height) rejects desk/bed tops
        and the -1.5 m cliff pits; points under a candidate plane are penalised."""
        for err in dict.fromkeys((tilt_err_deg, 20.0)):
            got = self._level_once(pts, prior, err, iters, seed, height_range)
            if got is not None:
                return got, True
        return prior, False

    def _level_once(self, pts, prior, max_tilt_err_deg, iters, seed, height_range):
        up = rot_rpy(prior.roll, prior.pitch, 0.0).T @ np.array([0.0, 0.0, 1.0])
        r = np.linalg.norm(pts, axis=1)
        h = pts @ up                                       # height relative to pelvis (approx.)
        band = 0.12 + r * np.tan(np.radians(max_tilt_err_deg))
        cand = pts[(r > 0.8) & (r < 12.0) & (np.abs(h + prior.z) < band) & (h < -0.3)]
        if len(cand) < 60:
            return None
        rng = np.random.default_rng(seed)
        sub = cand[rng.choice(len(cand), min(len(cand), 4000), replace=False)]
        cos_max = np.cos(np.radians(max_tilt_err_deg))
        best_n, best_cnt, best_d = None, 0, 0.0
        tri = sub[rng.integers(0, len(sub), (iters, 3))]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(nrm, axis=1)
        for k in range(iters):
            if ln[k] < 1e-6:
                continue
            n = nrm[k] / ln[k]
            if n @ up < 0:
                n = -n
            if n @ up < cos_max:
                continue
            d = -(tri[k, 0] @ n)                           # plane n.p + d = 0, d = pelvis height
            if not height_range[0] < d < height_range[1]:
                continue
            sd = sub @ n + d                               # signed height above the candidate plane
            # nothing should lie under the real floor (cliff pits are > 0.6 m deeper)
            cnt = int((np.abs(sd) < 0.03).sum()) - 3 * int(((sd < -0.06) & (sd > -0.6)).sum())
            if cnt > best_cnt:
                best_n, best_cnt, best_d = n, cnt, d
        if best_n is None or best_cnt < 40:
            return None
        f = cand[np.abs(cand @ best_n + best_d) < 0.03]
        for _ in range(2):                                 # least-squares refine on inliers
            c = f.mean(axis=0)
            _, _, vt = np.linalg.svd(f - c, full_matrices=False)
            n = vt[2] if vt[2] @ best_n > 0 else -vt[2]
            f = cand[np.abs((cand - c) @ n) < 0.02]
            if len(f) < 40:
                return None
        # n is world-up expressed in the pelvis frame: n = R^T e_z
        roll = float(np.arctan2(n[1], n[2]))
        pitch = float(np.arcsin(np.clip(-n[0], -1, 1)))
        height = float(-(c @ n))
        if abs(roll) > 0.5 or abs(pitch) > 0.6 or not height_range[0] < height < height_range[1]:
            return None
        return Pose(prior.x, prior.y, prior.yaw, height, roll, pitch)

    # -- 2. filtering -------------------------------------------------------------
    def prepare(self, pts: np.ndarray, lev: Pose) -> np.ndarray:
        r = np.linalg.norm(pts, axis=1)
        pts = pts[(r > self.min_range) & (r < self.max_range)]
        q = pts @ rot_rpy(lev.roll, lev.pitch, 0.0).T
        q[:, 2] += lev.z                                 # height above floor
        q = q[(q[:, 2] > self.z_band[0]) & (q[:, 2] < min(self.z_band[1], self.map.zmax - 0.05))]
        return voxel_down(q, self.voxel)

    # -- 3. ICP -------------------------------------------------------------------
    def _visible(self, idx, m, x, y):
        """Keep correspondences whose wall face points toward the sensor."""
        ref, nrm = self.map.p[idx[m]], self.map.n[idx[m]]
        v = np.array([x, y]) - ref[:, :2]
        return (np.einsum("ij,ij->i", nrm[:, :2], v) > 0) | ~self.map.one_sided[idx[m]]

    def icp(self, q: np.ndarray, x: float, y: float, yaw: float):
        n_tree = self.map.tree
        for gate in self.gates:
            c, s = np.cos(yaw), np.sin(yaw)
            w = np.empty_like(q)
            w[:, 0] = c * q[:, 0] - s * q[:, 1] + x
            w[:, 1] = s * q[:, 0] + c * q[:, 1] + y
            w[:, 2] = q[:, 2]
            d, idx = n_tree.query(w, distance_upper_bound=gate, workers=-1)
            m = np.isfinite(d)
            m[m] = self._visible(idx, m, x, y)
            if m.sum() < self.min_points // 2:
                return x, y, yaw, False
            nrm, ref = self.map.n[idx[m]], self.map.p[idx[m]]
            wm, qm = w[m], q[m]
            res = np.einsum("ij,ij->i", nrm[:, :2], wm[:, :2] - ref[:, :2])
            cgate = max(gate, 0.05)
            wt = np.clip(1 - (res / cgate) ** 2, 0, None) ** 2          # Tukey
            # Jacobian of residual w.r.t. (dx, dy, dyaw); rotation applied about map origin of q
            dxy = np.stack([-(s * qm[:, 0] + c * qm[:, 1]), c * qm[:, 0] - s * qm[:, 1]], 1)
            J = np.stack([nrm[:, 0], nrm[:, 1], np.einsum("ij,ij->i", nrm[:, :2], dxy)], 1)
            H = J.T @ (J * wt[:, None]) + np.diag([1e-3, 1e-3, 1e-3]) * wt.sum() * 1e-3
            g = J.T @ (wt * res)
            try:
                dlt = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return x, y, yaw, False
            x, y, yaw = x + dlt[0], y + dlt[1], wrap(yaw + dlt[2])
        return x, y, yaw, True

    def score(self, q: np.ndarray, x: float, y: float, yaw: float) -> tuple[float, float]:
        c, s = np.cos(yaw), np.sin(yaw)
        w = np.c_[c * q[:, 0] - s * q[:, 1] + x, s * q[:, 0] + c * q[:, 1] + y, q[:, 2]]
        d, idx = self.map.tree.query(w, distance_upper_bound=self.inlier_dist, workers=-1)
        m = np.isfinite(d)
        m[m] = self._visible(idx, m, x, y)
        return float(m.mean()), float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else 1.0

    # -- full step ------------------------------------------------------------------
    def localize(self, pts_pelvis: np.ndarray, prior: Pose, tilt_err_deg: float = 5.0,
                 restarts: bool = True) -> MatchResult:
        lev, leveled = self.level(pts_pelvis, prior, tilt_err_deg)
        q = self.prepare(pts_pelvis, lev)
        if len(q) < self.min_points:
            return MatchResult(lev, False, 0.0, 1.0, len(q), leveled)
        best = None
        starts = [(0.0, 0.0, 0.0)]
        if restarts:                                    # only used when the first try is poor
            starts += [(dx, dy, dyaw) for dyaw in (0.0, np.radians(-6), np.radians(6))
                       for dx, dy in ((0, 0), (0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2))][1:]
        for dx, dy, dyaw in starts:
            x, y, yaw, conv = self.icp(q, lev.x + dx, lev.y + dy, wrap(lev.yaw + dyaw))
            ratio, rmse = self.score(q, x, y, yaw)
            cand = (conv, ratio, -rmse, x, y, yaw)
            if best is None or cand[:3] > best[:3]:
                best = cand
            if conv and ratio >= self.min_inlier_ratio + 0.1 and rmse <= self.max_rmse:
                break                                   # clearly good: skip remaining restarts
        conv, ratio, nrmse, x, y, yaw = best
        ok = conv and ratio >= self.min_inlier_ratio and -nrmse <= self.max_rmse
        return MatchResult(Pose(x, y, yaw, lev.z, lev.roll, lev.pitch), ok, ratio, -nrmse, len(q), leveled)

    def relocalize(self, pts_pelvis: np.ndarray, guess: Pose, radius: float = 1.0,
                   step: float = 0.25, yaw_range_deg: float = 180.0, yaw_step_deg: float = 10.0,
                   top_k: int = 6) -> MatchResult:
        """Search around a rough guess (start, respawn, or lost tracking).

        Cheap scoring of every (x, y, yaw) candidate on a sparse cloud, then full ICP
        from the best top_k candidates."""
        lev, _ = self.level(pts_pelvis, guess, tilt_err_deg=20.0)
        q = self.prepare(pts_pelvis, lev)
        if len(q) < self.min_points:
            return MatchResult(lev, False, 0.0, 1.0, len(q), False)
        sparse = voxel_down(q, 0.3)
        offs = np.arange(-radius, radius + 1e-9, step)
        yaws = np.radians(np.arange(-yaw_range_deg, yaw_range_deg, yaw_step_deg)) if yaw_range_deg < 180 \
            else np.radians(np.arange(-180.0, 180.0, yaw_step_deg))
        cands = []
        for dyaw in yaws:
            yaw = wrap(guess.yaw + dyaw)
            c, s_ = np.cos(yaw), np.sin(yaw)
            rot = np.c_[c * sparse[:, 0] - s_ * sparse[:, 1], s_ * sparse[:, 0] + c * sparse[:, 1], sparse[:, 2]]
            for dx in offs:
                for dy in offs:
                    d, _ = self.map.tree.query(rot + [guess.x + dx, guess.y + dy, 0.0],
                                               distance_upper_bound=0.25, workers=-1)
                    cands.append((float(np.isfinite(d).mean()), guess.x + dx, guess.y + dy, yaw))
        cands.sort(reverse=True)
        best = None
        for _, x, y, yaw in cands[:top_k]:
            xi, yi, yawi, conv = self.icp(q, x, y, yaw)
            ratio, rmse = self.score(q, xi, yi, yawi)
            if best is None or (conv, ratio, -rmse) > best[:3]:
                best = (conv, ratio, -rmse, xi, yi, yawi)
        conv, ratio, nrmse, x, y, yaw = best
        ok = conv and ratio >= self.min_inlier_ratio and -nrmse <= self.max_rmse
        return MatchResult(Pose(x, y, yaw, lev.z, lev.roll, lev.pitch), ok, ratio, -nrmse, len(q), True)
