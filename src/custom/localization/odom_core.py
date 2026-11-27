"""LiDAR scan-to-keyframe odometry (no map, no ROS).

Each scan is leveled with the same floor fit as the localizer, cut to a height band,
and registered (point-to-plane, x/y/yaw) to a keyframe cloud built from earlier scans.
Everything static counts here (walls *and* furniture), so it also works where the
architectural map is incomplete. A new keyframe is taken every 0.5 m or 15 deg.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from icp_core import Localizer, Pose, rot_rpy, voxel_down, wrap


class _Target:
    """Duck-types WallMap for Localizer.icp/score: points, horizontal normals, KD-tree."""

    def __init__(self, pts: np.ndarray, k: int = 10):
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=min(k, len(pts)), workers=-1)
        nb = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
        _, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
        n = vec[:, :, 0]
        vertical = np.abs(n[:, 2]) < 0.5                  # planar-ish, vertical surfaces
        self.p, self.n = pts[vertical], n[vertical]
        self.n[:, 2] = 0.0
        self.n /= np.maximum(np.linalg.norm(self.n, axis=1, keepdims=True), 1e-9)
        self.tree = cKDTree(self.p)
        self.one_sided = np.zeros(len(self.p), bool)
        self.zmax = 10.0


class ScanOdometry:
    def __init__(self, key_dist: float = 0.5, key_yaw_deg: float = 15.0, voxel: float = 0.08):
        # reuse the localizer's leveling / filtering / ICP with odometry-friendly settings
        self.reg = Localizer.__new__(Localizer)
        Localizer.__init__(self.reg, wall_map=_Target(np.random.default_rng(0).normal(size=(50, 3))),
                           voxel=voxel, gates=(0.5, 0.3, 0.2, 0.12, 0.08, 0.08),
                           min_inlier_ratio=0.5, max_rmse=0.06)
        self.key_dist, self.key_yaw = key_dist, np.radians(key_yaw_deg)
        self.pose = Pose(0.0, 0.0, 0.0, 0.79, 0.0, np.radians(13.0))
        self.key_pose = None                                   # odom pose of the keyframe
        self.key = None                                        # _Target in keyframe-local coords
        self.n_fail = 0

    def reset(self, pose: Pose):
        self.pose, self.key, self.key_pose = pose, None, None

    def _local_cloud(self, pts_pelvis: np.ndarray, lev: Pose) -> np.ndarray:
        return self.reg.prepare(pts_pelvis, lev)               # leveled, height band, voxel

    def update(self, pts_pelvis: np.ndarray, dyaw_pred: float = 0.0,
               dxy_pred: tuple[float, float] = (0.0, 0.0)) -> tuple[Pose, bool]:
        """dyaw_pred / dxy_pred: motion since the last scan (gyro, velocity) in odom frame."""
        prior = Pose(self.pose.x + dxy_pred[0], self.pose.y + dxy_pred[1],
                     wrap(self.pose.yaw + dyaw_pred), self.pose.z, self.pose.roll, self.pose.pitch)
        lev, _ = self.reg.level(pts_pelvis, prior, tilt_err_deg=5.0)
        q = self._local_cloud(pts_pelvis, lev)
        if len(q) < 80:
            self.pose, self.n_fail = lev, self.n_fail + 1
            return self.pose, False
        if self.key is None:                                   # first scan: becomes the keyframe
            self.pose = lev
            self._set_key(q, lev)
            return self.pose, True
        # express the prior relative to the keyframe, register, map back
        kx, ky, kyaw = self.key_pose
        c, s = np.cos(-kyaw), np.sin(-kyaw)
        rx, ry = c * (lev.x - kx) - s * (lev.y - ky), s * (lev.x - kx) + c * (lev.y - ky)
        ryaw = wrap(lev.yaw - kyaw)
        self.reg.map = self.key
        x, y, yaw, conv = self.reg.icp(q, rx, ry, ryaw)
        ratio, rmse = self.reg.score(q, x, y, yaw)
        ok = conv and ratio >= self.reg.min_inlier_ratio and rmse <= self.reg.max_rmse
        if ok:
            c, s = np.cos(kyaw), np.sin(kyaw)
            self.pose = Pose(kx + c * x - s * y, ky + s * x + c * y, wrap(kyaw + yaw),
                             lev.z, lev.roll, lev.pitch)
            self.n_fail = 0
        else:                                                  # keep the prediction
            self.pose, self.n_fail = lev, self.n_fail + 1
        moved = np.hypot(self.pose.x - kx, self.pose.y - ky)
        if not ok or moved > self.key_dist or abs(wrap(self.pose.yaw - kyaw)) > self.key_yaw:
            self._set_key(q, self.pose)
        return self.pose, ok

    def _set_key(self, q: np.ndarray, pose: Pose):
        self.key = _Target(voxel_down(q, 0.05))
        self.key_pose = (pose.x, pose.y, pose.yaw)
