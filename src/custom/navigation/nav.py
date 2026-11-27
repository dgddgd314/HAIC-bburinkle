"""Grid path planning + path following for G1 on the structural map.

    from custom.navigation.nav import Navigator
    nav = Navigator(mission)                 # mission: a mission_api.Mission
    ok = nav.go_to(x, y, tol=0.2)            # plans, walks, stops; True when within tol

Planning: A* on architecture_nav_safe (0.25 m margin) at 10 cm, 8-connected, plus a cost
for walking near walls (clearance_m), then line-of-sight shortcutting.
Following: turn toward a lookahead point, walk forward when roughly facing it.
"""
from __future__ import annotations

import heapq
import math
import time
from pathlib import Path

import numpy as np

MAP_DIR = Path(__file__).resolve().parents[1] / "maps_structural"
RES0, OX, OY = 0.02, 5.0, -0.5            # source grid: 2 cm, lower-left world (5.0, -0.5)


class GridMap:
    def __init__(self, cell: float = 0.10, extra_margin: float = 0.0):
        safe = np.load(MAP_DIR / "architecture_nav_safe.npy")
        clear = np.load(MAP_DIR / "clearance_m.npy")
        k = int(round(cell / RES0))
        h, w = safe.shape[0] // k, safe.shape[1] // k
        blk = (safe[:h * k, :w * k] != 0).reshape(h, k, w, k).any(axis=(1, 3))
        cl = clear[:h * k, :w * k].reshape(h, k, w, k).min(axis=(1, 3))
        if extra_margin > 0:
            blk |= cl < 0.25 + extra_margin
        self.blocked, self.clear, self.cell = blk, cl, cell

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int((y - OY) // self.cell), int((x - OX) // self.cell)

    def to_world(self, r: int, c: int) -> tuple[float, float]:
        return OX + (c + 0.5) * self.cell, OY + (r + 0.5) * self.cell

    def inside(self, r, c) -> bool:
        return 0 <= r < self.blocked.shape[0] and 0 <= c < self.blocked.shape[1]

    def free(self, r, c) -> bool:
        return self.inside(r, c) and not self.blocked[r, c]

    def nearest_free(self, r, c, max_cells: int = 15):
        if self.free(r, c):
            return r, c
        best = None
        for d in range(1, max_cells + 1):
            for dr in range(-d, d + 1):
                for dc in range(-d, d + 1):
                    if max(abs(dr), abs(dc)) == d and self.free(r + dr, c + dc):
                        dist = math.hypot(dr, dc)
                        if best is None or dist < best[0]:
                            best = (dist, r + dr, c + dc)
            if best:
                return best[1], best[2]
        return None

    def plan(self, start_xy, goal_xy, wall_weight: float = 0.6):
        """World waypoints from start to goal, or None."""
        s = self.nearest_free(*self.to_cell(*start_xy))
        g = self.nearest_free(*self.to_cell(*goal_xy))
        if s is None or g is None:
            return None
        nbr = [(dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        # stepping costs more where clearance is small (prefer the middle of corridors/doors)
        pen = 1.0 + wall_weight * np.clip(1.0 - self.clear / 1.0, 0, 1)
        dist = {s: 0.0}
        prev = {}
        pq = [(0.0, s)]
        while pq:
            f, cur = heapq.heappop(pq)
            if cur == g:
                break
            d0 = dist[cur]
            if f - math.hypot(cur[0] - g[0], cur[1] - g[1]) > d0 + 1e-9:
                continue
            for dr, dc, step in nbr:
                n = (cur[0] + dr, cur[1] + dc)
                if not self.free(*n):
                    continue
                if dr and dc and not (self.free(cur[0] + dr, cur[1]) and self.free(cur[0], cur[1] + dc)):
                    continue                              # no corner cutting
                nd = d0 + step * pen[n]
                if nd < dist.get(n, math.inf):
                    dist[n], prev[n] = nd, cur
                    heapq.heappush(pq, (nd + math.hypot(n[0] - g[0], n[1] - g[1]), n))
        if g not in dist:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(prev[cells[-1]])
        cells.reverse()
        cells = self._shortcut(cells)
        pts = [self.to_world(*c) for c in cells]
        pts[-1] = tuple(goal_xy) if self.free(*self.to_cell(*goal_xy)) else pts[-1]
        return pts

    def _line_free(self, a, b, min_clear: float = 0.4) -> bool:
        """Straight segment stays on free cells and keeps min_clear from walls
        (relaxed to the endpoints' own clearance when they are closer than that)."""
        need = min(min_clear, self.clear[a] - 1e-3, self.clear[b] - 1e-3)
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        for t in np.linspace(0, 1, n + 1):
            r, c = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            for rr, cc in ((math.floor(r), math.floor(c)), (math.ceil(r), math.ceil(c))):
                if not self.free(rr, cc) or self.clear[rr, cc] < need:
                    return False
        return True

    def _shortcut(self, cells):
        out, i = [cells[0]], 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._line_free(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return out


class Navigator:
    """Walks a mission's G1 along planned paths using self.pose() and self.cmd_vel()."""

    def __init__(self, mission, grid: GridMap | None = None, *, v_max: float = 0.5,
                 w_max: float = 0.7, lookahead: float = 0.6, log=None):
        self.m = mission
        self.grid = grid or GridMap()
        self.v_max, self.w_max, self.look = v_max, w_max, lookahead
        self.log = log or getattr(mission, "log", print)

    def wait_localized(self, timeout: float = 20.0) -> bool:
        """Block until /localization/status says ok and not searching (e.g. right after
        a respawn the TF can still hold the pre-teleport pose for a moment)."""
        import json
        from std_msgs.msg import String
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            msg = self.m.latest("/localization/status", String, timeout=1.0)
            if msg is not None:
                try:
                    s = json.loads(msg.data)
                except ValueError:
                    s = {}
                if s.get("ok") and not s.get("searching"):
                    return True
            self.m.stop()
            time.sleep(0.1)
        self.log("[nav] localization not ready, continuing anyway")
        return False

    def pose(self, wait: float = 2.0):
        end = time.monotonic() + wait
        while time.monotonic() < end:
            p = self.m.pose(timeout=0.2)
            if p is not None:
                return p
            time.sleep(0.05)
        return None

    def go_to(self, x: float, y: float, tol: float = 0.2, timeout: float = 120.0,
              final_yaw: float | None = None) -> bool:
        self.wait_localized()
        p = self.pose(5.0)
        if p is None:
            self.log("[nav] no pose (map -> pelvis TF missing)")
            return False
        path = self.grid.plan((p[0], p[1]), (x, y))
        if path is None:
            self.log(f"[nav] no path from ({p[0]:.2f}, {p[1]:.2f}) to ({x:.2f}, {y:.2f})")
            return False
        self.log(f"[nav] path {len(path)} pts: " + " -> ".join(f"({a:.1f},{b:.1f})" for a, b in path))
        ok = self.follow(path, tol, timeout)
        if ok and final_yaw is not None:
            ok = self.turn_to(final_yaw)
        return ok

    def follow(self, path, tol: float, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        seg = 0
        gx, gy = path[-1]
        lost_since = None
        try:
            while time.monotonic() < deadline:
                p = self.m.pose(timeout=0.2)
                if p is None:
                    self.m.stop()
                    lost_since = lost_since or time.monotonic()
                    if time.monotonic() - lost_since > 5.0:
                        self.log("[nav] pose lost for 5 s")
                        return False
                    continue
                lost_since = None
                x, y, yaw = p
                dg = math.hypot(gx - x, gy - y)
                if dg < tol:
                    return True
                # advance along the path: lookahead point on the remaining polyline
                while seg < len(path) - 2 and math.hypot(path[seg + 1][0] - x, path[seg + 1][1] - y) < self.look:
                    seg += 1
                tx, ty = self._lookahead(path, seg, x, y)
                heading = math.atan2(ty - y, tx - x)
                err = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
                wz = max(-self.w_max, min(self.w_max, 1.5 * err))
                # forward speed: slow when facing away or close to the goal
                v = self.v_max * max(0.0, math.cos(err)) ** 3 * min(1.0, dg / 0.8)
                v = max(v, 0.08) if abs(err) < 0.35 else (0.0 if abs(err) > 0.8 else v)
                # small lateral correction near the goal (G1 can strafe)
                vy = 0.0
                if dg < 0.6:
                    ly = -math.sin(yaw) * (gx - x) + math.cos(yaw) * (gy - y)
                    vy = max(-0.2, min(0.2, 0.8 * ly))
                self.m.cmd_vel(v, vy, wz)
                time.sleep(0.05)
            self.log("[nav] timeout")
            return False
        finally:
            self.m.stop()

    def _lookahead(self, path, seg, x, y):
        for i in range(seg, len(path) - 1):
            ax, ay = path[i]
            bx, by = path[i + 1]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 < 1e-9:
                continue
            # farthest point on this segment within the lookahead circle
            fx, fy = ax - x, ay - y
            b = 2 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - self.look ** 2
            disc = b * b - 4 * L2 * c
            if disc >= 0:
                t = (-b + math.sqrt(disc)) / (2 * L2)
                if 0 <= t <= 1:
                    return ax + t * dx, ay + t * dy
        return path[-1]

    def turn_to(self, yaw_goal: float, tol_deg: float = 5.0, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                p = self.m.pose(timeout=0.2)
                if p is None:
                    continue
                err = math.atan2(math.sin(yaw_goal - p[2]), math.cos(yaw_goal - p[2]))
                if abs(math.degrees(err)) < tol_deg:
                    return True
                self.m.cmd_vel(0.0, 0.0, max(-self.w_max, min(self.w_max, 1.5 * err)))
                time.sleep(0.05)
            return False
        finally:
            self.m.stop()

    def settle(self, seconds: float = 1.0):
        """Stand still so the walking policy comes to rest before the referee judges."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.m.stop()
            time.sleep(0.1)
