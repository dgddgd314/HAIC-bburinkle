from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

COLORS = ("green", "blue", "orange")

GOAL_XY = (15.50, 18.00)
CONE_FRONT = (0.0, -1.0)
COURSE_WALL_X = 13.0
COURSE_ENTRANCE_Y = (19.525, 20.725)
COURSE_EXIT_Y = (10.775, 11.975)
EXIT_ZONE = (13.0, 14.2, 10.775, 11.975)
ROOM_X = (18.25, 23.0)
ROOM_HALF_Y = 3.4
BALL_RADIUS = 0.11
PENCIL_HALF_LENGTH = 0.1445
PENCIL_RADIUS = 0.0145
HOLDER_INNER_HALF = 0.0603571
HOLDER_HEIGHT = 0.150
HOLDER_TILT_DEG = 25.0
PENCIL_TIP_DEPTH = 0.02
TELEPORT_M = 0.5

REGION_RGBA = (0.10, 0.85, 0.30, 0.35)
ROOM_RGBA = (0.10, 0.85, 0.30, 0.12)
FLOOR_Z = 0.012


def _lt(value: float, limit: float) -> str:
    return "<" if value <= limit else ">"


def goal_mouth(model, data, goal_body: int, ball_radius: float = BALL_RADIUS) -> Optional[tuple[float, ...]]:
    import mujoco

    subtree = set()
    for b in range(model.nbody):
        a = b
        while a > 0:
            if a == goal_body:
                subtree.add(b)
                break
            a = int(model.body_parentid[a])
    posts, bar, back = [], None, None
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in subtree:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        rot = data.geom_xmat[g].reshape(3, 3)
        aabb = model.geom_aabb[g]
        centre = data.geom_xpos[g] + rot @ aabb[:3]
        half = np.abs(rot) @ aabb[3:]
        low, high = np.minimum(low, centre - half), np.maximum(high, centre + half)
        if name.endswith("post_collision"):
            posts.append((centre, half))
        elif name.endswith("crossbar_collision"):
            bar = (centre, half)
        elif name.endswith("back_net_collision"):
            back = (centre, half)
    if len(posts) != 2 or bar is None or not np.all(np.isfinite(low)):
        return None
    (c1, h1), (c2, h2) = sorted(posts, key=lambda ph: ph[0][0])
    x0, x1 = float(c1[0] + h1[0]), float(c2[0] - h2[0])
    line_y = float((c1[1] + c2[1]) / 2.0)
    net_centre_y = float((low[1] + high[1]) / 2.0)
    if net_centre_y < line_y:
        y1 = line_y - float(h1[1]) - ball_radius
        y0 = float(back[0][1] + back[1][1]) if back is not None else float(low[1])
    else:
        y0 = line_y + float(h1[1]) + ball_radius
        y1 = float(back[0][1] - back[1][1]) if back is not None else float(high[1])
    z1 = float(bar[0][2] - bar[1][2])
    return (x0, x1, y0, y1, z1)


class MissionJudge:
    def __init__(self, model, data, *, g1_prefix: str, m2_color: str = "random",
                 m1_goal: Optional[Sequence[float]] = None,
                 m4_room: str = "classroom", goal_radius: float = 0.5,
                 cone_box: float = 0.5, cone_offset: float = 0.75,
                 room_centers: Optional[Callable[[], dict[str, float]]] = None,
                 soccer=None, rng=None, show_regions: bool = True,
                 origin: Sequence[float] = (0.0, 0.0),
                 regions: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
                 hide_targets: bool = False) -> None:
        import mujoco

        self.model = model
        self.origin = (float(origin[0]), float(origin[1]))
        self.rng = rng or np.random.default_rng()
        self.m4_room = m4_room
        self.goal_radius = float(goal_radius)
        self.cone_box = float(cone_box)
        self.cone_offset = float(cone_offset)
        self.room_centers = room_centers
        self.show_regions = bool(show_regions)
        self.show_random = bool(show_regions)
        self.regions = [(tuple(float(v) for v in box), tuple(float(v) for v in rgba))
                        for box, rgba in regions]
        self.hide_targets = bool(hide_targets)
        self.public: dict[str, object] = {}
        self.public["m4_rooms"] = sorted(self.room_centers() if room_centers else ())
        self.set_m1_goal_world(*(m1_goal if m1_goal else GOAL_XY))
        self.set_m2_color(m2_color if m2_color in COLORS else COLORS[0])

        root = [j for j in range(model.njnt)
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                       int(model.jnt_bodyid[j])) or "").startswith(g1_prefix)]
        if not root:
            raise RuntimeError(f"no G1 free joint under {g1_prefix!r}")
        self._root_qadr = int(model.jnt_qposadr[root[0]])

        def body(fragment: str) -> int:
            ids = [b for b in range(model.nbody)
                   if fragment in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
            return ids[0] if len(ids) == 1 else -1

        self._cones = {c: body(f"extension_cone_{c}/") for c in COLORS}
        self._pencil = body("panda_task_random_pencil")
        self._holder = body("school_room7_empty_desk_pencil_cup")
        self._panda_bodies = {b for b in range(model.nbody)
                              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                                  or "").startswith("panda/")}
        self._soccer = soccer
        self._goal_box: Optional[tuple[float, ...]] = None
        self._course_reset()

    def local(self, xy) -> list[float]:
        return [round(float(xy[0]) - self.origin[0], 4), round(float(xy[1]) - self.origin[1], 4),
                *(float(v) for v in xy[2:])]

    def local_box(self, box) -> list[float]:
        ox, oy = self.origin
        return [round(float(box[0]) - ox, 4), round(float(box[1]) - ox, 4),
                round(float(box[2]) - oy, 4), round(float(box[3]) - oy, 4),
                *(float(v) for v in box[4:])]

    def set_m1_goal_world(self, x: float, y: float, yaw_deg: Optional[float] = None) -> None:
        self.goal_xy = (float(x), float(y))
        self.goal_yaw = yaw_deg
        if self.hide_targets:
            return
        yaw = [] if yaw_deg is None else [float(yaw_deg)]
        self.public["m1_goal_world"] = [self.goal_xy[0], self.goal_xy[1], *yaw]
        self.public["m1_goal"] = self.local(self.goal_xy) + yaw

    def set_m2_color(self, color: str) -> None:
        if color not in COLORS:
            raise ValueError(f"cone colour must be one of {COLORS}, got {color!r}")
        self.m2_color = color
        if not self.hide_targets:
            self.public["m2_color"] = color

    def mission_started(self, index: Optional[int]) -> None:
        self._course_reset()

    def _course_reset(self) -> None:
        self._side: Optional[str] = None
        self._entered_via: Optional[str] = None
        self.course_crossed = False
        self._last_xy: Optional[np.ndarray] = None

    def pelvis(self, data) -> np.ndarray:
        return np.asarray(data.qpos[self._root_qadr:self._root_qadr + 3], float).copy()

    @staticmethod
    def _door(y: float) -> Optional[str]:
        pad = 0.2
        if COURSE_ENTRANCE_Y[0] - pad <= y <= COURSE_ENTRANCE_Y[1] + pad:
            return "entrance"
        if COURSE_EXIT_Y[0] - pad <= y <= COURSE_EXIT_Y[1] + pad:
            return "exit"
        return None

    def track(self, data) -> None:
        xy = np.asarray(data.qpos[self._root_qadr:self._root_qadr + 2], float)
        if not np.all(np.isfinite(xy)):
            return
        if self._last_xy is not None and np.linalg.norm(xy - self._last_xy) > TELEPORT_M:
            self._side = None
        self._last_xy = xy.copy()
        x, y = float(xy[0]), float(xy[1])
        if x < COURSE_WALL_X - 0.1 and 10.5 < y < 21.0:
            side = "in"
        elif x > COURSE_WALL_X + 0.1:
            side = "out"
        else:
            return
        if side == self._side:
            return
        door = self._door(y)
        if self._side == "out" and side == "in":
            self._entered_via = door
        elif self._side == "in" and side == "out":
            if door == "exit" and self._entered_via == "entrance":
                self.course_crossed = True
            self._entered_via = None
        self._side = side

    def cone_box_center(self, data) -> Optional[np.ndarray]:
        body = self._cones.get(self.m2_color, -1)
        if body < 0:
            return None
        return data.xpos[body, :2] + self.cone_offset * np.asarray(CONE_FRONT)

    def room_box(self, room: str) -> Optional[tuple[float, float, float, float]]:
        centers = self.room_centers() if self.room_centers else {}
        if room not in centers:
            return None
        cy = centers[room]
        return (ROOM_X[0], ROOM_X[1], cy - ROOM_HALF_Y, cy + ROOM_HALF_Y)

    def goal_box(self, data) -> Optional[tuple[float, ...]]:
        if self._soccer is None:
            return None
        if self._goal_box is None:
            self._goal_box = goal_mouth(self.model, data, self._soccer.goal_body)
        return self._goal_box

    def judge(self, index: int, data) -> dict:
        p = self.pelvis(data)
        verdict = {"mission": int(index), "pelvis": [round(v, 3) for v in self.local(p)]}
        rule = getattr(self, f"_judge_{index}", None)
        if rule is None:
            success, summary, extra = None, f"no referee for mission {index}", {}
        else:
            success, summary, extra = rule(data, p)
        verdict.update({"success": success, "summary": summary, **extra})
        return verdict

    def _judge_1(self, data, p):
        wx, wy = self.goal_xy
        gx, gy = self.local(self.goal_xy)
        d = math.dist(p[:2], self.goal_xy)
        ok = d <= self.goal_radius
        where = f"world ({wx:.2f}, {wy:.2f})"
        if self.origin != (0.0, 0.0):
            where += f" = map ({gx:.2f}, {gy:.2f})"
        return ok, f"{d:.2f} m {_lt(d, self.goal_radius)} {self.goal_radius:.2f} m from goal {where}", {
            "goal": [gx, gy], "goal_world": [wx, wy], "distance_m": round(d, 3),
            "threshold_m": self.goal_radius}

    def _judge_2(self, data, p):
        centre = self.cone_box_center(data)
        if centre is None:
            return None, f"{self.m2_color} cone not in the scene", {}
        dx, dy = (float(v) for v in p[:2] - centre)
        half = self.cone_box / 2.0
        ok = abs(dx) <= half and abs(dy) <= half
        return ok, (f"{self.m2_color} cone box: |dx| {abs(dx):.2f} {_lt(abs(dx), half)} {half:.2f}, "
                    f"|dy| {abs(dy):.2f} {_lt(abs(dy), half)} {half:.2f} m"), {
            "color": self.m2_color, "box_center": [round(v, 3) for v in self.local(centre)],
            "box_m": self.cone_box, "dx_m": round(dx, 3), "dy_m": round(dy, 3)}

    def _judge_3(self, data, p):
        x0, x1, y0, y1 = EXIT_ZONE
        inside = x0 <= p[0] <= x1 and y0 <= p[1] <= y1
        ok = inside and self.course_crossed
        crossed = ("crossed entrance -> exit" if self.course_crossed
                   else "did not cross the course entrance -> exit")
        where = "inside" if inside else "outside"
        lp, zone = self.local(p), self.local_box(EXIT_ZONE)
        return ok, (f"{crossed}; pelvis ({lp[0]:.2f}, {lp[1]:.2f}) {where} the exit zone "
                    f"x {zone[0]:.2f}..{zone[1]:.2f}, y {zone[2]:.2f}..{zone[3]:.2f}"), {
            "crossed": self.course_crossed, "in_exit_zone": bool(inside), "exit_zone": zone}

    def _judge_4(self, data, p):
        box = self.room_box(self.m4_room)
        if box is None:
            return None, f"room {self.m4_room!r} unknown", {}
        x0, x1, y0, y1 = box
        inside = x0 <= p[0] <= x1 and y0 <= p[1] <= y1
        actual = next((room for room in (self.room_centers() if self.room_centers else {})
                       if (b := self.room_box(room)) and b[0] <= p[0] <= b[1] and b[2] <= p[1] <= b[3]),
                      "no room")
        lp, lbox = self.local(p), self.local_box(box)
        rooms = {room: [round(v, 3) for v in self.local_box(self.room_box(room))]
                 for room in (self.room_centers() if self.room_centers else {})}
        return inside, (f"pelvis ({lp[0]:.2f}, {lp[1]:.2f}) in {actual}; target {self.m4_room} "
                        f"x {lbox[0]:.2f}..{lbox[1]:.2f}, y {lbox[2]:.2f}..{lbox[3]:.2f}"), {
            "room": self.m4_room, "room_box": [round(v, 3) for v in lbox], "pelvis_room": actual,
            "rooms": rooms}

    def _judge_5(self, data, p):
        if self._pencil < 0 or self._holder < 0:
            return None, "no pencil/holder in the scene", {}
        pencil, holder = data.xpos[self._pencil], data.xpos[self._holder]
        axis = data.xmat[self._pencil].reshape(3, 3)[:, 0]
        az = float(axis[2])
        tilt = math.degrees(math.acos(min(1.0, abs(az))))
        rim_z = float(holder[2] + HOLDER_HEIGHT)
        depth = rim_z - float(pencil[2] - PENCIL_HALF_LENGTH * abs(az))
        in_opening = False
        if abs(az) > 1e-6:
            rim = pencil + (rim_z - float(pencil[2])) / az * axis
            local = data.xmat[self._holder].reshape(3, 3).T @ (rim - holder)
            in_opening = bool(np.all(np.abs(local[:2]) <= HOLDER_INNER_HALF - PENCIL_RADIUS))
        held = False
        for c in data.contact[:data.ncon]:
            bodies = {int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])}
            if self._pencil in bodies and bodies & self._panda_bodies:
                held = True
                break
        ok = depth >= PENCIL_TIP_DEPTH and in_opening and not held
        return ok, (f"pencil tip {depth:.3f} {'>' if depth >= PENCIL_TIP_DEPTH else '<'} "
                    f"{PENCIL_TIP_DEPTH:.2f} m below the rim, "
                    f"{'inside' if in_opening else 'outside'} the opening, "
                    f"{'still held by the gripper' if held else 'released'}"), {
            "tilt_deg": round(tilt, 2), "tip_depth_m": round(depth, 4),
            "in_opening": in_opening, "held": held}

    def _judge_6(self, data, p):
        box = self.goal_box(data)
        if box is None or self._soccer is None:
            return None, "no soccer goal in the scene", {}
        ball = data.xpos[self._soccer.ball_body]
        x0, x1, y0, y1, z1 = box
        ok = bool(x0 <= ball[0] <= x1 and y0 <= ball[1] <= y1 and ball[2] <= z1)
        lb, lbox = self.local(ball), self.local_box(box)
        return ok, (f"ball ({lb[0]:.2f}, {lb[1]:.2f}) {'inside' if ok else 'outside'} the goal "
                    f"(centre x {lbox[0]:.2f}..{lbox[1]:.2f}, y {lbox[2]:.2f}..{lbox[3]:.2f}: whole ball "
                    "past the line, between the posts)"), {
            "ball": [round(v, 3) for v in lb], "goal_box": [round(v, 3) for v in lbox]}

    @staticmethod
    def _draw_boxes(scene, boxes) -> None:
        import mujoco

        identity = np.eye(3).flatten()
        for cx, cy, hx, hy, rgba in boxes:
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([hx, hy, 0.004]), np.array([cx, cy, FLOOR_Z]),
                                identity, np.asarray(rgba, np.float32))
            scene.ngeom += 1

    def draw(self, scene, data) -> None:
        import mujoco

        boxes = []
        if self.show_random:
            for (x0, x1, y0, y1), rgba in self.regions:
                boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2, rgba))
        if not self.show_regions:
            scene.ngeom = 0
            self._draw_boxes(scene, boxes)
            return
        cone = self.cone_box_center(data)
        if cone is not None:
            half = self.cone_box / 2.0
            boxes.append((cone[0], cone[1], half, half, REGION_RGBA))
        x0, x1, y0, y1 = EXIT_ZONE
        boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, REGION_RGBA))
        room = self.room_box(self.m4_room)
        if room is not None:
            x0, x1, y0, y1 = room
            boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, ROOM_RGBA))
        goal = self.goal_box(data)
        if goal is not None:
            x0, x1, y0, y1, _ = goal
            boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, REGION_RGBA))
        scene.ngeom = 0
        self._draw_boxes(scene, boxes)
        identity = np.eye(3).flatten()
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                                np.array([self.goal_radius, 0.004, 0.0]),
                                np.array([self.goal_xy[0], self.goal_xy[1], FLOOR_Z]),
                                identity, np.asarray(REGION_RGBA, np.float32))
            scene.ngeom += 1
