# Geometry routines extracted from HAIC layout_loader.py, commit 052a00be32fdd818550c3b2d176b8c1c1db13b7d.
# Appearance and MuJoCo dependencies replaced by a geometry-only collector.
from __future__ import annotations
import math
import re
from types import SimpleNamespace
import numpy as np
BASE_WALL_THICKNESS = 0.005
DOOR_FRAME_THICKNESS = 0.012
mujoco = SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_BOX="box"))
def _wall_finish_appearance(*args):
    return [1, 1, 1, 1], ""
class Body:
    def __init__(self, scene, name): self.scene, self.name = scene, name
    def add_geom(self, **kwargs): self.scene.boxes.append(kwargs)
class Scene:
    def __init__(self): self.boxes=[]; self.worldbody=self
    def add_body(self, name): return Body(self, name)


def yaw_quaternion(yaw_degrees: float) -> list[float]:
    half = math.radians(yaw_degrees) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]

def _segment_key(segment: list[list[float]], precision: int = 4) -> tuple:
    points = [tuple(round(float(value), precision) for value in point) for point in segment]
    return tuple(sorted(points))

def _base_wall_groups(description: dict) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for index, wall in enumerate(description["walls"]):
        key = _segment_key(wall["segment"])
        group = grouped.setdefault(
            key,
            {
                "id": f"base_wall_{len(grouped):03d}",
                "kind": "base_wall",
                "start": list(key[0]),
                "end": list(key[1]),
                "source_ids": [],
                "body_names": [],
            },
        )
        group["source_ids"].append(wall["id"])
        room_number = int(re.search(r"\d+", wall["roomId"]).group())
        group["body_names"].append(f"wall_{room_number}_{index}")
    return list(grouped.values())

def _box_along_segment(
    body: mujoco.MjsBody,
    *,
    name: str,
    start: list[float],
    end: list[float],
    z_bottom: float,
    height: float,
    thickness: float,
    rgba: list[float],
    material: str = "",
) -> None:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    delta = end_array - start_array
    length = float(np.linalg.norm(delta))
    if length < 1e-4 or height < 1e-4:
        return
    midpoint = (start_array + end_array) / 2.0
    yaw = math.degrees(math.atan2(delta[1], delta[0]))
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[float(midpoint[0]), float(midpoint[1]), z_bottom + height / 2.0],
        quat=yaw_quaternion(yaw),
        size=[length / 2.0, thickness / 2.0, height / 2.0],
        rgba=rgba,
        material=material,
        contype=1,
        conaffinity=1,
        friction=[0.9, 0.01, 0.001],
        group=0,
    )

def _add_visual_wall_finish_panel(
    body: mujoco.MjsBody,
    *,
    name: str,
    start: list[float],
    end: list[float],
    height: float,
    thickness: float,
    rgba: list[float],
    material: str = "",
) -> None:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    delta = end_array - start_array
    length = float(np.linalg.norm(delta))
    if length < 1e-4 or height < 1e-4:
        return
    midpoint = (start_array + end_array) / 2.0
    yaw = math.degrees(math.atan2(delta[1], delta[0]))
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[float(midpoint[0]), float(midpoint[1]), height / 2.0],
        quat=yaw_quaternion(yaw),
        size=[length / 2.0, thickness / 2.0, height / 2.0],
        rgba=rgba,
        material=material,
        contype=0,
        conaffinity=0,
        group=0,
    )

def _door_interval_on_wall(wall: dict, door: dict) -> tuple[float, float] | None:
    wall_start = np.asarray(wall["start"], dtype=float)
    wall_end = np.asarray(wall["end"], dtype=float)
    delta = wall_end - wall_start
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return None
    direction = delta / length
    normal = np.array([-direction[1], direction[0]])
    door_points = [np.asarray(door["start"], dtype=float), np.asarray(door["end"], dtype=float)]
    if max(abs(float(np.dot(point - wall_start, normal))) for point in door_points) > 0.18:
        return None
    values = sorted(float(np.dot(point - wall_start, direction)) for point in door_points)
    if values[1] < -0.02 or values[0] > length + 0.02:
        return None
    return max(0.0, values[0]), min(length, values[1])

def _add_wall(
    scene_spec: mujoco.MjSpec,
    wall: dict,
    doors: list[dict],
    body_name: str,
    finish: dict | None = None,
) -> None:
    body = scene_spec.worldbody.add_body(name=body_name)
    start = np.asarray(wall["start"], dtype=float)
    end = np.asarray(wall["end"], dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-4:
        return
    direction = delta / length
    height = float(wall.get("height", 3.2))
    z_bottom = float(wall.get("z_bottom", 0.0))
    thickness = float(wall.get("thickness", BASE_WALL_THICKNESS))
    color, material = _wall_finish_appearance(
        scene_spec, wall["id"], finish
    )
    openings = []
    if not wall.get("ignore_doors", False):
        for door in doors:
            interval = _door_interval_on_wall(wall, door)
            if interval and interval[1] - interval[0] > 0.2:
                openings.append((interval[0], interval[1], float(door.get("height", 2.1))))
    openings.sort()
    cursor = 0.0
    piece_index = 0
    for opening_start, opening_end, door_height in openings:
        if opening_start > cursor + 1e-4:
            _box_along_segment(
                body,
                name=f"{body_name}_solid_{piece_index}",
                start=(start + direction * cursor).tolist(),
                end=(start + direction * opening_start).tolist(),
                z_bottom=z_bottom,
                height=height,
                thickness=thickness,
                rgba=color,
                material=material,
            )
            piece_index += 1
        if door_height < height:
            _box_along_segment(
                body,
                name=f"{body_name}_header_{piece_index}",
                start=(start + direction * opening_start).tolist(),
                end=(start + direction * opening_end).tolist(),
                z_bottom=z_bottom + door_height,
                height=height - door_height,
                thickness=thickness,
                rgba=color,
                material=material,
            )
            piece_index += 1
        cursor = max(cursor, opening_end)
    if cursor < length - 1e-4:
        _box_along_segment(
            body,
            name=f"{body_name}_solid_{piece_index}",
            start=(start + direction * cursor).tolist(),
            end=end.tolist(),
            z_bottom=z_bottom,
            height=height,
            thickness=thickness,
            rgba=color,
            material=material,
        )

def _add_open_door_frame(
    scene_spec: mujoco.MjSpec,
    door: dict,
    body_name: str,
) -> None:
    body = scene_spec.worldbody.add_body(name=body_name)
    start = np.asarray(door["start"], dtype=float)
    end = np.asarray(door["end"], dtype=float)
    delta = end - start
    width = float(np.linalg.norm(delta))
    if width < 0.2:
        return
    direction = delta / width
    height = float(door.get("height", 2.1))
    thickness = float(door.get("thickness", DOOR_FRAME_THICKNESS))
    jamb = min(0.07, width * 0.08)
    color = [0.30, 0.22, 0.15, 1.0]
    _box_along_segment(
        body,
        name=f"{body_name}_jamb_a",
        start=(start - direction * jamb).tolist(),
        end=start.tolist(),
        z_bottom=0.0,
        height=height,
        thickness=thickness * 1.35,
        rgba=color,
    )
    _box_along_segment(
        body,
        name=f"{body_name}_jamb_b",
        start=end.tolist(),
        end=(end + direction * jamb).tolist(),
        z_bottom=0.0,
        height=height,
        thickness=thickness * 1.35,
        rgba=color,
    )
    _box_along_segment(
        body,
        name=f"{body_name}_lintel",
        start=(start - direction * jamb).tolist(),
        end=(end + direction * jamb).tolist(),
        z_bottom=height,
        height=0.09,
        thickness=thickness * 1.35,
        rgba=color,
    )

def _base_door_record(door: dict) -> dict:
    height = float(door["holePolygon"][1]["y"] - door["holePolygon"][0]["y"])
    return {
        "id": door["id"],
        "kind": "base_door",
        "start": list(map(float, door["doorSegment"][0])),
        "end": list(map(float, door["doorSegment"][1])),
        "height": height,
        "thickness": DOOR_FRAME_THICKNESS,
    }

def _realign_custom_walls(layout: dict, description: dict) -> bool:
    walls = [wall for wall in layout["structure"]["walls"] if wall.get("active", True)]
    if not walls:
        return False
    before = [(tuple(wall["start"]), tuple(wall["end"])) for wall in walls]
    base_endpoints = [
        np.asarray(point, dtype=float)
        for group in _base_wall_groups(description)
        for point in (group["start"], group["end"])
    ]
    references = [(wall, key) for wall in walls for key in ("start", "end")]
    anchored: dict[tuple[int, str], np.ndarray] = {}
    for wall, key in references:
        point = np.asarray(wall[key], dtype=float)
        closest = min(base_endpoints, key=lambda value: float(np.linalg.norm(value - point)))
        if float(np.linalg.norm(closest - point)) <= 0.20:
            wall[key] = closest.copy().tolist()
            anchored[(id(wall), key)] = closest.copy()

    for _ in range(4):
        unused = set(range(len(references)))
        while unused:
            seed = unused.pop()
            cluster = {seed}
            grew = True
            while grew:
                grew = False
                for candidate in list(unused):
                    candidate_point = np.asarray(references[candidate][0][references[candidate][1]])
                    if any(
                        float(
                            np.linalg.norm(
                                candidate_point
                                - np.asarray(references[index][0][references[index][1]])
                            )
                        )
                        <= 0.20
                        for index in cluster
                    ):
                        cluster.add(candidate)
                        unused.remove(candidate)
                        grew = True
            anchored_points = [
                anchored[(id(references[index][0]), references[index][1])]
                for index in cluster
                if (id(references[index][0]), references[index][1]) in anchored
            ]
            if anchored_points:
                target = anchored_points[0]
            else:
                target = np.mean(
                    [np.asarray(references[index][0][references[index][1]]) for index in cluster],
                    axis=0,
                )
            for index in cluster:
                wall, key = references[index]
                wall[key] = target.round(4).tolist()

        for wall in walls:
            start = np.asarray(wall["start"], dtype=float)
            end = np.asarray(wall["end"], dtype=float)
            delta = end - start
            if float(np.linalg.norm(delta)) < 1e-5:
                continue
            if abs(delta[0]) >= abs(delta[1]) * 4.5:
                y = start[1] if (id(wall), "start") in anchored else end[1] if (id(wall), "end") in anchored else float((start[1] + end[1]) / 2.0)
                wall["start"][1] = wall["end"][1] = round(float(y), 4)
                if (id(wall), "start") in anchored:
                    anchored[(id(wall), "end")] = np.asarray(wall["end"], dtype=float)
                elif (id(wall), "end") in anchored:
                    anchored[(id(wall), "start")] = np.asarray(wall["start"], dtype=float)
            elif abs(delta[1]) >= abs(delta[0]) * 4.5:
                x = start[0] if (id(wall), "start") in anchored else end[0] if (id(wall), "end") in anchored else float((start[0] + end[0]) / 2.0)
                wall["start"][0] = wall["end"][0] = round(float(x), 4)
                if (id(wall), "start") in anchored:
                    anchored[(id(wall), "end")] = np.asarray(wall["end"], dtype=float)
                elif (id(wall), "end") in anchored:
                    anchored[(id(wall), "start")] = np.asarray(wall["start"], dtype=float)

    after = [(tuple(wall["start"]), tuple(wall["end"])) for wall in walls]
    return before != after
