from __future__ import annotations

import datetime
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

COLORS = ("green", "blue", "orange")
ROOMS = ("classroom", "medical", "cafeteria")
DEFAULT_M1_GOAL = (15.5, 18.0, 90.0)
DEFAULT_M2_COLOR = "green"
DEFAULT_CONE_ORDER = ("blue", "green", "orange")
DEFAULT_ROOM_ORDER = ("classroom", "medical", "cafeteria")
DEFAULT_BALL_XY = (9.25, 8.15)
DEFAULT_M1_GOAL_REGION = (14.0, 17.0, 1.0, 20.0)
DEFAULT_M1_GOAL_YAW_DEG = 90.0
DEFAULT_BALL_REGION = (7.25, 11.25, 5.5, 8.5)


@dataclass
class Scenario:
    m1_goal: list[float] = field(default_factory=lambda: list(DEFAULT_M1_GOAL))
    m2_color: str = DEFAULT_M2_COLOR
    cone_order: list[str] = field(default_factory=lambda: list(DEFAULT_CONE_ORDER))
    room_order: list[str] = field(default_factory=lambda: list(DEFAULT_ROOM_ORDER))
    ball_xy: list[float] = field(default_factory=lambda: list(DEFAULT_BALL_XY))
    panda_task: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "scenario") -> "Scenario":
        env = cls()
        if not isinstance(data, dict):
            raise ValueError(f"{source}: expected a mapping")
        if "m1_goal" in data:
            goal = [float(v) for v in data["m1_goal"]]
            if len(goal) == 2:
                goal.append(DEFAULT_M1_GOAL[2])
            if len(goal) != 3:
                raise ValueError(f"{source}: m1_goal must be [x, y, yaw_deg]")
            env.m1_goal = goal
        if "m2_color" in data:
            color = str(data["m2_color"]).strip().lower()
            if color not in COLORS:
                raise ValueError(f"{source}: m2_color must be one of {COLORS}, got {color!r}")
            env.m2_color = color
        if "cone_order" in data:
            order = [str(c).strip().lower() for c in data["cone_order"]]
            if sorted(order) != sorted(COLORS):
                raise ValueError(f"{source}: cone_order must list {COLORS} once each")
            env.cone_order = order
        if "room_order" in data:
            order = [str(r).strip().lower() for r in data["room_order"]]
            if sorted(order) != sorted(ROOMS):
                raise ValueError(f"{source}: room_order must list {ROOMS} once each")
            env.room_order = order
        if "ball_xy" in data:
            ball = [float(v) for v in data["ball_xy"]]
            if len(ball) != 2:
                raise ValueError(f"{source}: ball_xy must be [x, y]")
            env.ball_xy = ball
        task = data.get("panda_task")
        if task:
            env.panda_task = {
                "pencil_xy": [float(v) for v in task["pencil_xy"]],
                "pencil_yaw_deg": float(task.get("pencil_yaw_deg", 0.0)),
                "cup_xy": [float(v) for v in task["cup_xy"]],
                "cup_yaw_deg": float(task.get("cup_yaw_deg", 0.0)),
            }
        return env

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        import yaml

        path = Path(path)
        return cls.from_dict(yaml.safe_load(path.read_text()) or {}, str(path))

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["m1_goal"] = [round(float(v), 4) for v in self.m1_goal]
        data["ball_xy"] = [round(float(v), 4) for v in self.ball_xy]
        if self.panda_task:
            data["panda_task"] = {
                "pencil_xy": [round(float(v), 4) for v in self.panda_task["pencil_xy"]],
                "pencil_yaw_deg": round(float(self.panda_task["pencil_yaw_deg"]), 2),
                "cup_xy": [round(float(v), 4) for v in self.panda_task["cup_xy"]],
                "cup_yaw_deg": round(float(self.panda_task["cup_yaw_deg"]), 2),
            }
        return data

    def save(self, path: str | Path, note: str = "") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.as_dict()
        task = data["panda_task"]
        lines = [
            "# val_861 mission scenario -- the organiser's; loaded by the simulator",
            "# (main_config.yaml scenario.file) so every team gets the same scene.",
            f"# written {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}"
            + (f": {note}" if note else ""),
            "",
            f"m1_goal: {_flow(data['m1_goal'])}        # mission 1 goal: world x, y, yaw_deg",
            f"m2_color: {data['m2_color']}             # mission 2 target cone: green | blue | orange",
            f"cone_order: {_flow(data['cone_order'])}   # cone colours left -> right (x 14.0, 15.5, 17.0)",
            f"room_order: {_flow(data['room_order'])}   # rooms north -> south (y 17.5, 10.5, 3.5)",
            f"ball_xy: {_flow(data['ball_xy'])}          # soccer ball (the white ring follows it)",
        ]
        if task is None:
            lines.append("panda_task: null               # pencil / holder: as the scene builder draws them")
        else:
            lines += [
                "panda_task:                    # mission 5: pencil and holder on the desk (world frame)",
                f"  pencil_xy: {_flow(task['pencil_xy'])}",
                f"  pencil_yaw_deg: {task['pencil_yaw_deg']}",
                f"  cup_xy: {_flow(task['cup_xy'])}",
                f"  cup_yaw_deg: {task['cup_yaw_deg']}",
            ]
        path.write_text("\n".join(lines) + "\n")
        return path

    def summary(self) -> str:
        goal = ", ".join(f"{v:.2f}" for v in self.m1_goal[:2])
        return (f"m1 goal ({goal}), m2 {self.m2_color}, cones {' | '.join(self.cone_order)}, "
                f"rooms {' | '.join(self.room_order)}, ball ({self.ball_xy[0]:.2f}, {self.ball_xy[1]:.2f})")


DEFAULT = Scenario()


def _flow(values: Sequence[Any]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def sample_point(rng: np.random.Generator, region: Sequence[float]) -> tuple[float, float]:
    x0, x1, y0, y1 = (float(v) for v in region)
    return float(rng.uniform(min(x0, x1), max(x0, x1))), float(rng.uniform(min(y0, y1), max(y0, y1)))


def quat_yaw_deg(quat: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
