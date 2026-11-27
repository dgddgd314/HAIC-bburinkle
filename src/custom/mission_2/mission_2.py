#!/usr/bin/env python3

from __future__ import annotations

import sys
import math
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometry_msgs.msg import PointStamped  # noqa: E402
from mission_api import FAIL, Mission, MissionResult, run_standalone  # noqa: E402
from custom.navigation.nav import Navigator  # noqa: E402


class Mission2(Mission):
    MISSION = 2

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> MissionResult:
        """Observe the requested cone at the doorway and stop on its door-facing side."""
        self.begin_mission(2)
        color = self.params.get("color") or self.m2_color
        if color not in ("blue", "green", "orange"):
            return MissionResult(self.MISSION, status=FAIL, detail="Missing or invalid cone color")
        nav = Navigator(self)
        if not nav.go_to(15.5, 21.0, tol=0.15, final_yaw=math.pi / 2):
            self.stop()
            return MissionResult(self.MISSION, status=FAIL, detail="Could not reach cone doorway")
        nav.settle()
        observation = self.latest(f"/perception/cones/{color}", PointStamped, timeout=5.0)
        if observation is None or observation.header.frame_id != "map":
            return MissionResult(self.MISSION, status=FAIL, detail="No cone observation in map frame")
        x, y = observation.point.x, observation.point.y - 0.75
        if not (math.isfinite(x) and math.isfinite(y)):
            return MissionResult(self.MISSION, status=FAIL, detail="Invalid cone position")
        if not nav.go_to(x, y, tol=0.15):
            self.stop()
            return MissionResult(self.MISSION, status=FAIL, detail="Cone approach failed")
        nav.settle()
        pose = nav.pose()
        if pose is None or abs(pose[0] - x) > 0.25 or abs(pose[1] - y) > 0.25:
            return MissionResult(self.MISSION, status=FAIL, detail="Outside cone approach box")
        return self.result()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission2))
