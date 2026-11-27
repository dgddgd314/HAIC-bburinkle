#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import math  # noqa: E402
import time  # noqa: E402

from mission_api import Mission, MissionResult, run_standalone  # noqa: E402
from custom.navigation.nav import Navigator  # noqa: E402

GOAL_TOL = 0.15        # stop this close (success radius is 0.5 m)
RETRY_TOL = 0.30       # after settling, walk again if drifted farther than this


class Mission1(Mission):
    MISSION = 1

    def __init__(self) -> None:
        super().__init__()
        self.nav = Navigator(self)

    def run(self) -> MissionResult:
        start = self.begin_mission(1)
        goal = self.params.get("goal") or self.m1_goal
        self.log(f"[mission 1] goal {goal}, start {start.get('pos')}")
        if not goal:
            self.log("[mission 1] no goal given")
            return self.result()
        gx, gy = float(goal[0]), float(goal[1])

        t0 = time.monotonic()
        for attempt in range(3):
            self.nav.go_to(gx, gy, tol=GOAL_TOL, timeout=150.0)
            self.nav.settle(1.5)                     # come to rest before the referee judges
            p = self.nav.pose()
            if p is None:
                break
            d = math.hypot(gx - p[0], gy - p[1])
            self.log(f"[mission 1] attempt {attempt + 1}: at ({p[0]:.2f}, {p[1]:.2f}), "
                     f"{d:.2f} m from goal, {time.monotonic() - t0:.0f} s")
            if d <= RETRY_TOL:
                break
        self.stop()
        return self.result()


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission1))
