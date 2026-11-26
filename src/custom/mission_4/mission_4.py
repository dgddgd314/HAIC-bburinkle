#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mission_api import Mission, MissionResult, run_standalone  # noqa: E402


class Mission4(Mission):
    MISSION = 4

    def __init__(self) -> None:
        super().__init__()

    def run(self) -> MissionResult:
        self.begin_mission(4)
        # TODO: Identify rooms north-to-south and navigate into the classroom.
        answer = []
        return self.result(rooms=answer)


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission4))
