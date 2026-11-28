#!/usr/bin/env python3

from __future__ import annotations

import io
import os
import sys
import threading
from http.client import HTTPException
from pathlib import Path

from PIL import Image
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Empty

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from custom.mission_4.room_scores import assign_rooms, request_scores
from custom.navigation.nav import Navigator
from mission_api import FAIL, Mission, MissionResult, run_standalone

# Public doorway geometry in north-to-south order (world metres).
DOOR_Y = (17.5, 10.5, 3.59)


class Mission4(Mission):
    MISSION = 4

    def __init__(self) -> None:
        """Listen for RGB frames and layout resets during exploration."""
        super().__init__()
        self.frame_ready = threading.Condition()
        self.frame = None
        self.respawned = threading.Event()
        self.create_subscription(RosImage, "/camera/color/image_raw", self.on_frame,
                                 qos_profile_sensor_data)
        self.create_subscription(Empty, "/g1/respawned", self.on_respawn, 10)

    def on_frame(self, message):
        """Keep the latest color frame for the next doorway observation."""
        with self.frame_ready:
            self.frame = message
            self.frame_ready.notify_all()

    def on_respawn(self, _message):
        """Interrupt the current survey if the simulator resets the layout."""
        with self.frame_ready:
            self.respawned.set()
            self.frame_ready.notify_all()

    def capture(self):
        """Encode a fresh stationary camera view, respecting the ROS row stride."""
        with self.frame_ready:
            previous = self.frame
            if not self.frame_ready.wait_for(
                lambda: self.frame is not previous or self.respawned.is_set(), timeout=4
            ) or self.respawned.is_set():
                raise ValueError("No fresh doorway frame")
            msg = self.frame
        if msg.encoding != "rgb8":
            raise ValueError(f"Unexpected camera encoding: {msg.encoding}")
        image = Image.frombytes("RGB", (msg.width, msg.height), bytes(msg.data),
                                "raw", "RGB", msg.step, 1)
        jpeg = io.BytesIO()
        image.save(jpeg, format="JPEG", quality=90)
        return jpeg.getvalue()

    def run(self) -> MissionResult:
        """Survey the doors and repeat from scratch after a respawn."""
        self.begin_mission(4)
        nav = Navigator(self, interrupted=self.respawned)
        url = os.environ.get("M4_VLM_URL", "http://127.0.0.1:8044/scores")
        while True:
            self.respawned.clear()
            try:
                return_value = self.survey(nav, url)
            except (ValueError, OSError, HTTPException) as exc:
                return_value = MissionResult(self.MISSION, status=FAIL, detail=str(exc))
            finally:
                self.stop()
            if not self.respawned.is_set():
                return return_value
            self.log("[M4] Respawned. Repeating the doorway survey.")

    def doorway_views(self, nav):
        """Yield the same stationary JPEGs to the mission and practice recorder."""
        # Spawn is beside the middle door. Visit middle, north, then south.
        for y in (DOOR_Y[1], DOOR_Y[0], DOOR_Y[2]):
            if not nav.go_to(17.4, y, tol=0.15, final_yaw=0.0):
                raise ValueError(f"Doorway y={y} unreachable")
            nav.settle()
            yield y, self.capture()

    def survey(self, nav, url):
        """Classify each view, solve the layout and enter the identified classroom."""
        scores = {}
        for y, jpeg in self.doorway_views(nav):
            scores[y] = request_scores(url, jpeg)
            self.log(f"[M4] Door y={y}: {scores[y]}")
        rooms = assign_rooms([scores[y] for y in DOOR_Y])
        y = DOOR_Y[rooms.index("classroom")]
        if not nav.go_to(17.4, y, tol=0.15, final_yaw=0.0) or not nav.go_to(18.85, y, tol=0.15):
            return MissionResult(self.MISSION, status=FAIL, detail="Classroom entry failed",
                                 data={"rooms": rooms})
        nav.settle()
        return self.result(rooms=rooms)


if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission4))
