#!/usr/bin/env python3

from __future__ import annotations

import sys
import math
import time
import threading
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
from message_filters import Subscriber, TimeSynchronizer  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rclpy.time import Time  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import Empty  # noqa: E402
from tf2_ros import TransformException  # noqa: E402
from custom.mission_2.cone_vision import HUE, cone_points  # noqa: E402
from mission_api import FAIL, Mission, MissionResult, run_standalone  # noqa: E402
from custom.navigation.nav import Navigator  # noqa: E402


class Mission2(Mission):
    MISSION = 2

    def __init__(self) -> None:
        """Subscribe to timestamp-matched camera frames and public respawn events."""
        super().__init__()
        self.frames_ready = threading.Condition()
        self.frames = None
        self.respawned = threading.Event()
        streams = [
            Subscriber(self, msg_type, topic, qos_profile=qos_profile_sensor_data)
            for msg_type, topic in ((Image, "/camera/color/image_raw"),
                                    (Image, "/camera/depth/image_raw"),
                                    (CameraInfo, "/camera/camera_info"))
        ]
        self.camera_sync = TimeSynchronizer(streams, 5)
        self.camera_sync.registerCallback(self.on_frames)
        self.create_subscription(Empty, "/g1/respawned", self.on_respawn, 10)

    def on_frames(self, rgb, depth, info):
        """Wake the observer with a complete RGB-D sample."""
        with self.frames_ready:
            self.frames = (rgb, depth, info)
            self.frames_ready.notify_all()

    def on_respawn(self, _msg):
        """Invalidate the cone target when the simulator shuffles its placement."""
        with self.frames_ready:
            self.respawned.set()
            self.frames = None
            self.frames_ready.notify_all()

    def observe_cone(self, color):
        """Locate a floor-level color component inside the cone room using fresh RGB-D."""
        deadline = time.monotonic() + 4.0
        with self.frames_ready:
            previous = self.frames
        while time.monotonic() < deadline:
            with self.frames_ready:
                fresh = self.frames_ready.wait_for(
                    lambda: self.respawned.is_set() or self.frames is not previous,
                    timeout=deadline - time.monotonic())
                if not fresh or self.respawned.is_set():
                    return None
                previous = self.frames
                rgb, depth, info = previous
            if (rgb.encoding != "rgb8" or depth.encoding != "32FC1"
                    or rgb.header.frame_id != depth.header.frame_id
                    or rgb.header.frame_id != info.header.frame_id
                    or (rgb.width, rgb.height) != (info.width, info.height)
                    or (depth.width, depth.height) != (rgb.width, rgb.height)):
                continue
            pixels = np.ndarray((rgb.height, rgb.width, 3), np.uint8, buffer=rgb.data,
                                strides=(rgb.step, 3, 1))
            distances = np.ndarray((depth.height, depth.width), ">f4" if depth.is_bigendian else "<f4",
                                   buffer=depth.data, strides=(depth.step, 4))
            try:
                tf = self._tf().lookup_transform("map", rgb.header.frame_id,
                         Time.from_msg(rgb.header.stamp), timeout=Duration(seconds=0.4)).transform
            except TransformException:
                continue
            q, t = tf.rotation, tf.translation
            rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
            for point in cone_points(pixels, distances, info.k, color):
                x, y, z = rotation.apply(point) + [t.x, t.y, t.z]
                if 13 < x < 18 and 21 < y < 26 and 0 <= z <= 1:
                    return float(x), float(y)
        return None

    def run(self) -> MissionResult:
        """Observe the requested cone at the doorway and stop on its door-facing side."""
        self.begin_mission(2)
        color = self.params.get("color") or self.m2_color
        if color not in HUE:
            return MissionResult(self.MISSION, status=FAIL, detail="Missing or invalid cone color")
        nav = Navigator(self, interrupted=self.respawned)
        while True:
            self.respawned.clear()
            result = self.approach_cone(nav, color)
            if not self.respawned.is_set():
                return result
            self.stop()
            self.log("[mission 2] respawned; returning to observe the new placement")

    def approach_cone(self, nav, color):
        """Complete one approach using only observations from this placement."""
        if not nav.go_to(15.5, 21.0, tol=0.15, final_yaw=math.pi / 2):
            self.stop()
            return MissionResult(self.MISSION, status=FAIL, detail="Could not reach cone doorway")
        nav.settle()
        cone = self.observe_cone(color)
        for heading in (65, 115):
            if cone is not None:
                break
            if not nav.turn_to(math.radians(heading)):
                return MissionResult(self.MISSION, status=FAIL, detail="Cone search turn failed")
            nav.settle()
            cone = self.observe_cone(color)
        if cone is None:
            return MissionResult(self.MISSION, status=FAIL, detail=f"No RGB-D detection for {color} cone")
        x, y = cone[0], cone[1] - 0.75
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
