from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Set ROS_LOCALHOST_ONLY before rclpy.init(); all nodes must match.
os.environ["ROS_LOCALHOST_ONLY"] = os.environ.get("HAIC_LOCALHOST_ONLY", "1")

from rclpy.node import Node  # noqa: E402

TOPICS = {
    "mission": "/g1/mission",
    "mission_begin": "/g1/mission_begin",
    "respawn": "/g1/respawn",
    "randomize": "/g1/randomize",
    "cmd_vel": "/cmd_vel",
    "mission_status": "/g1/mission_status",
    "mission_end": "/g1/mission_end",
}

MISSION_LABELS = {
    1: "Point-goal navigation",
    2: "Cone colour navigation",
    3: "Cliff crossing",
    4: "Semantic delivery",
    5: "Pencil pick-and-place (Panda)",
    6: "Soccer goal scoring",
}

DONE = "done"
SUCCESS = "success"
FAIL = "fail"
ERROR = "error"
NOT_IMPLEMENTED = "not_implemented"

ACK_TIMEOUT = 60.0
M4_ROOMS = ("cafeteria", "classroom", "medical")
START_POSE_WORLD = (16.0, 11.0, 0.0)
GAP_FALL_SOURCE = "fell into a gap"
GAP_FALLS_ALLOWED = 2


@dataclass
class MissionResult:
    mission: int
    status: str = DONE
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    judge: Optional[dict[str, Any]] = None

    @property
    def label(self) -> str:
        return MISSION_LABELS.get(self.mission, f"mission {self.mission}")

    def as_dict(self) -> dict[str, Any]:
        return {"mission": self.mission, "label": self.label, "status": self.status,
                "detail": self.detail, "seconds": round(self.seconds, 2), "data": self.data,
                "judge": self.judge}


class MissionTimeout(TimeoutError):
    pass


class MissionAborted(BaseException):
    """BaseException prevents mission-level except Exception from swallowing an abort."""


class Mission(Node):
    """run() executes beside background ROS callbacks; the runner owns the node lifecycle."""

    MISSION: int = 0

    # DDS discovery can lose initial commands; wait for a match and retry until acknowledged.
    MATCH_TIMEOUT_S = 5.0
    REPUBLISH_S = 3.0

    def __init__(self, node_name: Optional[str] = None) -> None:
        from geometry_msgs.msg import Twist
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Empty, Int32, String

        super().__init__(node_name or (f"mission_{self.MISSION}" if self.MISSION else "mission"))
        self._api_group = ReentrantCallbackGroup()
        self._Empty, self._Int32, self._Twist = Empty, Int32, Twist
        self._pub = {
            "mission_begin": self.create_publisher(Int32, TOPICS["mission_begin"], 10),
            "respawn": self.create_publisher(Empty, TOPICS["respawn"], 10),
            "randomize": self.create_publisher(Empty, TOPICS["randomize"], 10),
            "cmd_vel": self.create_publisher(Twist, TOPICS["cmd_vel"], 10),
            "mission_end": self.create_publisher(Int32, TOPICS["mission_end"], 10),
        }
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._cv = threading.Condition()
        self._status: Optional[dict[str, Any]] = None
        self._status_seq = -1
        self.gap_falls = 0
        self._latest_subs: dict[str, Any] = {}
        self._latest_msgs: dict[str, Any] = {}
        self._tf_buffer = None
        self.params: dict[str, Any] = {}
        self._status_sub = self.create_subscription(
            String, TOPICS["mission_status"], self._on_status, latched,
            callback_group=self._api_group)

    def run(self) -> "MissionResult":
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented yet")

    def result(self, **data: Any) -> MissionResult:
        return MissionResult(self.MISSION, data=dict(data))

    def log(self, text: str) -> None:
        print(text, flush=True)

    def _on_status(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self._cv:
            first = self._status is None
            self._status = payload
            self._status_seq = int(payload.get("seq", self._status_seq + 1))
            if (not first and payload.get("event") == "respawned"
                    and payload.get("source") == GAP_FALL_SOURCE):
                self.gap_falls += 1
            self._cv.notify_all()

    def latest(self, topic: str, msg_type, timeout: float = 5.0, qos=None):
        """First call waits up to timeout; later calls return cached data. Latched topics need TRANSIENT_LOCAL QoS."""

        from rclpy.qos import QoSProfile, ReliabilityPolicy

        with self._cv:
            new = topic not in self._latest_subs
            if new:
                def store(message, topic=topic) -> None:
                    with self._cv:
                        self._latest_msgs[topic] = message
                        self._cv.notify_all()

                self._latest_subs[topic] = self.create_subscription(
                    msg_type, topic, store,
                    qos or QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
                    callback_group=self._api_group)
            deadline = time.monotonic() + (timeout if new else 0.0)
            while topic not in self._latest_msgs and time.monotonic() < deadline:
                self._cv.wait(timeout=0.1)
            return self._latest_msgs.get(topic)

    def _tf(self):
        if self._tf_buffer is None:
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from tf2_msgs.msg import TFMessage
            from tf2_ros import Buffer

            self._tf_buffer = Buffer()

            def add(message, static: bool) -> None:
                for transform in message.transforms:
                    if static:
                        self._tf_buffer.set_transform_static(transform, "mission_api")
                    else:
                        self._tf_buffer.set_transform(transform, "mission_api")

            static_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                                    durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self._tf_subs = [
                self.create_subscription(TFMessage, "/tf", lambda m: add(m, False), 100,
                                         callback_group=self._api_group),
                self.create_subscription(TFMessage, "/tf_static", lambda m: add(m, True), static_qos,
                                         callback_group=self._api_group),
            ]
        return self._tf_buffer

    def pose(self, frame: str = "map", child: str = "pelvis",
             timeout: float = 2.0) -> Optional[tuple[float, float, float]]:
        """Return (x, y, yaw_rad) from TF in the requested frame."""

        import math

        from rclpy.duration import Duration
        from rclpy.time import Time

        try:
            t = self._tf().lookup_transform(frame, child, Time(),
                                            timeout=Duration(seconds=timeout)).transform
        except Exception:                                     # noqa: BLE001 - tf2 lookup/connectivity errors
            return None
        q = t.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return float(t.translation.x), float(t.translation.y), yaw

    def last_status(self) -> Optional[dict[str, Any]]:
        with self._cv:
            return self._status

    def _wait_status(self, predicate, timeout: Optional[float], after_seq: int, what: str) -> dict:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while True:
                if self._status is not None and self._status_seq > after_seq and predicate(self._status):
                    return self._status
                if deadline is not None and time.monotonic() >= deadline:
                    raise MissionTimeout(f"no answer from the simulator: {what}")
                self._cv.wait(timeout=0.5)

    def _command(self, key: str, message, predicate, what: str,
                 timeout: Optional[float] = ACK_TIMEOUT) -> dict[str, Any]:
        publisher = self._pub[key]
        with self._cv:
            before = self._status_seq
        match_deadline = time.monotonic() + self.MATCH_TIMEOUT_S
        while publisher.get_subscription_count() == 0 and time.monotonic() < match_deadline:
            time.sleep(0.05)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            publisher.publish(message)
            slice_timeout = self.REPUBLISH_S if deadline is None else min(
                self.REPUBLISH_S, max(0.0, deadline - time.monotonic()))
            try:
                return self._wait_status(predicate, slice_timeout, before, what)
            except MissionTimeout:
                if deadline is not None and time.monotonic() >= deadline:
                    raise

    def wait_for_sim(self, timeout: Optional[float] = 600.0) -> dict[str, Any]:
        self.log(f"[mission] waiting for the simulator on {TOPICS['mission_status']} ...")
        status = self._wait_status(lambda m: True, timeout, -1, "simulator startup")
        self.log(f"[mission] simulator up (mission point {status.get('mission')}: {status.get('label')})")
        return status

    def begin_mission(self, index: int, timeout: Optional[float] = ACK_TIMEOUT) -> dict[str, Any]:
        """Reset mission objects and activate the mission without moving G1."""

        return self._command(
            "mission_begin", self._Int32(data=int(index)),
            lambda m: m.get("event") == "selected" and m.get("mission") == index,
            f"begin mission {index}", timeout)

    def respawn(self, timeout: Optional[float] = ACK_TIMEOUT) -> dict[str, Any]:
        return self._command("respawn", self._Empty(), lambda m: m.get("event") == "respawned",
                             "respawn", timeout)

    @property
    def m1_goal(self) -> Optional[list[float]]:
        """World [x, y, yaw_deg] in testing; None in instruction mode."""

        status = self.last_status()
        return None if status is None else status.get("m1_goal")

    @property
    def m2_color(self) -> Optional[str]:
        """Target cone colour in testing; None in instruction mode."""

        status = self.last_status()
        return None if status is None else status.get("m2_color")

    @property
    def m4_rooms(self) -> list[str]:
        status = self.last_status()
        return list((status or {}).get("m4_rooms") or M4_ROOMS)

    def judge(self, index: int, settle: float = 1.0,
              timeout: Optional[float] = ACK_TIMEOUT) -> dict[str, Any]:
        self.stop()
        time.sleep(settle)
        status = self._command(
            "mission_end", self._Int32(data=int(index)),
            lambda m: m.get("event") == "judged" and m.get("mission") == index,
            f"referee for mission {index}", timeout)
        return status.get("judge") or {}

    def randomize(self, timeout: Optional[float] = ACK_TIMEOUT) -> dict[str, Any]:
        return self._command("randomize", self._Empty(), lambda m: m.get("event") == "randomized",
                             "randomize", timeout)

    def cmd_vel(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        """Body-frame forward/left velocity in m/s and counter-clockwise yaw rate in rad/s."""

        import rclpy

        if not rclpy.ok():
            return
        msg = self._Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        try:
            self._pub["cmd_vel"].publish(msg)
        except Exception as exc:              # noqa: BLE001
            # Ctrl-C can shut down the ROS context between the check and the call.
            if "context is invalid" not in str(exc):
                raise

    def stop(self) -> None:
        self.cmd_vel(0.0, 0.0, 0.0)


class MissionRunner:
    def __init__(self, node_name: str = "mission_runner", *, log: Callable[[str], None] = print) -> None:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.executor = MultiThreadedExecutor()
        self.api = Mission(node_name)
        self.api.log = log
        self.executor.add_node(self.api)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, name="mission-spin", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        from rclpy.executors import ExternalShutdownException

        try:
            while not self._stop.is_set() and self._rclpy.ok():
                self.executor.spin_once(timeout_sec=0.1)
        except ExternalShutdownException:
            pass

    def start(self, mission_class: Callable[[], Mission]) -> Mission:
        mission = mission_class()
        if not isinstance(mission, Mission):
            raise TypeError(f"{mission_class!r} must create a mission_api.Mission, "
                            f"got {type(mission).__name__}")
        self.executor.add_node(mission)
        return mission

    def finish(self, mission: Mission) -> None:
        try:
            self.executor.remove_node(mission)
        except Exception:                                      # noqa: BLE001 - never added
            pass
        mission.destroy_node()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.executor.remove_node(self.api)
            self.executor.shutdown(timeout_sec=1.0)
            self.api.destroy_node()
        finally:
            if self._owns_rclpy and self._rclpy.ok():
                self._rclpy.shutdown()

    def __enter__(self) -> "MissionRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.api.stop()
        self.close()


class GapFallWatch:
    def __init__(self, mission: Mission, allowed: int = GAP_FALLS_ALLOWED) -> None:
        self.mission, self.allowed = mission, int(allowed)
        self.tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="gap-fall-watch", daemon=True)
        self._previous = None

    def _watch(self) -> None:
        import signal

        while not self._stop.wait(0.2):
            if self.mission.gap_falls > self.allowed:
                self.tripped = True
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    def __enter__(self) -> "GapFallWatch":
        import signal

        def abort(_signum, _frame) -> None:
            raise MissionAborted(f"fell into a gap {self.mission.gap_falls} times "
                                 f"({self.allowed} allowed): the mission is over")

        self._previous = signal.signal(signal.SIGUSR1, abort)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        import signal

        self._stop.set()
        self._thread.join(timeout=1.0)
        signal.signal(signal.SIGUSR1, self._previous or signal.SIG_DFL)


def check_result(mission: int, result: Any) -> MissionResult:
    if isinstance(result, MissionResult):
        result.mission = mission
        return result
    if result is None:
        return MissionResult(mission)
    return MissionResult(mission, ERROR,
                         f"run() returned {type(result).__name__}; return self.result()")


def run_standalone(mission_class: Callable[[], Mission]) -> int:
    from referee import Referee

    referee = Referee(None)
    with MissionRunner("mission_standalone") as runner:
        runner.api.wait_for_sim()
        mission = runner.start(mission_class)
        index = mission.MISSION
        referee.begin(index, rooms=mission.m4_rooms)
        try:
            with GapFallWatch(mission):
                result = check_result(index, mission.run())
        except MissionAborted as exc:
            result = MissionResult(index, FAIL, str(exc))
        except KeyboardInterrupt:
            mission.stop()
            print("interrupted")
            return 130
        mission.stop()
        record = referee.finish(mission, result)
        runner.finish(mission)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False, default=str))
    return 0 if record["success"] else 1
