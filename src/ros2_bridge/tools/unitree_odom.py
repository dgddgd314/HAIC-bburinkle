from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

__all__ = ["UnitreeOdomPublisher", "quaternion_to_rpy"]


def quaternion_to_rpy(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quat_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class _Tick:
    def __init__(self, hz: float) -> None:
        self.period = 1.0 / float(hz)
        self.next = 0.0

    def due(self, now: float) -> bool:
        if now + 1e-12 < self.next:
            return False
        self.next += self.period
        if now - self.next > self.period:
            self.next = now + self.period
        return True


class UnitreeOdomPublisher:
    HIGH_TOPIC = "rt/odommodestate"
    LOW_TOPIC = "rt/lf/odommodestate"

    def __init__(
        self,
        model: Any,
        prefix: str,
        *,
        high_hz: float = 500.0,
        low_hz: float = 20.0,
        domain_id: int = 0,
        network_interface: Optional[str] = None,
    ) -> None:
        import mujoco
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        if high_hz <= 0 or low_hz <= 0:
            raise ValueError("Unitree odometry frequencies must be positive")
        physics_hz = 1.0 / float(model.opt.timestep)
        if high_hz > physics_hz * (1.0 + 1e-9):
            raise ValueError(
                f"Unitree high-rate odometry {high_hz:g} Hz exceeds the "
                f"MuJoCo physics rate {physics_hz:g} Hz"
            )
        if low_hz > high_hz:
            raise ValueError("Unitree low-rate odometry cannot exceed the high-rate stream")

        roots = [
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").startswith(prefix)
        ]
        if len(roots) != 1:
            raise ValueError(f"expected one G1 free joint under {prefix!r}, found {len(roots)}")
        root = roots[0]
        self._qadr = int(model.jnt_qposadr[root])
        self._dadr = int(model.jnt_dofadr[root])
        self._model = model
        self._left_foot = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, prefix + "left_foot"
        )
        self._right_foot = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, prefix + "right_foot"
        )
        if min(self._left_foot, self._right_foot) < 0:
            raise KeyError("G1 left_foot/right_foot sites are required for SportModeState")

        # SDK2 ChannelFactory is process-global; initialize it once for both writers.
        ChannelFactoryInitialize(int(domain_id), network_interface)
        self._high = ChannelPublisher(self.HIGH_TOPIC, SportModeState_)
        self._low = ChannelPublisher(self.LOW_TOPIC, SportModeState_)
        self._high.Init()
        self._low.Init()
        self._new_message = unitree_go_msg_dds__SportModeState_
        self._high_tick = _Tick(high_hz)
        self._low_tick = _Tick(low_hz)
        self.high_hz = float(high_hz)
        self.low_hz = float(low_hz)
        self.domain_id = int(domain_id)
        self._closed = False

    @staticmethod
    def _rotation(quat_wxyz: np.ndarray) -> np.ndarray:
        import mujoco

        matrix = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(matrix, np.asarray(quat_wxyz, dtype=float))
        return matrix.reshape(3, 3)

    def _foot_state(
        self,
        data: Any,
        site_id: int,
        base_pos: np.ndarray,
        base_rot: np.ndarray,
        base_linear_world: np.ndarray,
        base_angular_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import mujoco

        foot_world = np.asarray(data.site_xpos[site_id], dtype=float)
        spatial = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self._model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, spatial, 0
        )
        relative_world = foot_world - base_pos
        relative_position = base_rot.T @ relative_world
        relative_velocity = base_rot.T @ (
            spatial[3:6]
            - base_linear_world
            - np.cross(base_angular_world, relative_world)
        )
        return relative_position, relative_velocity

    def make_message(self, data: Any, sim_time: Optional[float] = None):
        now = float(data.time if sim_time is None else sim_time)
        qadr, dadr = self._qadr, self._dadr
        base_pos = np.asarray(data.qpos[qadr:qadr + 3], dtype=float)
        quat = np.asarray(data.qpos[qadr + 3:qadr + 7], dtype=float)
        base_rot = self._rotation(quat)
        linear_world = np.asarray(data.qvel[dadr:dadr + 3], dtype=float)
        angular_body = np.asarray(data.qvel[dadr + 3:dadr + 6], dtype=float)
        angular_world = base_rot @ angular_body
        acceleration_world = np.asarray(data.qacc[dadr:dadr + 3], dtype=float)
        proper_accel_body = base_rot.T @ (
            acceleration_world - np.asarray(self._model.opt.gravity, dtype=float)
        )

        left_pos, left_vel = self._foot_state(
            data, self._left_foot, base_pos, base_rot, linear_world, angular_world
        )
        right_pos, right_vel = self._foot_state(
            data, self._right_foot, base_pos, base_rot, linear_world, angular_world
        )

        moving = bool(
            np.linalg.norm(linear_world[:2]) > 0.03 or abs(float(angular_world[2])) > 0.05
        )
        msg = self._new_message()
        sec = int(math.floor(now))
        nanosec = int(round((now - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        msg.stamp.sec = sec
        msg.stamp.nanosec = nanosec
        msg.error_code = 0
        msg.imu_state.quaternion = [float(value) for value in quat]
        msg.imu_state.gyroscope = [float(value) for value in angular_body]
        msg.imu_state.accelerometer = [float(value) for value in proper_accel_body]
        msg.imu_state.rpy = [float(value) for value in quaternion_to_rpy(quat)]
        msg.imu_state.temperature = 0
        msg.mode = 3 if moving else 1
        msg.progress = 0.0
        msg.gait_type = 1 if moving else 0
        msg.foot_raise_height = 0.08 if moving else 0.0
        msg.position = [float(value) for value in base_pos]
        msg.body_height = float(base_pos[2])
        # SportModeState velocity is expressed in the odometry frame.
        msg.velocity = [float(value) for value in linear_world]
        msg.yaw_speed = float(angular_world[2])
        msg.range_obstacle = [0.0] * 4
        msg.foot_force = [0] * 4
        # Map G1 feet into the first two legacy quadruped slots; leave the others neutral.
        msg.foot_position_body = [
            *(float(value) for value in left_pos),
            *(float(value) for value in right_pos),
            *([0.0] * 6),
        ]
        msg.foot_speed_body = [
            *(float(value) for value in left_vel),
            *(float(value) for value in right_vel),
            *([0.0] * 6),
        ]
        return msg

    def update(self, data: Any, sim_time: Optional[float] = None) -> None:
        now = float(data.time if sim_time is None else sim_time)
        high_due = self._high_tick.due(now)
        low_due = self._low_tick.due(now)
        if not high_due and not low_due:
            return
        msg = self.make_message(data, now)
        if high_due:
            self._high.Write(msg)
        if low_due:
            self._low.Write(msg)

    def close(self) -> None:
        if self._closed:
            return
        self._high.Close()
        self._low.Close()
        self._closed = True

