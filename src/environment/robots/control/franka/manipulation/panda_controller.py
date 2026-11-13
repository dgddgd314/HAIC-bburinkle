from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


PANDA_PREFIX = "panda/"
ARM_JOINTS = tuple(f"robot0_joint{index}" for index in range(1, 8))
ARM_ACTUATORS = tuple(f"robot0_torq_j{index}" for index in range(1, 8))
FINGER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
FINGER_ACTUATORS = (
    "gripper0_gripper_finger_joint1",
    "gripper0_gripper_finger_joint2",
)
BASE_BODY = "robot0_base"
EEF_SITE = "gripper0_grip_site"

PANDA_HOME = np.asarray(
    [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.22675220, np.pi / 4.0],
    dtype=float,
)

OSC_POSITION_SCALE = 0.05
OSC_ROTATION_SCALE = 0.5
OSC_KP = 150.0
OSC_DAMPING_RATIO = 1.0


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise RuntimeError(f"Panda model element is missing: {name}")
    return result


def initialize_panda_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    prefix: str = PANDA_PREFIX,
    opening: float = 0.02,
) -> None:
    for name, value in zip(ARM_JOINTS, PANDA_HOME):
        joint = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        data.qpos[model.jnt_qposadr[joint]] = value
        data.qvel[model.jnt_dofadr[joint]] = 0.0
    opening = float(np.clip(opening, 0.0, 0.04))
    for name, value in zip(FINGER_JOINTS, (opening, -opening)):
        joint = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        data.qpos[model.jnt_qposadr[joint]] = value
        data.qvel[model.jnt_dofadr[joint]] = 0.0
    mujoco.mj_forward(model, data)


class PandaController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        prefix: str = PANDA_PREFIX,
    ) -> None:
        self.model = model
        self.data = data
        self.prefix = prefix
        self.joint_ids = np.asarray(
            [_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name) for name in ARM_JOINTS],
            dtype=np.int32,
        )
        self.qpos_adrs = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.dof_adrs = model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.actuator_ids = np.asarray(
            [_named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name) for name in ARM_ACTUATORS],
            dtype=np.int32,
        )
        self.finger_joint_ids = np.asarray(
            [_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name) for name in FINGER_JOINTS],
            dtype=np.int32,
        )
        self.finger_qpos_adrs = model.jnt_qposadr[self.finger_joint_ids].astype(np.int32)
        self.finger_actuator_ids = np.asarray(
            [_named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name) for name in FINGER_ACTUATORS],
            dtype=np.int32,
        )
        self.base_body = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + BASE_BODY)
        self.eef_site = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + EEF_SITE)
        self.joint_lower = model.jnt_range[self.joint_ids, 0].copy()
        self.joint_upper = model.jnt_range[self.joint_ids, 1].copy()
        self.torque_lower = model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.torque_upper = model.actuator_ctrlrange[self.actuator_ids, 1].copy()

        self.mode = "joint"
        self.joint_target = data.qpos[self.qpos_adrs].copy()
        self.nullspace_target = self.joint_target.copy()
        self.gripper_opening = float(
            np.clip(abs(data.qpos[self.finger_qpos_adrs[0]]), 0.0, 0.04)
        )
        self.goal_position = data.site_xpos[self.eef_site].copy()
        self.goal_orientation = data.site_xmat[self.eef_site].reshape(3, 3).copy()

        self.joint_kp = np.asarray([90.0, 90.0, 75.0, 75.0, 45.0, 30.0, 20.0])
        self.joint_kd = 2.0 * np.sqrt(self.joint_kp)
        self.osc_kp = np.full(6, OSC_KP)
        self.osc_kd = 2.0 * np.sqrt(self.osc_kp) * OSC_DAMPING_RATIO

    def site_transform(self, data: mujoco.MjData | None = None) -> np.ndarray:
        data = self.data if data is None else data
        transform = np.eye(4)
        transform[:3, 3] = data.site_xpos[self.eef_site]
        transform[:3, :3] = data.site_xmat[self.eef_site].reshape(3, 3)
        return transform

    def base_transform(self, data: mujoco.MjData | None = None) -> np.ndarray:
        data = self.data if data is None else data
        transform = np.eye(4)
        transform[:3, 3] = data.xpos[self.base_body]
        transform[:3, :3] = data.xmat[self.base_body].reshape(3, 3)
        return transform

    def finger_positions(self, data: mujoco.MjData | None = None) -> np.ndarray:
        data = self.data if data is None else data
        return data.qpos[self.finger_qpos_adrs].astype(np.float32).copy()

    def set_gripper_opening(self, opening: float) -> None:
        self.gripper_opening = float(np.clip(opening, 0.0, 0.04))

    def set_joint_target(self, target: Sequence[float]) -> None:
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("Panda joint target must contain 7 finite values")
        self.joint_target = np.clip(target, self.joint_lower, self.joint_upper)
        self.mode = "joint"

    def command_joint_positions(self, names: Sequence[str], positions: Sequence[float]) -> None:
        if len(names) != len(positions):
            raise ValueError("Panda joint command names and positions must have equal lengths")
        target = self.joint_target.copy()
        by_name = {name: index for index, name in enumerate(ARM_JOINTS)}
        for name, value in zip(names, positions):
            index = by_name.get(name)
            if index is not None and np.isfinite(value):
                target[index] = float(value)
        self.set_joint_target(target)

    def set_eef_target(self, transform: np.ndarray) -> None:
        transform = np.asarray(transform, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("Panda EEF target must be one finite 4x4 transform")
        self.goal_position = transform[:3, 3].copy()
        self.goal_orientation = transform[:3, :3].copy()
        self.mode = "osc"

    def set_eef_delta(self, delta: Sequence[float], *, normalized: bool = False) -> None:
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (6,) or not np.all(np.isfinite(delta)):
            raise ValueError("Panda EE delta must contain xyz + rotation-vector (6 values)")
        if normalized:
            delta = np.clip(delta, -1.0, 1.0)
            delta = delta * np.asarray(
                [OSC_POSITION_SCALE] * 3 + [OSC_ROTATION_SCALE] * 3
            )
        current = self.site_transform()
        target = current.copy()
        target[:3, 3] += delta[:3]
        # Match robosuite: left-multiply the axis-angle delta.
        target[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ current[:3, :3]
        self.set_eef_target(target)

    @staticmethod
    def _orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
        return 0.5 * sum(
            (np.cross(current[:, axis], desired[:, axis]) for axis in range(3)),
            start=np.zeros(3),
        )

    def _mass_and_jacobian(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        full_mass = np.empty((self.model.nv, self.model.nv), dtype=float)
        mujoco.mj_fullM(self.model, full_mass, data.qM)
        mass = full_mass[np.ix_(self.dof_adrs, self.dof_adrs)]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.eef_site)
        jacobian = np.vstack((jacp[:, self.dof_adrs], jacr[:, self.dof_adrs]))
        return mass, jacobian

    def _joint_torques(self, data: mujoco.MjData) -> np.ndarray:
        q = data.qpos[self.qpos_adrs]
        qd = data.qvel[self.dof_adrs]
        return (
            self.joint_kp * (self.joint_target - q)
            - self.joint_kd * qd
            + data.qfrc_bias[self.dof_adrs]
        )

    def _osc_torques(self, data: mujoco.MjData) -> np.ndarray:
        current_position = data.site_xpos[self.eef_site]
        current_orientation = data.site_xmat[self.eef_site].reshape(3, 3)
        mass, jacobian = self._mass_and_jacobian(data)
        qd = data.qvel[self.dof_adrs]
        twist = jacobian @ qd
        error = np.concatenate(
            (
                self.goal_position - current_position,
                self._orientation_error(self.goal_orientation, current_orientation),
            )
        )
        desired_wrench = self.osc_kp * error - self.osc_kd * twist

        mass_inv = np.linalg.pinv(mass, rcond=1e-8)
        jac_pos, jac_ori = jacobian[:3], jacobian[3:]
        lambda_full = np.linalg.pinv(jacobian @ mass_inv @ jacobian.T, rcond=1e-6)
        lambda_pos = np.linalg.pinv(jac_pos @ mass_inv @ jac_pos.T, rcond=1e-6)
        lambda_ori = np.linalg.pinv(jac_ori @ mass_inv @ jac_ori.T, rcond=1e-6)
        wrench = np.concatenate(
            (lambda_pos @ desired_wrench[:3], lambda_ori @ desired_wrench[3:])
        )
        torques = jacobian.T @ wrench + data.qfrc_bias[self.dof_adrs]

        jbar = mass_inv @ jacobian.T @ lambda_full
        nullspace = np.eye(7) - jbar @ jacobian
        q = data.qpos[self.qpos_adrs]
        pose_accel = 10.0 * (self.nullspace_target - q) - 2.0 * np.sqrt(10.0) * qd
        torques += nullspace.T @ (mass @ pose_accel)
        return torques

    def before_step(self, data: mujoco.MjData | None = None) -> None:
        data = self.data if data is None else data
        torques = self._osc_torques(data) if self.mode == "osc" else self._joint_torques(data)
        data.ctrl[self.actuator_ids] = np.clip(
            torques, self.torque_lower, self.torque_upper
        )
        data.ctrl[self.finger_actuator_ids[0]] = self.gripper_opening
        data.ctrl[self.finger_actuator_ids[1]] = -self.gripper_opening

