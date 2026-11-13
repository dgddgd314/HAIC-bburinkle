from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np

from ros2_bridge.tools.g1_walk_policy import gravity_orientation


JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

DEFAULT_POS = np.array(
    [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0,
     0, 0, 0, 0.35, 0.18, 0, 0.87, 0, 0, 0,
     0.35, -0.18, 0, 0.87, 0, 0, 0], dtype=float,
)
KP = np.array(
    [40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
     40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
     40.179, 28.501, 28.501,
     14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
     14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778], dtype=float,
)
KD = np.array(
    [2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
     2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
     2.558, 1.814, 1.814,
     0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
     0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068], dtype=float,
)
ACTION_SCALE = np.array(
    [0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
     0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
     0.548, 0.439, 0.439,
     0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075,
     0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075], dtype=float,
)

NOMINAL_DT = 0.002
CONTROL_DECIMATION = 10
POLICY_DT = NOMINAL_DT * CONTROL_DECIMATION
GAIT_PERIOD = 0.6
NUM_OBS = 98


class G1WalkPolicy29DoF:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_path: str | Path,
        *,
        prefix: str = "",
        cmd: Sequence[float] = (0.0, 0.0, 0.0),
        stationary_idle: bool = True,
        command_deadband: float = 1e-3,
        command_slew: Sequence[float] = (1.2, 1.2, 2.4),
    ) -> None:
        import onnxruntime as ort

        if abs(float(model.opt.timestep) - NOMINAL_DT) > 1e-9:
            raise ValueError(
                f"29-DoF policy requires {NOMINAL_DT:g}s timestep; "
                f"model has {model.opt.timestep:g}s"
            )
        self.model, self.data, self.prefix = model, data, prefix
        self.session = ort.InferenceSession(
            str(policy_path), providers=["CPUExecutionProvider"]
        )
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        if input_info.shape != [1, NUM_OBS] or output_info.shape != [1, len(JOINTS)]:
            raise ValueError(
                f"unexpected ONNX contract: input={input_info.shape}, "
                f"output={output_info.shape}"
            )
        self.input_name = input_info.name
        self.output_name = output_info.name
        self.stationary_idle = bool(stationary_idle)
        self.command_deadband = max(0.0, float(command_deadband))
        self.command_slew = np.asarray(command_slew, dtype=float)
        if self.command_slew.shape != (3,) or np.any(self.command_slew <= 0):
            raise ValueError("command_slew must contain three positive values")

        self._resolve_indices()
        self._apply_policy_gains()
        self.target_cmd = np.zeros(3, dtype=float)
        self.cmd = np.zeros(3, dtype=float)
        self.action = np.zeros(len(JOINTS), dtype=np.float32)
        self.target_pos = DEFAULT_POS.copy()
        self.obs = np.zeros(NUM_OBS, dtype=np.float32)
        self.phase = 0.0
        self.counter = 0
        self._idle = True
        self._idle_pos = np.zeros(3, dtype=float)
        self._idle_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.set_command(*cmd)

    def _resolve_indices(self) -> None:
        roots = [
            joint for joint in range(self.model.njnt)
            if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            and (mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint
            ) or "").startswith(self.prefix)
        ]
        if len(roots) != 1:
            raise ValueError(
                f"expected one G1 free joint under {self.prefix!r}, found {len(roots)}"
            )
        self.root_jid = roots[0]
        self.root_qadr = int(self.model.jnt_qposadr[self.root_jid])
        self.root_dadr = int(self.model.jnt_dofadr[self.root_jid])
        joint_ids = []
        for name in JOINTS:
            joint = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + name
            )
            if joint < 0:
                raise KeyError(f"missing policy joint: {self.prefix + name}")
            joint_ids.append(joint)
        self.qadr = np.asarray(
            [self.model.jnt_qposadr[joint] for joint in joint_ids], dtype=np.int32
        )
        self.dadr = np.asarray(
            [self.model.jnt_dofadr[joint] for joint in joint_ids], dtype=np.int32
        )
        by_joint = {
            int(self.model.actuator_trnid[actuator, 0]): actuator
            for actuator in range(self.model.nu)
            if self.model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT
        }
        missing = [name for name, joint in zip(JOINTS, joint_ids) if joint not in by_joint]
        if missing:
            raise KeyError(f"missing policy actuators: {missing}")
        self.act = np.asarray([by_joint[joint] for joint in joint_ids], dtype=np.int32)

    def _apply_policy_gains(self) -> None:
        for actuator, kp, kd in zip(self.act, KP, KD):
            if self.model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE:
                raise ValueError("29-DoF policy requires affine position actuators")
            self.model.actuator_gainprm[actuator, 0] = kp
            self.model.actuator_biasprm[actuator, 1] = -kp
            self.model.actuator_biasprm[actuator, 2] = -kd

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        values = np.asarray((vx, vy, wz), dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("velocity command must contain three finite values")
        values[:] = np.clip(values, (-0.5, -0.5, -1.0), (1.0, 0.5, 1.0))
        was_idle = self._idle
        self.target_cmd[:] = values
        requested_idle = bool(np.linalg.norm(values) <= self.command_deadband)
        if was_idle and not requested_idle:
            self._idle = False
            self.counter = 0
            self.phase = 0.0
            self.action[:] = 0.0
        elif was_idle and requested_idle:
            self.cmd[:] = 0.0

    def _capture_idle_pose(self) -> None:
        self._idle_pos[:] = self.data.qpos[self.root_qadr:self.root_qadr + 3]
        self._idle_quat[:] = self.data.qpos[self.root_qadr + 3:self.root_qadr + 7]

    def _idle_control(self, data: mujoco.MjData) -> None:
        data.ctrl[self.act] = DEFAULT_POS
        if not self.stationary_idle:
            return
        qadr, dadr = self.root_qadr, self.root_dadr
        force = (
            np.array([6000.0, 6000.0, 2500.0])
            * (self._idle_pos - data.qpos[qadr:qadr + 3])
            - np.array([300.0, 300.0, 220.0]) * data.qvel[dadr:dadr + 3]
        )
        rotation_error = np.zeros(3, dtype=float)
        mujoco.mju_subQuat(
            rotation_error,
            self._idle_quat,
            data.qpos[qadr + 3:qadr + 7],
        )
        torque = 1000.0 * rotation_error - 80.0 * data.qvel[dadr + 3:dadr + 6]
        data.qfrc_applied[dadr:dadr + 3] = np.clip(force, -5000.0, 5000.0)
        data.qfrc_applied[dadr + 3:dadr + 6] = np.clip(torque, -1000.0, 1000.0)

    def reset(self, pos: Optional[Sequence[float]] = None, yaw: float = 0.0) -> None:
        data = self.data
        data.qpos[self.qadr] = DEFAULT_POS
        data.qvel[self.dadr] = 0.0
        if pos is not None:
            data.qpos[self.root_qadr:self.root_qadr + 3] = np.asarray(pos, dtype=float)
        data.qpos[self.root_qadr + 3:self.root_qadr + 7] = (
            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
        )
        data.qvel[self.root_dadr:self.root_dadr + 6] = 0.0
        self.action[:] = 0.0
        self.target_pos[:] = DEFAULT_POS
        self.obs[:] = 0.0
        self.phase = 0.0
        self.counter = 0
        self.cmd[:] = 0.0
        self._idle = bool(np.linalg.norm(self.target_cmd) <= self.command_deadband)
        self._capture_idle_pose()
        data.ctrl[self.act] = DEFAULT_POS
        mujoco.mj_forward(self.model, data)

    def adopt_current_state(self) -> None:
        self.target_cmd[:] = 0.0
        self.cmd[:] = 0.0
        self.action[:] = 0.0
        self.target_pos[:] = DEFAULT_POS
        self.phase = 0.0
        self.counter = 0
        self._idle = True
        self._capture_idle_pose()
        self.data.ctrl[self.act] = DEFAULT_POS

    def _advance_command(self) -> None:
        self.cmd += np.clip(
            self.target_cmd - self.cmd,
            -self.command_slew * POLICY_DT,
            self.command_slew * POLICY_DT,
        )
        if (
            np.linalg.norm(self.target_cmd) <= self.command_deadband
            and np.linalg.norm(self.cmd) <= self.command_deadband
        ):
            self.cmd[:] = 0.0
            self._idle = True
            self._capture_idle_pose()

    def _infer(self, data: mujoco.MjData) -> None:
        self.phase = (self.phase + POLICY_DT / GAIT_PERIOD) % 1.0
        moving = np.linalg.norm(self.cmd) >= 0.1
        phase_obs = (
            (np.sin(2.0 * np.pi * self.phase), np.cos(2.0 * np.pi * self.phase))
            if moving else (0.0, 0.0)
        )
        quat = data.qpos[self.root_qadr + 3:self.root_qadr + 7]
        self.obs[0:3] = data.qvel[self.root_dadr + 3:self.root_dadr + 6]
        self.obs[3:6] = gravity_orientation(quat)
        self.obs[6:9] = self.cmd
        self.obs[9:11] = phase_obs
        self.obs[11:40] = data.qpos[self.qadr] - DEFAULT_POS
        self.obs[40:69] = data.qvel[self.dadr]
        self.obs[69:98] = self.action
        output = self.session.run(
            [self.output_name], {self.input_name: self.obs[None, :]}
        )[0]
        self.action[:] = np.asarray(output, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(self.action)):
            raise FloatingPointError("29-DoF G1 policy produced a non-finite action")
        self.target_pos[:] = DEFAULT_POS + ACTION_SCALE * self.action

    def control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._idle:
            self._idle_control(data)
            return
        data.qfrc_applied[self.root_dadr:self.root_dadr + 6] = 0.0
        if self.counter % CONTROL_DECIMATION == 0:
            self._advance_command()
            if self._idle:
                self._idle_control(data)
                return
            self._infer(data)
        self.counter += 1
        data.ctrl[self.act] = self.target_pos

    def install(self) -> None:
        mujoco.set_mjcb_control(self.control)

    @staticmethod
    def uninstall() -> None:
        mujoco.set_mjcb_control(None)
