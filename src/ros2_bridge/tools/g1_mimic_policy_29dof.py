from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np


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
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0, 0.0, 0.0,
     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0], dtype=float,
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

NUM_OBS = 154
POLICY_DT = 0.02
NOMINAL_SIM_DT = 0.002
CONTROL_DECIMATION = 10
SOURCE_ROOT_Z = 0.793
CANONICAL_BODY_NAMES = (
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link",
    "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
    "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link",
    "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
    "right_ankle_roll_link", "waist_yaw_link", "waist_roll_link",
    "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
    "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
    "right_wrist_pitch_link", "right_wrist_yaw_link",
)
ANCHOR_INDEX = CANONICAL_BODY_NAMES.index("torso_link")


def _metadata(session) -> dict[str, str]:
    try:
        return dict(session.get_modelmeta().custom_metadata_map)
    except Exception:
        return {}


def _csv_floats(value: str | None, fallback: np.ndarray) -> np.ndarray:
    if not value:
        return fallback.copy()
    parsed = np.asarray([float(item) for item in value.split(",")], dtype=float)
    if parsed.shape != fallback.shape or not np.all(np.isfinite(parsed)):
        raise ValueError(f"invalid ONNX metadata vector of shape {parsed.shape}")
    return parsed


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=float).copy()
    result[1:] *= -1.0
    return result


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dtype=float)


def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))


def _rotation_6d_from_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    matrix = np.asarray((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ))
    # Match mjlab rotation encoding: matrix_from_quat(q)[..., :2].reshape(..., 6).
    return matrix[:, :2].reshape(6)


class G1MimicPolicy29DoF:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_path: str | Path,
        motion_path: str | Path,
        *,
        prefix: str = "",
        loop: bool = False,
        autostart: bool = True,
    ) -> None:
        import onnxruntime as ort

        self.model, self.data, self.prefix = model, data, prefix
        self.policy_path = Path(policy_path)
        self.motion_path = Path(motion_path)
        self.loop = bool(loop)
        self.autostart = bool(autostart)
        self.session = ort.InferenceSession(
            str(self.policy_path), providers=["CPUExecutionProvider"]
        )
        inputs = {item.name: item for item in self.session.get_inputs()}
        outputs = {item.name: item for item in self.session.get_outputs()}
        if "obs" not in inputs or inputs["obs"].shape != [1, NUM_OBS]:
            raise ValueError(
                f"mimic ONNX must have obs [1,{NUM_OBS}], got "
                f"{[(x.name, x.shape) for x in inputs.values()]}"
            )
        if "actions" not in outputs or outputs["actions"].shape != [1, len(JOINTS)]:
            raise ValueError(
                f"mimic ONNX must output actions [1,{len(JOINTS)}], got "
                f"{[(x.name, x.shape) for x in outputs.values()]}"
            )
        self.has_time_input = "time_step" in inputs

        meta = _metadata(self.session)
        if meta.get("joint_names"):
            exported_joints = tuple(meta["joint_names"].split(","))
            if exported_joints != JOINTS:
                raise ValueError("ONNX joint order does not match canonical G1-29 order")
        self.default_pos = _csv_floats(meta.get("default_joint_pos"), DEFAULT_POS)
        self.kp = _csv_floats(meta.get("joint_stiffness"), KP)
        self.kd = _csv_floats(meta.get("joint_damping"), KD)
        self.action_scale = _csv_floats(meta.get("action_scale"), ACTION_SCALE)

        motion = np.load(self.motion_path)
        required = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
        missing = [key for key in required if key not in motion]
        if missing:
            raise ValueError(f"motion NPZ is missing fields: {missing}")
        self.reference_joint_pos = np.asarray(motion["joint_pos"], dtype=np.float32)
        self.reference_joint_vel = np.asarray(motion["joint_vel"], dtype=np.float32)
        self.reference_body_pos = np.asarray(motion["body_pos_w"], dtype=np.float32)
        self.reference_body_quat = np.asarray(motion["body_quat_w"], dtype=np.float32)
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        frames = self.reference_joint_pos.shape[0]
        expected = {
            "joint_pos": (frames, len(JOINTS)),
            "joint_vel": (frames, len(JOINTS)),
        }
        for name, shape in expected.items():
            if getattr(self, f"reference_{name}").shape != shape:
                raise ValueError(f"motion {name} must have shape {shape}")
        if self.reference_body_pos.shape[:2] != (frames, len(CANONICAL_BODY_NAMES)):
            raise ValueError(
                "motion body_pos_w must use the canonical 30-body G1 ordering"
            )
        if self.reference_body_quat.shape != (frames, len(CANONICAL_BODY_NAMES), 4):
            raise ValueError("motion body_quat_w has an unexpected shape")
        if not math.isclose(fps, 1.0 / POLICY_DT, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(f"mimic motion must be 50 Hz, got {fps:g} Hz")
        if not all(np.all(np.isfinite(array)) for array in (
            self.reference_joint_pos, self.reference_joint_vel,
            self.reference_body_pos, self.reference_body_quat,
        )):
            raise ValueError("motion NPZ contains non-finite values")

        ratio = POLICY_DT / float(model.opt.timestep)
        self.control_decimation = int(round(ratio))
        if self.control_decimation < 1 or not math.isclose(
            ratio, self.control_decimation, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(
                f"model timestep {model.opt.timestep:g}s does not divide the 0.02s policy period"
            )
        self._resolve_indices()
        self._apply_policy_gains()
        self.obs = np.zeros(NUM_OBS, dtype=np.float32)
        self.action = np.zeros(len(JOINTS), dtype=np.float32)
        self.target_pos = self.default_pos.copy()
        self.frame = 0
        self.counter = 0
        self.active = False
        self.finished = False
        self.world_rotation = np.asarray((1.0, 0.0, 0.0, 0.0))
        self.spawn_xy = np.zeros(2)
        self.floor_z = 0.0

    @property
    def frame_count(self) -> int:
        return int(self.reference_joint_pos.shape[0])

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
        self.root_body_id = int(self.model.jnt_bodyid[self.root_jid])
        self.root_qadr = int(self.model.jnt_qposadr[self.root_jid])
        self.root_dadr = int(self.model.jnt_dofadr[self.root_jid])
        joint_ids: list[int] = []
        for name in JOINTS:
            joint = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + name
            )
            if joint < 0:
                raise KeyError(f"missing mimic joint: {self.prefix + name}")
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
            raise KeyError(f"missing mimic actuators: {missing}")
        self.act = np.asarray([by_joint[joint] for joint in joint_ids], dtype=np.int32)
        self.anchor_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + "torso_link"
        )
        self.gyro_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, self.prefix + "imu_ang_vel"
        )
        if self.anchor_body_id < 0 or self.gyro_sensor_id < 0:
            raise KeyError("G1 mimic requires torso_link and imu_ang_vel")
        self.gyro_adr = int(self.model.sensor_adr[self.gyro_sensor_id])
        if int(self.model.sensor_dim[self.gyro_sensor_id]) != 3:
            raise ValueError("G1 imu_ang_vel must be a three-axis gyro")

    def _apply_policy_gains(self) -> None:
        for actuator, kp, kd in zip(self.act, self.kp, self.kd):
            if self.model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE:
                raise ValueError("G1 mimic requires affine position actuators")
            self.model.actuator_gainprm[actuator, 0] = kp
            self.model.actuator_biasprm[actuator, 1] = -kp
            self.model.actuator_biasprm[actuator, 2] = -kd

    def start_state(
        self, pos: Optional[Sequence[float]] = None, yaw: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reference_pos = self.reference_body_pos[0, 0].astype(float)
        reference_quat = self.reference_body_quat[0, 0].astype(float)
        aligned_quat = _quat_mul(
            _yaw_quat(yaw - _yaw_from_quat(reference_quat)), reference_quat
        )
        aligned_quat /= np.linalg.norm(aligned_quat)
        current = self.data.qpos[self.root_qadr:self.root_qadr + 3].copy()
        requested = current if pos is None else np.asarray(pos, dtype=float).reshape(-1)
        if requested.size not in (2, 3):
            raise ValueError("mimic reset pos must contain x y or x y z")
        # Keep the reference height offset relative to the source MJCF root on uneven floors.
        standing_z = float(current[2] if requested.size == 2 else requested[2])
        root_pos = np.asarray((
            requested[0], requested[1], standing_z - SOURCE_ROOT_Z + reference_pos[2]
        ))
        return root_pos, aligned_quat, self.reference_joint_pos[0].astype(float).copy()

    def _align(self, pos: Optional[Sequence[float]], yaw: float) -> None:
        reference_quat = self.reference_body_quat[0, 0].astype(float)
        self.world_rotation = _yaw_quat(yaw - _yaw_from_quat(reference_quat))
        root_pos, _, _ = self.start_state(pos, yaw)
        self.spawn_xy[:] = root_pos[:2]
        self.floor_z = float(root_pos[2] - self.reference_body_pos[0, 0, 2])

    def _restart_counters(self) -> None:
        self.action[:] = 0.0
        self.target_pos[:] = self.default_pos
        self.obs[:] = 0.0
        self.frame = 0
        self.counter = 0
        self.finished = False
        self.active = self.autostart
        self.metric_samples = 0
        self.joint_error_squared_sum = 0.0
        self.min_root_z = float("inf")
        self.min_up_z = float("inf")

    def begin_from_current(self, pos: Optional[Sequence[float]] = None, yaw: float = 0.0) -> None:
        self._align(pos, yaw)
        self._restart_counters()

    def reset(self, pos: Optional[Sequence[float]] = None, yaw: float = 0.0) -> None:
        root_pos, aligned_quat, joints = self.start_state(pos, yaw)
        self._align(pos, yaw)
        self.data.qpos[self.root_qadr:self.root_qadr + 3] = root_pos
        self.data.qpos[self.root_qadr + 3:self.root_qadr + 7] = aligned_quat
        self.data.qvel[self.root_dadr:self.root_dadr + 6] = 0.0
        self.data.qpos[self.qadr] = joints
        self.data.qvel[self.dadr] = self.reference_joint_vel[0]
        self.action[:] = 0.0
        self.target_pos[:] = self.default_pos
        self.obs[:] = 0.0
        self.frame = 0
        self.counter = 0
        self.finished = False
        self.active = self.autostart
        self.metric_samples = 0
        self.joint_error_squared_sum = 0.0
        self.min_root_z = float("inf")
        self.min_up_z = float("inf")
        self.data.ctrl[self.act] = self.reference_joint_pos[0]
        mujoco.mj_forward(self.model, self.data)

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        if np.linalg.norm((vx, vy, wz)) > 1e-3:
            self.start()

    def _reference_anchor_quat(self, frame: int) -> np.ndarray:
        quat = _quat_mul(
            self.world_rotation,
            self.reference_body_quat[frame, ANCHOR_INDEX].astype(float),
        )
        return quat / np.linalg.norm(quat)

    def _build_obs(self, data: mujoco.MjData) -> None:
        frame = min(self.frame, self.frame_count - 1)
        robot_anchor_quat = np.asarray(data.xquat[self.anchor_body_id], dtype=float)
        relative_quat = _quat_mul(
            _quat_conjugate(robot_anchor_quat), self._reference_anchor_quat(frame)
        )
        self.obs[0:29] = self.reference_joint_pos[frame]
        self.obs[29:58] = self.reference_joint_vel[frame]
        self.obs[58:64] = _rotation_6d_from_quat(relative_quat)
        self.obs[64:67] = data.sensordata[self.gyro_adr:self.gyro_adr + 3]
        self.obs[67:96] = data.qpos[self.qadr] - self.default_pos
        self.obs[96:125] = data.qvel[self.dadr]
        self.obs[125:154] = self.action

    def _infer(self, data: mujoco.MjData) -> None:
        self._build_obs(data)
        frame = min(self.frame, self.frame_count - 1)
        joint_error = data.qpos[self.qadr] - self.reference_joint_pos[frame]
        self.joint_error_squared_sum += float(np.mean(joint_error * joint_error))
        self.metric_samples += 1
        self.min_root_z = min(
            self.min_root_z, float(data.qpos[self.root_qadr + 2])
        )
        self.min_up_z = min(
            self.min_up_z,
            float(data.xmat[self.root_body_id].reshape(3, 3)[2, 2]),
        )
        feed = {"obs": self.obs[None, :]}
        if self.has_time_input:
            feed["time_step"] = np.asarray([[float(self.frame)]], dtype=np.float32)
        output = self.session.run(["actions"], feed)[0]
        self.action[:] = np.asarray(output, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(self.action)):
            raise FloatingPointError("G1 mimic policy produced a non-finite action")
        self.target_pos[:] = self.default_pos + self.action_scale * self.action

    def metrics(self) -> dict[str, float | int | bool]:
        rms = math.sqrt(
            self.joint_error_squared_sum / max(self.metric_samples, 1)
        )
        return {
            "reference_frames": self.frame_count,
            "last_frame": self.frame,
            "finished": self.finished,
            "samples": self.metric_samples,
            "joint_rmse_rad": rms,
            "min_root_z_m": self.min_root_z,
            "min_up_z": self.min_up_z,
        }

    def control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model
        if not self.active:
            data.ctrl[self.act] = self.target_pos
            return
        if self.counter % self.control_decimation == 0:
            self._infer(data)
            if self.frame + 1 < self.frame_count:
                self.frame += 1
            elif self.loop:
                self.frame = 0
            else:
                self.finished = True
                self.active = False
        self.counter += 1
        data.ctrl[self.act] = self.target_pos

    def install(self) -> None:
        mujoco.set_mjcb_control(self.control)

    @staticmethod
    def uninstall() -> None:
        mujoco.set_mjcb_control(None)
