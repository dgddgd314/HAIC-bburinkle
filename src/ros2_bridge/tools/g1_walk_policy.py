"""Drive a spawned G1 with the unitree_rl_gym walking policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

__all__ = ["G1WalkPolicy", "LEG_JOINTS", "DEFAULT_ANGLES"]


LEG_JOINTS: Tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)

DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2, float)
KPS = np.array([100.0, 100.0, 100.0, 150.0, 40.0, 40.0] * 2, float)
KDS = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0] * 2, float)

ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], float)

NUM_ACTIONS = 12
NUM_OBS = 47
CONTROL_DECIMATION = 10
GAIT_PERIOD = 0.8
NOMINAL_DT = 0.002


def gravity_orientation(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = (float(v) for v in quat)
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


class G1WalkPolicy:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_path: str | Path,
        prefix: str = "",
        cmd: Sequence[float] = (0.0, 0.0, 0.0),
        override_gains: bool = True,
        hold_upper_body: bool = True,
        upper_body_stiffness: float = 10.0,
        stationary_idle: bool = True,
        command_deadband: float = 1e-3,
        command_slew: Sequence[float] = (0.8, 0.8, 1.2),
        heading_hold_gain: float = 3.0,
        heading_hold_damping: float = 0.20,
        heading_hold_limit: float = 0.25,
    ) -> None:
        import torch

        self.model, self.data, self.prefix = model, data, prefix
        self.cmd = np.zeros(3, float)
        self.target_cmd = np.zeros(3, float)
        self.command_slew = np.asarray(command_slew, dtype=float)
        if self.command_slew.shape != (3,) or np.any(self.command_slew <= 0):
            raise ValueError("command_slew must contain three positive values")
        self.heading_hold_gain = max(0.0, float(heading_hold_gain))
        self.heading_hold_damping = max(0.0, float(heading_hold_damping))
        self.heading_hold_limit = max(0.0, float(heading_hold_limit))
        self._heading_ref: Optional[float] = None
        self.policy = torch.jit.load(str(policy_path))
        self.policy.eval()
        self._torch = torch
        # Reset the in-place LSTM state between episodes to avoid stale gait memory.
        self._recurrent = hasattr(self.policy, "hidden_state") and hasattr(self.policy, "cell_state")

        if abs(model.opt.timestep - NOMINAL_DT) > 1e-9:
            raise ValueError(
                f"policy assumes a {NOMINAL_DT} s timestep, model has "
                f"{model.opt.timestep}; the 0.8 s gait clock and the PD rate "
                "both depend on it"
            )

        self._resolve_indices()
        if override_gains:
            self._apply_unitree_gains()

        self.hold_act = np.array([], np.int32)
        self.hold_qadr = np.array([], np.int32)
        self.hold_dadr = np.array([], np.int32)
        self.hold_target = np.array([], float)
        if hold_upper_body:
            self._setup_upper_body(upper_body_stiffness)

        self.stationary_idle = bool(stationary_idle)
        self.command_deadband = max(0.0, float(command_deadband))
        self._idle = True
        self._idle_pos = np.zeros(3, float)
        self._idle_quat = np.array([1.0, 0.0, 0.0, 0.0], float)
        self._idle_leg_start = DEFAULT_ANGLES.copy()
        self._idle_blend = 1.0

        self.obs = np.zeros(NUM_OBS, np.float32)
        self.action = np.zeros(NUM_ACTIONS, np.float32)
        self.target_dof_pos = DEFAULT_ANGLES.copy()
        self.counter = 0
        self.set_command(*cmd)


    def _jid(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + name)
        if jid < 0:
            raise KeyError(f"joint {self.prefix + name!r} not in model")
        return jid

    def _resolve_indices(self) -> None:
        m = self.model

        # Free-joint qvel uses world linear velocity and base-frame angular velocity.
        root = [
            j for j in range(m.njnt)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(self.prefix)
        ]
        if len(root) != 1:
            raise ValueError(
                f"expected exactly one free joint under prefix {self.prefix!r}, found {len(root)}. "
                "Did you spawn with freeze_root=True? The policy needs a floating base."
            )
        self.root_jid = root[0]
        self.root_qadr = int(m.jnt_qposadr[self.root_jid])
        self.root_dadr = int(m.jnt_dofadr[self.root_jid])

        jids = [self._jid(n) for n in LEG_JOINTS]
        self.leg_qadr = np.array([m.jnt_qposadr[j] for j in jids], np.int32)
        self.leg_dadr = np.array([m.jnt_dofadr[j] for j in jids], np.int32)

        by_joint = {
            int(m.actuator_trnid[a, 0]): a
            for a in range(m.nu)
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
        }
        missing = [n for n, j in zip(LEG_JOINTS, jids) if j not in by_joint]
        if missing:
            raise KeyError(f"no actuator drives leg joint(s): {missing}")
        self.leg_act = np.array([by_joint[j] for j in jids], np.int32)
        self._leg_jids = set(jids)
        self._act_by_joint = by_joint

    def _apply_unitree_gains(self) -> None:
        m = self.model
        for a, kp, kd in zip(self.leg_act, KPS, KDS):
            if m.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE:
                raise ValueError(
                    f"actuator {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)!r} "
                    "is not an affine-bias PD actuator; cannot retune it"
                )
            m.actuator_gainprm[a, 0] = kp
            m.actuator_biasprm[a, 1] = -kp
            m.actuator_biasprm[a, 2] = -kd

    def _setup_upper_body(self, stiffness: float) -> None:
        m, d = self.model, self.data
        acts, qadr, dadr, targets = [], [], [], []
        for jid, a in self._act_by_joint.items():
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if jid in self._leg_jids or not name.startswith(self.prefix):
                continue
            if m.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            t = 0.0
            if m.jnt_limited[jid]:
                t = float(np.clip(t, *m.jnt_range[jid]))
            if m.actuator_ctrllimited[a]:
                t = float(np.clip(t, *m.actuator_ctrlrange[a]))
            acts.append(a)
            qadr.append(m.jnt_qposadr[jid])
            dadr.append(m.jnt_dofadr[jid])
            targets.append(t)

            if stiffness != 1.0 and m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE:
                kp = float(m.actuator_gainprm[a, 0]) * stiffness
                kd = -float(m.actuator_biasprm[a, 2]) * float(np.sqrt(stiffness))
                m.actuator_gainprm[a, 0] = kp
                m.actuator_biasprm[a, 1] = -kp
                m.actuator_biasprm[a, 2] = -kd

        self.hold_act = np.array(acts, np.int32)
        self.hold_qadr = np.array(qadr, np.int32)
        self.hold_dadr = np.array(dadr, np.int32)
        self.hold_target = np.array(targets, float)


    def set_command(self, vx: float, vy: float, wz: float) -> None:
        values = np.asarray((vx, vy, wz), dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"velocity command must be three finite values, got {values!r}")
        was_idle = self._idle
        previous_wz = float(self.target_cmd[2])
        self.target_cmd[:] = values
        requested_idle = bool(np.linalg.norm(values) <= self.command_deadband)
        if requested_idle and was_idle:
            self.cmd[:] = 0.0
            self._heading_ref = None
        elif was_idle and not requested_idle:
            self._idle = False
            if self._recurrent:
                self.policy.hidden_state.zero_()
                self.policy.cell_state.zero_()
            self.action[:] = 0.0
            self.target_dof_pos = DEFAULT_ANGLES.copy()
            self.counter = 0
        if requested_idle or abs(float(values[2])) > self.command_deadband:
            self._heading_ref = None
        elif was_idle or abs(previous_wz) > self.command_deadband:
            self._heading_ref = self._base_yaw(self.data)

    def _base_yaw(self, data: mujoco.MjData) -> float:
        qw, qx, qy, qz = data.qpos[self.root_qadr + 3:self.root_qadr + 7]
        return float(
            np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
        )

    def _observation_command(self, data: mujoco.MjData) -> np.ndarray:
        result = self.cmd.copy()
        if (
            self._heading_ref is not None
            and abs(float(self.target_cmd[2])) <= self.command_deadband
            and np.linalg.norm(self.target_cmd[:2]) > self.command_deadband
        ):
            error = np.arctan2(
                np.sin(self._heading_ref - self._base_yaw(data)),
                np.cos(self._heading_ref - self._base_yaw(data)),
            )
            correction = (
                self.heading_hold_gain * error
                - self.heading_hold_damping * float(data.qvel[self.root_dadr + 5])
            )
            result[2] = np.clip(
                correction, -self.heading_hold_limit, self.heading_hold_limit
            )
        return result

    def _advance_command(self) -> None:
        delta = np.clip(
            self.target_cmd - self.cmd,
            -self.command_slew * (NOMINAL_DT * CONTROL_DECIMATION),
            self.command_slew * (NOMINAL_DT * CONTROL_DECIMATION),
        )
        self.cmd += delta
        if (
            np.linalg.norm(self.target_cmd) <= self.command_deadband
            and np.linalg.norm(self.cmd) <= self.command_deadband
        ):
            self.cmd[:] = 0.0
            self._idle = True
            self._heading_ref = None
            self._capture_idle_pose()

    def _capture_idle_pose(self) -> None:
        d = self.data
        self._idle_pos[:] = d.qpos[self.root_qadr:self.root_qadr + 3]
        self._idle_quat[:] = d.qpos[self.root_qadr + 3:self.root_qadr + 7]
        self._idle_leg_start[:] = d.qpos[self.leg_qadr]
        self._idle_blend = 0.0

    def _idle_control(self, data: mujoco.MjData) -> None:
        if not self.stationary_idle:
            if self.counter % CONTROL_DECIMATION == 0:
                self._infer(data)
            self.counter += 1
            data.ctrl[self.leg_act] = self.target_dof_pos
            return

        self._idle_blend = min(1.0, self._idle_blend + NOMINAL_DT / 0.5)
        target = ((1.0 - self._idle_blend) * self._idle_leg_start
                  + self._idle_blend * DEFAULT_ANGLES)
        data.ctrl[self.leg_act] = target

        qadr, dadr = self.root_qadr, self.root_dadr
        pos = data.qpos[qadr:qadr + 3]
        linear = data.qvel[dadr:dadr + 3]
        force = (np.array([6000.0, 6000.0, 2500.0]) * (self._idle_pos - pos)
                 - np.array([300.0, 300.0, 220.0]) * linear)

        rotation_error = np.zeros(3, float)
        mujoco.mju_subQuat(
            rotation_error,
            self._idle_quat,
            data.qpos[qadr + 3:qadr + 7],
        )
        angular = data.qvel[dadr + 3:dadr + 6]
        torque = 1000.0 * rotation_error - 80.0 * angular
        data.qfrc_applied[dadr:dadr + 3] = np.clip(force, -5000.0, 5000.0)
        data.qfrc_applied[dadr + 3:dadr + 6] = np.clip(torque, -1000.0, 1000.0)

    def reset(self, pos: Optional[Sequence[float]] = None, yaw: float = 0.0) -> None:
        m, d = self.model, self.data
        d.qpos[self.leg_qadr] = DEFAULT_ANGLES
        d.qvel[self.leg_dadr] = 0.0
        if self.hold_act.size:
            d.qpos[self.hold_qadr] = self.hold_target
            d.qvel[self.hold_dadr] = 0.0
        if pos is not None:
            d.qpos[self.root_qadr: self.root_qadr + 3] = np.asarray(pos, float)
        d.qpos[self.root_qadr + 3: self.root_qadr + 7] = [
            np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
        ]
        d.qvel[self.root_dadr: self.root_dadr + 6] = 0.0

        if self._recurrent:
            self.policy.hidden_state.zero_()
            self.policy.cell_state.zero_()
        self.action[:] = 0.0
        self.obs[:] = 0.0
        self.target_dof_pos = DEFAULT_ANGLES.copy()
        self.counter = 0
        self.cmd[:] = 0.0
        self._idle = bool(np.linalg.norm(self.target_cmd) <= self.command_deadband)
        self._heading_ref = (
            self._base_yaw(d)
            if not self._idle
            and abs(float(self.target_cmd[2])) <= self.command_deadband
            and np.linalg.norm(self.target_cmd[:2]) > self.command_deadband
            else None
        )
        self._idle_pos[:] = d.qpos[self.root_qadr:self.root_qadr + 3]
        self._idle_quat[:] = d.qpos[self.root_qadr + 3:self.root_qadr + 7]
        self._idle_leg_start[:] = DEFAULT_ANGLES
        self._idle_blend = 1.0
        if self.hold_act.size:
            d.ctrl[self.hold_act] = self.hold_target
        d.ctrl[self.leg_act] = self.target_dof_pos
        mujoco.mj_forward(m, d)


    def _infer(self, d: mujoco.MjData) -> None:
        qj = (d.qpos[self.leg_qadr] - DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj = d.qvel[self.leg_dadr] * DOF_VEL_SCALE
        quat = d.qpos[self.root_qadr + 3: self.root_qadr + 7]
        omega = d.qvel[self.root_dadr + 3: self.root_dadr + 6] * ANG_VEL_SCALE

        phase = (self.counter * NOMINAL_DT) % GAIT_PERIOD / GAIT_PERIOD

        self.obs[:3] = omega
        self.obs[3:6] = gravity_orientation(quat)
        self.obs[6:9] = self._observation_command(d) * CMD_SCALE
        self.obs[9:9 + NUM_ACTIONS] = qj
        self.obs[9 + NUM_ACTIONS:9 + 2 * NUM_ACTIONS] = dqj
        self.obs[9 + 2 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS] = self.action
        self.obs[9 + 3 * NUM_ACTIONS:] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]

        with self._torch.no_grad():
            t = self._torch.from_numpy(self.obs).unsqueeze(0)
            self.action = self.policy(t).detach().numpy().squeeze()
        self.target_dof_pos = self.action * ACTION_SCALE + DEFAULT_ANGLES

    def control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._idle:
            self._idle_control(data)
            if self.hold_act.size:
                data.ctrl[self.hold_act] = self.hold_target
            return

        data.qfrc_applied[self.root_dadr:self.root_dadr + 6] = 0.0
        if self.counter % CONTROL_DECIMATION == 0:
            self._advance_command()
            if self._idle:
                self._idle_control(data)
                if self.hold_act.size:
                    data.ctrl[self.hold_act] = self.hold_target
                return
            self._infer(data)
        self.counter += 1
        data.ctrl[self.leg_act] = self.target_dof_pos
        if self.hold_act.size:
            data.ctrl[self.hold_act] = self.hold_target

    def install(self) -> None:
        mujoco.set_mjcb_control(self.control)

    @staticmethod
    def uninstall() -> None:
        mujoco.set_mjcb_control(None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ros2_bridge.tools.interactive_sim import (  # noqa: PLC0415
        DEFAULT_KEEP_FREE,
        DEFAULT_ROBOT_SPAWN_YAW,
        InteractiveSim,
    )

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True,
                    help="base scene: an MJCF, or a val_861 asset-editor layout.json")
    ap.add_argument("--catalog", default=None,
                    help="asset catalog for a .json editor layout (default: the editor's)")
    ap.add_argument("--policy", required=True, help="unitree_rl_gym deploy/pre_train/g1/motion.pt")
    ap.add_argument("--pos", type=float, nargs="+", default=None, metavar="X",
                    help="spawn position (x y [z]); omit to ray-cast for open floor space")
    ap.add_argument("--yaw", type=float, default=None, metavar="RAD",
                    help=f"override the heading, in radians about +Z (positive turns left); "
                         f"defaults to DEFAULT_ROBOT_SPAWN_YAW "
                         f"({np.degrees(DEFAULT_ROBOT_SPAWN_YAW):g} deg)")
    ap.add_argument("--cmd", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("VX", "VY", "WZ"), help="base-frame velocity command")
    ap.add_argument("--assets-dir", default=None)
    ap.add_argument("--bound-mass", type=float, default=0.0,
                    help="in-memory mass floor, e.g. 1e-3; some scenes (val_6436) hold "
                         "degenerate free bodies that spam 'Nan/Inf in QACC' warnings")
    ap.add_argument("--bound-inertia", type=float, default=0.0, help="in-memory inertia floor")
    ap.add_argument("--headless", type=float, default=0.0, metavar="SECONDS",
                    help="run N seconds without a viewer and report where it ended up")
    ap.add_argument("--static-scene", action="store_true",
                    help="pin every scene prop by deleting its free joint (in memory). "
                         "Editor layouts are authored with physics paused, so their props "
                         "start interpenetrating and fly apart once the solver runs")
    ap.add_argument("--keep-free", action="append", default=None, metavar="SUBSTR",
                    help="body-name substring that keeps its free joint despite "
                         "--static-scene (repeatable). Giving any replaces the default, "
                         f"which is the val_861 soccer ball ({DEFAULT_KEEP_FREE[0][:12]}...)")
    ap.add_argument("--freeze-all", action="store_true",
                    help="freeze the soccer ball too -- clears the --keep-free default")
    ap.add_argument("--no-lift-free", action="store_true",
                    help="do not raise surviving free bodies out of interpenetration")
    ap.add_argument("--settle", type=float, default=0.0, metavar="SEC",
                    help="run SEC of physics before opening, so the scene starts at rest "
                         "instead of settling in front of you (ignored with --static-scene)")
    ap.add_argument("--window", default="auto", choices=["auto", "glfw", "opencv"],
                    help="window backend: 'glfw' (GPU blit, ~0.65 ms/frame) or 'opencv' "
                         "(CPU blit, ~3.1 ms; the scripts/view_g1_in_holodeck.py route). "
                         "'auto' honours $MJ_WINDOW, default glfw")
    ap.add_argument("--supersample", type=float, default=1.0, metavar="N",
                    help="render N times larger and box-downscale, for anti-aliasing; "
                         "Filament's own MSAA is inert in this build. 1.5 is a good "
                         "trade (1280x720: 7.9 -> 11.4 ms/frame)")
    ap.add_argument("--renderer", default="auto",
                    choices=["auto", "filament", "classic", "viewer"])
    ap.add_argument("--duration", type=float, default=None, help="close the viewer after N seconds")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="simulation speed relative to the wall clock (0.25 = slow motion)")
    ap.add_argument("--max-substeps", type=int, default=64,
                    help="cap on physics substeps per rendered frame")
    args = ap.parse_args(argv)

    sim = InteractiveSim(args.scene, assets_dir=args.assets_dir,
                         bound_mass=args.bound_mass or None,
                         bound_inertia=args.bound_inertia or None,
                         catalog_path=args.catalog,
                         static_scene=args.static_scene,
                         keep_free=() if args.freeze_all else args.keep_free,
                         lift_free=not args.no_lift_free,
                         settle=args.settle)
    pos, floor_z = args.pos, 0.0
    if pos is None:
        x, y, floor_z = sim.find_free_pos()
        pos = (x, y)
        print(f"auto placement: x={x:.2f} y={y:.2f} floor_z={floor_z:.3f}")
    yaw = DEFAULT_ROBOT_SPAWN_YAW if args.yaw is None else args.yaw
    spawn = sim.spawn_robot("g1", pos=pos, yaw=yaw, floor_z=floor_z, keyframe=None)

    walker = G1WalkPolicy(sim.model, sim.data, args.policy, prefix=spawn.prefix, cmd=args.cmd)
    walker.reset(pos=spawn.pos, yaw=yaw)
    walker.install()

    if args.headless:
        n = int(args.headless / sim.model.opt.timestep)
        z0 = float(sim.data.qpos[walker.root_qadr + 2])
        sim.step(n)
        p = sim.data.qpos[walker.root_qadr: walker.root_qadr + 3]
        print(f"after {args.headless}s: pelvis=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) "
              f"dz={p[2] - z0:+.3f}  {'UPRIGHT' if p[2] > z0 - 0.25 else 'FELL'}")
        return 0

    sim.run(renderer=args.renderer, duration=args.duration,
            speed=args.speed, max_substeps=args.max_substeps,
            window=args.window, supersample=args.supersample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
