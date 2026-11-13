from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

SCAN_X = np.arange(-0.5, 1.5 + 0.025, 0.05)
SCAN_Y = np.arange(-0.35, 0.35 + 0.025, 0.05)
# Match training flatten order: body-yaw frame, y outer / x inner.
SCAN_OFFSETS = np.array([(x, y) for y in SCAN_Y for x in SCAN_X])
RAY_START_HEIGHT = 2.0
RAY_MAX_DISTANCE = 5.0
SCAN_SCALE = 0.5
POLICY_DT = 0.02
FOOT_SITE_OFFSET = np.array([0.04, 0.0, -0.035])
MIN_GAP_SPACING_M = 1.0
NUM_OBS = 726


GAP_FOOT_CAPSULES = (
    ("1", (0.1, -0.026, -0.030, 0.05, -0.027, -0.030)), ("3", (-0.052, -0.01, -0.030, 0.13, -0.01, -0.030)),
    ("5", (-0.052, 0.01, -0.030, 0.13, 0.01, -0.030)), ("7", (0.1, 0.026, -0.030, 0.05, 0.026, -0.030)),
    ("8", (0.1, -0.026, -0.020, 0.05, -0.027, -0.020)), ("10", (-0.052, -0.01, -0.020, 0.13, -0.01, -0.020)),
    ("12", (-0.052, 0.01, -0.020, 0.13, 0.01, -0.020)), ("14", (0.1, 0.026, -0.020, 0.05, 0.026, -0.020)),
)


def add_gap_feet(spec) -> int:
    added = 0
    for side in ("left", "right"):
        body = spec.body(f"{side}_ankle_roll_link")
        if body is None:
            continue
        for tag, fromto in GAP_FOOT_CAPSULES:
            g = body.add_geom(name=f"{side}_foot_gap{tag}_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                              fromto=list(fromto), size=[0.005, 0.0, 0.0])
            # Compile with collisions enabled to build the BVH; disable immediately afterward.
            g.contype, g.conaffinity = 1, 1
            g.condim, g.priority, g.group = 3, 1, 3
            g.friction = [0.6, 0.005, 0.0001]
            g.rgba = [0.0, 0.0, 0.0, 0.0]
            g.density = 0.0
            added += 1
    return added


def disable_gap_feet(model) -> int:
    ids = [g for g in range(model.ngeom)
           if "_foot_gap" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    model.geom_contype[ids] = 0
    model.geom_conaffinity[ids] = 0
    return len(ids)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class G1GapPolicy29DoF:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, policy_path: str | Path,
                 *, prefix: str = "") -> None:
        import onnxruntime as ort

        self.model, self.data, self.prefix = model, data, prefix
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        floats = lambda key: np.array([float(v) for v in meta[key].split(",")])  # noqa: E731
        self.joint_names = meta["joint_names"].split(",")
        self.default_pos = floats("default_joint_pos")
        self.action_scale = floats("action_scale")
        self.kp = floats("joint_stiffness")
        self.kd = floats("joint_damping")
        inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
        if inp.shape != [1, NUM_OBS] or out.shape != [1, len(self.joint_names)]:
            raise ValueError(f"unexpected gap policy contract {inp.shape} -> {out.shape}")
        self.input_name, self.output_name = inp.name, out.name

        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n) for n in self.joint_names]
        if min(jid) < 0:
            raise KeyError(f"gap policy joints missing: {[n for n, j in zip(self.joint_names, jid) if j < 0]}")
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid], np.int32)
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid], np.int32)
        by_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)
                    if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        self.act = np.array([by_joint[j] for j in jid], np.int32)
        self.root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and model.jnt_bodyid[j] == self.root_body]
        if self.root_body < 0 or not free:
            raise KeyError("gap policy needs the G1 pelvis free joint")
        self.root_qadr = int(model.jnt_qposadr[free[0]])
        self.root_dadr = int(model.jnt_dofadr[free[0]])
        gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + "imu_ang_vel")
        if gyro < 0:
            raise KeyError("gap policy needs the imu_ang_vel gyro")
        self.gyro_adr = int(model.sensor_adr[gyro])
        self.foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{side}_ankle_roll_link")
                            for side in ("left", "right")]
        g1_bodies = {b for b in range(model.nbody) if self._ancestor(b, self.root_body)}
        self.g1_geom = np.array([int(model.geom_bodyid[g]) in g1_bodies for g in range(model.ngeom)])
        self.foot_geoms = []
        for side in ("left", "right"):
            root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{side}_ankle_roll_link")
            subtree = {b for b in range(model.nbody) if self._ancestor(b, root)}
            self.foot_geoms.append({g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in subtree})
        self.geomgroup = np.array([1, 1, 1, 0, 0, 0], np.uint8)
        feet = sorted(self.foot_geoms[0] | self.foot_geoms[1])
        name = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""  # noqa: E731
        self._gap_feet = np.array([g for g in feet if "_foot_gap" in name(g)], np.int32)
        self._own_feet = np.array([g for g in feet if "_foot_gap" not in name(g)
                                   and (model.geom_contype[g] or model.geom_conaffinity[g])], np.int32)
        self._own_feet_bits = (model.geom_contype[self._own_feet].copy(),
                               model.geom_conaffinity[self._own_feet].copy())
        scene = ~self.g1_geom
        collides = (model.geom_contype > 0) | (model.geom_conaffinity > 0)
        self._scene_visual = np.flatnonzero(scene & ~collides & (model.geom_group <= 2))
        self._scene_collision = np.flatnonzero(scene & collides)

        self.action = np.zeros(len(self.joint_names), np.float32)
        self.obs = np.zeros(NUM_OBS, np.float32)
        self.target = self.default_pos.copy()
        self.heading = 0.0
        self.command_active = False
        self.z_support = 0.0
        self.air_time = np.zeros(2)
        self._last_time = 0.0
        self.counter = 0
        self.decimation = max(1, round(POLICY_DT / float(model.opt.timestep)))
        self.last_scan = np.zeros(len(SCAN_OFFSETS))
        self.target_gaps: Optional[int] = None
        self._pending_turn: Optional[tuple[float, float]] = None
        self._run_in = 0.0
        self.gaps_crossed = 0
        self.finished: Optional[str] = None
        self._under_steps = 0
        self._free_steps = 8
        self._travel = MIN_GAP_SPACING_M
        self._clear_steps = 0
        self._tilt_steps = 0

    def _ancestor(self, body: int, root: int) -> bool:
        while body > 0:
            if body == root:
                return True
            body = int(self.model.body_parentid[body])
        return False

    def apply_gains(self) -> None:
        for a, kp, kd in zip(self.act, self.kp, self.kd):
            self.model.actuator_gainprm[a, 0] = kp
            self.model.actuator_biasprm[a, 1] = -kp
            self.model.actuator_biasprm[a, 2] = -kd

    def activate(self, heading: Optional[float]) -> None:
        self.apply_gains()
        self.use_training_feet(True)
        self.target_gaps = None
        self.gaps_crossed = 0
        self.finished = None
        self._under_steps = 0
        self._free_steps = 8
        self._travel = MIN_GAP_SPACING_M
        self._clear_steps = 0
        self._tilt_steps = 0
        self.action[:] = 0.0
        self.target[:] = self.default_pos
        self.air_time[:] = 0.0
        self._last_time = float(self.data.time)
        self.counter = 0
        self.z_support = self._support_height()
        self.set_heading(heading)
        r = self.root_dadr
        self.data.qfrc_applied[r:r + 6] = 0.0

    def use_training_feet(self, on: bool) -> None:
        if not len(self._gap_feet):
            return
        m = self.model
        contype, conaff = self._own_feet_bits
        if on:
            m.geom_contype[self._gap_feet] = int(contype.max())
            m.geom_conaffinity[self._gap_feet] = int(conaff.max())
            m.geom_contype[self._own_feet] = 0
            m.geom_conaffinity[self._own_feet] = 0
        else:
            m.geom_contype[self._gap_feet] = 0
            m.geom_conaffinity[self._gap_feet] = 0
            m.geom_contype[self._own_feet] = contype
            m.geom_conaffinity[self._own_feet] = conaff

    def deactivate(self) -> None:
        self.use_training_feet(False)

    def set_heading(self, heading: Optional[float], gaps: Optional[int] = None,
                    then: Optional[float] = None, after_m: Optional[float] = None) -> None:
        """World heading in radians; None stands still. gaps ends a crossing; then/after_m schedules a turn."""

        self._pending_turn = None if then is None else (float(then), float(after_m or 0.0))
        self._run_in = 0.0
        if gaps is not None:
            self.target_gaps = self.gaps_crossed + int(gaps)
        self.command_active = heading is not None
        if heading is not None:
            self.heading = float(heading)

    def yaw(self) -> float:
        w, x, y, z = self.data.qpos[self.root_qadr + 3:self.root_qadr + 7]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _support_height(self) -> float:
        heights = []
        gid = np.array([-1], np.int32)
        for p in self._foot_points():
            p = p.copy()
            p[2] += 0.3
            for _ in range(4):
                dist = mujoco.mj_ray(self.model, self.data, p, np.array([0.0, 0.0, -1.0]),
                                     None, 1, self.root_body, gid)
                if dist < 0:
                    break
                if not self.g1_geom[gid[0]]:
                    heights.append(p[2] - dist)
                    break
                p[2] -= dist + 1e-4
        return float(max(heights)) if heights else 0.0

    def _foot_points(self) -> list[np.ndarray]:
        d = self.data
        return [d.xpos[b] + d.xmat[b].reshape(3, 3) @ FOOT_SITE_OFFSET for b in self.foot_bodies]

    def _height_scan(self) -> np.ndarray:
        m = self.model
        vis, col = self._scene_visual, self._scene_collision
        saved = (m.geom_group[vis].copy(), m.geom_group[col].copy(), m.geom_rgba[col, 3].copy())
        try:
            m.geom_group[vis] = 5
            m.geom_group[col] = 1
            m.geom_rgba[col, 3] = 1.0  # mj_ray ignores fully transparent geoms.
            return self._cast_scan()
        finally:
            m.geom_group[vis], m.geom_group[col], m.geom_rgba[col, 3] = saved

    def _cast_scan(self) -> np.ndarray:
        d, m = self.data, self.model
        base = d.xpos[self.root_body]
        yaw = self.yaw()
        c, s = math.cos(yaw), math.sin(yaw)
        xs = base[0] + c * SCAN_OFFSETS[:, 0] - s * SCAN_OFFSETS[:, 1]
        ys = base[1] + s * SCAN_OFFSETS[:, 0] + c * SCAN_OFFSETS[:, 1]
        z0 = base[2] + RAY_START_HEIGHT
        down = np.array([0.0, 0.0, -1.0])
        gid = np.array([-1], np.int32)
        out = np.empty(len(SCAN_OFFSETS))
        for i in range(len(SCAN_OFFSETS)):
            dist = mujoco.mj_ray(m, d, np.array([xs[i], ys[i], z0]), down, self.geomgroup, 1,
                                 self.root_body, gid)
            if dist < 0 or dist > RAY_MAX_DISTANCE:
                out[i] = -1.0
            else:
                out[i] = (z0 - dist - self.z_support) / SCAN_SCALE
        np.clip(out, -1.0, 1.0, out=out)
        self.last_scan = out
        return out

    def _foot_contacts(self) -> tuple[np.ndarray, np.ndarray]:
        d, m = self.data, self.model
        found = np.zeros(2)
        force = np.zeros((2, 3))
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            for k in range(2):
                feet = self.foot_geoms[k]
                if g1 in feet and not self.g1_geom[g2]:
                    sign = 1.0  # Contact force is on geom2 (ground) from geom1 (foot).
                elif g2 in feet and not self.g1_geom[g1]:
                    sign = -1.0
                else:
                    continue
                mujoco.mj_contactForce(m, d, i, f6)
                force[k] += sign * (con.frame.reshape(3, 3).T @ f6[:3])
                found[k] += 1
        return found, force

    def _build_obs(self) -> None:
        d = self.data
        q = d.qpos[self.root_qadr + 3:self.root_qadr + 7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        found, force = self._foot_contacts()
        now = float(d.time)
        dt = now - self._last_time
        self._last_time = now
        self.air_time = np.where(found > 0, 0.0, self.air_time + dt)
        err = _wrap(self.heading - self.yaw())
        heading = ([math.cos(err), math.sin(err), 1.0] if self.command_active else [1.0, 0.0, 0.0])
        feet_z = [p[2] - self.z_support for p in self._foot_points()]
        o = self.obs
        o[0:3] = d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        o[3:6] = R.T @ np.array([0.0, 0.0, -1.0])
        o[6:35] = d.qpos[self.qadr] - self.default_pos
        o[35:64] = d.qvel[self.dadr]
        o[64:93] = self.action
        o[93:708] = self._height_scan()
        o[708:711] = heading
        o[711:714] = R.T @ d.qvel[self.root_dadr:self.root_dadr + 3]
        o[714:716] = feet_z
        o[716:718] = self.air_time
        o[718:720] = (found > 0).astype(np.float32)
        o[720:726] = (np.sign(force) * np.log1p(np.abs(force))).reshape(-1)

    def _progress(self) -> None:
        if self._pending_turn is not None:
            self._run_in += max(0.0, float(self.obs[711])) * POLICY_DT
            if self._run_in >= self._pending_turn[1]:
                self.heading = self._pending_turn[0]
                self._pending_turn = None

        scan = self.last_scan.reshape(len(SCAN_Y), len(SCAN_X))
        edges = scan[[0, -1]]
        pit = (edges < -0.5).any(axis=0)
        # Ignore outer-column voids swept by the scan while turning.
        squared = abs(_wrap(self.heading - self.yaw())) < math.radians(30.0)
        behind = squared and bool(pit[(SCAN_X >= -0.3) & (SCAN_X <= -0.1)].any())
        near = bool(pit[(SCAN_X >= -0.5) & (SCAN_X <= 0.1)].any())
        # Debounce pit edges and enforce MIN_GAP_SPACING_M to count each gap once.
        self._travel += max(0.0, float(self.obs[711])) * POLICY_DT
        if behind:
            self._under_steps += 1
            if self._under_steps == 2 and self._free_steps >= 8 and self._travel >= MIN_GAP_SPACING_M:
                self.gaps_crossed += 1
                self._travel = 0.0
            if self._under_steps >= 2:
                self._free_steps = 0
        else:
            self._under_steps = 0
            self._free_steps += 1
        self._clear_steps = 0 if near else self._clear_steps + 1
        if (self.target_gaps is not None and self.gaps_crossed >= self.target_gaps
                and self._clear_steps >= 3):
            self.finished = "done"
        tilted = self.obs[5] > -0.5
        self._tilt_steps = self._tilt_steps + 1 if tilted else 0
        if self._tilt_steps >= 10 or self.air_time.min() > 1.2:
            self.finished = "fell"

    def control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model
        data.qfrc_applied[self.root_dadr:self.root_dadr + 6] = 0.0
        self.decimation = max(1, round(POLICY_DT / float(self.model.opt.timestep)))
        if self.counter % self.decimation == 0:
            self._build_obs()
            out = self.session.run([self.output_name], {self.input_name: self.obs[None, :]})[0]
            self.action[:] = np.asarray(out, np.float32).reshape(-1)
            if not np.all(np.isfinite(self.action)):
                raise FloatingPointError("gap policy produced a non-finite action")
            self.target[:] = self.default_pos + self.action_scale * self.action
            if self.finished is None:
                self._progress()
        self.counter += 1
        data.ctrl[self.act] = self.target
