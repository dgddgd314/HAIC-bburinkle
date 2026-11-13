from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np

from ros2_bridge.tools.g1_mimic_policy_29dof import G1MimicPolicy29DoF
from ros2_bridge.tools.g1_walk_policy_29dof import G1WalkPolicy29DoF

READY_BLEND_S = 0.8
READY_SETTLE_S = 0.2
READY_SUPPORT_RAMP = 0.25
READY_LIFT_M = 0.015
READY_HANDOVER_VELOCITY = True


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)

DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[2] / "environment" / "robots" / "control" / "g1"
    / "custom" / "soccer" / "soccer_policies.json"
)


def load_registry(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    registry = json.loads(path.read_text(encoding="utf-8"))
    policies = registry.get("policies")
    if not isinstance(policies, dict) or not policies:
        raise ValueError(f"{path}: 'policies' must be a non-empty object")
    for name, entry in policies.items():
        for key in ("onnx", "motion"):
            value = entry.get(key)
            if not value:
                raise ValueError(f"{path}: policy '{name}' has no '{key}' path")
            resolved = Path(value)
            if not resolved.is_absolute():
                resolved = (path.parent / resolved).resolve()
            entry[key] = str(resolved)
            entry[f"{key}_exists"] = resolved.exists()
        spawn = entry.get("spawn") or {}
        has_spawn = (
            isinstance(spawn.get("pos"), (list, tuple))
            and len(spawn["pos"]) >= 2
            and "yaw_deg" in spawn
        )
        entry["ready"] = bool(entry["onnx_exists"] and entry["motion_exists"] and has_spawn)
    registry["path"] = str(path)
    return registry


def ready_policy_names(registry: dict) -> list[str]:
    return [name for name, entry in registry["policies"].items() if entry["ready"]]


def resolve_default(registry: dict) -> Optional[str]:
    ready = ready_policy_names(registry)
    default = registry.get("default")
    if default in ready:
        return default
    return ready[0] if ready else None


class G1SoccerKickSwitch29DoF:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        walk_policy_path: str | Path,
        registry_path: str | Path = DEFAULT_REGISTRY,
        *,
        prefix: str = "",
        policy_name: Optional[str] = None,
        spawn_override: Optional[Sequence[float]] = None,
        gap_policy_path: Optional[str | Path] = None,
    ) -> None:
        self.model, self.data, self.prefix = model, data, prefix
        self.walk = G1WalkPolicy29DoF(
            model, data, walk_policy_path, prefix=prefix, cmd=(0.0, 0.0, 0.0)
        )
        self.registry = load_registry(registry_path)
        self.ready_names = ready_policy_names(self.registry)
        if policy_name is not None:
            if policy_name not in self.registry["policies"]:
                raise ValueError(
                    f"unknown soccer policy '{policy_name}'; registry has "
                    f"{sorted(self.registry['policies'])}"
                )
            entry = self.registry["policies"][policy_name]
            if spawn_override is not None:
                entry["spawn"] = {
                    "pos": [float(spawn_override[0]), float(spawn_override[1])],
                    "yaw_deg": float(spawn_override[2]),
                }
                entry["ready"] = bool(entry["onnx_exists"] and entry["motion_exists"])
                self.ready_names = ready_policy_names(self.registry)
            if not entry["ready"]:
                raise ValueError(
                    f"soccer policy '{policy_name}' is not ready (files present: onnx="
                    f"{entry['onnx_exists']}, motion={entry['motion_exists']}, spawn="
                    f"{entry.get('spawn')}); ready: {self.ready_names}"
                )
            self.selected: Optional[str] = policy_name
        else:
            self.selected = resolve_default(self.registry)
        self._mimics: dict[str, G1MimicPolicy29DoF] = {}
        self.mimic: Optional[G1MimicPolicy29DoF] = None
        self.kicking = False
        self._ready: Optional[dict] = None
        self.ready_s = READY_BLEND_S
        self.ready_error: Optional[dict] = None
        self.standing_z: Optional[float] = None
        self.kick_count = 0
        self.last_kick: Optional[str] = None
        self.gap = None
        self.gap_active = False
        self.gap_result: Optional[str] = None
        if gap_policy_path is not None and Path(gap_policy_path).is_file():
            from ros2_bridge.tools.g1_gap_policy_29dof import G1GapPolicy29DoF

            self.gap = G1GapPolicy29DoF(model, data, gap_policy_path, prefix=prefix)

    def __getattr__(self, name: str):
        if name in {"walk", "mimic"} or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.walk, name)

    def select(self, name: str) -> str:
        if name not in self.ready_names:
            raise ValueError(f"soccer policy '{name}' is not ready; ready: {self.ready_names}")
        self.selected = name
        return name

    def cycle(self) -> Optional[str]:
        if not self.ready_names:
            return None
        if self.selected not in self.ready_names:
            self.selected = self.ready_names[0]
        else:
            index = (self.ready_names.index(self.selected) + 1) % len(self.ready_names)
            self.selected = self.ready_names[index]
        return self.selected

    def spawn_for(self, name: Optional[str] = None) -> tuple[np.ndarray, float]:
        name = name or self.selected
        if name is None:
            raise RuntimeError("no calibrated soccer policy is available")
        entry = self.registry["policies"][name]
        if not entry["ready"]:
            raise RuntimeError(f"soccer policy '{name}' is not ready")
        spawn = entry["spawn"]
        if self.standing_z is not None:
            z = float(self.standing_z)
        else:
            z = float(self.data.qpos[self.walk.root_qadr + 2])
        pos = np.asarray((float(spawn["pos"][0]), float(spawn["pos"][1]), z), dtype=float)
        return pos, math.radians(float(spawn["yaw_deg"]))

    def reset(self, pos: Optional[Sequence[float]] = None, yaw: float = 0.0) -> None:
        if pos is not None and len(pos) == 3:
            self.standing_z = float(pos[2])
        self.kicking = False
        self._ready = None
        if self.gap_active:
            self.gap_active = False
            self.gap.deactivate()
            self.walk._apply_policy_gains()
        self.walk.reset(pos=pos, yaw=yaw)

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        if not self.kicking and not self.gap_active:
            self.walk.set_command(vx, vy, wz)

    def current_pose(self) -> tuple[np.ndarray, float]:
        q = self.walk.root_qadr
        pos = np.asarray(self.data.qpos[q:q + 3], dtype=float).copy()
        w, x, y, z = (float(v) for v in self.data.qpos[q + 3:q + 7])
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return pos, yaw

    def start_kick(
        self,
        name: Optional[str] = None,
        *,
        pose: Optional[tuple[Sequence[float], float]] = None,
        teleport: bool = True,
        ready_s: Optional[float] = None,
    ) -> tuple[str, np.ndarray, float]:
        if self.gap_active:
            raise RuntimeError("the gap-traversal policy is running")
        name = name or self.selected
        if not teleport:
            here, yaw = self.current_pose()
            pos = np.asarray((here[0], here[1], float(self.standing_z) if self.standing_z else here[2]))
        elif pose is not None:
            xy, yaw = pose
            z = float(self.standing_z) if self.standing_z is not None else float(self.data.qpos[self.walk.root_qadr + 2])
            pos = np.asarray((float(xy[0]), float(xy[1]), float(xy[2]) if len(xy) >= 3 else z), dtype=float)
            yaw = float(yaw)
        else:
            pos, yaw = self.spawn_for(name)
        entry = self.registry["policies"][name]
        mimic = self._mimics.get(name)
        if mimic is None:
            mimic = G1MimicPolicy29DoF(
                self.model, self.data, entry["onnx"], entry["motion"],
                prefix=self.prefix, loop=False, autostart=True,
            )
            self._mimics[name] = mimic
        self.walk.set_command(0.0, 0.0, 0.0)
        self.walk.cmd[:] = 0.0
        self.walk.target_cmd[:] = 0.0
        blend = self.ready_s if ready_s is None else float(ready_s)
        if not teleport and blend > 0.0:
            q = self.walk.root_qadr
            root_pos, root_quat, joints = mimic.start_state(pos, yaw)
            self._ready = {
                "mimic": mimic, "pos": pos, "yaw": yaw, "t": 0.0, "T": blend,
                "q0": self.data.qpos[mimic.qadr].astype(float).copy(), "q1": joints,
                "p0": self.data.qpos[q:q + 3].astype(float).copy(), "p1": root_pos,
                "r0": self.data.qpos[q + 3:q + 7].astype(float).copy(), "r1": root_quat,
                "weight": float(self.model.body_subtreemass[int(self.model.jnt_bodyid[self.walk.root_jid])])
                * float(-self.model.opt.gravity[2]),
            }
            self.mimic = mimic
            self.kicking = True
            self.kick_count += 1
            self.last_kick = name
            return name, pos, yaw
        # mimic.reset() invokes mj_forward; detach the old callback and clear its root wrench
        # before teleporting, or it can throw G1 toward the previous pose.
        mujoco.set_mjcb_control(None)
        try:
            root = self.walk.root_dadr
            self.data.qfrc_applied[root:root + 6] = 0.0
            self.data.qfrc_applied[self.walk.dadr] = 0.0
            self.data.xfrc_applied[int(self.model.jnt_bodyid[self.walk.root_jid])] = 0.0
            mimic.reset(pos=pos, yaw=yaw)
            self.data.qacc_warmstart[root:root + 6] = 0.0
            self.data.qacc_warmstart[self.walk.dadr] = 0.0
            self.mimic = mimic
            self.kicking = True
        finally:
            mujoco.set_mjcb_control(self.control)
        self.kick_count += 1
        self.last_kick = name
        return name, pos, yaw

    def start_gap(self, heading: float, gaps: Optional[int] = None,
                  then: Optional[float] = None, after_m: Optional[float] = None) -> None:
        if self.gap is None:
            raise RuntimeError("no gap-traversal policy installed (--gap-policy)")
        if self.kicking:
            raise RuntimeError("a kick is in progress")
        if not self.gap_active:
            self.walk.set_command(0.0, 0.0, 0.0)
            self.walk.cmd[:] = 0.0
            self.walk.target_cmd[:] = 0.0
            self.gap.activate(heading)
            self.gap_active = True
            self.gap_result = None
        self.gap.set_heading(heading, gaps, then, after_m)

    def stop_gap(self) -> None:
        if not self.gap_active:
            return
        self.gap_active = False
        self.gap.deactivate()
        self.walk._apply_policy_gains()
        self.walk.adopt_current_state()

    def _ready_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        r = self._ready
        mimic = r["mimic"]
        r["t"] += float(model.opt.timestep)
        s = min(1.0, r["t"] / r["T"])
        w = _smooth(s)
        support = _smooth(s / READY_SUPPORT_RAMP) * (1.0 - _smooth((s - (1.0 - READY_SUPPORT_RAMP)) / READY_SUPPORT_RAMP))
        data.ctrl[mimic.act] = r["q0"] + w * (r["q1"] - r["q0"])
        data.qfrc_applied[mimic.dadr] = data.qfrc_bias[mimic.dadr]
        target_quat = r["r0"].copy()
        if w > 0.0:
            dq = np.zeros(3)
            mujoco.mju_subQuat(dq, r["r1"], r["r0"])
            mujoco.mju_quatIntegrate(target_quat, dq, w)
        q, d = self.walk.root_qadr, self.walk.root_dadr
        target_pos = r["p0"] + w * (r["p1"] - r["p0"])
        target_pos[2] += READY_LIFT_M * support
        force = (
            np.array([6000.0, 6000.0, 2500.0]) * (target_pos - data.qpos[q:q + 3])
            - np.array([300.0, 300.0, 220.0]) * data.qvel[d:d + 3]
        )
        force[2] += support * r["weight"]
        rotation_error = np.zeros(3)
        mujoco.mju_subQuat(rotation_error, target_quat, data.qpos[q + 3:q + 7])
        torque = 1000.0 * rotation_error - 80.0 * data.qvel[d + 3:d + 6]
        data.qfrc_applied[d:d + 3] = np.clip(force, -5000.0, 5000.0)
        data.qfrc_applied[d + 3:d + 6] = np.clip(torque, -1000.0, 1000.0)
        if r["t"] >= r["T"] + READY_SETTLE_S:
            self.ready_error = {
                "joint_max_deg": float(np.degrees(np.max(np.abs(data.qpos[mimic.qadr] - r["q1"])))),
                "root_m": float(np.linalg.norm(data.qpos[q:q + 3] - r["p1"])),
            }
            data.qfrc_applied[d:d + 6] = 0.0
            data.qfrc_applied[mimic.dadr] = 0.0
            if READY_HANDOVER_VELOCITY:
                data.qvel[mimic.dadr] = mimic.reference_joint_vel[0]
            mimic.begin_from_current(r["pos"], r["yaw"])
            self._ready = None

    def control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.gap_active:
            self.gap.control(model, data)
            if self.gap.finished is not None:
                self.gap_result = self.gap.finished
                self.stop_gap()
            return
        if self._ready is not None:
            self._ready_control(model, data)
            return
        if self.kicking and self.mimic is not None:
            self.mimic.control(model, data)
            if self.mimic.finished:
                self.walk.adopt_current_state()
                self.kicking = False
            return
        self.walk.control(model, data)

    def metrics(self) -> dict[str, float | int | bool | str | None]:
        if self.mimic is not None:
            values: dict = dict(self.mimic.metrics())
        else:
            values = {
                "reference_frames": 0, "last_frame": 0, "finished": False,
                "samples": 0, "joint_rmse_rad": 0.0,
                "min_root_z_m": float("nan"), "min_up_z": float("nan"),
            }
        values["switched_to_walk"] = bool(self.mimic is not None and not self.kicking)
        values["ready_blend"] = self.ready_error
        values["policy"] = self.last_kick or self.selected
        values["kick_count"] = self.kick_count
        return values

    def install(self) -> None:
        mujoco.set_mjcb_control(self.control)

    @staticmethod
    def uninstall() -> None:
        mujoco.set_mjcb_control(None)
