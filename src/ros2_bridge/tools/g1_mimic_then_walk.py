from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mujoco

from ros2_bridge.tools.g1_mimic_policy_29dof import G1MimicPolicy29DoF
from ros2_bridge.tools.g1_walk_policy_29dof import G1WalkPolicy29DoF


class G1MimicThenWalkPolicy29DoF:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        mimic_policy_path: str | Path,
        motion_path: str | Path,
        walk_policy_path: str | Path,
        *,
        prefix: str = "",
        loop: bool = False,
        autostart: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.mimic = G1MimicPolicy29DoF(
            model,
            data,
            mimic_policy_path,
            motion_path,
            prefix=prefix,
            loop=loop,
            autostart=autostart,
        )
        self.walk = G1WalkPolicy29DoF(
            model,
            data,
            walk_policy_path,
            prefix=prefix,
            cmd=(0.0, 0.0, 0.0),
        )
        self._using_walk = False

    def __getattr__(self, name: str):
        return getattr(self.mimic, name)

    @property
    def using_walk(self) -> bool:
        return self._using_walk

    def reset(self, pos: Optional[Sequence[float]] = None, yaw: float = 0.0) -> None:
        self._using_walk = False
        self.mimic.reset(pos=pos, yaw=yaw)

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        if self._using_walk:
            self.walk.set_command(vx, vy, wz)
        else:
            self.mimic.set_command(vx, vy, wz)

    def control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._using_walk:
            self.walk.control(model, data)
            return
        self.mimic.control(model, data)
        if self.mimic.finished:
            self.walk.adopt_current_state()
            self._using_walk = True

    def metrics(self) -> dict[str, float | int | bool]:
        values = dict(self.mimic.metrics())
        values["switched_to_walk"] = self._using_walk
        return values

    def install(self) -> None:
        mujoco.set_mjcb_control(self.control)

    @staticmethod
    def uninstall() -> None:
        mujoco.set_mjcb_control(None)
