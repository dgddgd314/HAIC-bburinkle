"""Lightweight MuJoCo-only interactive simulation utility."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

try:
    from . import mj_renderer
except ImportError:  # pragma: no cover - script execution
    import mj_renderer

__all__ = ["InteractiveSim", "Spawn", "default_assets_dir", "resolve_robot_xml"]


_SRC_ROOT = Path(__file__).resolve().parents[2]
_G1_MODELS = _SRC_ROOT / "environment" / "robots" / "mujoco" / "g1" / "models"
ROBOT_XML: Dict[str, str] = {
    "g1": str(_G1_MODELS / "g1_dex.xml"),
    "g1_sensors": str(_G1_MODELS / "g1_dex_sensors_missions.xml"),
    "panda": str(
        _SRC_ROOT / "environment" / "robots" / "description" / "franka_description"
        / "libero_panda" / "panda.xml"
    ),
}

DEFAULT_ROBOT_SPAWN_YAW: float = math.radians(80.0)

DEFAULT_KEEP_FREE: Tuple[str, ...] = ("soccer_ball_center",)


def load_editor_spec(
    layout_path: str | os.PathLike,
    assets_dir: Optional[str | os.PathLike] = None,
    catalog_path: Optional[str | os.PathLike] = None,
) -> "mujoco.MjSpec":
    import importlib.util

    layout_path = _abspath(layout_path)
    if not layout_path.exists():
        raise FileNotFoundError(layout_path)

    from environment.test_zone.src import layout_loader as editor

    resolved_assets = editor.resolve_assets_dir(Path(assets_dir) if assets_dir else None)
    os.environ["MLSPACES_ASSETS_DIR"] = str(resolved_assets.resolve())

    catalog = _abspath(catalog_path) if catalog_path else editor.DEFAULT_CATALOG
    _, catalog_assets = editor.load_catalog(Path(catalog))
    editor.relocate_catalog_paths(catalog_assets, resolved_assets)
    layout = editor.load_layout(layout_path)
    editor._portableize_layout_texture_paths(layout, resolved_assets)
    editor._realign_custom_walls(layout, editor._scene_description(resolved_assets))
    editor.hydrate_layout_assets(layout, catalog_assets, resolved_assets)
    # Simulation must not call the editor save_layout() path and overwrite the layout.

    scene_spec, _runtime, _structures, _removed, _changed = editor.build_editor_spec(
        resolved_assets, catalog_assets, layout
    )
    return scene_spec


def freeze_scene_props(
    spec: "mujoco.MjSpec", keep_free: Sequence[str] = ()
) -> Tuple[int, List[str]]:
    patterns = [p.lower() for p in keep_free if p]
    removed, kept = 0, []
    for body in spec.bodies:
        name = body.name or ""
        for joint in list(body.joints):
            if joint.type != mujoco.mjtJoint.mjJNT_FREE:
                continue
            if any(p in name.lower() for p in patterns):
                kept.append(name)
                continue
            spec.delete(joint)
            removed += 1
    return removed, kept


def lift_free_bodies(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clearance: float = 2e-3,
    step: float = 5e-3,
    max_lift: float = 3.0,
    rounds: int = 8,
) -> Dict[str, float]:
    free_jnts = [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if not free_jnts:
        return {}
    bodies = {int(model.jnt_bodyid[j]): j for j in free_jnts}
    lifted: Dict[str, float] = {}

    for _ in range(rounds):
        mujoco.mj_forward(model, data)
        deepest: Dict[int, float] = {}
        for c in range(data.ncon):
            con = data.contact[c]
            if con.dist >= -clearance:
                continue
            for gid in (con.geom1, con.geom2):
                b = int(model.geom_bodyid[gid])
                while b not in bodies and model.body_parentid[b] not in (0, b):
                    b = int(model.body_parentid[b])
                if b in bodies:
                    deepest[b] = min(deepest.get(b, 0.0), float(con.dist))
        if not deepest:
            break
        for b, dist in deepest.items():
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            if lifted.get(name, 0.0) >= max_lift:
                continue
            adr = model.jnt_qposadr[bodies[b]]
            dz = max(step, -dist + clearance)
            data.qpos[adr + 2] += dz
            model.qpos0[adr + 2] += dz
            lifted[name] = lifted.get(name, 0.0) + dz
    mujoco.mj_forward(model, data)
    return lifted


def _fix_gridlayout(xml: str) -> str:
    def repl(match: "re.Match[str]") -> str:
        element = match.group(0)
        size = re.search(r'gridsize="(\d+)\s+(\d+)"', element)
        layout = re.search(r'gridlayout="([^"]*)"', element)
        if not (size and layout):
            return element
        cells = int(size.group(1)) * int(size.group(2))
        return element.replace(
            f'gridlayout="{layout.group(1)}"', f'gridlayout="{layout.group(1)[:cells]}"'
        )

    return re.sub(r'<texture[^>]*gridlayout="[^"]*"[^>]*/?>', repl, xml)


def _asset_roots(assets_dir: Optional[Path]) -> List[Path]:
    roots = []
    if assets_dir:
        roots.append(Path(assets_dir))
    roots.append(_SRC_ROOT / "environment" / "models" / "assets" / "custom")
    roots.append(Path.home() / ".cache" / "molmo-spaces-resources" / "objects")
    return [r for r in roots if r.is_dir()]


def export_scene_xml(
    spec: "mujoco.MjSpec",
    path: str | os.PathLike,
    assets_dir: Optional[Path] = None,
) -> Dict[str, int]:
    path = _abspath(path)
    base = Path(spec.modelfiledir or ".")
    xml = _fix_gridlayout(spec.to_xml())

    index: Dict[str, Path] = {}
    def lookup(name: str) -> Optional[Path]:
        if not index:
            for root in _asset_roots(assets_dir):
                for dirpath, _dirs, files in os.walk(root):
                    for f in files:
                        index.setdefault(f, Path(dirpath) / f)
        return index.get(name)

    stats = {"total": 0, "already_absolute": 0, "from_model_dir": 0,
             "from_asset_roots": 0, "unresolved": 0}
    cache: Dict[str, str] = {}

    def resolve(match: "re.Match[str]") -> str:
        ref = match.group(1)
        stats["total"] += 1
        if ref in cache:
            return f'file="{cache[ref]}"'
        out = ref
        if os.path.isabs(ref) and os.path.exists(ref):
            stats["already_absolute"] += 1
        elif (base / ref).exists():
            out = str((base / ref).resolve())
            stats["from_model_dir"] += 1
        else:
            found = lookup(os.path.basename(ref))
            if found is not None:
                out = str(found.resolve())
                stats["from_asset_roots"] += 1
            else:
                stats["unresolved"] += 1
        cache[ref] = out
        return f'file="{out}"'

    xml = re.sub(r'file="([^"]+)"', resolve, xml)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)

    model = mujoco.MjModel.from_xml_path(str(path))
    stats.update(nbody=model.nbody, ngeom=model.ngeom, nv=model.nv,
                 nmesh=model.nmesh, ntex=model.ntex, bytes=path.stat().st_size)
    return stats


def _abspath(p: str | os.PathLike) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(p))))


def default_assets_dir() -> Optional[Path]:
    raw = os.environ.get("MLSPACES_ASSETS_DIR")
    return _abspath(raw) if raw else None


def resolve_robot_xml(name: str, assets_dir: Optional[Path] = None) -> Path:
    p = _abspath(name)
    if p.suffix == ".xml" and p.is_file():
        return p

    assets_dir = assets_dir or default_assets_dir()
    if assets_dir is None:
        raise FileNotFoundError(
            f"cannot resolve robot {name!r}: no assets dir "
            "(set MLSPACES_ASSETS_DIR or pass assets_dir=)"
        )

    rel = ROBOT_XML.get(name)
    if rel is not None:
        xml = Path(rel) if Path(rel).is_absolute() else assets_dir / rel
        if xml.exists():
            return xml

    root = assets_dir / "robots" / name
    if root.is_dir():
        cands = sorted(root.glob(f"{name}*.xml")) or sorted(root.glob("*.xml"))
        if len(cands) >= 1:
            return cands[0]

    raise FileNotFoundError(
        f"cannot resolve robot {name!r} under {assets_dir}; "
        f"known robots: {sorted(ROBOT_XML)}"
    )


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], float)


def _sanitize(text: str) -> str:
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in text).strip("_") or "asset"


def _free_joint(body: "mujoco.MjsBody") -> Optional["mujoco.MjsJoint"]:
    for j in body.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            return j
    return None


@dataclass
class Spawn:
    xml: Path
    prefix: str
    pos: np.ndarray
    quat: np.ndarray
    freeze_root: bool = False
    keyframe: Optional[str] = None
    hold_keyframe: bool = True
    floor_z: float = 0.0
    root_body: str = ""
    has_freejoint: bool = False
    source_root_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def name(self) -> str:
        return self.prefix.rstrip("_")


class InteractiveSim:
    def __init__(
        self,
        scene_xml: str | os.PathLike,
        assets_dir: Optional[str | os.PathLike] = None,
        preserve_state: bool = True,
        bound_mass: Optional[float] = None,
        bound_inertia: Optional[float] = None,
        filament_options: Optional[Dict[str, Any]] = None,
        catalog_path: Optional[str | os.PathLike] = None,
        static_scene: bool = False,
        keep_free: Optional[Sequence[str]] = None,
        lift_free: bool = True,
        settle: float = 0.0,
    ) -> None:
        self.scene_xml = _abspath(scene_xml)
        if not self.scene_xml.exists():
            raise FileNotFoundError(self.scene_xml)
        self.assets_dir = _abspath(assets_dir) if assets_dir else default_assets_dir()
        if self.assets_dir is None:
            for parent in self.scene_xml.parents:
                if parent.name == "scenes" and (parent.parent / "robots").is_dir():
                    self.assets_dir = parent.parent
                    break
        self.preserve_state = preserve_state

        if self.scene_xml.suffix.lower() == ".json":
            self._base_spec = load_editor_spec(
                self.scene_xml, assets_dir=self.assets_dir, catalog_path=catalog_path
            )
        else:
            self._base_spec = mujoco.MjSpec.from_file(str(self.scene_xml))
        self.bound_mass, self.bound_inertia = bound_mass, bound_inertia
        if bound_mass is not None:
            self._base_spec.compiler.boundmass = bound_mass
        if bound_inertia is not None:
            self._base_spec.compiler.boundinertia = bound_inertia

        self.static_scene = static_scene
        self.keep_free = DEFAULT_KEEP_FREE if keep_free is None else tuple(keep_free)
        self.lift_free = lift_free
        self.frozen_joints, self.free_bodies = 0, []
        if static_scene:
            self.frozen_joints, self.free_bodies = freeze_scene_props(
                self._base_spec, self.keep_free
            )

        self.filament_options = filament_options
        mj_renderer.apply_filament_options(self._base_spec, filament_options)

        self._child_cache: Dict[str, mujoco.MjSpec] = {}

        self.spawns: List[Spawn] = []
        self._counters: Dict[str, int] = {}
        self._lock = threading.RLock()
        self.generation = 0

        self.spec: mujoco.MjSpec = self._base_spec.copy()
        self.model: mujoco.MjModel = self.spec.compile()
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.lifted: Dict[str, float] = {}
        if lift_free:
            self.lifted = lift_free_bodies(self.model, self.data)
        if settle > 0 and not static_scene:
            self.step(int(settle / self.model.opt.timestep))


    def spawn_mjcf(
        self,
        asset_xml: str | os.PathLike,
        pos: Optional[Sequence[float]] = None,
        quat: Optional[Sequence[float]] = None,
        yaw: Optional[float] = None,
        prefix: Optional[str] = None,
        name: Optional[str] = None,
        freeze_root: bool = False,
        keyframe: Optional[str] = None,
        hold_keyframe: bool = True,
        floor_z: float = 0.0,
    ) -> Spawn:
        xml = _abspath(asset_xml)
        if not xml.exists():
            raise FileNotFoundError(xml)
        if quat is not None and yaw is not None:
            raise ValueError("pass either quat or yaw, not both")

        base = _sanitize(name or xml.stem)
        if prefix is None:
            idx = self._counters.get(base, 0)
            self._counters[base] = idx + 1
            prefix = f"{base}_{idx}_"
        if not prefix.endswith("_"):
            prefix += "_"
        if any(s.prefix == prefix for s in self.spawns):
            raise ValueError(f"prefix {prefix!r} already used")

        q = np.array(quat, float) if quat is not None else yaw_to_quat(yaw or 0.0)
        q = q / np.linalg.norm(q)

        rec = Spawn(
            xml=xml,
            prefix=prefix,
            pos=np.zeros(3),
            quat=q,
            freeze_root=freeze_root,
            keyframe=keyframe,
            hold_keyframe=hold_keyframe,
            floor_z=floor_z,
        )
        rec.pos = np.asarray(pos, float) if pos is not None else None  # type: ignore[assignment]

        with self._lock:
            self.spawns.append(rec)
            try:
                self._rebuild()
            except Exception:
                self.spawns.pop()
                self._rebuild()
                raise
        return rec

    def spawn_robot(
        self,
        name: str = "g1",
        pos: Optional[Sequence[float]] = None,
        quat: Optional[Sequence[float]] = None,
        yaw: Optional[float] = None,
        keyframe: Optional[str] = "stand",
        freeze_root: bool = False,
        hold_keyframe: bool = True,
        prefix: Optional[str] = None,
        floor_z: float = 0.0,
    ) -> Spawn:
        if quat is None and yaw is None:
            yaw = DEFAULT_ROBOT_SPAWN_YAW
        xml = resolve_robot_xml(name, self.assets_dir)
        return self.spawn_mjcf(
            xml,
            pos=pos,
            quat=quat,
            yaw=yaw,
            prefix=prefix,
            name=name if Path(name).suffix != ".xml" else None,
            freeze_root=freeze_root,
            keyframe=keyframe,
            hold_keyframe=hold_keyframe,
            floor_z=floor_z,
        )

    def remove(self, spawn: Spawn | str) -> None:
        with self._lock:
            key = spawn if isinstance(spawn, str) else spawn.prefix
            key = key if key.endswith("_") else key + "_"
            before = len(self.spawns)
            self.spawns = [s for s in self.spawns if s.prefix != key]
            if len(self.spawns) == before:
                raise KeyError(f"no spawn with prefix {key!r}")
            self._rebuild()


    def _load_child(self, rec: Spawn) -> mujoco.MjSpec:
        cached = self._child_cache.get(str(rec.xml))
        if cached is None:
            cached = mujoco.MjSpec.from_file(str(rec.xml))
            self._child_cache[str(rec.xml)] = cached
        child = cached.copy()

        if not child.worldbody.bodies:
            raise ValueError(f"{rec.xml} has no body in <worldbody>")
        root = child.worldbody.bodies[0]
        rec.root_body = rec.prefix + (root.name or "")
        rec.source_root_pos = np.array(root.pos, float)

        if rec.pos is None:
            rec.pos = rec.source_root_pos + np.array([0.0, 0.0, rec.floor_z])
        else:
            p = np.asarray(rec.pos, float).ravel()
            if p.size == 2:
                p = np.array([p[0], p[1], rec.floor_z + rec.source_root_pos[2]])
            elif p.size != 3:
                raise ValueError("pos must have 2 or 3 elements")
            rec.pos = p

        fj = _free_joint(root)
        rec.has_freejoint = fj is not None
        if rec.freeze_root and fj is not None:
            child.delete(fj)
            # Removing a free joint also requires trimming its seven keyframe qpos entries.
            for key in list(child.keys):
                qp = np.asarray(key.qpos, float)
                if qp.size:
                    key.qpos = qp[7:]
            rec.has_freejoint = False

        root.pos = [0.0, 0.0, 0.0]
        root.quat = [1.0, 0.0, 0.0, 0.0]
        return child

    def _rebuild(self) -> None:
        spec = self._base_spec.copy()
        for rec in self.spawns:
            child = self._load_child(rec)
            frame = spec.worldbody.add_frame(
                name=f"{rec.prefix}site", pos=list(rec.pos), quat=list(rec.quat)
            )
            spec.attach(child, prefix=rec.prefix, frame=frame)

        model = spec.compile()
        data = mujoco.MjData(model)

        old = (self.model, self.data) if getattr(self, "model", None) is not None else None
        if old is not None and self.preserve_state:
            self._migrate_state(old[0], old[1], model, data)

        self.spec, self.model, self.data = spec, model, data
        for rec in self.spawns:
            self._apply_spawn_state(rec, fresh_only=(old is not None and self.preserve_state))
        mujoco.mj_forward(self.model, self.data)
        if self.lift_free:
            # Recompilation restores spec qpos, so reapply the pose correction.
            self.lifted = lift_free_bodies(self.model, self.data)
        self.generation += 1

    @staticmethod
    def _migrate_state(om: mujoco.MjModel, od: mujoco.MjData, nm: mujoco.MjModel, nd: mujoco.MjData) -> None:
        for j in range(om.njnt):
            nj_name = mujoco.mj_id2name(om, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not nj_name:
                continue
            k = mujoco.mj_name2id(nm, mujoco.mjtObj.mjOBJ_JOINT, nj_name)
            if k < 0 or nm.jnt_type[k] != om.jnt_type[j]:
                continue
            nq = {0: 7, 1: 4}.get(int(om.jnt_type[j]), 1)
            nv = {0: 6, 1: 3}.get(int(om.jnt_type[j]), 1)
            nd.qpos[nm.jnt_qposadr[k]: nm.jnt_qposadr[k] + nq] = \
                od.qpos[om.jnt_qposadr[j]: om.jnt_qposadr[j] + nq]
            nd.qvel[nm.jnt_dofadr[k]: nm.jnt_dofadr[k] + nv] = \
                od.qvel[om.jnt_dofadr[j]: om.jnt_dofadr[j] + nv]
        for a in range(om.nu):
            an = mujoco.mj_id2name(om, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if not an:
                continue
            k = mujoco.mj_name2id(nm, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
            if k >= 0:
                nd.ctrl[k] = od.ctrl[a]
        nd.time = od.time

    def _prefixed_joints(self, prefix: str) -> List[int]:
        out = []
        for j in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if n and n.startswith(prefix):
                out.append(j)
        return out

    def _apply_spawn_state(self, rec: Spawn, fresh_only: bool = False) -> None:
        m, d = self.model, self.data
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rec.root_body)
        if bid < 0:
            return
        if fresh_only and rec is not self.spawns[-1]:
            return

        jids = self._prefixed_joints(rec.prefix)

        if rec.keyframe:
            kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, rec.prefix + rec.keyframe)
            if kid < 0:
                raise KeyError(
                    f"keyframe {rec.keyframe!r} not found in {rec.xml.name} "
                    f"(available: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)]})"
                )
            kq = m.key_qpos[kid]
            for j in jids:
                nq = {0: 7, 1: 4}.get(int(m.jnt_type[j]), 1)
                a = m.jnt_qposadr[j]
                d.qpos[a: a + nq] = kq[a: a + nq]

        if rec.has_freejoint:
            jadr = m.body_jntadr[bid]
            if jadr >= 0 and m.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                a = m.jnt_qposadr[jadr]
                d.qpos[a: a + 3] = rec.pos
                d.qpos[a + 3: a + 7] = rec.quat
                d.qvel[m.jnt_dofadr[jadr]: m.jnt_dofadr[jadr] + 6] = 0.0

        if rec.hold_keyframe:
            for a in range(m.nu):
                an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                if not an or not an.startswith(rec.prefix):
                    continue
                if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                    continue
                j = int(m.actuator_trnid[a, 0])
                if m.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                    continue
                if m.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE:
                    continue
                target = float(d.qpos[m.jnt_qposadr[j]])
                if m.actuator_ctrllimited[a]:
                    target = float(np.clip(target, *m.actuator_ctrlrange[a]))
                d.ctrl[a] = target


    def find_free_pos(
        self,
        radius: float = 0.4,
        height: float = 1.4,
        grid: float = 0.4,
        margin: float = 0.5,
        enclosure: float = 15.0,
    ) -> Tuple[float, float, float]:
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        lo = d.geom_xpos.min(axis=0)
        hi = d.geom_xpos.max(axis=0)
        centroid = d.geom_xpos.mean(axis=0)[:2]
        gid = np.zeros(1, np.int32)
        dirs = [
            np.array([math.cos(t), math.sin(t), 0.0])
            for t in np.linspace(0.0, 2 * math.pi, 12, endpoint=False)
        ]
        up = np.array([0.0, 0.0, 1.0])
        down = -up

        best, best_score, best_clear = None, -np.inf, 0.0
        for x in np.arange(lo[0] + margin, hi[0] - margin + 1e-9, grid):
            for y in np.arange(lo[1] + margin, hi[1] - margin + 1e-9, grid):
                top = np.array([float(x), float(y), float(hi[2]) + 1.0])
                dist = mujoco.mj_ray(m, d, top, down, None, 1, -1, gid)
                if dist < 0:
                    continue
                floor_z = top[2] - dist
                if not (-0.05 <= floor_z <= 0.15):
                    continue
                probe = np.array([float(x), float(y), floor_z + height * 0.5])
                if 0 <= mujoco.mj_ray(m, d, probe, up, None, 1, -1, gid) < height * 0.5:
                    continue
                hits = [mujoco.mj_ray(m, d, probe, v, None, 1, -1, gid) for v in dirs]
                if any(h < 0 or h > enclosure for h in hits):
                    continue
                clear = min(hits)
                if clear < radius:
                    continue
                score = min(clear, 2.0) - 0.02 * float(np.linalg.norm([x, y] - centroid))
                if score > best_score:
                    best, best_score, best_clear = (float(x), float(y), floor_z), score, clear
        if best is None:
            raise RuntimeError(f"no free spot with {radius} m clearance found in {self.scene_xml.name}")
        return best

    def reset(self) -> None:
        with self._lock:
            self.spawns.clear()
            self._counters.clear()
            self.spec = self._base_spec.copy()
            self.model = self.spec.compile()
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
            self.generation += 1

    def reset_state(self) -> None:
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            for rec in self.spawns:
                self._apply_spawn_state(rec)
            mujoco.mj_forward(self.model, self.data)


    def step(self, n: int = 1) -> None:
        with self._lock:
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)

    def run(self, realtime: bool = True, duration: Optional[float] = None,
            hard_exit: bool = False, renderer: str = "auto",
            speed: float = 1.0, max_substeps: int = 64, **kwargs: Any) -> None:
        if renderer == "auto":
            renderer = "filament" if mj_renderer.IS_FILAMENT else "viewer"
        if renderer != "viewer":
            mj_renderer.run_interactive(
                self, renderer=renderer, realtime=realtime, duration=duration,
                hard_exit=hard_exit, speed=speed, max_substeps=max_substeps, **kwargs,
            )
            return
        if kwargs:
            raise TypeError(f"unsupported options for renderer='viewer': {sorted(kwargs)}")
        self._run_stock_viewer(realtime=realtime, duration=duration, hard_exit=hard_exit,
                               speed=speed, max_substeps=max_substeps)

    def run_filament(self, **kwargs: Any) -> None:
        self.run(renderer="filament", **kwargs)

    def render_rgb(
        self,
        width: int = 1280,
        height: int = 720,
        camera: Any = -1,
        renderer: str = "auto",
        shadows: bool = True,
        max_geom: int = 30000,
    ) -> np.ndarray:
        mj_renderer.resolve_renderer(renderer)
        with mj_renderer.OffscreenRenderer(
            self.model, width=width, height=height, max_geom=max_geom, shadows=shadows
        ) as r:
            return r.render_rgb(self.data, camera=camera)

    def _run_stock_viewer(self, realtime: bool = True, duration: Optional[float] = None,
                          hard_exit: bool = False, speed: float = 1.0,
                          max_substeps: int = 64) -> None:
        try:
            from mujoco import viewer as mj_viewer
        except ImportError as exc:
            raise RuntimeError(
                f"mujoco.viewer is unusable in this build ({exc}).\n"
                f"The Filament fork (mujoco {mujoco.__version__}, mjRENDERER="
                f"{mj_renderer.RENDERER_NAME!r}) exports no mjui_* symbols, so its "
                "_simulate module cannot load.\n"
                "Use renderer='filament' instead -- it provides the interactive viewer."
            ) from exc

        t_end = None if duration is None else time.time() + duration
        while True:
            gen = self.generation
            model, data = self.model, self.data
            viewer = mj_viewer.launch_passive(model, data)
            done = False
            sim_debt, prev_wall = 0.0, time.time()
            try:
                while viewer.is_running() and self.generation == gen:
                    if t_end is not None and time.time() > t_end:
                        done = True
                        break
                    t0 = time.time()
                    dt = model.opt.timestep
                    if realtime:
                        sim_debt += (t0 - prev_wall) * speed
                        n = int(sim_debt / dt)
                        if n > max_substeps:
                            n, sim_debt = max_substeps, 0.0
                        else:
                            sim_debt -= n * dt
                    else:
                        n = max_substeps
                    prev_wall = t0
                    with self._lock:
                        if self.generation != gen:
                            break
                        for _ in range(n):
                            mujoco.mj_step(model, data)
                    viewer.sync()
                    if realtime:
                        lag = 1.0 / 120.0 - (time.time() - t0)
                        if lag > 0:
                            time.sleep(lag)
                if not viewer.is_running():
                    done = True
            except BaseException:
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                if hard_exit:
                    os._exit(1)
                raise
            finally:
                if done and hard_exit:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(0)
                viewer.close()
            if done:
                return


    def to_xml(self) -> str:
        return self.spec.to_xml()

    def summary(self) -> str:
        m = self.model
        lines = [
            f"scene   : {self.scene_xml.name}",
            f"model   : nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} nkey={m.nkey}",
        ]
        for s in self.spawns:
            lines.append(
                f"  spawn {s.prefix:<12} {s.xml.name:<24} pos={np.round(s.pos, 3).tolist()} "
                f"free={s.has_freejoint} key={s.keyframe}"
            )
        return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True,
                    help="base scene: an MJCF, or a val_861 asset-editor layout.json "
                         "(composed exactly as scripts/edit_val861_assets.py composes it)")
    ap.add_argument("--catalog", default=None,
                    help="asset catalog for a .json editor layout (default: the editor's)")
    ap.add_argument("--spawn-g1", action="store_true", help="spawn a Unitree G1")
    ap.add_argument("--pos", type=float, nargs="+", default=None, metavar="X",
                    help="spawn position (x y [z]); z defaults to the asset's own root height")
    ap.add_argument("--yaw", type=float, default=None, metavar="RAD",
                    help=f"override the heading, in radians about +Z (positive turns left). "
                         f"Robots default to DEFAULT_ROBOT_SPAWN_YAW "
                         f"({math.degrees(DEFAULT_ROBOT_SPAWN_YAW):g} deg)")
    ap.add_argument("--auto-pos", action="store_true",
                    help="ray-cast for open floor space instead of using --pos")
    ap.add_argument("--keyframe", default="stand", help="robot keyframe ('none' to skip)")
    ap.add_argument("--freeze-root", action="store_true", help="pin the robot in place (in memory only)")
    ap.add_argument("--spawn-mjcf", action="append", default=[], metavar="XML",
                    help="extra MJCF to attach (repeatable)")
    ap.add_argument("--assets-dir", default=None)
    ap.add_argument("--bound-mass", type=float, default=0.0,
                    help="optional in-memory mass floor (e.g. 1e-3); 0 keeps the scene's own setting")
    ap.add_argument("--bound-inertia", type=float, default=0.0, help="optional in-memory inertia floor")
    ap.add_argument("--no-viewer", action="store_true", help="compile and report, do not open the viewer")
    ap.add_argument("--steps", type=int, default=0, help="headless steps to run with --no-viewer")
    ap.add_argument("--duration", type=float, default=None, help="close the viewer after N seconds")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="simulation speed relative to the wall clock (0.25 = slow motion)")
    ap.add_argument("--max-substeps", type=int, default=64,
                    help="cap on physics substeps per rendered frame")

    ap.add_argument("--export-xml", default=None, metavar="PATH",
                    help="write the composed scene (after --static-scene / spawns) to a "
                         "standalone MJCF and exit. Asset paths are made absolute")
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
    ap.add_argument("--renderer", default="auto", choices=["auto", "filament", "classic", "viewer"],
                    help="rendering layer: 'filament' (MuJoCo Filament renderer), 'classic' "
                         "(classic OpenGL, same render loop), 'viewer' (stock mujoco.viewer), "
                         "or 'auto'")
    ap.add_argument("--width", type=int, default=1280, help="render width")
    ap.add_argument("--height", type=int, default=720, help="render height")
    ap.add_argument("--camera", default="-1",
                    help="initial camera: -1 for the free camera, or an id / name")
    ap.add_argument("--no-shadows", action="store_true", help="disable shadow rendering")
    ap.add_argument("--paused", action="store_true", help="start paused (SPACE to resume)")
    ap.add_argument("--env-light", type=float, default=None, metavar="I",
                    help="filament.fallback.environment_light_intensity (default 15000); "
                         "Filament-only, Holodeck scenes are near-black without it")
    ap.add_argument("--filament-option", action="append", default=[], metavar="KEY=V[,V...]",
                    help="extra filament.* model numeric, e.g. filament.out.exposure=1.2 "
                         "(repeatable; --list-filament-options prints the keys)")
    ap.add_argument("--list-filament-options", action="store_true",
                    help="print every filament.* key this MuJoCo build understands and exit")
    ap.add_argument("--save-rgb", default=None, metavar="PNG",
                    help="render one offscreen frame to PNG and exit (headless, no window)")
    args = ap.parse_args(argv)

    if args.list_filament_options:
        print(f"mujoco {mujoco.__version__}  mjRENDERER={mj_renderer.RENDERER_NAME}")
        for key in mj_renderer.FILAMENT_OPTION_KEYS:
            default = mj_renderer.DEFAULT_FILAMENT_OPTIONS.get(key)
            print(f"  {key}" + (f"   [default {default}]" if default is not None else ""))
        return 0

    filament_options = dict(mj_renderer.DEFAULT_FILAMENT_OPTIONS)
    if args.env_light is not None:
        filament_options["filament.fallback.environment_light_intensity"] = args.env_light
    for item in args.filament_option:
        key, _, raw = item.partition("=")
        if not raw:
            ap.error(f"--filament-option needs KEY=VALUE, got {item!r}")
        filament_options[key.strip()] = [float(v) for v in raw.split(",")]

    camera: Any = args.camera
    try:
        camera = int(camera)
    except ValueError:
        pass

    sim = InteractiveSim(args.scene, assets_dir=args.assets_dir,
                         bound_mass=args.bound_mass or None,
                         bound_inertia=args.bound_inertia or None,
                         filament_options=filament_options,
                         catalog_path=args.catalog,
                         static_scene=args.static_scene,
                         keep_free=() if args.freeze_all else args.keep_free,
                         lift_free=not args.no_lift_free,
                         settle=args.settle)
    if args.static_scene:
        print(f"static scene: froze {sim.frozen_joints} free joints"
              + (f"; still dynamic: {', '.join(sim.free_bodies)}" if sim.free_bodies else ""))
    if sim.lifted:
        print("lifted out of penetration: "
              + ", ".join(f"{n.split('/')[-2] if '/' in n else n} +{dz*100:.1f} cm"
                          for n, dz in sim.lifted.items()))
    print(sim.summary())

    pos, floor_z = args.pos, 0.0
    if args.auto_pos or (pos is None and (args.spawn_g1 or args.spawn_mjcf)):
        x, y, floor_z = sim.find_free_pos()
        pos = (x, y)
        print(f"auto placement: x={x:.2f} y={y:.2f} floor_z={floor_z:.3f}")

    if args.spawn_g1:
        sim.spawn_robot(
            "g1",
            pos=pos,
            yaw=args.yaw,
            keyframe=None if args.keyframe.lower() == "none" else args.keyframe,
            freeze_root=args.freeze_root,
            floor_z=floor_z,
        )
    for extra in args.spawn_mjcf:
        sim.spawn_mjcf(extra, pos=pos, yaw=args.yaw, floor_z=floor_z)

    print(sim.summary())

    if args.steps:
        t0 = time.time()
        sim.step(args.steps)
        print(f"stepped {args.steps} ({sim.data.time:.2f}s sim) in {time.time() - t0:.2f}s wall")

    if args.export_xml:
        stats = export_scene_xml(sim.spec, args.export_xml, sim.assets_dir)
        print(f"wrote {args.export_xml}  ({stats['bytes']/1e6:.1f} MB)")
        print(f"  asset refs: {stats['total']} total | {stats['already_absolute']} already absolute | "
              f"{stats['from_model_dir']} from the model dir | "
              f"{stats['from_asset_roots']} from asset roots | {stats['unresolved']} unresolved")
        print(f"  reloads as: nbody={stats['nbody']} ngeom={stats['ngeom']} nv={stats['nv']} "
              f"nmesh={stats['nmesh']} ntex={stats['ntex']}")
        return 0

    if args.save_rgb:
        t0 = time.time()
        rgb = sim.render_rgb(width=args.width, height=args.height, camera=camera,
                             renderer="auto" if args.renderer == "viewer" else args.renderer,
                             shadows=not args.no_shadows)
        mj_renderer._save_png(rgb, args.save_rgb)
        print(f"wrote {args.save_rgb} ({rgb.shape[1]}x{rgb.shape[0]}, "
              f"{mj_renderer.RENDERER_NAME} renderer) in {time.time() - t0:.2f}s")
        sys.stdout.flush()
        os._exit(0)  # Filament engine teardown at interpreter exit is racy in this build.

    if args.no_viewer:
        return 0

    sim.run(hard_exit=True, duration=args.duration, renderer=args.renderer,
            speed=args.speed, max_substeps=args.max_substeps,
            **({} if args.renderer == "viewer" else dict(
                width=args.width, height=args.height, camera=camera,
                shadows=not args.no_shadows, paused=args.paused,
                window=args.window, supersample=args.supersample)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
