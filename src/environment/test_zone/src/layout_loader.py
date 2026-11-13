"""Runtime layout assembler for the fixed HAIC val_861 test zone."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from collections import deque
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

try:
    from .asset_taxonomy import (
        ASSET_BROWSE_CATEGORIES,
        asset_browse_category,
        asset_default_dynamic,
    )
except ImportError:
    from asset_taxonomy import (
        ASSET_BROWSE_CATEGORIES,
        asset_browse_category,
        asset_default_dynamic,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCENE_SOURCE = "holodeck-objaverse-val"
TEST_ZONE_ROOT = Path(__file__).resolve().parents[1]
VENDORED_ASSETS = PROJECT_ROOT / "src" / "environment" / "models" / "assets" / "molmo"
EDITOR_ROOT = TEST_ZONE_ROOT
CUSTOM_SURFACE_ROOT = TEST_ZONE_ROOT / "surface_textures"
DEFAULT_CATALOG = TEST_ZONE_ROOT / "world" / "asset_catalog.json"
DEFAULT_OBJATHOR_DB = TEST_ZONE_ROOT / "world" / "objathor_catalog.sqlite"
DEFAULT_LAYOUT = TEST_ZONE_ROOT / "world" / "layout.json"
THUMBNAIL_DIR = TEST_ZONE_ROOT / "thumbnails"
BASE_WALL_THICKNESS = 0.005
DOOR_FRAME_THICKNESS = 0.012

KEY_ENTER = 257
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265
KEY_PAGE_UP = 266
KEY_PAGE_DOWN = 267
KEY_DELETE = 261
KEY_SPACE = 32
KEY_Q = ord("Q")
KEY_E = ord("E")
KEY_A = ord("A")
KEY_C = ord("C")
KEY_D = ord("D")
KEY_O = ord("O")
KEY_S = ord("S")
KEY_U = ord("U")
KEY_X = ord("X")
KEY_B = ord("B")
KEY_M = ord("M")
KEY_P = ord("P")
KEY_R = ord("R")
KEY_H = ord("H")
KEY_F2 = 291
KEY_MINUS = ord("-")
KEY_EQUAL = ord("=")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--search", help="Print catalog matches and exit.")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="ASSET[:COUNT]",
        help="Add one or more catalog assets before opening the editor.",
    )
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Open the graphical asset picker before the MuJoCo viewer.",
    )
    parser.add_argument(
        "--reset-layout",
        action="store_true",
        help="Clear the editable layout; the downloaded val_861 remains untouched.",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def resolve_assets_dir(cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value.expanduser().resolve()
    env_value = os.environ.get("MLSPACES_ASSETS_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    if (VENDORED_ASSETS / "scenes" / SCENE_SOURCE / "val_861.xml").is_file():
        return VENDORED_ASSETS.resolve()
    raise RuntimeError("Set MLSPACES_ASSETS_DIR or pass --assets-dir.")


def yaw_quaternion(yaw_degrees: float) -> list[float]:
    half = math.radians(yaw_degrees) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def quaternion_conjugate(quaternion: list[float]) -> list[float]:
    w, x, y, z = quaternion
    return [w, -x, -y, -z]


def quaternion_rotate(quaternion: list[float], vector: list[float]) -> list[float]:
    pure = [0.0, *vector]
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, pure),
        quaternion_conjugate(quaternion),
    )
    return rotated[1:]


def load_catalog(path: Path) -> tuple[dict, list[dict]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Asset catalog is missing: {path}\n"
            f"It is part of this checkout and is not generated at runtime; "
            f"restore the file from a backup or another copy of the repo."
        )
    document = json.loads(path.read_text())
    return document, document["assets"]


def relocate_catalog_paths(catalog_assets: list[dict], assets_dir: Path) -> None:
    custom_root = assets_dir.parent / "custom"
    for asset in catalog_assets:
        source = str(asset.get("xml_path", ""))
        if not source:
            continue
        candidate = Path(source)
        if not candidate.is_absolute():
            asset["xml_path"] = str(PROJECT_ROOT / candidate)
            continue
        if candidate.is_file():
            continue
        if asset.get("provider") == "custom":
            asset["xml_path"] = str(custom_root / candidate.parent.name / "model.xml")
            continue
        marker = "/objects/thor/"
        if marker in source:
            relative = source.split(marker, 1)[1]
            parts = Path(relative).parts
            if parts and re.fullmatch(r"20\d{6}", parts[0]):
                relative = str(Path(*parts[1:]))
            target = assets_dir / "objects" / "thor" / relative
            packaged = target.parent / candidate.parent.name / target.name
            asset["xml_path"] = str(packaged if packaged.is_file() else target)
            continue
        uid = asset.get("asset_id", "")
        if uid:
            hits = sorted(assets_dir.rglob(f"{uid}.xml"))
            if hits:
                asset["xml_path"] = str(hits[0])


def load_layout(path: Path) -> dict:
    if not path.exists():
        layout = {"version": 2, "base_scene": 861, "objects": []}
        _ensure_structure_layout(layout)
        return layout
    layout = json.loads(path.read_text())
    if layout.get("base_scene") != 861:
        raise RuntimeError("The editor layout must use val_861 as its base scene.")
    _ensure_structure_layout(layout)
    return layout


def _ensure_structure_layout(layout: dict) -> None:
    previous_version = int(layout.get("version", 1))
    layout["version"] = max(6, previous_version)
    structure = layout.setdefault("structure", {})
    structure.setdefault("removed_base_walls", [])
    structure.setdefault("removed_base_doors", [])
    structure.setdefault("removed_base_floors", [])
    structure.setdefault("walls", [])
    structure.setdefault("doors", [])
    structure.setdefault("floors", [])
    structure.setdefault("wall_finishes", {})
    structure.setdefault("room_wall_finishes", {})
    structure.setdefault("room_finish_panels", [])
    structure.setdefault("window_replacements", [])
    structure.setdefault("floor_finishes", {})
    structure.setdefault("wall_default_finish", None)
    structure.setdefault("floor_default_finish", None)
    structure.setdefault("auto_floor", True)
    structure.setdefault(
        "wall_defaults", {"height": 3.2, "thickness": BASE_WALL_THICKNESS}
    )
    structure.setdefault(
        "door_defaults", {"height": 2.1, "thickness": DOOR_FRAME_THICKNESS}
    )
    # Migrate only the old 12 cm default; preserve explicitly customized wall thickness.
    if previous_version < 3:
        if abs(float(structure["wall_defaults"].get("thickness", 0.12)) - 0.12) < 1e-6:
            structure["wall_defaults"]["thickness"] = BASE_WALL_THICKNESS
        if abs(float(structure["door_defaults"].get("thickness", 0.12)) - 0.12) < 1e-6:
            structure["door_defaults"]["thickness"] = DOOR_FRAME_THICKNESS
        for wall in structure["walls"]:
            if abs(float(wall.get("thickness", 0.12)) - 0.12) < 1e-6:
                wall["thickness"] = BASE_WALL_THICKNESS
        for door in structure["doors"]:
            if abs(float(door.get("thickness", 0.12)) - 0.12) < 1e-6:
                door["thickness"] = DOOR_FRAME_THICKNESS


def save_layout(path: Path, layout: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = copy.deepcopy(layout)
    document.pop("_needs_save", None)
    document["objects"] = [
        item for item in document["objects"] if not item.get("draft", False)
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def searchable_text(asset: dict) -> str:
    fields = [
        asset.get("catalog_id", ""),
        asset.get("asset_id", ""),
        asset.get("display_name", ""),
        asset.get("provider", ""),
        asset.get("path_category", ""),
        *asset.get("aliases", []),
        *asset.get("categories", []),
        *asset.get("search_terms", []),
    ]
    return " ".join(fields).lower()


def search_assets(
    assets: list[dict],
    query: str,
    *,
    limit: int = 100,
    importable_only: bool = True,
) -> list[dict]:
    words = query.lower().split()
    scored: list[tuple[tuple[int, int, str], dict]] = []
    for asset in assets:
        if importable_only and not (
            asset.get("importable") or asset.get("downloadable")
        ):
            continue
        haystack = searchable_text(asset)
        if not all(word in haystack for word in words):
            continue
        display = asset.get("display_name", "").lower()
        asset_id = asset.get("asset_id", "").lower()
        exact = int(query.lower() not in (display, asset_id, asset.get("catalog_id", "").lower()))
        starts = int(not display.startswith(query.lower()) and not asset_id.startswith(query.lower()))
        scored.append(((exact, starts, display), asset))
    scored.sort(key=lambda item: item[0])
    return [asset for _, asset in scored[:limit]]


def _objathor_row_to_asset(row: tuple) -> dict:
    return {
        "catalog_id": row[0],
        "asset_id": row[1],
        "display_name": row[2],
        "provider": "objathor",
        "format": "mjcf",
        "aliases": row[3].split("\x1f") if row[3] else [],
        "categories": row[4].split("\x1f") if row[4] else [],
        "search_terms": row[5].split("\x1f") if row[5] else [],
        "xml_path": row[6],
        "root_body": row[1],
        "importable": bool(int(row[7])),
        "browse_category": row[8],
        "downloadable": True,
        "path_category": "Objathor / Objaverse (downloads on selection)",
        "objathor_metadata": True,
    }


def search_objathor_assets(
    query: str,
    *,
    limit: int = 500,
    db_path: Path = DEFAULT_OBJATHOR_DB,
) -> list[dict]:
    if not db_path.exists():
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        words = re.findall(r"[\w-]+", query.lower(), flags=re.UNICODE)
        if words:
            expression = " AND ".join(f'"{word.replace(chr(34), "")}"*' for word in words)
            rows = connection.execute(
                "SELECT catalog_id, asset_id, display_name, aliases, categories, "
                "search_terms, xml_path, importable, browse_category FROM assets "
                "WHERE assets MATCH ? ORDER BY bm25(assets) LIMIT ?",
                (expression, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT catalog_id, asset_id, display_name, aliases, categories, "
                "search_terms, xml_path, importable, browse_category FROM assets "
                "ORDER BY display_name LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        connection.close()
    return [_objathor_row_to_asset(row) for row in rows]


def search_objathor_category(
    category: str,
    *,
    limit: int = 500,
    db_path: Path = DEFAULT_OBJATHOR_DB,
) -> list[dict]:
    if not db_path.exists():
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT a.catalog_id, a.asset_id, a.display_name, a.aliases, "
            "a.categories, a.search_terms, a.xml_path, a.importable, "
            "a.browse_category FROM assets AS a "
            "JOIN category_index AS c ON c.rowid = a.rowid "
            "WHERE c.browse_category = ? ORDER BY a.display_name LIMIT ?",
            (category, limit),
        ).fetchall()
    finally:
        connection.close()
    return [_objathor_row_to_asset(row) for row in rows]


def resolve_objathor_asset(
    identifier: str,
    db_path: Path = DEFAULT_OBJATHOR_DB,
) -> dict | None:
    if not db_path.exists():
        return None
    lookup_asset_id = (
        identifier.split(":", 1)[1]
        if identifier.startswith(("objaverse:", "objathor:"))
        else identifier
    )
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT catalog_id, asset_id, display_name, aliases, categories, "
            "search_terms, xml_path, importable, browse_category FROM assets "
            "WHERE catalog_id = ? OR asset_id = ? LIMIT 1",
            (identifier, lookup_asset_id),
        ).fetchone()
    finally:
        connection.close()
    return None if row is None else _objathor_row_to_asset(row)


def resolve_catalog_asset(assets: list[dict], identifier: str) -> dict:
    exact = [
        asset
        for asset in assets
        if asset.get("catalog_id") == identifier or asset.get("asset_id") == identifier
    ]
    exact = [
        asset for asset in exact
        if asset.get("importable") or asset.get("downloadable")
    ]
    if len(exact) != 1:
        raise ValueError(
            f"Expected one MJCF/downloadable catalog match for {identifier!r}, found {len(exact)}."
        )
    return exact[0]


def _register_catalog_asset(assets: list[dict], asset: dict) -> dict:
    for existing in assets:
        if existing["catalog_id"] == asset["catalog_id"]:
            existing.update(asset)
            return existing
    assets.append(asset)
    return asset


def materialize_catalog_asset(asset: dict, assets_dir: Path) -> dict:
    xml_path = Path(asset.get("xml_path", ""))
    if asset.get("importable") and xml_path.is_file():
        return asset
    if not asset.get("downloadable"):
        raise RuntimeError(f"Asset is not MuJoCo-ready: {asset['catalog_id']}")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(assets_dir.resolve()))
    inferred_cache = assets_dir.resolve().parent / "cache"
    os.environ.setdefault("MLSPACES_CACHE_DIR", str(inferred_cache))
    from molmo_spaces.molmo_spaces_constants import get_resource_manager
    from molmo_spaces.utils.lazy_loading_utils import locate_uid_package

    source, package, located_xml = locate_uid_package(asset["asset_id"])
    if source is None or located_xml is None:
        raise RuntimeError(f"Objathor MJCF was not found for {asset['asset_id']}")
    if not located_xml.exists():
        if package is None:
            raise RuntimeError(f"No downloadable package for {asset['asset_id']}")
        get_resource_manager().install_packages("objects", {source: [package]})
    if not located_xml.exists():
        raise RuntimeError(f"Download completed but MJCF is missing: {located_xml}")
    root_body = _root_body_from_xml(located_xml)
    if root_body is None:
        raise RuntimeError(f"Downloaded MJCF has no root body: {located_xml}")
    asset.update(
        {
            "xml_path": str(located_xml.resolve()),
            "root_body": root_body,
            "importable": True,
        }
    )
    return asset


def hydrate_layout_assets(
    layout: dict,
    catalog_assets: list[dict],
    assets_dir: Path,
) -> None:
    known = _catalog_index(catalog_assets)
    for instance in layout.get("objects", []):
        if not instance.get("active", True):
            continue
        catalog_id = instance.get("catalog_id", "")
        if catalog_id in known:
            continue
        asset = resolve_objathor_asset(catalog_id)
        if asset is None:
            uid = catalog_id.split(":", 1)[-1]
            package = assets_dir / "objects" / "objaverse" / uid
            candidates = sorted(package.rglob("*.xml"))
            if not candidates:
                raise RuntimeError(f"Packaged Objathor asset is missing: {catalog_id}")
            xml_path = candidates[0]
            root_body = _root_body_from_xml(xml_path)
            if root_body is None:
                raise RuntimeError(f"Packaged Objathor asset has no root body: {xml_path}")
            asset = {"catalog_id": catalog_id, "asset_id": uid, "display_name": uid,
                     "provider": "objathor", "format": "mjcf", "xml_path": str(xml_path),
                     "root_body": root_body, "importable": True, "dynamic": False}
        else:
            materialize_catalog_asset(asset, assets_dir)
        # Saved objaverse IDs alias objathor metadata IDs for portable layouts.
        asset = dict(asset)
        asset["catalog_id"] = catalog_id
        if catalog_id.startswith("objaverse:"):
            asset["provider"] = "objaverse"
        registered = _register_catalog_asset(catalog_assets, asset)
        known[registered["catalog_id"]] = registered


def _root_body_from_xml(xml_path: Path) -> str | None:
    import xml.etree.ElementTree as ET

    try:
        body = ET.parse(xml_path).getroot().find("./worldbody/body")
    except (ET.ParseError, OSError):
        return None
    return None if body is None else body.get("name")


def make_instance(asset: dict, index: int, *, count_offset: int = 0) -> dict:
    column = (index + count_offset) % 3
    row = ((index + count_offset) // 3) % 3
    return {
        "id": f"{asset['asset_id'][:32]}_{uuid.uuid4().hex[:8]}",
        "catalog_id": asset["catalog_id"],
        "asset_id": asset["asset_id"],
        "active": True,
        "pos": [8.5 + column * 1.5, 1.2 + row * 1.5, float(asset.get("floor_z", 0.0))],
        "yaw": 0.0,
        "auto_ground": "floor_z" not in asset,
    }


def append_asset(layout: dict, asset: dict, count: int) -> None:
    offset = sum(bool(item.get("active", True)) for item in layout["objects"])
    for index in range(count):
        layout["objects"].append(make_instance(asset, index, count_offset=offset))


def _thumbnail_path(
    asset: dict,
    azimuth: float = 135.0,
    elevation: float = -22.0,
) -> Path:
    xml_path = Path(asset["xml_path"])
    signature = (
        f"neutral-v5:{asset['catalog_id']}:{xml_path.stat().st_mtime_ns}:"
        f"{round(azimuth, 1)}:{round(elevation, 1)}"
    )
    digest = hashlib.sha1(signature.encode()).hexdigest()[:16]
    return THUMBNAIL_DIR / f"{digest}.png"


def _geom_world_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
        world_vertices = data.geom_xpos[geom_id] + vertices @ rotation.T
        return world_vertices.min(axis=0), world_vertices.max(axis=0)

    center = model.geom_aabb[geom_id, :3]
    half = model.geom_aabb[geom_id, 3:]
    world_center = data.geom_xpos[geom_id] + rotation @ center
    world_half = np.abs(rotation) @ half
    return world_center - world_half, world_center + world_half


def _model_visual_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] != 0:
            continue
        geom_minimum, geom_maximum = _geom_world_bounds(model, data, geom_id)
        minimum = np.minimum(minimum, geom_minimum)
        maximum = np.maximum(maximum, geom_maximum)
    if not np.all(np.isfinite(minimum)):
        return np.array([-0.5, -0.5, 0.0]), np.array([0.5, 0.5, 1.0])
    return minimum, maximum


def render_asset_thumbnail(
    asset: dict,
    *,
    azimuth: float = 135.0,
    elevation: float = -22.0,
) -> Path:
    path = _thumbnail_path(asset, azimuth, elevation)
    if path.exists():
        return path
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw

    try:
        spec = mujoco.MjSpec.from_file(asset["xml_path"])
        for site in spec.sites:
            site.group = 4
        root = spec.body(asset["root_body"])
        if root is None:
            raise RuntimeError(f"Missing root body: {asset['root_body']}")
        root.quat = asset.get("base_quat", [1.0, 0.0, 0.0, 0.0])
        root.pos = [0.0, 0.0, 0.0]
        floor_z = float(asset.get("floor_z", _visual_floor_offset(spec)))
        root.pos = [0.0, 0.0, floor_z]
        spec.add_texture(
            name="asset_preview_sky",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.26, 0.28, 0.32],
            rgb2=[0.10, 0.11, 0.14],
            width=512,
            height=512,
        )
        spec.worldbody.add_geom(
            name="asset_preview_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[5.0, 5.0, 0.05],
            pos=[0.0, 0.0, -0.01],
            rgba=[0.12, 0.13, 0.15, 1.0],
            contype=0,
            conaffinity=0,
            group=2,
        )
        spec.worldbody.add_light(
            name="asset_preview_key",
            pos=[-3.0, -4.0, 6.0],
            dir=[0.35, 0.45, -1.0],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            castshadow=1,
            ambient=[0.45, 0.45, 0.45],
            diffuse=[0.95, 0.95, 0.95],
            specular=[0.25, 0.25, 0.25],
        )
        spec.worldbody.add_light(
            name="asset_preview_fill",
            pos=[4.0, 2.0, 4.0],
            dir=[-0.5, -0.2, -1.0],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            castshadow=0,
            ambient=[0.25, 0.25, 0.25],
            diffuse=[0.75, 0.78, 0.85],
            specular=[0.1, 0.1, 0.1],
        )
        model = spec.compile()
        model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
        model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        model.vis.headlight.specular[:] = [0.15, 0.15, 0.15]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        minimum, maximum = _model_visual_bounds(model, data)
        center = (minimum + maximum) / 2.0
        extent = np.maximum(maximum - minimum, 0.1)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = center
        camera.distance = max(1.25, float(np.max(extent)) * 2.3)
        camera.azimuth = azimuth
        camera.elevation = elevation
        option = mujoco.MjvOption()
        with mujoco.Renderer(model, width=560, height=420) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            image = renderer.render()
        Image.fromarray(image).save(path)
    except Exception as error:
        image = Image.new("RGB", (560, 420), (35, 37, 43))
        draw = ImageDraw.Draw(image)
        draw.text((24, 30), asset["display_name"], fill=(235, 235, 235))
        draw.text((24, 70), "Preview unavailable", fill=(255, 180, 90))
        draw.text((24, 105), str(error)[:220], fill=(200, 200, 200))
        image.save(path)
    return path


def graphical_picker(
    assets: list[dict],
    assets_dir: Path,
) -> tuple[dict, int] | None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
        from PIL import Image, ImageTk
    except ImportError:
        return terminal_picker(assets, assets_dir)

    result: list[tuple[dict, int]] = []
    filtered: list[dict] = []
    window = tk.Tk()
    window.title("val_861 MuJoCo Asset Picker")
    window.geometry("1280x720")
    window.lift()
    window.attributes("-topmost", True)
    window.after(400, lambda: window.attributes("-topmost", False))

    query = tk.StringVar()
    source_filter = tk.StringVar(value="All")
    category_filter = tk.StringVar(value="All categories")
    copies = tk.IntVar(value=1)
    status = tk.StringVar()

    header = ttk.Frame(window, padding=10)
    header.pack(fill="x")
    ttk.Label(header, text="Search local + Objathor MuJoCo assets:").pack(side="left")
    entry = ttk.Entry(header, textvariable=query)
    entry.pack(side="left", fill="x", expand=True, padx=8)
    ttk.Combobox(
        header,
        textvariable=source_filter,
        values=("All", "Objathor", "Local"),
        state="readonly",
        width=11,
    ).pack(side="right")
    ttk.Combobox(
        header,
        textvariable=category_filter,
        values=("All categories", *ASSET_BROWSE_CATEGORIES),
        state="readonly",
        width=24,
    ).pack(side="right", padx=(0, 8))

    body = ttk.Panedwindow(window, orient="horizontal")
    body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    list_frame = ttk.Frame(body)
    preview_frame = ttk.Frame(body, padding=(14, 0, 0, 0))
    body.add(list_frame, weight=3)
    body.add(preview_frame, weight=2)

    listbox = tk.Listbox(list_frame, font=("monospace", 11), exportselection=False)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    preview_label = ttk.Label(preview_frame, anchor="center")
    preview_label.pack(fill="both", expand=True)
    metadata_label = ttk.Label(preview_frame, justify="left", wraplength=480)
    metadata_label.pack(fill="x", pady=(10, 8))
    rotation_controls = ttk.Frame(preview_frame)
    rotation_controls.pack(fill="x", pady=(0, 8))
    controls = ttk.Frame(preview_frame)
    controls.pack(fill="x")
    ttk.Label(controls, text="Copies").pack(anchor="w")
    ttk.Spinbox(controls, from_=1, to=50, textvariable=copies, width=8).pack(anchor="w")
    ttk.Label(controls, textvariable=status, wraplength=220).pack(anchor="w", pady=18)
    preview_state = {
        "asset": None,
        "azimuth": 135.0,
        "elevation": -22.0,
        "drag": None,
    }

    def refresh(*_args) -> None:
        nonlocal filtered
        search_text = query.get().strip()
        browse_category = category_filter.get()
        if search_text:
            local_matches = search_assets(assets, search_text, limit=500)
            objathor_matches = search_objathor_assets(search_text, limit=500)
        elif browse_category != "All categories":
            local_matches = sorted(
                (
                    asset
                    for asset in assets
                    if (asset.get("importable") or asset.get("downloadable"))
                    and asset.get(
                        "browse_category", asset_browse_category(asset)
                    ) == browse_category
                ),
                key=lambda asset: asset.get("display_name", "").lower(),
            )[:500]
            objathor_matches = search_objathor_category(
                browse_category, limit=500
            )
        else:
            local_matches = search_assets(assets, "", limit=500)
            objathor_matches = search_objathor_assets("", limit=500)
        if source_filter.get() == "Objathor":
            local_matches = []
        elif source_filter.get() == "Local":
            objathor_matches = []
        combined = []
        for index in range(max(len(local_matches), len(objathor_matches))):
            if index < len(local_matches):
                combined.append(local_matches[index])
            if index < len(objathor_matches):
                combined.append(objathor_matches[index])
        filtered = list({asset["catalog_id"]: asset for asset in combined}.values())[:500]
        listbox.delete(0, "end")
        for asset in filtered:
            browse_group = asset.get(
                "browse_category", asset_browse_category(asset)
            )
            label = (
                f"[{asset['provider']:<9}] [{browse_group[:20]:<20}] "
                f"{asset['display_name'][:34]:<34} "
                f"{asset['asset_id']}"
            )
            listbox.insert("end", label)
        status.set(
            f"{len(filtered)} matches shown\n"
            + (
                "Search covers all browse categories.\n"
                if search_text
                else f"Browse category: {browse_category}.\n"
            )
            + "Objathor assets download their MJCF when previewed."
        )

    def accept(*_args) -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo("Select asset", "Select one asset from the list.")
            return
        asset = filtered[selection[0]]
        try:
            materialize_catalog_asset(asset, assets_dir)
        except Exception as error:
            messagebox.showerror("Objathor download failed", str(error))
            return
        result.append((asset, max(1, int(copies.get()))))
        window.destroy()

    def display_preview() -> None:
        asset = preview_state["asset"]
        if asset is None:
            return
        status.set(
            "Downloading Objathor MJCF..."
            if not asset.get("importable")
            else "Rendering preview..."
        )
        window.update_idletasks()
        try:
            materialize_catalog_asset(asset, assets_dir)
        except Exception as error:
            status.set(f"Objathor download failed: {error}")
            return
        thumbnail = Image.open(
            render_asset_thumbnail(
                asset,
                azimuth=preview_state["azimuth"],
                elevation=preview_state["elevation"],
            )
        ).convert("RGB")
        thumbnail.thumbnail((540, 430))
        photo = ImageTk.PhotoImage(thumbnail)
        preview_label.configure(image=photo)
        preview_label.image = photo
        status.set(
            f"3D preview  az={preview_state['azimuth']:.0f}° "
            f"el={preview_state['elevation']:.0f}°"
        )

    def show_preview(*_args) -> None:
        selection = listbox.curselection()
        if not selection:
            return
        asset = filtered[selection[0]]
        if preview_state["asset"] is not asset:
            preview_state.update(asset=asset, azimuth=135.0, elevation=-22.0)
        display_preview()
        aliases = ", ".join(asset.get("aliases", [])[:8]) or "-"
        categories = ", ".join(asset.get("categories", [])) or asset.get(
            "path_category", "-"
        )
        browse_group = asset.get(
            "browse_category", asset_browse_category(asset)
        )
        metadata_label.configure(
            text=(
                f"{asset['display_name']}\n"
                f"ID: {asset['catalog_id']}\n"
                f"Browse group: {browse_group}\n"
                f"Category: {categories}\n"
                f"Aliases: {aliases}\n\n"
                "After clicking Preview in MuJoCo, move and rotate the draft "
                "inside the still-open MuJoCo window, then confirm or cancel it."
            )
        )
    def rotate_preview(delta: float) -> None:
        if preview_state["asset"] is None:
            return
        preview_state["azimuth"] = (preview_state["azimuth"] + delta) % 360.0
        display_preview()

    def start_preview_drag(event) -> None:
        preview_state["drag"] = (event.x, event.y)

    def drag_preview(event) -> None:
        if preview_state["drag"] is None or preview_state["asset"] is None:
            return
        old_x, old_y = preview_state["drag"]
        delta_x, delta_y = event.x - old_x, event.y - old_y
        if abs(delta_x) + abs(delta_y) < 14:
            return
        preview_state["azimuth"] = (preview_state["azimuth"] - delta_x * 0.7) % 360.0
        preview_state["elevation"] = min(
            10.0, max(-80.0, preview_state["elevation"] + delta_y * 0.45)
        )
        preview_state["drag"] = (event.x, event.y)
        display_preview()

    preview_label.bind("<ButtonPress-1>", start_preview_drag)
    preview_label.bind("<B1-Motion>", drag_preview)
    preview_label.bind("<ButtonRelease-1>", lambda _event: preview_state.update(drag=None))
    ttk.Button(rotation_controls, text="↶ Rotate 30°", command=lambda: rotate_preview(-30.0)).pack(side="left", fill="x", expand=True, padx=(0, 3))
    ttk.Label(rotation_controls, text="Drag preview to orbit", anchor="center").pack(side="left", fill="x", expand=True, padx=3)
    ttk.Button(rotation_controls, text="Rotate 30° ↷", command=lambda: rotate_preview(30.0)).pack(side="left", fill="x", expand=True, padx=(3, 0))

    ttk.Button(controls, text="Preview in MuJoCo", command=accept).pack(
        side="left", fill="x", expand=True, padx=(0, 4)
    )
    ttk.Button(controls, text="Cancel", command=window.destroy).pack(
        side="left", fill="x", expand=True, padx=(4, 0)
    )
    listbox.bind("<Double-Button-1>", accept)
    listbox.bind("<<ListboxSelect>>", show_preview)
    query.trace_add("write", refresh)
    source_filter.trace_add("write", refresh)
    category_filter.trace_add("write", refresh)
    refresh()
    entry.focus_set()
    window.mainloop()
    return result[0] if result else None


def terminal_picker(
    assets: list[dict],
    assets_dir: Path,
) -> tuple[dict, int] | None:
    query = input("Asset search (blank to cancel): ").strip()
    if not query:
        return None
    local = search_assets(assets, query, limit=40)
    objathor = search_objathor_assets(query, limit=40)
    matches = []
    for index in range(max(len(local), len(objathor))):
        if index < len(local):
            matches.append(local[index])
        if index < len(objathor):
            matches.append(objathor[index])
    matches = matches[:40]
    for index, asset in enumerate(matches, 1):
        print(f"{index:2}: [{asset['provider']}] {asset['display_name']} ({asset['asset_id']})")
    choice = int(input("Select number: ")) - 1
    count = max(1, int(input("Copies [1]: ") or "1"))
    asset = materialize_catalog_asset(matches[choice], assets_dir)
    return asset, count


def _clear_val861_objects(
    scene_spec: mujoco.MjSpec,
    assets_dir: Path,
) -> int:
    scene_dir = assets_dir / "scenes" / SCENE_SOURCE
    description = json.loads((scene_dir / "val_861.json").read_text())
    metadata = json.loads((scene_dir / "val_861_metadata.json").read_text())["objects"]
    object_ids: set[str] = set()

    def collect_object_tree(item: dict) -> None:
        object_ids.add(item["id"])
        for child in item.get("children", []):
            collect_object_tree(child)

    for item in description["objects"]:
        collect_object_tree(item)
    body_names = [
        item["hash_name"]
        for item in metadata.values()
        if item.get("object_id") in object_ids
    ]
    mapped_ids = {
        item.get("object_id")
        for item in metadata.values()
        if item.get("object_id") in object_ids
    }
    if mapped_ids != object_ids:
        raise RuntimeError(
            f"Could not map all val_861 objects: {len(mapped_ids)} IDs for "
            f"{len(object_ids)} recursive objects."
        )
    removed = 0
    for name in body_names:
        body = scene_spec.body(name)
        if body is not None:
            scene_spec.delete(body)
            removed += 1
    return removed


def _scene_description(assets_dir: Path) -> dict:
    path = assets_dir / "scenes" / SCENE_SOURCE / "val_861.json"
    return json.loads(path.read_text())


def _segment_key(segment: list[list[float]], precision: int = 4) -> tuple:
    points = [tuple(round(float(value), precision) for value in point) for point in segment]
    return tuple(sorted(points))


def _base_wall_groups(description: dict) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for index, wall in enumerate(description["walls"]):
        key = _segment_key(wall["segment"])
        group = grouped.setdefault(
            key,
            {
                "id": f"base_wall_{len(grouped):03d}",
                "kind": "base_wall",
                "start": list(key[0]),
                "end": list(key[1]),
                "source_ids": [],
                "body_names": [],
            },
        )
        group["source_ids"].append(wall["id"])
        room_number = int(re.search(r"\d+", wall["roomId"]).group())
        group["body_names"].append(f"wall_{room_number}_{index}")
    return list(grouped.values())


def _base_floor_records(description: dict) -> list[dict]:
    records = []
    for index, room in enumerate(description["rooms"]):
        polygon = [
            [float(point["x"]), float(point["z"])]
            for point in room["floorPolygon"]
        ]
        coordinates = np.asarray(polygon, dtype=float)
        records.append(
            {
                "id": f"base_floor_{room['id']}",
                "room_id": room["id"],
                "kind": "base_floor",
                "start": coordinates.min(axis=0).round(4).tolist(),
                "end": coordinates.max(axis=0).round(4).tolist(),
                "polygon": polygon,
                "body_names": [f"room_{index}", f"editor_floor_collision_{index}"],
                "source_ids": [room["id"]],
            }
        )
    return records


def _delete_base_walls(
    scene_spec: mujoco.MjSpec,
    description: dict,
    removed_ids: set[str],
) -> None:
    for index, wall in enumerate(description["walls"]):
        if wall["id"] not in removed_ids:
            continue
        room_number = int(re.search(r"\d+", wall["roomId"]).group())
        body = scene_spec.body(f"wall_{room_number}_{index}")
        if body is not None:
            scene_spec.delete(body)


def _delete_base_floors(
    scene_spec: mujoco.MjSpec,
    description: dict,
    removed_room_ids: set[str],
) -> None:
    for index, room in enumerate(description["rooms"]):
        if room["id"] not in removed_room_ids:
            continue
        body = scene_spec.body(f"room_{index}")
        if body is not None:
            scene_spec.delete(body)


def _remove_original_door_assets(
    scene_spec: mujoco.MjSpec,
    description: dict,
) -> None:
    door_numbers = {int(door["id"].split("|")[1]) for door in description["doors"]}
    roots = []
    for body in scene_spec.bodies:
        if not body.name.startswith(("doorway_", "doorframe_")):
            continue
        match = re.search(r"_\d+_0_(\d+)$", body.name)
        if match and int(match.group(1)) in door_numbers:
            roots.append(body)
    for body in roots:
        scene_spec.delete(body)


def _remove_windows_on_deleted_walls(
    scene_spec: mujoco.MjSpec,
    assets_dir: Path,
    description: dict,
    removed_wall_ids: set[str],
) -> int:
    removed_window_ids = {
        window["id"]
        for window in description["windows"]
        if window.get("wall0") in removed_wall_ids
        or window.get("wall1") in removed_wall_ids
    }
    if not removed_window_ids:
        return 0
    metadata_path = (
        assets_dir / "scenes" / SCENE_SOURCE / "val_861_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())["objects"]
    roots = [
        scene_spec.body(item["hash_name"])
        for item in metadata.values()
        if not item.get("parent")
        and item.get("category") == "Window"
        and item.get("object_id") in removed_window_ids
    ]
    for body in roots:
        if body is not None:
            scene_spec.delete(body)
    return sum(body is not None for body in roots)


def _box_along_segment(
    body: mujoco.MjsBody,
    *,
    name: str,
    start: list[float],
    end: list[float],
    z_bottom: float,
    height: float,
    thickness: float,
    rgba: list[float],
    material: str = "",
) -> None:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    delta = end_array - start_array
    length = float(np.linalg.norm(delta))
    if length < 1e-4 or height < 1e-4:
        return
    midpoint = (start_array + end_array) / 2.0
    yaw = math.degrees(math.atan2(delta[1], delta[0]))
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[float(midpoint[0]), float(midpoint[1]), z_bottom + height / 2.0],
        quat=yaw_quaternion(yaw),
        size=[length / 2.0, thickness / 2.0, height / 2.0],
        rgba=rgba,
        material=material,
        contype=1,
        conaffinity=1,
        friction=[0.9, 0.01, 0.001],
        group=0,
    )


def _add_visual_wall_finish_panel(
    body: mujoco.MjsBody,
    *,
    name: str,
    start: list[float],
    end: list[float],
    height: float,
    thickness: float,
    rgba: list[float],
    material: str = "",
) -> None:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    delta = end_array - start_array
    length = float(np.linalg.norm(delta))
    if length < 1e-4 or height < 1e-4:
        return
    midpoint = (start_array + end_array) / 2.0
    yaw = math.degrees(math.atan2(delta[1], delta[0]))
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[float(midpoint[0]), float(midpoint[1]), height / 2.0],
        quat=yaw_quaternion(yaw),
        size=[length / 2.0, thickness / 2.0, height / 2.0],
        rgba=rgba,
        material=material,
        contype=0,
        conaffinity=0,
        group=0,
    )


def _add_wide_window_wall(
    scene_spec: mujoco.MjSpec,
    record: dict,
    finish: dict | None,
) -> str:
    body_name = f"editor_window_wall_{record['id']}"
    body = scene_spec.worldbody.add_body(name=body_name)
    start_x, end_x = sorted((float(record["start"][0]), float(record["end"][0])))
    wall_y = float(record["start"][1])
    wall_height = float(record.get("height", 3.2))
    thickness = float(record.get("thickness", 0.1))
    openings = sorted(record.get("openings", []), key=lambda item: item[0])
    rgba, material = _wall_finish_appearance(
        scene_spec, record["id"], finish
    )

    bottom = min(float(item[2]) for item in openings)
    top = max(float(item[3]) for item in openings)
    _box_along_segment(
        body,
        name=f"{body_name}_lower",
        start=[start_x, wall_y],
        end=[end_x, wall_y],
        z_bottom=0.0,
        height=bottom,
        thickness=thickness,
        rgba=rgba,
        material=material,
    )
    _box_along_segment(
        body,
        name=f"{body_name}_upper",
        start=[start_x, wall_y],
        end=[end_x, wall_y],
        z_bottom=top,
        height=wall_height - top,
        thickness=thickness,
        rgba=rgba,
        material=material,
    )
    cursor = start_x
    for index, (left, right, opening_bottom, opening_top) in enumerate(openings):
        if left > cursor:
            _box_along_segment(
                body,
                name=f"{body_name}_pier_{index}",
                start=[cursor, wall_y],
                end=[left, wall_y],
                z_bottom=bottom,
                height=top - bottom,
                thickness=thickness,
                rgba=rgba,
                material=material,
            )
        cursor = right
        center_x = (left + right) / 2.0
        center_z = (opening_bottom + opening_top) / 2.0
        frame_rgba = [0.84, 0.86, 0.88, 1.0]
        frame_depth = thickness / 2.0 + 0.018
        frame_width = 0.035
        for suffix, x in (("left", left), ("right", right), ("center", center_x)):
            body.add_geom(
                name=f"{body_name}_{index}_{suffix}_frame",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, wall_y, center_z],
                size=[frame_width, frame_depth, (opening_top - opening_bottom) / 2.0],
                rgba=frame_rgba,
                contype=0,
                conaffinity=0,
                group=0,
            )
        for suffix, z in (("bottom", opening_bottom), ("top", opening_top)):
            body.add_geom(
                name=f"{body_name}_{index}_{suffix}_frame",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[center_x, wall_y, z],
                size=[(right - left) / 2.0, frame_depth, frame_width],
                rgba=frame_rgba,
                contype=0,
                conaffinity=0,
                group=0,
            )
        body.add_geom(
            name=f"{body_name}_{index}_glass",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[center_x, wall_y, center_z],
            size=[(right - left) / 2.0 - 0.04, 0.006, (opening_top - opening_bottom) / 2.0 - 0.04],
            rgba=[0.55, 0.74, 0.82, 0.18],
            contype=0,
            conaffinity=0,
            group=0,
        )
    if cursor < end_x:
        _box_along_segment(
            body,
            name=f"{body_name}_pier_end",
            start=[cursor, wall_y],
            end=[end_x, wall_y],
            z_bottom=bottom,
            height=top - bottom,
            thickness=thickness,
            rgba=rgba,
            material=material,
        )
    return body_name


def _surface_finish_appearance(
    scene_spec: mujoco.MjSpec,
    surface_id: str,
    finish: dict | None,
    *,
    default_color: list[float],
    namespace: str,
) -> tuple[list[float], str]:
    if not finish:
        return default_color, ""
    if finish.get("mode") == "color":
        rgba = [float(value) for value in finish.get("rgba", default_color)]
        if len(rgba) == 3:
            rgba.append(1.0)
        return rgba[:4], ""
    source_texture_path = _resolve_surface_texture_path(
        finish.get("texture_path", "")
    )
    if finish.get("mode") != "texture" or not source_texture_path.is_file():
        return default_color, ""
    normal_source = _surface_companion_texture_path(source_texture_path, "normal")
    texture_path = _mujoco_texture_path(source_texture_path)
    normal_path = (
        _mujoco_texture_path(normal_source) if normal_source is not None else None
    )
    digest_source = f"{surface_id}:{texture_path}:{normal_path or ''}"
    digest = hashlib.sha1(digest_source.encode()).hexdigest()[:12]
    texture_name = f"editor_{namespace}_tex_{digest}"
    normal_name = f"editor_{namespace}_normal_{digest}"
    material_name = f"editor_{namespace}_mat_{digest}"
    if scene_spec.texture(texture_name) is None:
        scene_spec.add_texture(
            name=texture_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(texture_path.resolve()),
        )
    if normal_path is not None and scene_spec.texture(normal_name) is None:
        scene_spec.add_texture(
            name=normal_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(normal_path.resolve()),
        )
    if scene_spec.material(material_name) is None:
        category = str(finish.get("texture_category", ""))
        is_outdoor = category == "Outdoor"
        material = scene_spec.add_material(
            name=material_name,
            rgba=[1.0, 1.0, 1.0, 1.0],
            texrepeat=[float(finish.get("repeat", 2.0))] * 2,
            texuniform=1,
            roughness=0.92 if is_outdoor else 0.8,
            specular=0.08 if is_outdoor else 0.2,
            shininess=0.05 if is_outdoor else 0.2,
        )
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = texture_name
        if normal_path is not None:
            material.textures[
                mujoco.mjtTextureRole.mjTEXROLE_NORMAL.value
            ] = normal_name
    return [1.0, 1.0, 1.0, 1.0], material_name


def _resolve_surface_texture_path(reference: str) -> Path:
    if reference.startswith("project://"):
        relative = reference.removeprefix("project://")
        marker = "src/environment/test_zone/world/surface_textures/"
        if relative.startswith(marker):
            return CUSTOM_SURFACE_ROOT / relative.removeprefix(marker)
        return PROJECT_ROOT / relative
    if reference.startswith("assets://"):
        assets_dir = os.environ.get("MLSPACES_ASSETS_DIR")
        if assets_dir:
            return Path(assets_dir).expanduser() / reference.removeprefix("assets://")
        return VENDORED_ASSETS / reference.removeprefix("assets://")
    path = Path(reference).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _surface_companion_texture_path(source: Path, role: str) -> Path | None:
    if role != "normal":
        return None
    stem = source.stem
    replacements = (
        ("_albedo_", "_normal_"),
        ("_albedo", "_normal"),
        ("albedo", "normal"),
        ("basecolor", "normal"),
        ("base_color", "normal"),
    )
    candidate_stems: list[str] = []
    lowered = stem.lower()
    for old, new in replacements:
        index = lowered.find(old)
        if index >= 0:
            candidate_stems.append(
                f"{stem[:index]}{new}{stem[index + len(old):]}"
            )
    extensions = (source.suffix, ".png", ".jpg", ".jpeg")
    for candidate_stem in candidate_stems:
        for extension in dict.fromkeys(extensions):
            candidate = source.with_name(f"{candidate_stem}{extension}")
            if candidate.is_file():
                return candidate
    return None


def _mujoco_texture_path(source: Path) -> Path:
    if source.suffix.lower() == ".png":
        return source
    stat = source.stat()
    signature = f"{source.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    digest = hashlib.sha1(signature.encode()).hexdigest()[:20]
    cache_root = EDITOR_ROOT / "texture_cache"
    converted = cache_root / f"{digest}.png"
    if converted.is_file():
        return converted

    from PIL import Image

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = converted.with_suffix(".tmp.png")
    with Image.open(source) as image:
        image.convert("RGBA").save(temporary, format="PNG")
    temporary.replace(converted)
    return converted


def _portable_surface_texture_reference(path_text: str, assets_dir: Path) -> str:
    path = Path(path_text).expanduser()
    try:
        return f"project://{path.resolve().relative_to(PROJECT_ROOT.resolve())}"
    except ValueError:
        pass

    thor_root = (assets_dir / "objects" / "thor").resolve()
    try:
        relative = path.resolve().relative_to(thor_root)
        return f"assets://objects/thor/{relative}"
    except ValueError:
        pass

    normalized = path.resolve().as_posix()
    marker = "/objects/thor/"
    if marker in normalized:
        parts = Path(normalized.split(marker, 1)[1]).parts
        if parts and re.fullmatch(r"20\d{6}", parts[0]):
            parts = parts[1:]
        return f"assets://objects/thor/{Path(*parts)}"
    try:
        relative = path.resolve().relative_to(assets_dir.resolve())
        return f"assets://{relative}"
    except ValueError:
        return str(path.resolve())


def _portableize_layout_texture_paths(layout: dict, assets_dir: Path) -> None:
    changed = False
    structure = layout.get("structure", {})
    for finish_group in (
        "wall_finishes",
        "room_wall_finishes",
        "floor_finishes",
    ):
        for finish in structure.get(finish_group, {}).values():
            reference = finish.get("texture_path")
            if not reference or str(reference).startswith("project://"):
                continue
            if str(reference).startswith("assets://objects/thor/"):
                relative = str(reference).removeprefix("assets://objects/thor/")
                parts = Path(relative).parts
                if parts and re.fullmatch(r"20\d{6}", parts[0]):
                    relative = str(Path(*parts[1:]))
                portable = f"assets://objects/thor/{relative}"
            elif str(reference).startswith("assets://"):
                portable = str(reference)
            else:
                portable = _portable_surface_texture_reference(
                    str(reference), assets_dir
                )
            if portable != reference:
                finish["texture_path"] = portable
                changed = True
    if changed:
        layout["_needs_save"] = True


def _wall_finish_appearance(
    scene_spec: mujoco.MjSpec,
    wall_id: str,
    finish: dict | None,
) -> tuple[list[float], str]:
    return _surface_finish_appearance(
        scene_spec,
        wall_id,
        finish,
        default_color=[0.76, 0.73, 0.67, 1.0],
        namespace="wall",
    )


def _floor_finish_appearance(
    scene_spec: mujoco.MjSpec,
    floor_id: str,
    finish: dict | None,
) -> tuple[list[float], str]:
    return _surface_finish_appearance(
        scene_spec,
        floor_id,
        finish,
        default_color=[0.62, 0.58, 0.50, 1.0],
        namespace="floor",
    )


def _apply_finish_to_bodies(
    scene_spec: mujoco.MjSpec,
    body_names: list[str],
    rgba: list[float],
    material: str,
) -> None:
    for body_name in body_names:
        body = scene_spec.body(body_name)
        if body is None:
            continue
        for geom in body.geoms:
            if int(geom.group) != 0:
                continue
            geom.material = material
            geom.rgba = rgba


def _door_interval_on_wall(wall: dict, door: dict) -> tuple[float, float] | None:
    wall_start = np.asarray(wall["start"], dtype=float)
    wall_end = np.asarray(wall["end"], dtype=float)
    delta = wall_end - wall_start
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return None
    direction = delta / length
    normal = np.array([-direction[1], direction[0]])
    door_points = [np.asarray(door["start"], dtype=float), np.asarray(door["end"], dtype=float)]
    if max(abs(float(np.dot(point - wall_start, normal))) for point in door_points) > 0.18:
        return None
    values = sorted(float(np.dot(point - wall_start, direction)) for point in door_points)
    if values[1] < -0.02 or values[0] > length + 0.02:
        return None
    return max(0.0, values[0]), min(length, values[1])


def _add_wall(
    scene_spec: mujoco.MjSpec,
    wall: dict,
    doors: list[dict],
    body_name: str,
    finish: dict | None = None,
) -> None:
    body = scene_spec.worldbody.add_body(name=body_name)
    start = np.asarray(wall["start"], dtype=float)
    end = np.asarray(wall["end"], dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-4:
        return
    direction = delta / length
    height = float(wall.get("height", 3.2))
    z_bottom = float(wall.get("z_bottom", 0.0))
    thickness = float(wall.get("thickness", BASE_WALL_THICKNESS))
    color, material = _wall_finish_appearance(
        scene_spec, wall["id"], finish
    )
    openings = []
    if not wall.get("ignore_doors", False):
        for door in doors:
            interval = _door_interval_on_wall(wall, door)
            if interval and interval[1] - interval[0] > 0.2:
                openings.append((interval[0], interval[1], float(door.get("height", 2.1))))
    openings.sort()
    cursor = 0.0
    piece_index = 0
    for opening_start, opening_end, door_height in openings:
        if opening_start > cursor + 1e-4:
            _box_along_segment(
                body,
                name=f"{body_name}_solid_{piece_index}",
                start=(start + direction * cursor).tolist(),
                end=(start + direction * opening_start).tolist(),
                z_bottom=z_bottom,
                height=height,
                thickness=thickness,
                rgba=color,
                material=material,
            )
            piece_index += 1
        if door_height < height:
            _box_along_segment(
                body,
                name=f"{body_name}_header_{piece_index}",
                start=(start + direction * opening_start).tolist(),
                end=(start + direction * opening_end).tolist(),
                z_bottom=z_bottom + door_height,
                height=height - door_height,
                thickness=thickness,
                rgba=color,
                material=material,
            )
            piece_index += 1
        cursor = max(cursor, opening_end)
    if cursor < length - 1e-4:
        _box_along_segment(
            body,
            name=f"{body_name}_solid_{piece_index}",
            start=(start + direction * cursor).tolist(),
            end=end.tolist(),
            z_bottom=z_bottom,
            height=height,
            thickness=thickness,
            rgba=color,
            material=material,
        )


def _add_open_door_frame(
    scene_spec: mujoco.MjSpec,
    door: dict,
    body_name: str,
) -> None:
    body = scene_spec.worldbody.add_body(name=body_name)
    start = np.asarray(door["start"], dtype=float)
    end = np.asarray(door["end"], dtype=float)
    delta = end - start
    width = float(np.linalg.norm(delta))
    if width < 0.2:
        return
    direction = delta / width
    height = float(door.get("height", 2.1))
    thickness = float(door.get("thickness", DOOR_FRAME_THICKNESS))
    jamb = min(0.07, width * 0.08)
    color = [0.30, 0.22, 0.15, 1.0]
    _box_along_segment(
        body,
        name=f"{body_name}_jamb_a",
        start=(start - direction * jamb).tolist(),
        end=start.tolist(),
        z_bottom=0.0,
        height=height,
        thickness=thickness * 1.35,
        rgba=color,
    )
    _box_along_segment(
        body,
        name=f"{body_name}_jamb_b",
        start=end.tolist(),
        end=(end + direction * jamb).tolist(),
        z_bottom=0.0,
        height=height,
        thickness=thickness * 1.35,
        rgba=color,
    )
    _box_along_segment(
        body,
        name=f"{body_name}_lintel",
        start=(start - direction * jamb).tolist(),
        end=(end + direction * jamb).tolist(),
        z_bottom=height,
        height=0.09,
        thickness=thickness * 1.35,
        rgba=color,
    )


def _base_door_record(door: dict) -> dict:
    height = float(door["holePolygon"][1]["y"] - door["holePolygon"][0]["y"])
    return {
        "id": door["id"],
        "kind": "base_door",
        "start": list(map(float, door["doorSegment"][0])),
        "end": list(map(float, door["doorSegment"][1])),
        "height": height,
        "thickness": DOOR_FRAME_THICKNESS,
    }


def _iter_polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for child in geometry.geoms:
            yield from _iter_polygons(child)


def _add_floor_triangle(
    scene_spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    triangle,
    name: str,
    *,
    visible: bool = True,
    rgba: list[float] | None = None,
    material: str = "",
    uv_bounds: tuple[float, float, float, float] | None = None,
    collidable: bool = True,
) -> None:
    coordinates = list(triangle.exterior.coords)[:3]
    if len(coordinates) != 3:
        return
    thickness = 0.06
    vertices = [[x, y, -thickness] for x, y in coordinates]
    vertices += [[x, y, 0.0] for x, y in coordinates]
    faces = [
        [3, 4, 5], [2, 1, 0],
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
        [2, 0, 3], [2, 3, 5],
    ]
    mesh_kwargs = {
        "name": f"{name}_mesh",
        "uservert": np.asarray(vertices, dtype=float).reshape(-1).tolist(),
        "userface": np.asarray(faces, dtype=int).reshape(-1).tolist(),
    }
    if uv_bounds is not None:
        min_x, min_y, max_x, max_y = uv_bounds
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)
        physical_scale = min(width, height)
        texcoords = [
            [(x - min_x) / physical_scale, (y - min_y) / physical_scale]
            for x, y, _ in vertices
        ]
        # Use shared room-space UVs to avoid per-triangle texture stretching.
        mesh_kwargs["usertexcoord"] = (
            np.asarray(texcoords, dtype=float).reshape(-1).tolist()
        )
        mesh_kwargs["userfacetexcoord"] = (
            np.asarray(faces, dtype=int).reshape(-1).tolist()
        )
    mesh = scene_spec.add_mesh(**mesh_kwargs)
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,
        rgba=(rgba or [0.62, 0.58, 0.50, 1.0]) if visible else [0.0, 0.0, 0.0, 0.0],
        material=material if visible else "",
        contype=1 if collidable else 0,
        conaffinity=1 if collidable else 0,
        friction=[1.0, 0.01, 0.001],
        group=0 if visible else 4,
    )


def _add_floor_polygon_mesh(
    scene_spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    polygon,
    name: str,
    *,
    visible: bool = True,
    rgba: list[float] | None = None,
    material: str = "",
    uv_bounds: tuple[float, float, float, float] | None = None,
    collidable: bool = True,
) -> None:
    from shapely import constrained_delaunay_triangles

    triangles = list(constrained_delaunay_triangles(polygon).geoms)
    if not triangles:
        return

    coordinate_index: dict[tuple[float, float], int] = {}
    coordinates: list[tuple[float, float]] = []

    def vertex_index(x: float, y: float) -> int:
        key = (round(float(x), 9), round(float(y), 9))
        if key not in coordinate_index:
            coordinate_index[key] = len(coordinates)
            coordinates.append((float(x), float(y)))
        return coordinate_index[key]

    triangle_indices: list[tuple[int, int, int]] = []
    for triangle in triangles:
        points = list(triangle.exterior.coords)[:3]
        if len(points) != 3:
            continue
        cross = (
            (points[1][0] - points[0][0]) * (points[2][1] - points[0][1])
            - (points[1][1] - points[0][1]) * (points[2][0] - points[0][0])
        )
        if cross < 0.0:
            points[1], points[2] = points[2], points[1]
        triangle_indices.append(tuple(vertex_index(x, y) for x, y in points))

    rings = [polygon.exterior, *polygon.interiors]
    for ring in rings:
        for x, y in list(ring.coords)[:-1]:
            vertex_index(x, y)

    count = len(coordinates)
    thickness = 0.06
    vertices = [[x, y, -thickness] for x, y in coordinates]
    vertices += [[x, y, 0.0] for x, y in coordinates]
    faces: list[list[int]] = []
    for a, b, c in triangle_indices:
        faces.append([a + count, b + count, c + count])
        faces.append([c, b, a])
    for ring in rings:
        points = list(ring.coords)
        for first, second in zip(points[:-1], points[1:]):
            a = vertex_index(*first)
            b = vertex_index(*second)
            faces.extend(([a, b, b + count], [a, b + count, a + count]))

    mesh_kwargs = {
        "name": f"{name}_mesh",
        "uservert": np.asarray(vertices, dtype=float).reshape(-1).tolist(),
        "userface": np.asarray(faces, dtype=int).reshape(-1).tolist(),
    }
    if uv_bounds is not None:
        min_x, min_y, max_x, max_y = uv_bounds
        physical_scale = max(min(max_x - min_x, max_y - min_y), 1e-6)
        texcoords = [
            [(x - min_x) / physical_scale, (y - min_y) / physical_scale]
            for x, y, _ in vertices
        ]
        mesh_kwargs["usertexcoord"] = (
            np.asarray(texcoords, dtype=float).reshape(-1).tolist()
        )
        mesh_kwargs["userfacetexcoord"] = (
            np.asarray(faces, dtype=int).reshape(-1).tolist()
        )
    mesh = scene_spec.add_mesh(**mesh_kwargs)
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,
        rgba=(rgba or [0.62, 0.58, 0.50, 1.0]) if visible else [0.0, 0.0, 0.0, 0.0],
        material=material if visible else "",
        contype=1 if collidable else 0,
        conaffinity=1 if collidable else 0,
        friction=[1.0, 0.01, 0.001],
        group=0 if visible else 4,
    )


def _add_base_floor_collisions(
    scene_spec: mujoco.MjSpec,
    description: dict,
    removed_room_ids: set[str],
) -> None:
    from shapely.geometry import Polygon
    from shapely.ops import triangulate

    for index, room in enumerate(description["rooms"]):
        if room["id"] in removed_room_ids:
            continue
        polygon = Polygon(
            [(float(point["x"]), float(point["z"])) for point in room["floorPolygon"]]
        )
        if polygon.is_empty or polygon.area <= 0.01:
            continue
        body = scene_spec.worldbody.add_body(name=f"editor_floor_collision_{index}")
        triangle_index = 0
        for triangle in triangulate(polygon):
            if not polygon.covers(triangle.representative_point()):
                continue
            _add_floor_triangle(
                scene_spec,
                body,
                triangle,
                f"base_floor_{index}_{triangle_index}",
                visible=False,
            )
            triangle_index += 1


def _add_automatic_floors(
    scene_spec: mujoco.MjSpec,
    description: dict,
    structure: dict,
) -> tuple[str | None, int]:
    if not structure.get("auto_floor", True):
        return None, 0
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, triangulate, unary_union

    removed = set(structure["removed_base_walls"])
    custom_segments = [
        [wall["start"], wall["end"]]
        for wall in structure["walls"]
        if wall.get("active", True)
    ]
    if not custom_segments:
        return None, 0
    segments: dict[tuple, list[list[float]]] = {}
    for wall in description["walls"]:
        if wall["id"] not in removed:
            segments[_segment_key(wall["segment"])] = wall["segment"]
    for segment in custom_segments:
        segments[_segment_key(segment)] = segment
    if not segments:
        return None, 0
    lines = unary_union([LineString(segment) for segment in segments.values()])
    custom_lines = unary_union([LineString(segment) for segment in custom_segments])
    candidate_rooms = list(polygonize(lines))
    base_floors = unary_union(
        [Polygon([(point["x"], point["z"]) for point in room["floorPolygon"]])
         for room in description["rooms"]]
    )
    extra_polygons = []
    for room in candidate_rooms:
        if room.boundary.intersection(custom_lines).length < 0.05:
            continue
        extra = room.difference(base_floors.buffer(1e-4))
        extra_polygons.extend(poly for poly in _iter_polygons(extra) if poly.area > 0.05)
    if not extra_polygons:
        return None, 0
    body_name = "editor_auto_floors"
    body = scene_spec.worldbody.add_body(name=body_name)
    floor_color, floor_material = _floor_finish_appearance(
        scene_spec,
        "auto_floors",
        structure["floor_finishes"].get(
            "auto_floors", structure.get("floor_default_finish")
        ),
    )
    triangle_count = 0
    for polygon_index, polygon in enumerate(extra_polygons):
        for triangle in triangulate(polygon):
            if not polygon.covers(triangle.representative_point()):
                continue
            _add_floor_triangle(
                scene_spec,
                body,
                triangle,
                f"auto_floor_{polygon_index}_{triangle_count}",
                rgba=floor_color,
                material=floor_material,
            )
            triangle_count += 1
    return body_name, len(extra_polygons)


def _add_custom_floor(
    scene_spec: mujoco.MjSpec,
    floor: dict,
    finish: dict | None,
) -> tuple[str, int]:
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import Polygon

    polygon = Polygon(floor["polygon"])
    if not polygon.is_valid or polygon.area <= 0.05:
        raise ValueError(f"Invalid custom floor polygon: {floor['id']}")
    body_name = f"editor_floor_{floor['id']}"
    body = scene_spec.worldbody.add_body(
        name=body_name,
        pos=[0.0, 0.0, float(floor.get("z", 0.0))],
    )
    color, material = _floor_finish_appearance(scene_spec, floor["id"], finish)
    uv_bounds = tuple(float(value) for value in polygon.bounds)
    collidable = bool(floor.get("collision", True))
    # Use separate convex collision pieces; a combined convex hull would bridge cliff gaps.
    _add_floor_polygon_mesh(
        scene_spec,
        body,
        polygon,
        f"custom_floor_{floor['id']}_visual",
        rgba=color,
        material=material,
        uv_bounds=uv_bounds,
        collidable=False,
    )
    collision_count = 0
    if collidable:
        for triangle in constrained_delaunay_triangles(polygon).geoms:
            _add_floor_triangle(
                scene_spec,
                body,
                triangle,
                f"custom_floor_{floor['id']}_collision_{collision_count}",
                visible=False,
                collidable=True,
            )
            collision_count += 1
    return body_name, 1 + collision_count


def _add_editor_structures(
    scene_spec: mujoco.MjSpec,
    assets_dir: Path,
    layout: dict,
) -> tuple[list[dict], int]:
    description = _scene_description(assets_dir)
    structure = layout["structure"]
    removed_walls = set(structure["removed_base_walls"])
    removed_doors = set(structure["removed_base_doors"])
    removed_floors = set(structure["removed_base_floors"])
    _delete_base_walls(scene_spec, description, removed_walls)
    _delete_base_floors(scene_spec, description, removed_floors)
    _add_base_floor_collisions(scene_spec, description, removed_floors)
    _remove_original_door_assets(scene_spec, description)
    _remove_windows_on_deleted_walls(
        scene_spec, assets_dir, description, removed_walls
    )
    blueprints: list[dict] = []

    for floor in _base_floor_records(description):
        if floor["room_id"] not in removed_floors:
            finish = structure["floor_finishes"].get(
                floor["id"], structure.get("floor_default_finish")
            )
            if finish:
                rgba, material = _floor_finish_appearance(
                    scene_spec, floor["id"], finish
                )
                _apply_finish_to_bodies(
                    scene_spec, floor["body_names"], rgba, material
                )
            blueprints.append(floor)

    for floor in structure.get("floors", []):
        if not floor.get("active", True):
            continue
        finish = structure["floor_finishes"].get(
            floor["id"], structure.get("floor_default_finish")
        )
        body_name, triangle_count = _add_custom_floor(scene_spec, floor, finish)
        points = np.asarray(floor["polygon"], dtype=float)
        blueprints.append(
            {
                **floor,
                "kind": "floor",
                "start": points.min(axis=0).tolist(),
                "end": points.max(axis=0).tolist(),
                "body_names": [body_name],
                "source_ids": [],
                "floor_count": triangle_count,
            }
        )

    for group in _base_wall_groups(description):
        active_sources = [
            (source_id, name)
            for source_id, name in zip(group["source_ids"], group["body_names"])
            if source_id not in removed_walls
        ]
        body_names = [name for _, name in active_sources]
        if not body_names:
            continue
        explicit_finish = structure["wall_finishes"].get(group["id"])
        for source_id, body_name in active_sources:
            room_match = re.match(r"wall\|(room\d+)\|", source_id)
            room_id = room_match.group(1) if room_match else ""
            finish = explicit_finish
            if finish is None and room_id:
                finish = structure["room_wall_finishes"].get(room_id)
            if finish is None:
                finish = structure.get("wall_default_finish")
            if finish:
                rgba, material = _wall_finish_appearance(
                    scene_spec,
                    f"{group['id']}_{room_id or 'shared'}_{body_name}",
                    finish,
                )
                _apply_finish_to_bodies(
                    scene_spec, [body_name], rgba, material
                )
        blueprints.append({**group, "body_names": body_names})

    for replacement in structure.get("window_replacements", []):
        if not replacement.get("active", True) or not replacement.get("openings"):
            continue
        room_id = str(replacement.get("room_id", ""))
        finish = structure["wall_finishes"].get(replacement["id"])
        if finish is None and room_id:
            finish = structure["room_wall_finishes"].get(room_id)
        if finish is None:
            finish = structure.get("wall_default_finish")
        body_name = _add_wide_window_wall(scene_spec, replacement, finish)
        blueprints.append(
            {
                **replacement,
                "kind": "window_wall",
                "body_names": [body_name],
                "source_ids": list(replacement.get("converted_from", [])),
            }
        )

    for panel in structure.get("room_finish_panels", []):
        if not panel.get("active", True):
            continue
        room_id = str(panel.get("room_id", ""))
        finish = structure["room_wall_finishes"].get(
            room_id, structure.get("wall_default_finish")
        )
        rgba, material = _wall_finish_appearance(
            scene_spec, panel["id"], finish
        )
        body = scene_spec.worldbody.add_body(
            name=f"editor_room_finish_{panel['id']}"
        )
        _add_visual_wall_finish_panel(
            body,
            name=f"editor_room_finish_geom_{panel['id']}",
            start=panel["start"],
            end=panel["end"],
            height=float(panel.get("height", 3.2)),
            thickness=float(panel.get("thickness", 0.002)),
            rgba=rgba,
            material=material,
        )

    for door_index, source in enumerate(description["doors"]):
        door = _base_door_record(source)
        if door["id"] in removed_doors:
            continue
        body_name = f"editor_base_door_{door_index}"
        _add_open_door_frame(scene_spec, door, body_name)
        blueprints.append({**door, "body_names": [body_name], "source_ids": [door["id"]]})

    active_doors = [door for door in structure["doors"] if door.get("active", True)]
    for wall in structure["walls"]:
        if not wall.get("active", True):
            continue
        body_name = f"editor_wall_{wall['id']}"
        finish = structure["wall_finishes"].get(wall["id"])
        if finish is None and wall.get("room_id"):
            finish = structure["room_wall_finishes"].get(wall["room_id"])
        if finish is None:
            finish = structure.get("wall_default_finish")
        _add_wall(
            scene_spec,
            wall,
            active_doors,
            body_name,
            finish,
        )
        blueprints.append(
            {
                **wall,
                "kind": "wall",
                "body_names": [body_name],
                "source_ids": [],
            }
        )
    for door in active_doors:
        body_name = f"editor_door_{door['id']}"
        _add_open_door_frame(scene_spec, door, body_name)
        blueprints.append(
            {
                **door,
                "kind": "door",
                "body_names": [body_name],
                "source_ids": [],
            }
        )
    floor_body, floor_count = _add_automatic_floors(
        scene_spec, description, structure
    )
    if floor_body:
        blueprints.append(
            {
                "id": "auto_floors",
                "kind": "auto_floor",
                "start": [0.0, 0.0],
                "end": [0.0, 0.0],
                "body_names": [floor_body],
                "source_ids": [],
                "floor_count": floor_count,
            }
        )
    return blueprints, floor_count


def _visual_floor_offset(asset_spec: mujoco.MjSpec) -> float:
    try:
        model = asset_spec.compile()
    except ValueError:
        return 0.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    minimum, _ = _model_visual_bounds(model, data)
    return 0.0 if not np.all(np.isfinite(minimum)) else -float(minimum[2])


def _has_freejoint(body: mujoco.MjsBody) -> bool:
    return any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in body.joints)


def _catalog_index(assets: list[dict]) -> dict[str, dict]:
    return {asset["catalog_id"]: asset for asset in assets}


def build_editor_spec(
    assets_dir: Path,
    catalog_assets: list[dict],
    layout: dict,
    *,
    movable_instances: bool = True,
) -> tuple[mujoco.MjSpec, list[dict], list[dict], int, bool]:
    scene_path = assets_dir / "scenes" / SCENE_SOURCE / "val_861.xml"
    if not scene_path.exists():
        raise FileNotFoundError(f"val_861 is missing: {scene_path}")
    scene_spec = mujoco.MjSpec.from_file(str(scene_path))
    safety_floor = scene_spec.geom("floor")
    if safety_floor is not None:
        scene_spec.delete(safety_floor)
    removed = _clear_val861_objects(scene_spec, assets_dir)
    _ensure_structure_layout(layout)
    structure_blueprints, _ = _add_editor_structures(
        scene_spec, assets_dir, layout
    )
    scene_spec.copy_during_attach = True

    catalog = _catalog_index(catalog_assets)
    asset_specs: dict[tuple, mujoco.MjSpec] = {}
    floor_offsets: dict[str, float] = {}
    runtime_blueprint: list[dict] = []
    layout_changed = False

    for instance in layout["objects"]:
        if not instance.get("active", True):
            continue
        catalog_id = instance["catalog_id"]
        if catalog_id not in catalog:
            raise RuntimeError(f"Layout asset is not in the catalog: {catalog_id}")
        asset = catalog[catalog_id]
        if not asset.get("importable"):
            raise RuntimeError(f"Asset requires conversion before MuJoCo import: {catalog_id}")
        is_dynamic = bool(
            instance.get(
                "dynamic",
                asset.get("dynamic", asset_default_dynamic(asset)),
            )
        )
        needs_freejoint = movable_instances or is_dynamic
        mapping_collision = instance.get("mapping_collision")
        mapping_signature = ()
        if mapping_collision:
            mapping_signature = (
                tuple(float(value) for value in mapping_collision["full_size"]),
                float(mapping_collision.get("z_bottom", 0.0)),
                float(instance["yaw"]),
                float(instance["pos"][2]),
                tuple(
                    float(value)
                    for value in instance.get(
                        "base_quat",
                        asset.get("base_quat", [1.0, 0.0, 0.0, 0.0]),
                    )
                ),
            )
        asset_key = (catalog_id, needs_freejoint, mapping_signature)

        if asset_key not in asset_specs:
            asset_spec = mujoco.MjSpec.from_file(asset["xml_path"])
            for site in asset_spec.sites:
                site.group = 4
            root = asset_spec.body(asset["root_body"])
            if root is None:
                raise RuntimeError(f"Missing root body {asset['root_body']!r}: {catalog_id}")
            base_quat = instance.get(
                "base_quat", asset.get("base_quat", [1.0, 0.0, 0.0, 0.0])
            )
            root.pos = [0.0, 0.0, 0.0]
            root.quat = base_quat
            floor_offsets[catalog_id] = float(
                asset.get("floor_z", _visual_floor_offset(asset_spec))
            )
            if needs_freejoint and not _has_freejoint(root):
                if root.joints:
                    raise RuntimeError(
                        f"Asset root has non-free joints and cannot be moved by the editor: {catalog_id}"
                    )
                root.add_freejoint(name=f"{asset['root_body']}_editor_free")
            elif not needs_freejoint and _has_freejoint(root):
                for joint in list(root.joints):
                    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                        asset_spec.delete(joint)
            if mapping_collision:
                for geom in asset_spec.geoms:
                    if geom.group != 0 or geom.contype or geom.conaffinity:
                        geom.contype = 0
                        geom.conaffinity = 0
                        geom.rgba = [0.0, 0.0, 0.0, 0.0]
                full_size = [
                    float(value) for value in mapping_collision["full_size"]
                ]
                z_bottom = float(mapping_collision.get("z_bottom", 0.0))
                placed_quat = quaternion_multiply(
                    yaw_quaternion(float(instance["yaw"])), base_quat
                )
                inverse_quat = quaternion_conjugate(placed_quat)
                local_center = quaternion_rotate(
                    inverse_quat,
                    [
                        0.0,
                        0.0,
                        z_bottom
                        + full_size[2] / 2.0
                        - float(instance["pos"][2]),
                    ],
                )
                root.add_geom(
                    name="mapping_collision_proxy",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=local_center,
                    quat=inverse_quat,
                    size=[value / 2.0 for value in full_size],
                    # mj_ray requires nonzero alpha; hide collision geometry through group filters.
                    rgba=[0.18, 0.62, 0.88, 1.0],
                    contype=1,
                    conaffinity=1,
                    group=4,
                    friction=[0.9, 0.02, 0.001],
                    density=20.0,
                )
            asset_specs[asset_key] = asset_spec

        asset_spec = asset_specs[asset_key]
        root = asset_spec.body(asset["root_body"])
        if root is None:
            raise RuntimeError(f"Asset root disappeared while attaching: {catalog_id}")
        if instance.pop("auto_ground", False):
            instance["pos"][2] = floor_offsets[catalog_id]
            layout_changed = True
        base_quat = instance.get(
            "base_quat", asset.get("base_quat", [1.0, 0.0, 0.0, 0.0])
        )
        root.pos = instance["pos"]
        root.quat = quaternion_multiply(yaw_quaternion(instance["yaw"]), base_quat)
        prefix = f"editor/{instance['id']}/"
        frame = scene_spec.worldbody.add_frame()
        frame.attach_body(root, prefix, "")
        runtime_blueprint.append(
            {
                "instance_id": instance["id"],
                "catalog_id": catalog_id,
                "root_name": f"{prefix}{asset['root_body']}",
                "floor_z": floor_offsets[catalog_id],
                "dynamic": is_dynamic,
            }
        )

    return (
        scene_spec,
        runtime_blueprint,
        structure_blueprints,
        removed,
        layout_changed,
    )


def build_editor_model(
    assets_dir: Path,
    catalog_assets: list[dict],
    layout: dict,
) -> tuple[mujoco.MjModel, mujoco.MjData, list[dict], list[dict], int]:
    (
        scene_spec,
        runtime_blueprint,
        structure_blueprints,
        removed,
        layout_changed,
    ) = build_editor_spec(assets_dir, catalog_assets, layout)

    model = scene_spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    runtime: list[dict] = []
    for blueprint in runtime_blueprint:
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, blueprint["root_name"]
        )
        free_joint_id = -1
        joint_start = model.body_jntadr[body_id]
        for joint_id in range(joint_start, joint_start + model.body_jntnum[body_id]):
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                free_joint_id = joint_id
                break
        if free_joint_id < 0:
            raise RuntimeError(f"Editor freejoint is missing: {blueprint['root_name']}")
        blueprint.update(
            {
                "body_id": body_id,
                "qpos_adr": int(model.jnt_qposadr[free_joint_id]),
                "qvel_adr": int(model.jnt_dofadr[free_joint_id]),
                "geom_ids": _subtree_geom_ids(model, body_id),
            }
        )
        runtime.append(blueprint)
    structure_runtime: list[dict] = []
    for blueprint in structure_blueprints:
        body_ids = []
        geom_ids = []
        for body_name in blueprint["body_names"]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            body_ids.append(body_id)
            geom_ids.extend(_subtree_geom_ids(model, body_id))
        structure_runtime.append(
            {**blueprint, "body_ids": body_ids, "geom_ids": sorted(set(geom_ids))}
        )
    if layout_changed:
        layout["_needs_save"] = True
    return model, data, runtime, structure_runtime, removed


def _subtree_geom_ids(model: mujoco.MjModel, root_body_id: int) -> list[int]:
    descendants = {root_body_id}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if body_id not in descendants and model.body_parentid[body_id] in descendants:
                descendants.add(body_id)
                changed = True
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in descendants and model.geom_group[geom_id] == 0
    ]


def _layout_by_id(layout: dict) -> dict[str, dict]:
    return {item["id"]: item for item in layout["objects"]}


def apply_instance_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime_item: dict,
    instance: dict,
    asset: dict,
) -> None:
    address = runtime_item["qpos_adr"]
    data.qpos[address : address + 3] = instance["pos"]
    base_quat = instance.get(
        "base_quat", asset.get("base_quat", [1.0, 0.0, 0.0, 0.0])
    )
    data.qpos[address + 3 : address + 7] = quaternion_multiply(
        yaw_quaternion(instance["yaw"]), base_quat
    )
    velocity_address = runtime_item.get("qvel_adr")
    if velocity_address is not None:
        data.qvel[velocity_address : velocity_address + 6] = 0.0
    mujoco.mj_normalizeQuat(model, data.qpos)
    mujoco.mj_forward(model, data)


def _pin_static_instances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: list[dict],
    layout_items: dict[str, dict],
    catalog: dict[str, dict],
) -> None:
    for runtime_item in runtime:
        if runtime_item.get("dynamic", False):
            continue
        instance = layout_items[runtime_item["instance_id"]]
        asset = catalog[runtime_item["catalog_id"]]
        address = runtime_item["qpos_adr"]
        data.qpos[address : address + 3] = instance["pos"]
        base_quat = instance.get(
            "base_quat", asset.get("base_quat", [1.0, 0.0, 0.0, 0.0])
        )
        data.qpos[address + 3 : address + 7] = quaternion_multiply(
            yaw_quaternion(instance["yaw"]), base_quat
        )
        velocity_address = runtime_item["qvel_adr"]
        data.qvel[velocity_address : velocity_address + 6] = 0.0
    mujoco.mj_normalizeQuat(model, data.qpos)


def step_physics_preview(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: list[dict],
    layout_items: dict[str, dict],
    catalog: dict[str, dict],
    step_count: int,
) -> None:
    for _ in range(max(0, step_count)):
        _pin_static_instances(model, data, runtime, layout_items, catalog)
        mujoco.mj_step(model, data)
        _pin_static_instances(model, data, runtime, layout_items, catalog)
    if step_count:
        mujoco.mj_forward(model, data)


def _set_overview_camera(camera: mujoco.MjvCamera) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [12.0, 10.5, 0.4]
    camera.distance = 29.0
    camera.azimuth = 180.0
    camera.elevation = -72.0


def _camera_state(camera: mujoco.MjvCamera) -> dict:
    return {
        "type": int(camera.type),
        "fixedcamid": int(camera.fixedcamid),
        "trackbodyid": int(camera.trackbodyid),
        "lookat": np.asarray(camera.lookat, dtype=float).copy(),
        "distance": float(camera.distance),
        "azimuth": float(camera.azimuth),
        "elevation": float(camera.elevation),
    }


def _restore_camera(camera: mujoco.MjvCamera, state: dict) -> None:
    camera.type = state["type"]
    camera.fixedcamid = state["fixedcamid"]
    camera.trackbodyid = state["trackbodyid"]
    camera.lookat[:] = state["lookat"]
    camera.distance = state["distance"]
    camera.azimuth = state["azimuth"]
    camera.elevation = state["elevation"]


def _project_to_segment(
    point: list[float] | np.ndarray,
    start: list[float] | np.ndarray,
    end: list[float] | np.ndarray,
) -> tuple[np.ndarray, float, float]:
    point_array = np.asarray(point, dtype=float)
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    delta = end_array - start_array
    squared_length = float(np.dot(delta, delta))
    if squared_length < 1e-10:
        return start_array.copy(), float(np.linalg.norm(point_array - start_array)), 0.0
    fraction = float(np.dot(point_array - start_array, delta) / squared_length)
    fraction = min(1.0, max(0.0, fraction))
    projection = start_array + delta * fraction
    return projection, float(np.linalg.norm(point_array - projection)), fraction


def _available_wall_segments(layout: dict, description: dict) -> list[dict]:
    removed = set(layout["structure"]["removed_base_walls"])
    segments = [
        group
        for group in _base_wall_groups(description)
        if not all(source_id in removed for source_id in group["source_ids"])
    ]
    segments.extend(
        {**wall, "kind": "wall"}
        for wall in layout["structure"]["walls"]
        if wall.get("active", True)
    )
    return segments


def _nearest_wall_projection(
    point: list[float],
    segments: list[dict],
) -> tuple[np.ndarray, float, dict | None]:
    best_point = np.asarray(point, dtype=float)
    best_distance = math.inf
    best_segment = None
    for segment in segments:
        projection, distance, _ = _project_to_segment(
            point, segment["start"], segment["end"]
        )
        if distance < best_distance:
            best_point, best_distance, best_segment = projection, distance, segment
    return best_point, best_distance, best_segment


def _align_wall_points(
    layout: dict,
    description: dict,
    start: list[float],
    end: list[float],
) -> tuple[list[float], list[float]]:
    segments = _available_wall_segments(layout, description)
    endpoints = [
        np.asarray(point, dtype=float)
        for segment in segments
        for point in (segment["start"], segment["end"])
    ]
    points = [np.asarray(start, dtype=float), np.asarray(end, dtype=float)]
    snapped = [False, False]
    for index, point in enumerate(points):
        if endpoints:
            closest = min(endpoints, key=lambda value: float(np.linalg.norm(value - point)))
            if float(np.linalg.norm(closest - point)) <= 0.30:
                points[index] = closest.copy()
                snapped[index] = True
                continue
        projection, distance, _ = _nearest_wall_projection(point.tolist(), segments)
        if distance <= 0.15:
            points[index] = projection
            snapped[index] = True

    delta = points[1] - points[0]
    if float(np.linalg.norm(delta)) > 1e-5:
        angle = math.atan2(delta[1], delta[0])
        cardinal = round(angle / (math.pi / 2.0)) * (math.pi / 2.0)
        deviation = abs(math.atan2(math.sin(angle - cardinal), math.cos(angle - cardinal)))
        if deviation <= math.radians(12.0):
            if abs(math.cos(cardinal)) > 0.5:
                if snapped[1] and not snapped[0]:
                    points[0][1] = points[1][1]
                else:
                    points[1][1] = points[0][1]
            else:
                if snapped[1] and not snapped[0]:
                    points[0][0] = points[1][0]
                else:
                    points[1][0] = points[0][0]
    return points[0].round(4).tolist(), points[1].round(4).tolist()


def _align_door_points(
    layout: dict,
    description: dict,
    start: list[float],
    end: list[float],
) -> tuple[list[float], list[float]]:
    best = None
    for segment in _available_wall_segments(layout, description):
        projected_start, distance_start, _ = _project_to_segment(
            start, segment["start"], segment["end"]
        )
        projected_end, distance_end, _ = _project_to_segment(
            end, segment["start"], segment["end"]
        )
        score = max(distance_start, distance_end)
        if best is None or score < best[0]:
            best = (score, projected_start, projected_end)
    if best is not None and best[0] <= 0.45:
        return best[1].round(4).tolist(), best[2].round(4).tolist()
    return [round(float(v), 4) for v in start], [round(float(v), 4) for v in end]


def _realign_custom_walls(layout: dict, description: dict) -> bool:
    walls = [wall for wall in layout["structure"]["walls"] if wall.get("active", True)]
    if not walls:
        return False
    before = [(tuple(wall["start"]), tuple(wall["end"])) for wall in walls]
    base_endpoints = [
        np.asarray(point, dtype=float)
        for group in _base_wall_groups(description)
        for point in (group["start"], group["end"])
    ]
    references = [(wall, key) for wall in walls for key in ("start", "end")]
    anchored: dict[tuple[int, str], np.ndarray] = {}
    for wall, key in references:
        point = np.asarray(wall[key], dtype=float)
        closest = min(base_endpoints, key=lambda value: float(np.linalg.norm(value - point)))
        if float(np.linalg.norm(closest - point)) <= 0.20:
            wall[key] = closest.copy().tolist()
            anchored[(id(wall), key)] = closest.copy()

    for _ in range(4):
        unused = set(range(len(references)))
        while unused:
            seed = unused.pop()
            cluster = {seed}
            grew = True
            while grew:
                grew = False
                for candidate in list(unused):
                    candidate_point = np.asarray(references[candidate][0][references[candidate][1]])
                    if any(
                        float(
                            np.linalg.norm(
                                candidate_point
                                - np.asarray(references[index][0][references[index][1]])
                            )
                        )
                        <= 0.20
                        for index in cluster
                    ):
                        cluster.add(candidate)
                        unused.remove(candidate)
                        grew = True
            anchored_points = [
                anchored[(id(references[index][0]), references[index][1])]
                for index in cluster
                if (id(references[index][0]), references[index][1]) in anchored
            ]
            if anchored_points:
                target = anchored_points[0]
            else:
                target = np.mean(
                    [np.asarray(references[index][0][references[index][1]]) for index in cluster],
                    axis=0,
                )
            for index in cluster:
                wall, key = references[index]
                wall[key] = target.round(4).tolist()

        for wall in walls:
            start = np.asarray(wall["start"], dtype=float)
            end = np.asarray(wall["end"], dtype=float)
            delta = end - start
            if float(np.linalg.norm(delta)) < 1e-5:
                continue
            if abs(delta[0]) >= abs(delta[1]) * 4.5:
                y = start[1] if (id(wall), "start") in anchored else end[1] if (id(wall), "end") in anchored else float((start[1] + end[1]) / 2.0)
                wall["start"][1] = wall["end"][1] = round(float(y), 4)
                if (id(wall), "start") in anchored:
                    anchored[(id(wall), "end")] = np.asarray(wall["end"], dtype=float)
                elif (id(wall), "end") in anchored:
                    anchored[(id(wall), "start")] = np.asarray(wall["start"], dtype=float)
            elif abs(delta[1]) >= abs(delta[0]) * 4.5:
                x = start[0] if (id(wall), "start") in anchored else end[0] if (id(wall), "end") in anchored else float((start[0] + end[0]) / 2.0)
                wall["start"][0] = wall["end"][0] = round(float(x), 4)
                if (id(wall), "start") in anchored:
                    anchored[(id(wall), "end")] = np.asarray(wall["end"], dtype=float)
                elif (id(wall), "end") in anchored:
                    anchored[(id(wall), "start")] = np.asarray(wall["start"], dtype=float)

    after = [(tuple(wall["start"]), tuple(wall["end"])) for wall in walls]
    return before != after


def _new_structure_item(
    layout: dict,
    kind: str,
    start: list[float],
    end: list[float],
) -> dict:
    defaults = layout["structure"][f"{kind}_defaults"]
    item = {
        "id": f"{kind}_{uuid.uuid4().hex[:10]}",
        "active": True,
        "start": [round(float(value), 4) for value in start],
        "end": [round(float(value), 4) for value in end],
        "height": float(defaults["height"]),
        "thickness": float(defaults["thickness"]),
    }
    layout["structure"][f"{kind}s"].append(item)
    return item


def _ensure_door_host_wall(layout: dict, description: dict, door: dict) -> None:
    for wall in layout["structure"]["walls"]:
        if wall.get("active", True) and _door_interval_on_wall(wall, door):
            return
    removed = set(layout["structure"]["removed_base_walls"])
    for group in _base_wall_groups(description):
        if all(source_id in removed for source_id in group["source_ids"]):
            continue
        host = {"start": group["start"], "end": group["end"]}
        if not _door_interval_on_wall(host, door):
            continue
        for source_id in group["source_ids"]:
            if source_id not in layout["structure"]["removed_base_walls"]:
                layout["structure"]["removed_base_walls"].append(source_id)
        replacement = _new_structure_item(layout, "wall", group["start"], group["end"])
        replacement["converted_from"] = list(group["source_ids"])
        return


def _append_structure_from_points(
    layout: dict,
    description: dict,
    kind: str,
    start: list[float],
    end: list[float],
) -> dict:
    if kind == "wall":
        start, end = _align_wall_points(layout, description, start, end)
    elif kind == "door":
        start, end = _align_door_points(layout, description, start, end)
    if float(np.linalg.norm(np.asarray(end) - np.asarray(start))) < 0.2:
        raise ValueError("Endpoints must be at least 0.2 m apart.")
    item = _new_structure_item(layout, kind, start, end)
    if kind == "wall":
        _realign_custom_walls(layout, description)
    elif kind == "door":
        _ensure_door_host_wall(layout, description, item)
    return item


def _remove_structure_item(layout: dict, description: dict, item: dict) -> None:
    structure = layout["structure"]
    if item["kind"] == "base_wall":
        for source_id in item["source_ids"]:
            if source_id not in structure["removed_base_walls"]:
                structure["removed_base_walls"].append(source_id)
        related = {
            door["id"]
            for door in description["doors"]
            if door.get("wall0") in item["source_ids"] or door.get("wall1") in item["source_ids"]
        }
        for door_id in related:
            if door_id not in structure["removed_base_doors"]:
                structure["removed_base_doors"].append(door_id)
    elif item["kind"] == "base_door":
        for source_id in item["source_ids"]:
            if source_id not in structure["removed_base_doors"]:
                structure["removed_base_doors"].append(source_id)
    elif item["kind"] == "base_floor":
        for room_id in item["source_ids"]:
            if room_id not in structure["removed_base_floors"]:
                structure["removed_base_floors"].append(room_id)
    elif item["kind"] == "auto_floor":
        structure["auto_floor"] = False
    elif item["kind"] == "floor":
        matches = [entry for entry in structure.get("floors", []) if entry["id"] == item["id"]]
        if matches:
            matches[0]["active"] = False
    elif item["kind"] in ("wall", "door"):
        matches = [entry for entry in structure[f"{item['kind']}s"] if entry["id"] == item["id"]]
        if matches:
            matches[0]["active"] = False


SURFACE_TEXTURE_GROUPS = {
    "wall": {
        "Paint & Plaster": ("drywall", "plaster", "stucco", "paintwall"),
        "Brick & Masonry": ("brick", "concrete", "masonry"),
        "Tile & Stone": ("tile", "marble", "granite", "stone"),
        "Wood Panel": (
            "woodgrain", "woodfine", "woodwhite", "wornwood", "woodplank",
            "woodslash", "crosshatchwood",
        ),
        "Fabric & Pattern": (
            "wallpaper", "wall_panel_fabric", "fabric", "pattern", "textile", "cloth",
        ),
        "Metal & Industrial": ("metal", "steel", "iron", "aluminum", "copper"),
        "Custom / Imported": (),
    },
    "floor": {
        "Wood & Parquet": (
            "woodgrain", "woodfine", "woodwhite", "wornwood", "woodplank",
            "woodslash", "crosshatchwood", "woodfloor", "bedroomfloor",
        ),
        "Tile & Stone": ("tile", "marble", "granite", "stone", "checkerboard"),
        "Carpet & Rug": ("carpet", "rug"),
        "Concrete & Industrial": ("concrete", "cement", "rubber", "metal", "steel"),
        "Outdoor": ("asphalt", "grass", "dirt", "gravel", "sand"),
        "Custom / Imported": (),
    },
}


def _surface_texture_display_name(path: Path) -> str:
    name = re.sub(r"_resized$", "", path.stem, flags=re.IGNORECASE)
    name = re.sub(r"^TexturesCom[_ -]*", "", name, flags=re.IGNORECASE)
    name = re.sub(
        r"[_ -]*(AlbedoTransparency|Albedo|Diffuse|Seamless)([_ -]*S)?$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    name = re.sub(r"[_-]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


@lru_cache(maxsize=8)
def _surface_texture_catalog(
    assets_dir_text: str,
    surface_kind: str,
) -> tuple[dict, ...]:
    groups = SURFACE_TEXTURE_GROUPS[surface_kind]
    texture_root = Path(assets_dir_text) / "objects" / "thor" / "Textures"
    extensions = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp"}
    excluded = (
        "normal", "nrm", "metallic", "smoothness", "roughness",
        "ambientocclusion", "emission", "opacity", "mask", "heightmap",
        "decal", "poster", "painting", "photo",
    )
    candidates: dict[str, tuple[tuple, dict]] = {}

    def add_candidate(
        path: Path,
        category: str,
        *,
        source: str,
        duplicate_scope: str = "builtin",
    ) -> None:
        lowered = path.stem.lower()
        duplicate_key = f"{duplicate_scope}:" + re.sub(
            r"[ _-]+", "", re.sub(r"_resized$", "", lowered)
        )
        preference = (
            lowered.endswith("_resized"),
            path.suffix.lower() != ".png",
            "cropped" in lowered,
            len(path.name),
        )
        entry = {
            "path": str(path.resolve()),
            "display_name": _surface_texture_display_name(path),
            "category": category,
            "source": source,
        }
        if source == "Built-in":
            entry["texture_ref"] = f"assets://objects/thor/Textures/{path.name}"
        else:
            entry["texture_ref"] = _portable_surface_texture_reference(
                str(path), Path(assets_dir_text)
            )
        previous = candidates.get(duplicate_key)
        if previous is None or preference < previous[0]:
            candidates[duplicate_key] = (preference, entry)

    if texture_root.exists():
        for path in texture_root.iterdir():
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            lowered = path.stem.lower()
            if (
                any(token in lowered for token in excluded)
                or lowered.endswith("ao")
                or (surface_kind == "floor" and "glass" in lowered)
                or (
                    surface_kind == "wall"
                    and any(
                        token in lowered
                        for token in (
                            "plate_", "paperroll", "sofa", "tablecloth", "blanket"
                        )
                    )
                )
            ):
                continue
            category = next(
                (
                    category
                    for category, tokens in groups.items()
                    if tokens and any(token in lowered for token in tokens)
                ),
                None,
            )
            if category is not None:
                add_candidate(path, category, source="Built-in")

    custom_root = CUSTOM_SURFACE_ROOT / surface_kind
    if custom_root.exists():
        for path in custom_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            relative = path.relative_to(custom_root)
            requested_category = relative.parts[0] if len(relative.parts) > 1 else ""
            category = (
                requested_category
                if requested_category in groups
                else "Custom / Imported"
            )
            add_candidate(
                path,
                category,
                source="Imported",
                duplicate_scope=f"custom:{category}",
            )
    return tuple(
        entry
        for _, entry in sorted(
            candidates.values(),
            key=lambda item: (
                list(groups).index(item[1]["category"]),
                item[1]["display_name"].lower(),
            ),
        )
    )


def surface_finish_dialog(
    layout: dict,
    surface_id: str,
    surface_kind: str,
    assets_dir: Path,
) -> bool:
    try:
        import tkinter as tk
        from tkinter import colorchooser, filedialog, messagebox, ttk
        from PIL import Image, ImageTk
    except ImportError:
        return False

    is_floor = surface_kind == "floor"
    finish_key = "floor_finishes" if is_floor else "wall_finishes"
    surface_label = "Floor" if is_floor else "Wall"
    default_rgba = (
        [0.62, 0.58, 0.50, 1.0]
        if is_floor
        else [0.76, 0.73, 0.67, 1.0]
    )
    finishes = layout["structure"][finish_key]
    textures = list(
        _surface_texture_catalog(str(assets_dir.resolve()), surface_kind)
    )
    texture_groups = tuple(SURFACE_TEXTURE_GROUPS[surface_kind])
    filtered: list[dict] = []
    changed = [False]
    window = tk.Tk()
    window.title(f"{surface_label} color / texture — {surface_id}")
    window.geometry("1050x700")
    window.lift()
    window.attributes("-topmost", True)
    window.after(350, lambda: window.attributes("-topmost", False))

    query = tk.StringVar()
    texture_group = tk.StringVar(value="All materials")
    current_finish = finishes.get(
        surface_id,
        layout["structure"].get(f"{surface_kind}_default_finish") or {},
    )
    default_repeat = 8.0 if is_floor else 2.0
    repeat = tk.DoubleVar(
        value=float(current_finish.get("repeat", default_repeat))
    )
    status = tk.StringVar(value=f"{len(textures)} compatible {surface_kind} textures")
    count_summary = tk.StringVar()
    header = ttk.Frame(window, padding=10)
    header.pack(fill="x")
    ttk.Label(header, text=f"Search compatible {surface_kind} textures:").pack(side="left")
    entry = ttk.Entry(header, textvariable=query)
    entry.pack(side="left", fill="x", expand=True, padx=8)
    ttk.Combobox(
        header,
        textvariable=texture_group,
        values=("All materials", *texture_groups),
        state="readonly",
        width=24,
    ).pack(side="right")

    body = ttk.Panedwindow(window, orient="horizontal")
    body.pack(fill="both", expand=True, padx=10)
    left = ttk.Frame(body)
    right = ttk.Frame(body, padding=(12, 0, 0, 0))
    body.add(left, weight=3)
    body.add(right, weight=2)
    ttk.Label(
        left,
        textvariable=count_summary,
        justify="left",
        wraplength=590,
    ).pack(fill="x", pady=(0, 6))
    results_frame = ttk.Frame(left)
    results_frame.pack(fill="both", expand=True)
    listbox = tk.Listbox(results_frame, font=("monospace", 10), exportselection=False)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(results_frame, command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    preview = ttk.Label(right, anchor="center")
    preview.pack(fill="both", expand=True)
    ttk.Label(right, textvariable=status, wraplength=390).pack(fill="x", pady=8)

    controls = ttk.Frame(window, padding=10)
    controls.pack(fill="x")
    ttk.Label(controls, text="Texture repeat").pack(side="left")
    ttk.Spinbox(controls, from_=0.25, to=40.0, increment=0.25, textvariable=repeat, width=7).pack(side="left", padx=6)

    def refresh(*_args) -> None:
        nonlocal filtered
        words = query.get().lower().split()
        selected_group = texture_group.get()
        filtered = [
            item
            for item in textures
            if (
                selected_group == "All materials"
                or item["category"] == selected_group
            )
            and all(
                word in (
                    f"{item['display_name']} {item['category']} {item['path']}"
                ).lower()
                for word in words
            )
        ]
        listbox.delete(0, "end")
        for item in filtered:
            source_mark = "+" if item["source"] == "Imported" else " "
            listbox.insert(
                "end",
                f"{source_mark} [{item['category']:<23}] {item['display_name']}",
            )
        category_counts = {
            group: sum(item["category"] == group for item in textures)
            for group in texture_groups
        }
        count_summary.set(
            f"Showing {len(filtered)} / Total {len(textures)}   "
            "(+ = imported)\n"
            + "  ·  ".join(
                f"{group}: {count}"
                for group, count in category_counts.items()
                if count
            )
        )
        status.set(
            f"{len(filtered)} of {len(textures)} compatible {surface_kind} textures"
        )

    def show_texture(*_args) -> None:
        selected = listbox.curselection()
        if not selected:
            return
        item = filtered[selected[0]]
        path = Path(item["path"])
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((420, 430))
            photo = ImageTk.PhotoImage(image)
            preview.configure(image=photo)
            preview.image = photo
            status.set(f"{item['category']} — {item['display_name']}")
        except Exception as error:
            status.set(f"Preview unavailable: {error}")

    def apply_texture() -> None:
        selected = listbox.curselection()
        if not selected:
            messagebox.showinfo("Select texture", "Select a texture first.")
            return
        item = filtered[selected[0]]
        finishes[surface_id] = {
            "mode": "texture",
            "texture_path": item.get("texture_ref", item["path"]),
            "texture_category": item["category"],
            "repeat": max(0.25, float(repeat.get())),
        }
        changed[0] = True
        window.destroy()

    def choose_color() -> None:
        current = finishes.get(surface_id, current_finish)
        rgba = current.get("rgba", default_rgba)
        initial = "#%02x%02x%02x" % tuple(
            max(0, min(255, round(float(value) * 255))) for value in rgba[:3]
        )
        rgb, _ = colorchooser.askcolor(initialcolor=initial, parent=window)
        if rgb is None:
            return
        finishes[surface_id] = {
            "mode": "color",
            "rgba": [component / 255.0 for component in rgb] + [1.0],
        }
        changed[0] = True
        window.destroy()

    def reset_finish() -> None:
        if surface_id in finishes:
            del finishes[surface_id]
            changed[0] = True
        window.destroy()

    def import_external_texture() -> None:
        nonlocal textures
        category = texture_group.get()
        if category == "All materials":
            messagebox.showinfo(
                "Choose material group",
                "Choose the wall/floor material group first, then import the image.",
            )
            return
        selected_path = filedialog.askopenfilename(
            parent=window,
            title=f"Import {surface_kind} base-color texture",
            filetypes=(
                ("Texture images", "*.png *.jpg *.jpeg *.webp *.bmp *.tga"),
                ("All files", "*"),
            ),
        )
        if not selected_path:
            return
        source_path = Path(selected_path).expanduser().resolve()
        try:
            with Image.open(source_path) as image:
                image.verify()
        except Exception as error:
            messagebox.showerror("Invalid image", str(error))
            return
        folder_name = category.replace("/", "-")
        destination_dir = CUSTOM_SURFACE_ROOT / surface_kind / folder_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source_path.name
        source_digest = hashlib.sha1(source_path.read_bytes()).hexdigest()
        if destination.exists() and source_path != destination.resolve():
            destination_digest = hashlib.sha1(destination.read_bytes()).hexdigest()
            if destination_digest != source_digest:
                destination = destination.with_name(
                    f"{destination.stem}_{source_digest[:8]}"
                    f"{destination.suffix.lower()}"
                )
        should_copy = source_path != destination.resolve()
        if destination.exists():
            should_copy = (
                hashlib.sha1(destination.read_bytes()).hexdigest()
                != source_digest
            )
        if should_copy:
            shutil.copy2(source_path, destination)
        _surface_texture_catalog.cache_clear()
        textures = list(
            _surface_texture_catalog(str(assets_dir.resolve()), surface_kind)
        )
        refresh()
        status.set(
            f"Imported {destination.name} into {category}. Select it to preview."
        )

    ttk.Button(controls, text="Apply selected texture", command=apply_texture).pack(side="left", padx=5)
    ttk.Button(controls, text="Choose solid color…", command=choose_color).pack(side="left", padx=5)
    ttk.Button(controls, text="Import external image…", command=import_external_texture).pack(side="left", padx=5)
    shared_finish = layout["structure"].get(
        f"{surface_kind}_default_finish"
    )
    ttk.Button(
        controls,
        text=(
            "Use shared/default finish"
            if shared_finish
            else "Restore original finish"
        ),
        command=reset_finish,
    ).pack(side="left", padx=5)
    ttk.Button(controls, text="Cancel", command=window.destroy).pack(side="right")
    query.trace_add("write", refresh)
    texture_group.trace_add("write", refresh)
    listbox.bind("<<ListboxSelect>>", show_texture)
    listbox.bind("<Double-Button-1>", lambda _event: apply_texture())
    refresh()
    entry.focus_set()
    window.mainloop()
    return changed[0]


def _highlight_selected(viewer, model: mujoco.MjModel, data: mujoco.MjData, item: dict | None) -> None:
    viewer.user_scn.ngeom = 0
    if item is None or not item["geom_ids"]:
        return
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    for geom_id in item["geom_ids"]:
        center = model.geom_aabb[geom_id, :3]
        half = model.geom_aabb[geom_id, 3:]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world_center = data.geom_xpos[geom_id] + rotation @ center
        world_half = np.abs(rotation) @ half
        minimum = np.minimum(minimum, world_center - world_half)
        maximum = np.maximum(maximum, world_center + world_half)
    if not np.all(np.isfinite(minimum)):
        return
    scene_geom = viewer.user_scn.geoms[0]
    size = np.maximum((maximum - minimum) / 2.0 + 0.025, 0.03)
    position = (minimum + maximum) / 2.0
    mujoco.mjv_initGeom(
        scene_geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        size,
        position,
        np.eye(3).reshape(-1),
        np.array([1.0, 0.82, 0.05, 0.12], dtype=np.float32),
    )
    scene_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    viewer.user_scn.ngeom = 1


def _hide_viewer_debug_geometry(viewer) -> None:
    viewer.opt.geomgroup[:] = 0
    viewer.opt.geomgroup[0] = 1
    for flag_name in (
        "mjVIS_ACTIVATION",
        "mjVIS_CONTACTPOINT",
        "mjVIS_CONTACTFORCE",
        "mjVIS_CONTACTSPLIT",
        "mjVIS_CONSTRAINT",
        "mjVIS_PERTFORCE",
        "mjVIS_PERTOBJ",
        "mjVIS_SELECT",
        "mjVIS_JOINT",
        "mjVIS_ACTUATOR",
        "mjVIS_AUTOCONNECT",
        "mjVIS_BODYBVH",
        "mjVIS_CAMERA",
        "mjVIS_COM",
        "mjVIS_CONVEXHULL",
        "mjVIS_INERTIA",
        "mjVIS_ISLAND",
        "mjVIS_LIGHT",
        "mjVIS_MESHBVH",
        "mjVIS_RANGEFINDER",
        "mjVIS_SCLINERTIA",
        "mjVIS_SDFITER",
        "mjVIS_TRANSPARENT",
    ):
        flag = getattr(mujoco.mjtVisFlag, flag_name, None)
        if flag is not None:
            viewer.opt.flags[flag] = 0
    viewer.perturb.active = 0


def _sync_viewer_without_debug(viewer) -> None:
    viewer.sync()
    with viewer.lock():
        _hide_viewer_debug_geometry(viewer)


def _consume_mouse_pick(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[int, np.ndarray] | None:
    with viewer.lock():
        body_id = int(viewer.perturb.select)
        if body_id <= 0 or body_id >= model.nbody:
            return None
        local_point = np.asarray(viewer.perturb.localpos, dtype=float).copy()
        rotation = data.xmat[body_id].reshape(3, 3)
        world_point = data.xpos[body_id] + rotation @ local_point
        viewer.perturb.active = 0
        viewer.perturb.select = 0
    return body_id, world_point


class _ViewerMousePoller:
    def __init__(self) -> None:
        self.display = None
        self.window = None
        self.left_mask = 0
        self.any_property_type = 0
        self.pid_atom = None
        self.target_pid = os.getpid()
        self.was_down = False
        try:
            from Xlib import X, display

            self.display = display.Display()
            self.left_mask = int(X.Button1Mask)
            self.any_property_type = int(X.AnyPropertyType)
            self.pid_atom = self.display.intern_atom("_NET_WM_PID")
        except Exception:
            self.display = None

    def _find_window(self):
        if self.display is None:
            return None
        pending = [(self.display.screen().root, 0)]
        while pending:
            window, depth = pending.pop(0)
            try:
                name = window.get_wm_name() or ""
                if str(name).startswith("MuJoCo") and self.pid_atom is not None:
                    pid_property = window.get_full_property(
                        self.pid_atom, self.any_property_type
                    )
                    window_pid = (
                        int(pid_property.value[0])
                        if pid_property is not None and len(pid_property.value)
                        else -1
                    )
                    if window_pid == self.target_pid:
                        return window
                if depth < 3:
                    pending.extend(
                        (child, depth + 1) for child in window.query_tree().children
                    )
            except Exception:
                continue
        return None

    def poll(self) -> tuple[float, float, float, float] | None:
        if self.display is None:
            return None
        if self.window is None:
            self.window = self._find_window()
            if self.window is None:
                return None
        try:
            pointer = self.window.query_pointer()
            geometry = self.window.get_geometry()
            down = bool(int(pointer.mask) & self.left_mask)
            clicked = down and not self.was_down
            self.was_down = down
            x = float(pointer.win_x)
            y = float(pointer.win_y)
            width = float(geometry.width)
            height = float(geometry.height)
            if clicked and 0.0 <= x < width and 0.0 <= y < height:
                return x, y, width, height
        except Exception:
            self.window = None
            self.was_down = False
        return None

    def arm(self) -> None:
        if self.display is None:
            return
        if self.window is None:
            self.window = self._find_window()
        if self.window is None:
            return
        try:
            pointer = self.window.query_pointer()
            self.was_down = bool(int(pointer.mask) & self.left_mask)
        except Exception:
            self.window = None
            self.was_down = False

    def close(self) -> None:
        if self.display is not None:
            try:
                self.display.close()
            except Exception:
                pass
            self.display = None


def _pick_viewer_point(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    click: tuple[float, float, float, float],
) -> tuple[int, np.ndarray] | None:
    mouse_x, mouse_y, window_width, window_height = click
    if window_width <= 0.0 or window_height <= 0.0:
        return None
    relx = float(np.clip(mouse_x / window_width, 0.0, 1.0))
    rely = float(np.clip(1.0 - mouse_y / window_height, 0.0, 1.0))
    with viewer.lock():
        viewport = viewer.viewport
        if viewport is None or viewport.width <= 0 or viewport.height <= 0:
            return None
        scene = mujoco.MjvScene(model, maxgeom=max(1000, model.ngeom * 2 + 100))
        mujoco.mjv_updateScene(
            model,
            data,
            viewer.opt,
            viewer.perturb,
            viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        selected_point = np.zeros(3, dtype=np.float64)
        geom_id = np.full(1, -1, dtype=np.int32)
        flex_id = np.full(1, -1, dtype=np.int32)
        skin_id = np.full(1, -1, dtype=np.int32)
        body_id = int(
            mujoco.mjv_select(
                model,
                data,
                viewer.opt,
                float(viewport.width) / float(viewport.height),
                relx,
                rely,
                scene,
                selected_point,
                geom_id,
                flex_id,
                skin_id,
            )
        )
    if body_id < 0:
        return None
    return body_id, selected_point


def _item_index_for_clicked_body(
    model: mujoco.MjModel,
    body_id: int,
    items: list[dict],
    root_field: str,
) -> int | None:
    ancestry = set()
    current = body_id
    while current > 0:
        ancestry.add(current)
        current = int(model.body_parentid[current])
    for index, item in enumerate(items):
        roots = item[root_field] if isinstance(item[root_field], list) else [item[root_field]]
        if any(root in ancestry for root in roots):
            return index
    return None


def _draw_endpoint_cursor(viewer, drawing: dict | None) -> None:
    if drawing is None:
        return

    def add_pin(point: list[float], color: np.ndarray) -> None:
        stem = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            stem,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([0.06, 0.55, 0.0]),
            np.array([point[0], point[1], 0.60]),
            np.eye(3).reshape(-1),
            color,
        )
        stem.category = mujoco.mjtCatBit.mjCAT_DECOR
        viewer.user_scn.ngeom += 1
        head = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            head,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.16, 0.0, 0.0]),
            np.array([point[0], point[1], 1.22]),
            np.eye(3).reshape(-1),
            color,
        )
        head.category = mujoco.mjtCatBit.mjCAT_DECOR
        viewer.user_scn.ngeom += 1

    first_point = drawing.get("first")
    if first_point is not None:
        add_pin(
            first_point,
            np.array([0.05, 1.0, 0.2, 0.95], dtype=np.float32),
        )
    if not drawing.get("cursor_visible", False):
        return
    cursor = np.array([drawing["cursor"][0], drawing["cursor"][1], 1.18])
    if first_point is None or float(
        np.linalg.norm(np.asarray(first_point) - np.asarray(drawing["cursor"]))
    ) > 1e-5:
        add_pin(
            drawing["cursor"],
            np.array([1.0, 0.35, 0.02, 0.95], dtype=np.float32),
        )
    if first_point is None:
        return
    first = np.array([first_point[0], first_point[1], 1.18])
    delta = cursor - first
    length = float(np.linalg.norm(delta[:2]))
    if length < 1e-4:
        return
    yaw = math.atan2(delta[1], delta[0])
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0],
         [math.sin(yaw), math.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]]
    )
    line = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    color = [0.05, 0.95, 1.0, 0.7] if drawing["kind"] == "wall" else [1.0, 0.45, 0.05, 0.8]
    mujoco.mjv_initGeom(
        line,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.array([length / 2.0, 0.035, 0.035]),
        (first + cursor) / 2.0,
        rotation.reshape(-1),
        np.array(color, dtype=np.float32),
    )
    line.category = mujoco.mjtCatBit.mjCAT_DECOR
    viewer.user_scn.ngeom += 1


def _overlay_text(
    viewer,
    runtime: list[dict],
    selected: int,
    layout_items: dict[str, dict],
    catalog: dict[str, dict],
    step: float,
    status: str,
    *,
    structures: list[dict] | None = None,
    structure_selected: int = 0,
    edit_mode: str = "objects",
    drawing: dict | None = None,
    object_move: dict | None = None,
    show_help: bool = True,
    physics_running: bool = False,
) -> None:
    physics_label = "PLAYING" if physics_running else "PAUSED"
    if not show_help:
        viewer.set_texts(
            [
                (
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"VAL_861 EDITOR  |  H: show help  |  Physics: {physics_label}",
                    status,
                )
            ]
        )
        return
    controls = (
        "VAL_861 OBJECT + ROOM EDITOR\n"
        f"H: hide help   R or left Run button: physics {physics_label}\n"
        "F2: object/structure mode   Double-click: direct selection/placement\n"
        "B: click wall endpoints     M: click doorway endpoints\n"
        "B/M: click candidate -> arrows -> Space (START, then END)\n"
        "A: visual asset picker      Space/Enter: confirm object move/pick point\n"
        "D: duplicate as draft   Delete: remove   U: undo previous edit\n"
        "Arrows: move XY   PgUp/PgDn: move Z   Q/E: rotate\n"
        "P: selected wall/floor finish\n"
        "C: focus          O: overview        S: save"
    )
    if drawing is not None:
        if drawing.get("first") is None:
            phase = "CLICK START CANDIDATE"
        elif not drawing.get("first_locked", False):
            phase = "ADJUST START — SPACE CONFIRMS"
        elif not drawing.get("cursor_visible", False):
            phase = "CLICK END CANDIDATE"
        else:
            phase = "ADJUST END — SPACE BUILDS"
        details = (
            f"{drawing['kind'].upper()} ENDPOINT MODE — {phase}\n"
            f"cursor {np.round(drawing['cursor'], 3).tolist()}\n"
            f"first {drawing.get('first', '-')}   step {step:.3f} m\n"
            "Click sets/replaces a candidate; arrows adjust; Space/Enter confirms. X cancels.\n"
            f"{status}"
        )
    elif edit_mode == "structures" and structures:
        item = structures[structure_selected % len(structures)]
        if item["kind"] in ("base_floor", "auto_floor", "floor"):
            geometry = (
                f"bounds {item['start']} -> {item['end']}"
                if item["kind"] in ("base_floor", "floor")
                else f"generated regions {item.get('floor_count', 0)}"
            )
            details = (
                f"STRUCTURE MODE   Selected {structure_selected % len(structures) + 1}/{len(structures)}\n"
                f"{item['kind']}  {item['id']}\n{geometry}\n"
                "Double-click floor to select; P changes finish; Delete removes it.\n"
                f"{status}"
            )
        else:
            length = float(np.linalg.norm(np.asarray(item["end"]) - np.asarray(item["start"])))
            details = (
                f"STRUCTURE MODE   Selected {structure_selected % len(structures) + 1}/{len(structures)}\n"
                f"{item['kind']}  {item['id']}\n"
                f"{item['start']} -> {item['end']}   length {length:.2f} m\n"
                "Double-click selects directly; P changes wall finish; Delete removes.\n"
                f"{status}"
            )
    elif edit_mode == "structures":
        details = f"STRUCTURE MODE\nNo selectable structures.\n{status}"
    elif runtime:
        selected_item = runtime[selected]
        instance = layout_items[selected_item["instance_id"]]
        asset = catalog[selected_item["catalog_id"]]
        if instance.get("draft"):
            mode = "DRAFT PREVIEW"
        elif object_move and object_move.get("instance_id") == instance["id"]:
            mode = "MOVE PREVIEW — SPACE CONFIRMS / X CANCELS"
        else:
            mode = "PLACED"
        details = (
            f"{mode}   Selected {selected + 1}/{len(runtime)}\n"
            f"{asset['display_name']}\n"
            f"{asset['catalog_id']}\n"
            f"xyz {np.round(instance['pos'], 3).tolist()}\n"
            f"yaw {instance['yaw']:.1f} deg   step {step:.3f} m\n"
            f"{status}"
        )
    else:
        details = f"OBJECT MODE\nNo placed assets\nPress A to choose one.\n{status}"
    details = f"PHYSICS {physics_label} — preview is not auto-saved\n{details}"
    viewer.set_texts(
        [
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                controls,
                details,
            )
        ]
    )


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: list[dict],
    structure_runtime: list[dict],
    catalog_assets: list[dict],
    layout: dict,
    layout_path: Path,
    assets_dir: Path,
) -> None:
    import mujoco.viewer

    key_queue: deque[int] = deque()
    queue_lock = threading.Lock()

    def key_callback(key: int) -> None:
        with queue_lock:
            key_queue.append(key)

    catalog = _catalog_index(catalog_assets)
    layout_items = _layout_by_id(layout)
    selected = 0
    structure_selected = 0
    edit_mode = "objects"
    drawing: dict | None = None
    object_move: dict | None = None
    undo_stack: list[dict] = []
    help_visible = True
    description = _scene_description(assets_dir)
    step = 0.1
    status = "Objects, walls, open doorways, and automatic floors are ready."
    initial_model = model
    initial_data = data
    retained_models: list[object] = [initial_model, initial_data, model, data]

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        mouse_poller = _ViewerMousePoller()
        _hide_viewer_debug_geometry(viewer)
        _set_overview_camera(viewer.cam)
        stable_camera = _camera_state(viewer.cam)
        simulator = viewer._get_sim()
        if simulator is not None:
            try:
                simulator.run = False
            except Exception:
                pass
        ui_physics_running = False
        last_ui_physics_running = False
        physics_override: bool | None = False
        physics_accumulator = 0.0
        physics_was_running = False
        previous_loop_time = time.perf_counter()

        def pause_native_simulator() -> None:
            nonlocal physics_override, physics_accumulator, physics_was_running
            nonlocal last_ui_physics_running
            current_simulator = viewer._get_sim()
            if current_simulator is not None:
                try:
                    current_simulator.run = False
                except Exception:
                    pass
            physics_override = False
            physics_accumulator = 0.0
            physics_was_running = False
            last_ui_physics_running = False

        def apply_pose_safely(runtime_item: dict, instance: dict) -> bool:
            nonlocal status
            try:
                with viewer.lock():
                    apply_instance_pose(
                        model,
                        data,
                        runtime_item,
                        instance,
                        catalog[runtime_item["catalog_id"]],
                    )
                return True
            except Exception as error:
                status = f"Move preview rejected; editor kept open: {error}"
                return False

        def hot_reload(
            select_instance_id: str | None = None,
            select_structure_id: str | None = None,
        ) -> bool:
            nonlocal model, data, runtime, structure_runtime
            nonlocal layout_items, selected, structure_selected, status
            nonlocal physics_accumulator, physics_was_running, physics_override
            nonlocal last_ui_physics_running
            pause_native_simulator()
            status = "Loading draft into the open MuJoCo window..."
            _overlay_text(
                viewer,
                runtime,
                selected % len(runtime) if runtime else 0,
                layout_items,
                catalog,
                step,
                status,
                show_help=help_visible,
                physics_running=False,
            )
            try:
                _sync_viewer_without_debug(viewer)
            except Exception:
                pass
            try:
                new_model, new_data, new_runtime, new_structures, _ = build_editor_model(
                    assets_dir, catalog_assets, layout
                )
            except Exception as error:
                status = f"Could not load asset: {error}"
                return False
            simulator = viewer._get_sim()
            if simulator is None:
                return False
            try:
                simulator.run = False
                viewer.user_scn.ngeom = 0
                simulator.load(new_model, new_data, "val_861 asset editor")
                simulator.run = False
            except Exception as error:
                status = f"Hot reload failed; editor kept open: {error}"
                return False
            retained_models[:] = [
                initial_model,
                initial_data,
                new_model,
                new_data,
            ]
            model, data, runtime = new_model, new_data, new_runtime
            structure_runtime = new_structures
            layout_items = _layout_by_id(layout)
            if select_instance_id is not None:
                matches = [
                    index
                    for index, item in enumerate(runtime)
                    if item["instance_id"] == select_instance_id
                ]
                selected = matches[0] if matches else 0
            else:
                selected = min(selected, max(0, len(runtime) - 1))
            if select_structure_id is not None:
                matches = [
                    index
                    for index, item in enumerate(structure_runtime)
                    if item["id"] == select_structure_id
                ]
                structure_selected = matches[0] if matches else 0
            else:
                structure_selected = min(
                    structure_selected, max(0, len(structure_runtime) - 1)
                )
            time.sleep(0.05)
            try:
                _sync_viewer_without_debug(viewer)
            except Exception as error:
                status = f"Reloaded; viewer sync will retry: {error}"
            last_ui_physics_running = False
            status = "Hot-loaded without closing MuJoCo."
            return True

        def remember_state(snapshot: dict | None = None) -> None:
            state = copy.deepcopy(layout if snapshot is None else snapshot)
            state.pop("_needs_save", None)
            undo_stack.append(state)
            del undo_stack[:-100]

        def undo_previous() -> bool:
            nonlocal drawing, object_move, edit_mode, status
            if not undo_stack:
                status = "Nothing to undo yet."
                return False
            previous = undo_stack.pop()
            layout.clear()
            layout.update(previous)
            _ensure_structure_layout(layout)
            drawing = None
            object_move = None
            save_layout(layout_path, layout)
            hot_reload()
            status = "Previous editor state restored."
            return True

        def finish_drawing(end: list[float]) -> bool:
            nonlocal drawing, edit_mode, status
            if drawing is None or drawing.get("first") is None:
                return False
            previous = copy.deepcopy(layout)
            try:
                item = _append_structure_from_points(
                    layout,
                    description,
                    drawing["kind"],
                    drawing["first"],
                    end,
                )
            except ValueError as error:
                status = str(error)
                return False
            remember_state(previous)
            drawing = None
            edit_mode = "structures"
            save_layout(layout_path, layout)
            hot_reload(select_structure_id=item["id"])
            status = f"Added {item['id']} from two mouse endpoints; auto-saved."
            return True

        while viewer.is_running():
            loop_time = time.perf_counter()
            elapsed_wall_time = min(0.1, max(0.0, loop_time - previous_loop_time))
            previous_loop_time = loop_time
            simulator = viewer._get_sim()
            ui_physics_running = bool(simulator is not None and simulator.run)
            if ui_physics_running != last_ui_physics_running:
                physics_override = None
                last_ui_physics_running = ui_physics_running
            physics_running = (
                ui_physics_running
                if physics_override is None
                else physics_override
            )
            if physics_running and (
                drawing is not None
                or object_move is not None
                or any(item.get("draft") for item in layout["objects"])
            ):
                physics_override = False
                physics_running = False
                status = "Confirm/cancel the current draft or endpoint edit before physics playback."
            if physics_running != physics_was_running:
                physics_accumulator = 0.0
                status = (
                    "Physics playing: dynamic assets respond to gravity and collision."
                    if physics_running
                    else "Physics paused; preview motion was not written to the layout."
                )
                physics_was_running = physics_running

            picked = None
            pending: list[int] = []
            with queue_lock:
                while key_queue:
                    pending.append(key_queue.popleft())

            for key in pending:
                if key == KEY_H:
                    help_visible = not help_visible
                    status = "Help expanded." if help_visible else "Help collapsed; press H to show it."
                    continue
                if key == KEY_R:
                    if not physics_running and (
                        drawing is not None
                        or object_move is not None
                        or any(item.get("draft") for item in layout["objects"])
                    ):
                        status = "Confirm/cancel the current preview or endpoint edit first."
                        continue
                    physics_override = not physics_running
                    physics_running = physics_override
                    physics_was_running = physics_running
                    physics_accumulator = 0.0
                    status = (
                        "Physics playing: dynamic assets respond to gravity and collision."
                        if physics_running
                        else "Physics paused; preview motion was not written to the layout."
                    )
                    continue
                if physics_running and key not in (KEY_O, KEY_C, KEY_F2, KEY_S):
                    status = "Pause physics with R or the left Run button before editing."
                    continue
                if key == KEY_U and drawing is None and object_move is None:
                    undo_previous()
                    continue
                if drawing is not None:
                    moved_cursor = False
                    if key == KEY_X:
                        drawing = None
                        edit_mode = "structures"
                        status = "Endpoint picking canceled."
                    elif key == KEY_LEFT:
                        drawing["cursor"][0] -= step
                        moved_cursor = True
                    elif key == KEY_RIGHT:
                        drawing["cursor"][0] += step
                        moved_cursor = True
                    elif key == KEY_UP:
                        drawing["cursor"][1] += step
                        moved_cursor = True
                    elif key == KEY_DOWN:
                        drawing["cursor"][1] -= step
                        moved_cursor = True
                    elif key == KEY_MINUS:
                        step = max(0.01, step / 2.0)
                    elif key == KEY_EQUAL:
                        step = min(2.0, step * 2.0)
                    elif key in (KEY_SPACE, KEY_ENTER):
                        if drawing.get("first") is None:
                            if not drawing.get("cursor_visible", False):
                                status = "Single-click near a wall edge first."
                                continue
                            drawing["first"] = list(drawing["cursor"])
                            drawing["first_locked"] = True
                            drawing["cursor_visible"] = False
                            status = "Start fixed; single-click the end point or move arrows and press Space."
                        elif not drawing.get("first_locked", False):
                            drawing["first_locked"] = True
                            drawing["cursor_visible"] = False
                            status = "Start fixed; single-click the end point."
                        else:
                            if not drawing.get("cursor_visible", False):
                                status = "Single-click the end point or move it with arrows first."
                                continue
                            finish_drawing(list(drawing["cursor"]))
                    if moved_cursor:
                        drawing["cursor"] = [round(value, 4) for value in drawing["cursor"]]
                        drawing["cursor_visible"] = True
                        if drawing.get("first") is not None and not drawing.get("first_locked", False):
                            drawing["first"] = list(drawing["cursor"])
                    continue

                if key == KEY_F2:
                    if object_move is not None:
                        status = "Confirm the object move with Space/Enter or cancel it with X first."
                        continue
                    edit_mode = "structures" if edit_mode == "objects" else "objects"
                    status = f"Switched to {edit_mode} mode."
                    continue
                if key in (KEY_B, KEY_M):
                    if object_move is not None or any(
                        item.get("draft") for item in layout["objects"]
                    ):
                        status = "Confirm or cancel the current object preview first."
                        continue
                    if structure_runtime:
                        cursor = list(
                            structure_runtime[structure_selected % len(structure_runtime)]["end"]
                        )
                    else:
                        cursor = [float(viewer.cam.lookat[0]), float(viewer.cam.lookat[1])]
                    drawing = {
                        "kind": "wall" if key == KEY_B else "door",
                        "cursor": cursor,
                        "first": None,
                        "first_locked": False,
                        "cursor_visible": False,
                        "accept_click_after": time.monotonic() + 0.15,
                    }
                    mouse_poller.arm()
                    edit_mode = "structures"
                    status = (
                        "Click a START candidate, adjust with arrows, then press Space; "
                        "repeat for END."
                    )
                    continue
                if key == KEY_O:
                    _set_overview_camera(viewer.cam)
                    continue
                if key == KEY_S:
                    if object_move is not None:
                        status = "Space/Enter confirms this move; X restores the previous pose."
                        continue
                    save_layout(layout_path, layout)
                    status = f"Saved {time.strftime('%H:%M:%S')}"
                    continue

                if edit_mode == "structures":
                    if not structure_runtime:
                        continue
                    structure_selected %= len(structure_runtime)
                    structure_item = structure_runtime[structure_selected]
                    if key == KEY_C:
                        midpoint = (
                            np.asarray(structure_item["start"], dtype=float)
                            + np.asarray(structure_item["end"], dtype=float)
                        ) / 2.0
                        viewer.cam.lookat[:] = [midpoint[0], midpoint[1], 1.2]
                        viewer.cam.distance = 5.0
                        viewer.cam.elevation = -30.0
                    elif key == KEY_P:
                        if structure_item["kind"] not in (
                            "base_wall", "wall", "base_floor", "auto_floor", "floor"
                        ):
                            status = "Select a wall or floor first; P changes its finish."
                            continue
                        surface_kind = (
                            "floor"
                            if structure_item["kind"] in ("base_floor", "auto_floor", "floor")
                            else "wall"
                        )
                        before = copy.deepcopy(layout)
                        pause_native_simulator()
                        try:
                            finish_changed = surface_finish_dialog(
                                layout,
                                structure_item["id"],
                                surface_kind,
                                assets_dir,
                            )
                        except Exception as error:
                            finish_changed = False
                            status = f"Finish picker error; editor kept open: {error}"
                        if finish_changed:
                            remember_state(before)
                            save_layout(layout_path, layout)
                            hot_reload(select_structure_id=structure_item["id"])
                            status = (
                                f"{surface_kind.title()} finish applied; "
                                "collision and dimensions unchanged."
                            )
                    elif key == KEY_DELETE:
                        remember_state()
                        _remove_structure_item(layout, description, structure_item)
                        save_layout(layout_path, layout)
                        hot_reload()
                        status = (
                            "Floor removed with its collision."
                            if structure_item["kind"] in ("base_floor", "auto_floor", "floor")
                            else "Structure removed; connected generated floors recalculated."
                        )
                    continue

                if key == KEY_A:
                    if object_move is not None or any(
                        item.get("draft") for item in layout["objects"]
                    ):
                        status = "Confirm or cancel the current object preview first."
                        continue
                    status = "MuJoCo stays open while the visual picker is active."
                    _overlay_text(
                        viewer,
                        runtime,
                        selected % len(runtime) if runtime else 0,
                        layout_items,
                        catalog,
                        step,
                        status,
                        show_help=help_visible,
                        physics_running=physics_running,
                    )
                    _sync_viewer_without_debug(viewer)
                    pause_native_simulator()
                    try:
                        picked = graphical_picker(catalog_assets, assets_dir)
                    except Exception as error:
                        picked = None
                        status = f"Asset picker error; editor kept open: {error}"
                    if picked:
                        remember_state()
                        asset, count = picked
                        asset = _register_catalog_asset(catalog_assets, asset)
                        catalog[asset["catalog_id"]] = asset
                        before = len(layout["objects"])
                        append_asset(layout, asset, count)
                        drafts = layout["objects"][before:]
                        for draft in drafts:
                            draft["draft"] = True
                        draft_id = drafts[0]["id"]
                        if hot_reload(draft_id) and runtime:
                            viewer.cam.lookat[:] = data.xpos[runtime[selected]["body_id"]]
                            viewer.cam.distance = 3.5
                            viewer.cam.elevation = -25.0
                            status = (
                                "DRAFT: move/rotate, then Space/Enter to keep "
                                "or X to cancel."
                            )
                    continue
                if key in (KEY_SPACE, KEY_ENTER):
                    drafts = [item for item in layout["objects"] if item.get("draft")]
                    if drafts:
                        remember_state()
                        for draft in drafts:
                            draft.pop("draft", None)
                        save_layout(layout_path, layout)
                        status = f"Confirmed {len(drafts)} asset(s); auto-saved."
                    elif object_move is not None:
                        remember_state(object_move["snapshot"])
                        save_layout(layout_path, layout)
                        object_move = None
                        status = "Object move confirmed and auto-saved."
                    continue
                if key == KEY_X:
                    drafts = [item for item in layout["objects"] if item.get("draft")]
                    if drafts:
                        remember_state()
                        draft_ids = {item["id"] for item in drafts}
                        layout["objects"] = [
                            item
                            for item in layout["objects"]
                            if item["id"] not in draft_ids
                        ]
                        hot_reload()
                        status = "Draft canceled; committed layout restored."
                    elif object_move is not None:
                        moving_id = object_move["instance_id"]
                        original = object_move["snapshot_item"]
                        moving_instance = layout_items[moving_id]
                        moving_instance["pos"] = copy.deepcopy(original["pos"])
                        moving_instance["yaw"] = float(original["yaw"])
                        moving_runtime = next(
                            item for item in runtime if item["instance_id"] == moving_id
                        )
                        if not apply_pose_safely(moving_runtime, moving_instance):
                            hot_reload(moving_id)
                        object_move = None
                        status = "Object move canceled; previous pose restored."
                    continue
                if not runtime:
                    continue

                if object_move is not None and key in (KEY_D, KEY_DELETE):
                    status = "Confirm or cancel the current object move before duplicating or deleting."
                    continue

                selected %= len(runtime)
                runtime_item = runtime[selected]
                instance = layout_items[runtime_item["instance_id"]]
                before_object_edit = copy.deepcopy(layout)
                changed = False
                if key == KEY_LEFT:
                    instance["pos"][0] -= step
                    changed = True
                elif key == KEY_RIGHT:
                    instance["pos"][0] += step
                    changed = True
                elif key == KEY_UP:
                    instance["pos"][1] += step
                    changed = True
                elif key == KEY_DOWN:
                    instance["pos"][1] -= step
                    changed = True
                elif key == KEY_PAGE_UP:
                    instance["pos"][2] += step
                    changed = True
                elif key == KEY_PAGE_DOWN:
                    instance["pos"][2] -= step
                    changed = True
                elif key == KEY_Q:
                    instance["yaw"] = (instance["yaw"] + 5.0) % 360.0
                    changed = True
                elif key == KEY_E:
                    instance["yaw"] = (instance["yaw"] - 5.0) % 360.0
                    changed = True
                elif key == KEY_MINUS:
                    step = max(0.01, step / 2.0)
                elif key == KEY_EQUAL:
                    step = min(2.0, step * 2.0)
                elif key == KEY_C:
                    viewer.cam.lookat[:] = data.xpos[runtime_item["body_id"]]
                    viewer.cam.distance = 3.5
                    viewer.cam.elevation = -25.0
                elif key == KEY_O:
                    _set_overview_camera(viewer.cam)
                elif key == KEY_S:
                    save_layout(layout_path, layout)
                    status = f"Saved {time.strftime('%H:%M:%S')}"
                elif key == KEY_D:
                    remember_state(before_object_edit)
                    duplicate = copy.deepcopy(instance)
                    duplicate["id"] = f"{instance['asset_id'][:32]}_{uuid.uuid4().hex[:8]}"
                    duplicate["pos"][0] += 0.5
                    duplicate["pos"][1] += 0.5
                    duplicate["draft"] = True
                    layout["objects"].append(duplicate)
                    save_layout(layout_path, layout)
                    hot_reload(duplicate["id"])
                    status = "Duplicate is a draft; Space/Enter confirms, X cancels."
                    continue
                elif key == KEY_DELETE:
                    remember_state(before_object_edit)
                    if instance.get("draft"):
                        layout["objects"] = [
                            item
                            for item in layout["objects"]
                            if item["id"] != instance["id"]
                        ]
                    else:
                        instance["active"] = False
                    save_layout(layout_path, layout)
                    hot_reload()
                    status = "Selected asset removed."
                    continue

                if changed:
                    if not instance.get("draft") and object_move is None:
                        object_move = {
                            "instance_id": instance["id"],
                            "snapshot": before_object_edit,
                            "snapshot_item": copy.deepcopy(
                                next(
                                    item
                                    for item in before_object_edit["objects"]
                                    if item["id"] == instance["id"]
                                )
                            ),
                        }
                    if not apply_pose_safely(runtime_item, instance):
                        original = next(
                            item for item in before_object_edit["objects"]
                            if item["id"] == instance["id"]
                        )
                        instance["pos"] = copy.deepcopy(original["pos"])
                        instance["yaw"] = float(original["yaw"])
                        apply_pose_safely(runtime_item, instance)
                        object_move = None
                        continue
                    status = (
                        "Draft adjusted; Space/Enter confirms, X cancels."
                        if instance.get("draft")
                        else "Move preview; Space/Enter confirms and saves, X cancels."
                    )

            direct_drawing_click = False
            if drawing is not None:
                click = mouse_poller.poll()
                click_ready = time.monotonic() >= drawing.get("accept_click_after", 0.0)
                picked = (
                    _pick_viewer_point(viewer, model, data, click)
                    if click and click_ready
                    else None
                )
                direct_drawing_click = picked is not None
            else:
                picked = _consume_mouse_pick(viewer, model, data)
            if picked is not None:
                if not direct_drawing_click:
                    _restore_camera(viewer.cam, stable_camera)
                clicked_body, clicked_point = picked
                if drawing is not None:
                    wall_segments = _available_wall_segments(layout, description)
                    projection, distance, segment = _nearest_wall_projection(
                        clicked_point[:2].tolist(), wall_segments
                    )
                    if (
                        drawing["kind"] == "door"
                        and drawing.get("first") is not None
                        and drawing.get("first_locked", False)
                        and drawing.get("segment_id") is not None
                    ):
                        host_segment = next(
                            (
                                candidate
                                for candidate in wall_segments
                                if candidate["id"] == drawing["segment_id"]
                            ),
                            None,
                        )
                        if host_segment is not None:
                            projection, distance, _ = _project_to_segment(
                                clicked_point[:2],
                                host_segment["start"],
                                host_segment["end"],
                            )
                            segment = host_segment
                    if segment is None:
                        status = "No wall edge is available to snap to."
                    elif (
                        drawing.get("first") is None
                        or not drawing.get("first_locked", False)
                    ):
                        drawing["cursor"] = projection.round(4).tolist()
                        drawing["first"] = list(drawing["cursor"])
                        drawing["first_locked"] = False
                        drawing["cursor_visible"] = True
                        drawing["segment_id"] = segment["id"]
                        status = (
                            f"START candidate on {segment['id']} (green). "
                            "Click again to replace it, use arrows to adjust, then press Space."
                        )
                    else:
                        drawing["cursor"] = projection.round(4).tolist()
                        drawing["cursor_visible"] = True
                        status = (
                            f"END candidate on {segment['id']} (orange). "
                            "Click again to replace it, use arrows to adjust, then press Space to build."
                        )
                elif edit_mode == "structures":
                    clicked_index = _item_index_for_clicked_body(
                        model,
                        clicked_body,
                        structure_runtime,
                        "body_ids",
                    )
                    if clicked_index is None:
                        nearest = []
                        for index, item in enumerate(structure_runtime):
                            if item["kind"] in ("auto_floor", "floor"):
                                continue
                            _, distance, _ = _project_to_segment(
                                clicked_point[:2], item["start"], item["end"]
                            )
                            nearest.append((distance, index))
                        if nearest and min(nearest)[0] <= 0.75:
                            clicked_index = min(nearest)[1]
                    if clicked_index is not None:
                        structure_selected = clicked_index
                        status = f"Selected {structure_runtime[clicked_index]['id']} by mouse."
                elif runtime:
                    clicked_index = _item_index_for_clicked_body(
                        model,
                        clicked_body,
                        runtime,
                        "body_id",
                    )
                    if clicked_index is not None:
                        clicked_instance_id = runtime[clicked_index]["instance_id"]
                        if (
                            object_move is not None
                            and clicked_instance_id != object_move["instance_id"]
                        ):
                            status = "Confirm or cancel the current object move before selecting another object."
                        else:
                            selected = clicked_index
                            status = "Object selected by mouse."
                    elif not physics_running and float(clicked_point[2]) <= 0.45:
                        selected %= len(runtime)
                        runtime_item = runtime[selected]
                        instance = layout_items[runtime_item["instance_id"]]
                        if not instance.get("draft") and object_move is None:
                            snapshot = copy.deepcopy(layout)
                            object_move = {
                                "instance_id": instance["id"],
                                "snapshot": snapshot,
                                "snapshot_item": copy.deepcopy(
                                    next(
                                        item
                                        for item in snapshot["objects"]
                                        if item["id"] == instance["id"]
                                    )
                                ),
                            }
                        instance["pos"][0] = round(float(clicked_point[0]), 4)
                        instance["pos"][1] = round(float(clicked_point[1]), 4)
                        instance["pos"][2] = runtime_item["floor_z"]
                        if not apply_pose_safely(runtime_item, instance):
                            original = object_move["snapshot_item"]
                            instance["pos"] = copy.deepcopy(original["pos"])
                            instance["yaw"] = float(original["yaw"])
                            apply_pose_safely(runtime_item, instance)
                            object_move = None
                            continue
                        status = (
                            "Object move preview at clicked floor position; arrows fine-tune, "
                            "Space/Enter confirms, X cancels."
                        )

            simulator = viewer._get_sim()
            ui_physics_running = bool(simulator is not None and simulator.run)
            physics_running = (
                ui_physics_running
                if physics_override is None
                else physics_override
            )
            if physics_running:
                physics_accumulator += elapsed_wall_time
                physics_timestep = max(float(model.opt.timestep), 1e-6)
                step_count = min(50, int(physics_accumulator / physics_timestep))
                if step_count:
                    try:
                        with viewer.lock():
                            step_physics_preview(
                                model,
                                data,
                                runtime,
                                layout_items,
                                catalog,
                                step_count,
                            )
                        physics_accumulator -= step_count * physics_timestep
                    except Exception as error:
                        pause_native_simulator()
                        status = f"Physics paused after recoverable error: {error}"
            else:
                physics_accumulator = 0.0

            if edit_mode == "structures" and structure_runtime:
                selected_item = structure_runtime[structure_selected % len(structure_runtime)]
            else:
                selected_item = runtime[selected % len(runtime)] if runtime else None
            with viewer.lock():
                _hide_viewer_debug_geometry(viewer)
                _highlight_selected(viewer, model, data, selected_item)
                _draw_endpoint_cursor(viewer, drawing)
            _overlay_text(
                viewer,
                runtime,
                selected % len(runtime) if runtime else 0,
                layout_items,
                catalog,
                step,
                status,
                structures=structure_runtime,
                structure_selected=structure_selected,
                edit_mode=edit_mode,
                drawing=drawing,
                object_move=object_move,
                show_help=help_visible,
                physics_running=physics_running,
            )
            _sync_viewer_without_debug(viewer)
            if picked is None:
                stable_camera = _camera_state(viewer.cam)
            time.sleep(1.0 / 60.0)
        mouse_poller.close()
    for _ in range(100):
        if not viewer.is_running():
            break
        time.sleep(0.01)


def print_matches(assets: list[dict], query: str, limit: int) -> None:
    matches = search_assets(assets, query, limit=limit, importable_only=False)
    for asset in matches:
        state = (
            "ready"
            if asset.get("importable")
            else "download on preview"
            if asset.get("downloadable")
            else "conversion required"
        )
        print(
            f"{asset['catalog_id']:<58} | {asset['display_name']:<38} | {state}"
        )
    print(f"Matches shown: {len(matches)}")


def _parse_add(value: str) -> tuple[str, int]:
    identifier, separator, count_text = value.rpartition(":")
    if separator and count_text.isdigit():
        return identifier, max(1, int(count_text))
    return value, 1


def main() -> None:
    args = parse_args()
    assets_dir = resolve_assets_dir(args.assets_dir)
    os.environ["MLSPACES_ASSETS_DIR"] = str(assets_dir.resolve())
    catalog_document, catalog_assets = load_catalog(args.catalog.expanduser().resolve())
    relocate_catalog_paths(catalog_assets, assets_dir)
    layout_path = args.layout.expanduser().resolve()
    layout = load_layout(layout_path)
    _portableize_layout_texture_paths(layout, assets_dir)
    _realign_custom_walls(layout, _scene_description(assets_dir))

    if args.search is not None:
        local_matches = search_assets(
            catalog_assets,
            args.search,
            limit=args.limit,
            importable_only=False,
        )
        objathor_matches = search_objathor_assets(args.search, limit=args.limit)
        matches = []
        for index in range(max(len(local_matches), len(objathor_matches))):
            if index < len(local_matches):
                matches.append(local_matches[index])
            if index < len(objathor_matches):
                matches.append(objathor_matches[index])
        print_matches(matches[: args.limit], "", args.limit)
        return
    if args.reset_layout:
        layout = {"version": 2, "base_scene": 861, "objects": []}
        _ensure_structure_layout(layout)

    for value in args.add:
        identifier, count = _parse_add(value)
        try:
            asset = resolve_catalog_asset(catalog_assets, identifier)
        except ValueError:
            asset = resolve_objathor_asset(identifier)
            if asset is None:
                raise
        materialize_catalog_asset(asset, assets_dir)
        asset = _register_catalog_asset(catalog_assets, asset)
        append_asset(layout, asset, count)
    if args.pick:
        picked = graphical_picker(catalog_assets, assets_dir)
        if picked:
            asset, count = picked
            asset = _register_catalog_asset(catalog_assets, asset)
            append_asset(layout, asset, count)
    hydrate_layout_assets(layout, catalog_assets, assets_dir)
    save_layout(layout_path, layout)

    print("Building empty val_861 editor scene...")
    model, data, runtime, structure_runtime, removed = build_editor_model(
        assets_dir, catalog_assets, layout
    )
    if layout.pop("_needs_save", False):
        save_layout(layout_path, layout)
    print(
        f"Ready: removed={removed} original objects, placed={len(runtime)}, "
        f"structures={len(structure_runtime)}, "
        f"nbody={model.nbody}, ngeom={model.ngeom}"
    )
    print(f"Catalog: {len(catalog_assets)} assets ({catalog_document['summary']})")
    print(f"Layout: {layout_path}")
    if args.check_only:
        return

    run_viewer(
        model,
        data,
        runtime,
        structure_runtime,
        catalog_assets,
        layout,
        layout_path,
        assets_dir,
    )


if __name__ == "__main__":
    raise SystemExit("layout_loader is a runtime module; use the HAIC ROS launch entry point.")
