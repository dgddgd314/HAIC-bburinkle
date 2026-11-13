from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import mujoco
import numpy as np

__all__ = [
    "RENDERER_NAME",
    "IS_FILAMENT",
    "DEFAULT_FILAMENT_OPTIONS",
    "FILAMENT_OPTION_KEYS",
    "apply_filament_options",
    "resolve_renderer",
    "OffscreenRenderer",
    "Presenter",
    "GlPresenter",
    "CvPresenter",
    "make_presenter",
    "DEFAULT_WINDOW",
    "run_interactive",
    "FilamentPassiveViewer",
    "launch_passive",
]


RENDERER_NAME: str = getattr(mujoco, "mjRENDERER", "classic")
IS_FILAMENT: bool = RENDERER_NAME == "filament"


def resolve_renderer(requested: str = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "filament" if IS_FILAMENT else "classic"
    if requested not in ("filament", "classic"):
        raise ValueError(f"renderer must be 'auto', 'filament' or 'classic'; got {requested!r}")
    if requested == "filament" and not IS_FILAMENT:
        raise RuntimeError(
            "renderer='filament' but the installed mujoco wheel is the classic build "
            f"(mujoco {mujoco.__version__}, mjRENDERER={RENDERER_NAME!r}).\n"
            "Install the Filament wheel shipped with this repo:\n"
            "    pip install bin/wheels/mujoco-3.7.1-cp311-cp311-"
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        )
    if requested == "classic" and IS_FILAMENT:
        raise RuntimeError(
            "renderer='classic' but the installed mujoco wheel is the Filament fork "
            f"(mujoco {mujoco.__version__}, mjRENDERER='filament'); its mjr_* API *is* "
            "Filament, so the classic renderer is not reachable in this process.\n"
            "Renderer choice is a wheel choice in this fork. To compare, swap wheels:\n"
            "    pip install --force-reinstall 'mujoco~=3.5.0'   # classic (stock PyPI)\n"
            "    pip install --force-reinstall bin/wheels/mujoco-3.7.1-cp311-cp311-"
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl   # back to Filament\n"
            "See tools/compare_renderers.sh, which runs both side by side."
        )
    return requested


FILAMENT_OPTION_KEYS: Tuple[str, ...] = (
    "filament.ao.bent_normals",
    "filament.ao.enabled",
    "filament.ao.ssct",
    "filament.clearColor",
    "filament.fallback.environment_light_intensity",
    "filament.fallback.head_light_intensity",
    "filament.fallback.scene_light_intensity",
    "filament.fog.color",
    "filament.fog.cutOffDistance",
    "filament.fog.density",
    "filament.fog.distance",
    "filament.fog.enabled",
    "filament.fog.height",
    "filament.fog.heightFalloff",
    "filament.fog.inScatteringSize",
    "filament.fog.inScatteringStart",
    "filament.fog.maximumOpacity",
    "filament.msaa.enabled",
    "filament.out.contrast",
    "filament.out.exposure",
    "filament.out.saturation",
    "filament.out.temperature",
    "filament.out.tint",
    "filament.out.tone_mapping",
    "filament.out.vibrance",
    "filament.phong.emissive_multiplier",
    "filament.phong.shininess_multiplier",
    "filament.phong.specular_multiplier",
    "filament.shadows.map_size",
    "filament.shadows.type",
    "filament.shadows.vsm_blur_width",
)

DEFAULT_FILAMENT_OPTIONS: Dict[str, float] = {
    "filament.fallback.environment_light_intensity": 15000.0,
}


def apply_filament_options(
    spec: "mujoco.MjSpec", options: Optional[Dict[str, Any]] = None
) -> None:
    if options is None:
        options = DEFAULT_FILAMENT_OPTIONS
    if not options:
        return

    existing = {n.name for n in spec.numerics}
    for key, value in options.items():
        if key in existing:
            spec.delete(spec.numeric(key))
        values = [float(v) for v in (value if isinstance(value, Iterable) else [value])]
        spec.add_numeric(key, values, len(values), f"filament option {key}")


def _load_engine_cls() -> Optional[type]:
    try:
        from molmo_spaces.renderer.filament_rendering import MjFilamentRenderer
    except Exception:
        return None
    return MjFilamentRenderer


class OffscreenRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 1280,
        height: int = 720,
        max_geom: int = 30000,
        font_scale: int = mujoco.mjtFontScale.mjFONTSCALE_150.value,
        owns_gl_context: bool = True,
        shadows: bool = True,
        supersample: float = 1.0,
    ) -> None:
        self.width, self.height = int(width), int(height)
        # Filament MSAA is inert in this build; supersample and downscale instead.
        self.supersample = max(1.0, float(supersample))
        self.max_geom = int(max_geom)
        self.font_scale = int(font_scale)
        self.shadows = bool(shadows)

        # Classic rendering requires a current GL context; Filament does not.
        self._gl: Optional[Any] = None
        if not IS_FILAMENT and owns_gl_context:
            self._gl = mujoco.GLContext(self.width, self.height)
            self._gl.make_current()

        self._engine_cls = _load_engine_cls()
        self.model: Optional[mujoco.MjModel] = None
        self._engine: Optional[Any] = None
        self._ctx: Optional[mujoco.MjrContext] = None
        self.scene: Optional[mujoco.MjvScene] = None
        self.scene_option = mujoco.MjvOption()
        self._rendered_once = False
        self._buf: Optional[np.ndarray] = None

        self.rebind(model)

    @property
    def render_size(self) -> Tuple[int, int]:
        if self.supersample == 1.0:
            return self.width, self.height
        return int(self.width * self.supersample), int(self.height * self.supersample)

    @property
    def engine_name(self) -> str:
        return "MjFilamentRenderer" if self._engine is not None else "mjr_* (builtin)"


    def rebind(self, model: mujoco.MjModel) -> None:
        self._free_context()
        self.model = model

        if self._engine_cls is not None:
            rw, rh = self.render_size
            self._engine = self._engine_cls(
                model=model, width=rw, height=rh, max_geom=self.max_geom
            )
            self._ctx = self._engine._mjr_context
            self.scene = self._engine.scene
        else:
            self._engine = None
            self._ctx = mujoco.MjrContext(model, self.font_scale)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._ctx)
            self.scene = mujoco.MjvScene(model=model, maxgeom=self.max_geom)

        # Both backends use reverse-Z: far=0, near=1.
        self._ctx.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR

        self._rendered_once = False
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.shadows
        if not IS_FILAMENT:
            mujoco.mjr_resizeOffscreen(*self.render_size, self._ctx)

    def resize(self, width: int, height: int) -> None:
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = width, height
        self._buf = None
        rw, rh = self.render_size
        if self._engine is not None:
            self._engine._width, self._engine._height = rw, rh
        if not IS_FILAMENT and self._ctx is not None:
            mujoco.mjr_resizeOffscreen(rw, rh, self._ctx)

    def _free_context(self) -> None:
        if self._ctx is None:
            return
        # Freeing an unrendered Filament context segfaults in this build.
        if self._rendered_once or not IS_FILAMENT:
            if self._engine is not None:
                self._engine.close()
            else:
                self._ctx.free()
        self._engine = None
        self._ctx = None

    def close(self) -> None:
        self._free_context()
        self.scene = None
        if self._gl is not None:
            try:
                self._gl.free()
            except Exception:
                pass
            self._gl = None

    def __enter__(self) -> "OffscreenRenderer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


    def make_camera(self, camera: Any = -1) -> mujoco.MjvCamera:
        if isinstance(camera, mujoco.MjvCamera):
            return camera
        cam_id = camera
        if isinstance(cam_id, str):
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA.value, cam_id)
            if cam_id == -1:
                raise ValueError(f'camera "{camera}" does not exist in this model')
        cam_id = int(cam_id)
        if cam_id < -1 or cam_id >= self.model.ncam:
            raise ValueError(f"camera id {cam_id} out of range [-1, {self.model.ncam})")
        cam = mujoco.MjvCamera()
        if cam_id == -1:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            mujoco.mjv_defaultFreeCamera(self.model, cam)
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cam_id
        return cam

    def update(
        self,
        data: mujoco.MjData,
        camera: Any = -1,
        scene_option: Optional[mujoco.MjvOption] = None,
    ) -> None:
        mujoco.mjv_updateScene(
            self.model,
            data,
            scene_option or self.scene_option,
            None,
            self.make_camera(camera),
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

    def render(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if width is not None or height is not None:
            self.resize(width or self.width, height or self.height)
        rw, rh = self.render_size
        if self._buf is None or self._buf.shape != (rh, rw, 3):
            self._buf = np.empty((rh, rw, 3), np.uint8)

        rect = mujoco.MjrRect(0, 0, rw, rh)
        mujoco.mjr_render(rect, self.scene, self._ctx)
        mujoco.mjr_readPixels(rgb=self._buf, depth=None, viewport=rect, con=self._ctx)
        self._rendered_once = True

        img = self._buf if IS_FILAMENT else self._buf[::-1]
        if self.supersample != 1.0:
            import cv2

            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        h, w = self.height, self.width
        if out is None:
            return img.copy()
        if out.shape != (h, w, 3):
            raise ValueError(f"expected out.shape == {(h, w, 3)}, got {out.shape}")
        out[:] = img
        return out

    def render_rgb(
        self,
        data: mujoco.MjData,
        width: Optional[int] = None,
        height: Optional[int] = None,
        camera: Any = -1,
        scene_option: Optional[mujoco.MjvOption] = None,
    ) -> np.ndarray:
        if width is not None or height is not None:
            self.resize(width or self.width, height or self.height)
        self.update(data, camera=camera, scene_option=scene_option)
        return self.render()

    def render_depth(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return (H, W) float32 depth in metres."""
        if width is not None or height is not None:
            self.resize(width or self.width, height or self.height)
        rw, rh = self.render_size
        buf = np.empty((rh, rw), np.float32)

        flags = self.scene.flags.copy()
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        self.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        try:
            rect = mujoco.MjrRect(0, 0, rw, rh)
            mujoco.mjr_render(rect, self.scene, self._ctx)
            mujoco.mjr_readPixels(rgb=None, depth=buf, viewport=rect, con=self._ctx)
            self._rendered_once = True
        finally:
            np.copyto(self.scene.flags, flags)

        depth = self._depth_to_metres(buf)
        if not IS_FILAMENT:
            depth = depth[::-1]
        if self.supersample != 1.0:
            import cv2

            depth = cv2.resize(depth, (self.width, self.height),
                               interpolation=cv2.INTER_NEAREST)
        h, w = self.height, self.width
        if out is None:
            return np.ascontiguousarray(depth)
        if out.shape != (h, w):
            raise ValueError(f"expected out.shape == {(h, w)}, got {out.shape}")
        out[:] = depth
        return out

    def _depth_to_metres(self, buf: np.ndarray) -> np.ndarray:
        extent = self.model.stat.extent
        znear = np.float32(self.model.vis.map.znear * extent)
        zfar = np.float32(self.model.vis.map.zfar * extent)

        c = -(zfar + znear) / (zfar - znear)
        d = -(np.float32(2) * zfar * znear) / (zfar - znear)
        c = np.float32(-0.5) * c - np.float32(0.5)
        d = np.float32(-0.5) * d

        return (d / (buf.astype(np.float64) + c)).astype(np.float32)

    def render_rgbd(
        self,
        data: mujoco.MjData,
        width: Optional[int] = None,
        height: Optional[int] = None,
        camera: Any = -1,
        scene_option: Optional[mujoco.MjvOption] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if width is not None or height is not None:
            self.resize(width or self.width, height or self.height)
        self.update(data, camera=camera, scene_option=scene_option)
        return self.render(), self.render_depth()


class _BasePresenter:
    ROT_STEP = 5.0
    ZOOM_STEP = 1.1
    PAN_STEP = 0.25

    def __init__(self, width: int, height: int, title: str) -> None:
        self.width, self.height = int(width), int(height)
        self.title = title
        self._closed = False
        self._drag = [0.0, 0.0, 0]
        self._scroll = 0.0
        self._last_xy: Optional[Tuple[float, float]] = None
        self._press_xy: Optional[Tuple[float, float]] = None
        self._on_click: Optional[Any] = None


    def request_close(self) -> None:
        self._closed = True

    @property
    def image_size(self) -> Tuple[int, int]:
        return self.width, self.height


    def apply_key(self, camera: mujoco.MjvCamera, key: int) -> bool:
        if key == ord("A"):
            camera.azimuth -= self.ROT_STEP
        elif key == ord("D"):
            camera.azimuth += self.ROT_STEP
        elif key == ord("W"):
            camera.elevation = min(camera.elevation + self.ROT_STEP, 85.0)
        elif key == ord("S"):
            camera.elevation = max(camera.elevation - self.ROT_STEP, -85.0)
        elif key == ord("Z"):
            camera.distance = max(0.25, camera.distance / self.ZOOM_STEP)
        elif key == ord("X"):
            camera.distance *= self.ZOOM_STEP
        elif key == ord("J"):
            camera.lookat[0] -= self.PAN_STEP
        elif key == ord("L"):
            camera.lookat[0] += self.PAN_STEP
        elif key == ord("I"):
            camera.lookat[1] += self.PAN_STEP
        elif key == ord("K"):
            camera.lookat[1] -= self.PAN_STEP
        elif key == ord("U"):
            camera.lookat[2] += self.PAN_STEP / 2.5
        elif key == ord("O"):
            camera.lookat[2] -= self.PAN_STEP / 2.5
        else:
            return False
        return True

    def apply_mouse(
        self, model: mujoco.MjModel, scene: mujoco.MjvScene, camera: mujoco.MjvCamera
    ) -> None:
        if self._scroll:
            mujoco.mjv_moveCamera(
                model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * self._scroll, scene, camera
            )
            self._scroll = 0.0
        dx, dy, button = self._drag
        if button:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_V if button == 2
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
            h = max(self.height, 1)
            mujoco.mjv_moveCamera(model, action, dx / h, dy / h, scene, camera)
            self._drag[:] = [0.0, 0.0, 0]

    def _note_drag(self, x: float, y: float, pan: bool) -> None:
        if self._last_xy is not None:
            self._drag[0] += x - self._last_xy[0]
            self._drag[1] += y - self._last_xy[1]
            self._drag[2] = 2 if pan else 1
        self._last_xy = (x, y)


    def draw_overlay(self, rgb: np.ndarray, lines: Sequence[str]) -> None:
        if not lines:
            return
        import cv2

        scale, thick, pad, lh = 0.4, 1, 8, 15
        width = max(
            cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for l in lines
        )
        box = rgb[: min(len(lines) * lh + 2 * pad, rgb.shape[0]),
                  : min(width + 2 * pad, rgb.shape[1])]
        box[:] = (box.astype(np.uint16) * 3 // 10).astype(np.uint8)
        for i, line in enumerate(lines):
            y = pad + lh * (i + 1) - 4
            if y >= rgb.shape[0]:
                break
            cv2.putText(rgb, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (245, 245, 245), thick, cv2.LINE_AA)


class GlPresenter(_BasePresenter):
    def __init__(self, width: int = 1280, height: int = 720, title: str = "MuJoCo") -> None:
        super().__init__(width, height, title)
        try:
            import glfw
            from OpenGL import GL
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "window='glfw' needs glfw and PyOpenGL (both are mujoco wheel "
                "dependencies); pass window='opencv' to use the OpenCV window"
            ) from exc

        self._glfw, self._GL = glfw, GL
        if not glfw.init():
            raise RuntimeError(
                "glfw.init() failed -- no display?  "
                f"DISPLAY={os.environ.get('DISPLAY', '')!r}"
            )
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("could not create a GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # Vsync would double-pace the loop.

        self._tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        for k in (GL.GL_TEXTURE_MIN_FILTER, GL.GL_TEXTURE_MAG_FILTER):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, k, GL.GL_LINEAR)
        for k in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, k, GL.GL_CLAMP_TO_EDGE)
        self._tex_shape: Optional[Tuple[int, int]] = None

        self._keys: list = []
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_scroll_callback(self.window, self._on_scroll)


    def _on_key(self, _w, key, _sc, action, _mods) -> None:
        if action == self._glfw.PRESS:
            self._keys.append(key)

    def _on_scroll(self, _w, _dx, dy) -> None:
        self._scroll += dy


    @property
    def image_size(self) -> Tuple[int, int]:
        w, h = self._glfw.get_framebuffer_size(self.window)
        return (int(w), int(h)) if w > 0 and h > 0 else (self.width, self.height)

    def should_close(self) -> bool:
        return self._closed or bool(self._glfw.window_should_close(self.window))

    def set_title(self, text: str) -> None:
        self._glfw.set_window_title(self.window, text)

    def poll(self, delay_ms: int = 1) -> int:
        glfw = self._glfw
        glfw.poll_events()
        self._poll_mouse()
        return self._keys.pop(0) if self._keys else -1

    def _poll_mouse(self) -> None:
        glfw = self._glfw
        x, y = glfw.get_cursor_pos(self.window)
        left = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        shift = (
            glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if not (left or right):
            if self._press_xy is not None and self._on_click is not None:
                px, py = self._press_xy
                if abs(x - px) <= 3 and abs(y - py) <= 3:
                    self._on_click(int(x), int(y))
            self._press_xy = None
            self._last_xy = None
            return
        if self._press_xy is None:
            self._press_xy = (x, y)
        self._note_drag(x, y, pan=bool(right or (left and shift)))


    def present(self, rgb: np.ndarray) -> None:
        GL = self._GL
        h, w = rgb.shape[:2]
        fb_w, fb_h = self.image_size
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if self._tex_shape != (h, w):
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0,
                            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)
            self._tex_shape = (h, w)
        else:
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                               GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)
        GL.glViewport(0, 0, fb_w, fb_h)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glColor3f(1.0, 1.0, 1.0)
        GL.glMatrixMode(GL.GL_PROJECTION); GL.glLoadIdentity()
        GL.glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
        GL.glMatrixMode(GL.GL_MODELVIEW); GL.glLoadIdentity()
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(0.0, 0.0)
        GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f(1.0, 0.0)
        GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f(1.0, 1.0)
        GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(0.0, 1.0)
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)
        self._glfw.swap_buffers(self.window)

    def close(self) -> None:
        self._closed = True
        try:
            if getattr(self, "window", None) is not None:
                self._glfw.destroy_window(self.window)
                self.window = None
            self._glfw.terminate()
        except Exception:
            pass


class CvPresenter(_BasePresenter):
    def __init__(self, width: int = 1280, height: int = 720, title: str = "MuJoCo") -> None:
        super().__init__(width, height, title)
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "window='opencv' needs OpenCV (`pip install opencv-python`); "
                "use window='glfw', or render_rgb()/--save-rgb for headless work"
            ) from exc
        self._cv2 = cv2
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, self.width, self.height)
        cv2.setMouseCallback(title, self._on_mouse)


    def _on_mouse(self, event, x, y, flags, _param) -> None:
        cv2 = self._cv2
        if event == cv2.EVENT_MOUSEWHEEL:
            self._scroll += 1.0 if flags > 0 else -1.0
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._press_xy = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            press, self._press_xy = self._press_xy, None
            if (press is not None and self._on_click is not None
                    and abs(x - press[0]) <= 3 and abs(y - press[1]) <= 3):
                self._on_click(x, y)
        left = bool(flags & cv2.EVENT_FLAG_LBUTTON)
        right = bool(flags & cv2.EVENT_FLAG_RBUTTON)
        if not (left or right):
            self._last_xy = None
            return
        self._note_drag(x, y, pan=bool(right or (flags & cv2.EVENT_FLAG_SHIFTKEY)))


    @property
    def image_size(self) -> Tuple[int, int]:
        try:
            _x, _y, w, h = self._cv2.getWindowImageRect(self.title)
        except Exception:
            return self.width, self.height
        return (int(w), int(h)) if w > 0 and h > 0 else (self.width, self.height)

    def should_close(self) -> bool:
        if self._closed:
            return True
        try:
            return self._cv2.getWindowProperty(self.title, self._cv2.WND_PROP_VISIBLE) < 1
        except self._cv2.error:
            return True

    def set_title(self, text: str) -> None:
        try:
            self._cv2.setWindowTitle(self.title, text)
        except self._cv2.error:
            pass

    def poll(self, delay_ms: int = 1) -> int:
        key = self._cv2.waitKeyEx(max(1, delay_ms))
        return -1 if key == -1 else _to_glfw_key(key)


    def present(self, rgb: np.ndarray) -> None:
        self._cv2.imshow(self.title, self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._closed = True
        try:
            self._cv2.destroyWindow(self.title)
            self._cv2.waitKey(1)
        except Exception:
            pass


DEFAULT_WINDOW: str = os.environ.get("MJ_WINDOW", "glfw").lower()


def make_presenter(
    window: str = "auto", width: int = 1280, height: int = 720, title: str = "MuJoCo"
) -> _BasePresenter:
    window = (window or "auto").lower()
    if window == "auto":
        window = DEFAULT_WINDOW
    if window == "glfw":
        return GlPresenter(width, height, title)
    if window in ("opencv", "cv2"):
        return CvPresenter(width, height, title)
    raise ValueError(f"window must be 'auto', 'glfw' or 'opencv'; got {window!r}")


Presenter = CvPresenter


_HELP = """\
  keys : A/D rotate | W/S tilt | Z/X zoom
         J/L move x | I/K move y | U/O move z | R reset camera
         SPACE pause | TAB / [ ] cycle cameras | BACKSPACE reset state
         P screenshot | Q or ESC quit
  mouse: left-drag orbit | right-drag (or shift+left) pan | wheel zoom"""


def run_interactive(
    sim: Any,
    renderer: str = "auto",
    width: int = 1280,
    height: int = 720,
    realtime: bool = True,
    speed: float = 1.0,
    max_substeps: int = 64,
    max_fps: float = 120.0,
    duration: Optional[float] = None,
    camera: Any = -1,
    max_geom: int = 30000,
    shadows: bool = True,
    paused: bool = False,
    hard_exit: bool = False,
    window: str = "auto",
    supersample: float = 1.0,
    screenshot_dir: str = ".",
    title: Optional[str] = None,
) -> None:
    kind = resolve_renderer(renderer)

    # OpenCV owns no GL context; create one for classic offscreen rendering.
    offscreen = OffscreenRenderer(
        sim.model,
        width=width,
        height=height,
        max_geom=max_geom,
        shadows=shadows,
        supersample=supersample,
    )
    presenter = make_presenter(window, width, height, title or f"MuJoCo [{kind}]")
    cam = offscreen.make_camera(camera)
    cam_id = camera if isinstance(camera, int) else -1
    initial_cam = (cam.lookat.copy(), cam.distance, cam.azimuth, cam.elevation)

    generation = getattr(sim, "generation", 0)
    t_end = None if duration is None else time.time() + duration
    frames = 0
    sim_debt = 0.0
    sim_t0 = sim.data.time
    prev_wall = time.time()
    fps_t0, fps_n = time.time(), 0
    print(f"[mj_renderer] renderer={kind} backend={'vulkan' if kind == 'filament' else 'opengl'} "
          f"engine={offscreen.engine_name} window={type(presenter).__name__} "
          f"{width}x{height}"
          + (f" ss={supersample:g}x" if supersample != 1.0 else "")
          + f"\n{_HELP}", flush=True)

    try:
        while not presenter.should_close():
            if t_end is not None and time.time() > t_end:
                break

            gen = getattr(sim, "generation", 0)
            if gen != generation:
                generation = gen
                offscreen.rebind(sim.model)
                cam_id = min(cam_id, sim.model.ncam - 1)

            win_w, win_h = presenter.image_size
            offscreen.resize(win_w, win_h)
            presenter.width, presenter.height = win_w, win_h

            key = presenter.poll(1)
            if key != -1 and not presenter.apply_key(cam, key):
                if key in (_GLFW_ESCAPE, ord("Q")):
                    presenter.request_close()
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("R"):
                    lookat, distance, azimuth, elevation = initial_cam
                    cam.lookat[:] = lookat
                    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
                elif key in (_GLFW_TAB, ord("]")):
                    cam_id = -1 if cam_id + 1 >= sim.model.ncam else cam_id + 1
                    cam = offscreen.make_camera(cam_id)
                elif key == ord("["):
                    cam_id = sim.model.ncam - 1 if cam_id - 1 < -1 else cam_id - 1
                    cam = offscreen.make_camera(cam_id)
                elif key == _GLFW_BACKSPACE and hasattr(sim, "reset_state"):
                    sim.reset_state()
                elif key == ord("P"):
                    path = os.path.join(screenshot_dir, f"mj_{kind}_{frames:05d}.png")
                    _save_png(offscreen.render(), path)
                    print(f"[mj_renderer] wrote {path}", flush=True)

            presenter.apply_mouse(sim.model, offscreen.scene, cam)

            t0 = time.time()
            if paused:
                sim_debt = 0.0
            else:
                dt = sim.model.opt.timestep
                if realtime:
                    sim_debt += (t0 - prev_wall) * speed
                    n = int(sim_debt / dt)
                    if n > max_substeps:
                        n, sim_debt = max_substeps, 0.0
                    else:
                        sim_debt -= n * dt
                else:
                    n = max_substeps
                if n:
                    lock = getattr(sim, "_lock", None)
                    if lock is not None:
                        with lock:
                            if getattr(sim, "generation", 0) != generation:
                                prev_wall = t0
                                continue
                            sim.step(n)
                    else:
                        sim.step(n)
            prev_wall = t0

            offscreen.update(sim.data, camera=cam)
            presenter.present(offscreen.render())
            frames += 1

            fps_n += 1
            if time.time() - fps_t0 > 1.0:
                wall = time.time() - fps_t0
                presenter.set_title(
                    f"MuJoCo [{kind}] {fps_n / wall:.0f} fps  "
                    f"x{(sim.data.time - sim_t0) / wall:.2f} realtime  "
                    f"t={sim.data.time:.2f}s{'  PAUSED' if paused else ''}"
                )
                fps_t0, fps_n, sim_t0 = time.time(), 0, sim.data.time

            if realtime and max_fps > 0:
                lag = 1.0 / max_fps - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
    finally:
        offscreen.close()
        presenter.close()
        sys.stdout.flush()
        sys.stderr.flush()
        if hard_exit:
            # Skip the racy Filament engine teardown at interpreter exit.
            os._exit(0)


def _save_png(rgb: np.ndarray, path: str) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(rgb).save(path)


# Translate OpenCV ASCII/X11 keysyms to the GLFW codes expected by viewer callbacks.
_GLFW_ESCAPE, _GLFW_TAB, _GLFW_BACKSPACE = 256, 258, 259

_GLFW_KEY = {
    13: 257, 10: 257,
    9: 258,
    8: 259, 65288: 259,
    65535: 261, 255: 261,
    65363: 262, 65361: 263,
    65364: 264, 65362: 265,
    65365: 266, 65366: 267,
    65360: 268, 65367: 269,
    27: 256,
    65470: 290, 65471: 291, 65472: 292, 65473: 293,
    65474: 294, 65475: 295, 65476: 296, 65477: 297,
}


def _to_glfw_key(key: int) -> int:
    if key in _GLFW_KEY:
        return _GLFW_KEY[key]
    if 97 <= key <= 122:
        return key - 32
    return key & 0xFFFF if key > 0xFFFF else key


class _SimHandle:
    def __init__(self, viewer: "FilamentPassiveViewer") -> None:
        self._viewer = viewer
        self.run = False

    def load(self, model: mujoco.MjModel, data: mujoco.MjData, name: str = "") -> None:
        self._viewer.load(model, data)


class FilamentPassiveViewer:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        key_callback: Optional[Any] = None,
        width: int = 1280,
        height: int = 720,
        title: str = "MuJoCo [filament]",
        max_geom: int = 30000,
        show_left_ui: bool = False,
        show_right_ui: bool = False,
        window: str = "auto",
        supersample: float = 1.0,
        **_ignored: Any,
    ) -> None:
        self.model, self.data = model, data
        self._key_callback = key_callback
        self._off = OffscreenRenderer(
            model, width, height, max_geom=max_geom, supersample=supersample
        )
        self._presenter = make_presenter(window, width, height, title)
        self._debug_keys = bool(os.environ.get("MJ_RENDERER_DEBUG_KEYS"))

        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.opt = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.user_scn = mujoco.MjvScene(model=model, maxgeom=max_geom)
        self.user_scn.ngeom = 0
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        self._texts: list = []
        self._sim = _SimHandle(self)
        self._closed = False
        self._click: Optional[Tuple[int, int]] = None
        self._presenter._on_click = self._note_click


    def is_running(self) -> bool:
        return not (self._closed or self._presenter.should_close())

    def lock(self):
        from contextlib import nullcontext

        return nullcontext()

    def set_texts(self, texts: Any) -> None:
        self._texts = list(texts or [])

    def _get_sim(self) -> _SimHandle:
        return self._sim

    def load(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model, self.data = model, data
        self._off.rebind(model)
        self.user_scn = mujoco.MjvScene(model=model, maxgeom=self._off.max_geom)
        self.user_scn.ngeom = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._off.close()
        self._presenter.close()

    def __enter__(self) -> "FilamentPassiveViewer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


    def _note_click(self, x: int, y: int) -> None:
        self._click = (x, y)

    def _handle_select(self) -> None:
        if self._click is None:
            return
        x, y = self._click
        self._click = None
        w, h = max(self._off.width, 1), max(self._off.height, 1)
        relx, rely = x / w, 1.0 - y / h
        if not (0.0 <= relx <= 1.0 and 0.0 <= rely <= 1.0):
            return
        selpnt = np.zeros(3)
        geomid = np.zeros(1, np.int32)
        flexid = np.zeros(1, np.int32)
        skinid = np.zeros(1, np.int32)
        body = mujoco.mjv_select(
            self.model, self.data, self.opt, w / h, relx, rely,
            self._off.scene, selpnt, geomid, flexid, skinid,
        )
        if body > 0:
            rot = self.data.xmat[body].reshape(3, 3)
            self.perturb.select = body
            self.perturb.localpos[:] = rot.T @ (selpnt - self.data.xpos[body])
            self.perturb.active = 0


    def sync(self) -> None:
        if not self.is_running():
            return

        win_w, win_h = self._presenter.image_size
        if (win_w, win_h) != (self._off.width, self._off.height):
            self._off.resize(win_w, win_h)
            self._presenter.width, self._presenter.height = win_w, win_h
            self.viewport = mujoco.MjrRect(0, 0, win_w, win_h)

        key = self._presenter.poll(1)
        if key != -1:
            if self._debug_keys:
                print(f"[mj_renderer] glfw key={key}", flush=True)
            if not self._presenter.apply_key(self.cam, key):
                if self._key_callback is not None:
                    self._key_callback(key)
        self._presenter.apply_mouse(self.model, self._off.scene, self.cam)
        self._handle_select()

        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, self.perturb, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value, self._off.scene,
        )
        self._merge_user_scn()
        rgb = self._off.render()
        self._draw_overlay(rgb)
        self._presenter.present(rgb)

    def _merge_user_scn(self) -> None:
        scn = self._off.scene
        for i in range(int(self.user_scn.ngeom)):
            if scn.ngeom >= scn.maxgeom:
                break
            src = self.user_scn.geoms[i]
            dst = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                dst, int(src.type), src.size, src.pos, src.mat.flatten(), src.rgba
            )
            dst.category = src.category
            dst.objtype, dst.objid, dst.dataid = src.objtype, src.objid, src.dataid
            dst.segid = src.segid
            scn.ngeom += 1

    def _draw_overlay(self, rgb: np.ndarray) -> None:
        if not self._texts:
            return
        lines: list = []
        for entry in self._texts:
            for chunk in tuple(entry)[2:]:
                if chunk:
                    lines.extend(str(chunk).splitlines())
        self._presenter.draw_overlay(rgb, lines)


def launch_passive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    key_callback: Optional[Any] = None,
    **kwargs: Any,
) -> FilamentPassiveViewer:
    return FilamentPassiveViewer(model, data, key_callback=key_callback, **kwargs)
