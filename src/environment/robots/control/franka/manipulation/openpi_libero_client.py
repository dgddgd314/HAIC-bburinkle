from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_VENDOR_DIR = str(Path(__file__).resolve().parent)
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
from openpi_client import image_tools, websocket_client_policy  # noqa: E402


class OpenPILiberoClient:
    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self.client = None
        self.inference_count = 0

    def _connect(self):
        if self.client is None:
            # Disable keepalive timeouts while the first JAX inference compiles kernels.
            self.client = websocket_client_policy.WebsocketClientPolicy(
                host=self.host,
                port=self.port,
                ping_interval=None,
                ping_timeout=None,
            )
            metadata = self.client.get_server_metadata()
            print(f"[OpenPI client] connected to ws://{self.host}:{self.port}; metadata={metadata}")
        return self.client

    def act(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        prompt = str(observation.get("prompt") or observation["task_instruction"])
        # MuJoCo images are already upright; do not apply the robosuite 180-degree flip.
        image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.asarray(observation["image"]), 224, 224)
        )
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.asarray(observation["wrist_image"]), 224, 224
            )
        )
        request = {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": np.asarray(observation["state"], dtype=np.float32),
            "prompt": prompt,
        }
        result = self._connect().infer(request)
        self.inference_count += 1
        if self.inference_count <= 3 or self.inference_count % 10 == 0:
            print(f"[OpenPI client] action chunk #{self.inference_count} received")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise RuntimeError(
                f"pi0.5-LIBERO server returned {actions.shape}, expected (horizon, 7)"
            )
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("pi0.5-LIBERO server returned NaN/Inf actions")
        return {"actions": actions}

    def reset(self) -> None:
        self._connect().reset()


def create_policy() -> OpenPILiberoClient:
    return OpenPILiberoClient(
        host=os.environ.get("OPENPI_HOST", "127.0.0.1"),
        port=int(os.environ.get("OPENPI_PORT", "8000")),
    )
