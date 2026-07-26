import asyncio
import base64
from collections.abc import AsyncIterator
from io import BytesIO

import numpy as np
from PIL import Image

from ..config import Settings
from ..schemas import GenerationMode
from .base import FramePacket, GenerationControl, GenerationSpec, InputFrameBuffer, VideoAdapter


class SelfForcingAdapter(VideoAdapter):
    """Client for the isolated official Self-Forcing worker environment.

    The upstream worker uses Socket.IO and has dependency pins that conflict with the
    WebRTC gateway. Keeping this adapter as a network boundary lets the GPU deployment
    run its Python 3.10 environment independently.
    """

    name = "self-forcing-1.3b"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    async def load(self) -> None:
        try:
            import socketio
        except ImportError as error:
            raise RuntimeError("Install the gpu-worker optional dependencies") from error
        self._client = socketio.AsyncClient(reconnection=True, request_timeout=20)
        await self._client.connect(self.settings.self_forcing_url, transports=["websocket"])

    async def generate(
        self,
        spec: GenerationSpec,
        control: GenerationControl,
        inputs: InputFrameBuffer,
    ) -> AsyncIterator[FramePacket]:
        if spec.mode != GenerationMode.TEXT:
            raise RuntimeError("Self-Forcing 1.3B worker supports text mode only")
        if self._client is None:
            raise RuntimeError("Self-Forcing worker is not connected")

        frames: asyncio.Queue[FramePacket | None | Exception] = asyncio.Queue(maxsize=8)

        async def on_frame(payload: dict[str, object]) -> None:
            raw = str(payload["data"])
            encoded = raw.split(",", 1)[-1]
            image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
            packet = FramePacket(np.asarray(image, dtype=np.uint8))
            if frames.full():
                frames.get_nowait()
            frames.put_nowait(packet)

        async def on_complete(_: object) -> None:
            await frames.put(None)

        async def on_error(payload: object) -> None:
            await frames.put(RuntimeError(f"Self-Forcing worker error: {payload}"))

        self._client.on("frame_ready", on_frame)
        self._client.on("generation_complete", on_complete)
        self._client.on("error", on_error)

        while await control.checkpoint():
            prompt, transition, _ = await control.prompt_state()
            await self._client.emit(
                "start_generation",
                {
                    "prompt": prompt,
                    "seed": 42,
                    "enable_torch_compile": True,
                    "enable_fp8": True,
                    "use_taehv": True,
                    "fps": spec.native_fps,
                },
            )
            while await control.checkpoint():
                packet = await frames.get()
                if packet is None:
                    break
                if isinstance(packet, Exception):
                    raise packet
                yield packet
            if transition.value == "restart":
                continue

    def telemetry(self) -> dict[str, float | bool | str]:
        return {
            "backend": self.name,
            "gpu_available": self._client is not None and self._client.connected,
            "vram_used_gb": 0.0,
        }

    async def close(self) -> None:
        if self._client is not None and self._client.connected:
            await self._client.disconnect()
