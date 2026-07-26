import asyncio
import math
from collections.abc import AsyncIterator
from pathlib import Path
from time import monotonic

import av
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from ..schemas import GenerationMode, Resolution, TransitionMode
from .base import FramePacket, GenerationControl, GenerationSpec, InputFrameBuffer, VideoAdapter


class MockVideoAdapter(VideoAdapter):
    name = "mock"

    def __init__(self) -> None:
        self._loaded = False

    async def load(self) -> None:
        await asyncio.sleep(0.15)
        self._loaded = True

    def telemetry(self) -> dict[str, float | bool | str]:
        return {"backend": self.name, "gpu_available": False, "vram_used_gb": 0.0}

    async def generate(
        self,
        spec: GenerationSpec,
        control: GenerationControl,
        inputs: InputFrameBuffer,
    ) -> AsyncIterator[FramePacket]:
        width, height = (1280, 720) if spec.resolution == Resolution.HD_720 else (832, 480)
        source = self._load_source(spec.source_path, width, height)
        frame_index = 0
        prompt, transition, version = await control.prompt_state()
        previous_prompt = prompt
        transition_progress = 1.0
        frame_interval = 1 / spec.native_fps

        while await control.checkpoint():
            started = monotonic()
            new_prompt, new_transition, new_version = await control.prompt_state()
            if new_version != version:
                previous_prompt = prompt
                prompt = new_prompt
                transition = new_transition
                version = new_version
                transition_progress = 0.0
                if transition == TransitionMode.RESTART:
                    frame_index = 0
                    source = self._load_source(spec.source_path, width, height)

            if spec.mode == GenerationMode.CAMERA:
                camera_frame = await inputs.latest()
                if camera_frame is not None:
                    source = Image.fromarray(camera_frame).convert("RGB").resize((width, height))

            image = self._render(
                width,
                height,
                frame_index,
                prompt,
                previous_prompt,
                transition_progress,
                spec.motion_strength,
                source,
                spec.mode,
            )
            transition_progress = min(1.0, transition_progress + 0.08)
            elapsed_ms = (monotonic() - started) * 1000
            yield FramePacket(np.asarray(image, dtype=np.uint8), inference_ms=elapsed_ms)
            frame_index += 1
            await asyncio.sleep(max(0.0, frame_interval - (monotonic() - started)))

    @staticmethod
    def _load_source(path: Path | None, width: int, height: int) -> Image.Image | None:
        if path is None or not path.exists():
            return None
        if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    return Image.fromarray(frame.to_ndarray(format="rgb24")).resize((width, height))
            return None
        with Image.open(path) as image:
            return image.convert("RGB").resize((width, height))

    @staticmethod
    def _render(
        width: int,
        height: int,
        frame_index: int,
        prompt: str,
        previous_prompt: str,
        transition_progress: float,
        motion: float,
        source: Image.Image | None,
        mode: GenerationMode,
    ) -> Image.Image:
        t = frame_index / 18.0
        y, x = np.mgrid[0:height, 0:width]
        wave = np.sin(x / 95 + t * (0.7 + motion)) + np.cos(y / 75 - t * 0.45)
        r = np.clip(22 + 31 * wave + 22 * np.sin(t), 0, 255)
        g = np.clip(16 + 22 * wave + 28 * np.cos(t * 0.7), 0, 255)
        b = np.clip(38 + 40 * wave + 30 * np.sin(t * 0.4 + 1.5), 0, 255)
        generated = Image.fromarray(np.stack((r, g, b), axis=-1).astype(np.uint8), "RGB")

        if source is not None:
            stylized = ImageEnhance.Color(source).enhance(1.35)
            stylized = stylized.filter(ImageFilter.GaussianBlur(radius=0.6))
            overlay_strength = 0.72 if mode == GenerationMode.CAMERA else 0.58
            generated = Image.blend(generated, stylized, overlay_strength)

        draw = ImageDraw.Draw(generated, "RGBA")
        for index in range(12):
            angle = t * (0.3 + motion) + index * 0.63
            radius = min(width, height) * (0.12 + index * 0.025)
            cx = width / 2 + math.cos(angle) * radius
            cy = height / 2 + math.sin(angle * 1.2) * radius * 0.55
            size = 6 + 12 * motion
            draw.ellipse((cx - size, cy - size, cx + size, cy + size), fill=(180, 110, 255, 35))

        draw.rounded_rectangle((24, 24, width - 24, 92), radius=16, fill=(5, 5, 8, 150))
        label = prompt if transition_progress >= 1 else f"{previous_prompt}  →  {prompt}"
        draw.text((48, 45), label[:110], fill=(245, 242, 255, 230), font=ImageFont.load_default())
        draw.text(
            (48, height - 44),
            f"NEELVERSE MOCK • {mode.value.upper()} • FRAME {frame_index:05d}",
            fill=(220, 190, 255, 190),
            font=ImageFont.load_default(),
        )
        return generated
