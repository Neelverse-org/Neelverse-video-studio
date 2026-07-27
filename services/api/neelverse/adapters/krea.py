import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from PIL import Image

from ..config import Settings
from ..schemas import GenerationMode, QualityProfile, TransitionMode
from .base import FramePacket, GenerationControl, GenerationSpec, InputFrameBuffer, VideoAdapter

logger = logging.getLogger(__name__)


class KreaRealtimeAdapter(VideoAdapter):
    """Lazy wrapper around Krea's Modular Diffusers realtime pipeline.

    Heavy imports and model loading happen only when this backend is selected, so the
    authenticated control plane remains runnable without CUDA.
    """

    name = "krea-realtime-14b"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipe: Any = None
        self._torch: Any = None
        self._pipeline_state: Any = None

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        import torch
        from diffusers import ModularPipeline
        from diffusers.modular_pipelines import PipelineState

        if not torch.cuda.is_available():
            raise RuntimeError("Krea backend requires an NVIDIA CUDA GPU")
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.settings.krea_model_revision:
            kwargs["revision"] = self.settings.krea_model_revision
        if self.settings.hf_token:
            kwargs["token"] = self.settings.hf_token
        pipe = ModularPipeline.from_pretrained(self.settings.krea_model_id, **kwargs)
        pipe.load_components(
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
        )
        if pipe.transformer is None:
            raise RuntimeError(
                "Krea transformer failed to load. Check that einops, flash-attn, "
                "sageattention, and kernels are installed correctly."
            )
        for block in pipe.transformer.blocks:
            block.self_attn.fuse_projections()
        if self.settings.krea_enable_fp8:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

            for block in pipe.transformer.blocks:
                quantize_(block, Float8DynamicActivationFloat8WeightConfig())
        if self.settings.krea_enable_compile:
            for module in pipe.transformer.modules():
                if module.__class__.__name__ == "CausalWanAttentionBlock":
                    module.compile(fullgraph=False)
        self.pipe = pipe
        self._torch = torch
        self._pipeline_state = PipelineState

    async def generate(
        self,
        spec: GenerationSpec,
        control: GenerationControl,
        inputs: InputFrameBuffer,
    ) -> AsyncIterator[FramePacket]:
        if self.pipe is None:
            raise RuntimeError("Krea backend has not been loaded")
        state = self._pipeline_state()
        block_index = 0
        prompt, _, prompt_version = await control.prompt_state()
        source_video: list[Image.Image] | None = self._source_frames(spec.source_path)

        while await control.checkpoint():
            current_prompt, transition, current_version = await control.prompt_state()
            if current_version != prompt_version:
                prompt = current_prompt
                prompt_version = current_version
                if transition == TransitionMode.RESTART:
                    state = self._pipeline_state()
                    block_index = 0

            video_stream: list[Image.Image] | None = None
            if spec.mode == GenerationMode.CAMERA:
                count = 9 if block_index == 0 else 12
                try:
                    arrays = await inputs.wait_for_frames(count, timeout=5.0)
                    video_stream = [Image.fromarray(frame).convert("RGB") for frame in arrays]
                except TimeoutError:
                    await asyncio.sleep(0.05)
                    continue

            started = monotonic()
            state = await asyncio.to_thread(
                self._run_block,
                state,
                spec,
                prompt,
                block_index,
                source_video,
                video_stream,
            )
            videos = state.values.get("videos")
            if videos is None:
                raise RuntimeError("Krea pipeline returned no video frames")
            frames = videos[0]
            inference_ms = (monotonic() - started) * 1000
            for frame in frames:
                yield FramePacket(self._as_rgb(frame), inference_ms=inference_ms / max(1, len(frames)))
            source_video = None
            block_index += 1

    def _run_block(
        self,
        state: Any,
        spec: GenerationSpec,
        prompt: str,
        block_index: int,
        source_video: list[Image.Image] | None,
        video_stream: list[Image.Image] | None,
    ) -> Any:
        steps = {QualityProfile.TURBO: 4, QualityProfile.BALANCED: 5, QualityProfile.QUALITY: 6}[spec.quality]
        kwargs: dict[str, Any] = {
            "prompt": [prompt],
            "num_inference_steps": steps,
            "strength": max(0.05, min(1.0, spec.motion_strength)),
            "block_idx": block_index,
            "generator": self._torch.Generator("cuda").manual_seed(42),
        }
        if video_stream is not None:
            kwargs["video_stream"] = video_stream
        elif source_video is not None:
            kwargs["video"] = source_video
        else:
            kwargs["num_blocks"] = 1_000_000
        return self.pipe(state, **kwargs)

    @staticmethod
    def _source_frames(path: Path | None) -> list[Image.Image] | None:
        if path is None:
            return None
        if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            from diffusers.utils import load_video

            return [frame.convert("RGB") for frame in load_video(str(path))]
        with Image.open(path) as image:
            return [image.convert("RGB")]

    @staticmethod
    def _as_rgb(frame: Any) -> np.ndarray:
        if isinstance(frame, Image.Image):
            return np.asarray(frame.convert("RGB"), dtype=np.uint8)
        if hasattr(frame, "detach"):
            frame = frame.detach().float().cpu().numpy()
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating):
            if array.min() < 0:
                array = (array + 1) / 2
            array = np.clip(array * 255, 0, 255)
        return array[..., :3].astype(np.uint8)

    def telemetry(self) -> dict[str, float | bool | str]:
        if self._torch is None or not self._torch.cuda.is_available():
            return {"backend": self.name, "gpu_available": False, "vram_used_gb": 0.0}
        return {
            "backend": self.name,
            "gpu_available": True,
            "vram_used_gb": self._torch.cuda.memory_allocated() / 1024**3,
        }

    async def close(self) -> None:
        self.pipe = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
