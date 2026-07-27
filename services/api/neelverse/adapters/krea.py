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
        import gc

        import torch
        from diffusers import ModularPipeline
        from diffusers.modular_pipelines import PipelineState

        if not torch.cuda.is_available():
            raise RuntimeError("Krea backend requires an NVIDIA CUDA GPU")

        # Disable SageAttention if not installed — fall back to PyTorch SDPA
        import os
        os.environ.setdefault("DISABLE_SAGEATTENTION", "1")

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.settings.krea_model_revision:
            kwargs["revision"] = self.settings.krea_model_revision
        if self.settings.hf_token:
            kwargs["token"] = self.settings.hf_token
        pipe = ModularPipeline.from_pretrained(self.settings.krea_model_id, **kwargs)
        try:
            if self.settings.krea_enable_fp8:
                # Stage the largest component separately so host RAM never needs to
                # hold the transformer, text encoder, and VAE at the same time.
                pipe.load_components(
                    names="transformer",
                    trust_remote_code=True,
                    device_map="cpu",
                    torch_dtype=torch.bfloat16,
                )
            else:
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

                config = Float8DynamicActivationFloat8WeightConfig()
                logger.info("Quantizing Krea transformer to FP8 on CPU")
                for block_index, block in enumerate(pipe.transformer.blocks):
                    quantize_(block, config)
                    gc.collect()
                    logger.info(
                        "Quantized Krea transformer block %d/%d",
                        block_index + 1,
                        len(pipe.transformer.blocks),
                    )
                logger.info("Moving FP8 Krea transformer to CUDA")
                pipe.transformer.to("cuda")
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Loading remaining Krea components on CUDA")
                pipe.load_components(
                    trust_remote_code=True,
                    device_map="cuda",
                    torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
                )
            if self.settings.krea_enable_compile:
                for module in pipe.transformer.modules():
                    if module.__class__.__name__ == "CausalWanAttentionBlock":
                        module.compile(fullgraph=False)
        except BaseException:
            # Clean up GPU memory on any failure to prevent permanent OOM on retry
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            raise
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
        generator = self._torch.Generator("cuda").manual_seed(42)
        prompt, _, prompt_version = await control.prompt_state()
        source_video: list[Image.Image] | None = self._source_frames(spec.source_path)

        try:
            while await control.checkpoint():
                current_prompt, transition, current_version = await control.prompt_state()
                if current_version != prompt_version:
                    prompt = current_prompt
                    prompt_version = current_version
                    if transition == TransitionMode.RESTART:
                        self._release_state(state)
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
                    generator,
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
                if spec.mode == GenerationMode.TEXT and block_index >= 9:
                    # Official text-to-video inference allocates a nine-block latent
                    # window. Release it before starting the next continuous window.
                    self._release_state(state)
                    state = self._pipeline_state()
                    block_index = 0
        finally:
            self._release_state(state)

    def _run_block(
        self,
        state: Any,
        spec: GenerationSpec,
        prompt: str,
        block_index: int,
        source_video: list[Image.Image] | None,
        video_stream: list[Image.Image] | None,
        generator: Any,
    ) -> Any:
        steps = {QualityProfile.TURBO: 4, QualityProfile.BALANCED: 5, QualityProfile.QUALITY: 6}[spec.quality]
        kwargs: dict[str, Any] = {
            "prompt": [prompt],
            "num_inference_steps": steps,
            "strength": max(0.05, min(1.0, spec.motion_strength)),
            "block_idx": block_index,
            "generator": generator,
        }
        if video_stream is not None:
            kwargs["video_stream"] = video_stream
        elif source_video is not None:
            kwargs["video"] = source_video
        else:
            # Text-to-video: use the model's official nine-block latent window.
            kwargs["num_blocks"] = 9
        return self._call_in_place(state, kwargs)

    def _call_in_place(self, state: Any, kwargs: dict[str, Any]) -> Any:
        """Run Modular Diffusers without deepcopying multi-gigabyte CUDA caches.

        The public ModularPipeline call deep-copies PipelineState on every invocation.
        That is safe on 80 GB accelerators but duplicates the KV cache and OOMs a
        48 GB L40S. Generation is serialized by GenerationManager, so mutating the
        current session's state in place is safe here.
        """
        passed_kwargs = kwargs.copy()
        for expected_input in self.pipe._blocks.inputs:
            name = expected_input.name
            kwargs_type = expected_input.kwargs_type
            if name in passed_kwargs:
                state.set(name, passed_kwargs.pop(name), kwargs_type)
            elif kwargs_type is not None and kwargs_type in passed_kwargs:
                for key, value in passed_kwargs.pop(kwargs_type).items():
                    state.set(key, value, kwargs_type)
            elif name is not None and name not in state.values:
                state.set(name, expected_input.default, kwargs_type)
        if passed_kwargs:
            logger.warning("Ignoring unexpected Krea inputs: %s", sorted(passed_kwargs))
        with self._torch.no_grad():
            _, state = self.pipe._blocks(self.pipe, state)
        return state

    def _release_state(self, state: Any) -> None:
        state.values.clear()
        state.kwargs_mapping.clear()
        import gc

        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

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
