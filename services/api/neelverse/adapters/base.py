import asyncio
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

import numpy as np

from ..schemas import GenerationMode, QualityProfile, Resolution, TransitionMode


@dataclass(slots=True)
class GenerationSpec:
    session_id: str
    mode: GenerationMode
    prompt: str
    negative_prompt: str
    resolution: Resolution
    quality: QualityProfile
    motion_strength: float
    duration_seconds: int | None
    source_path: Path | None
    native_fps: int


@dataclass(slots=True)
class FramePacket:
    image: np.ndarray
    generated_at: float = field(default_factory=monotonic)
    inference_ms: float = 0


class InputFrameBuffer:
    def __init__(self, maxlen: int = 24) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=maxlen)
        self._condition = asyncio.Condition()

    async def push(self, frame: np.ndarray) -> None:
        async with self._condition:
            self._frames.append(frame)
            self._condition.notify_all()

    async def latest(self) -> np.ndarray | None:
        async with self._condition:
            return self._frames[-1].copy() if self._frames else None

    async def wait_for_frames(self, count: int, timeout: float = 3.0) -> list[np.ndarray]:
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(lambda: len(self._frames) >= count), timeout=timeout
            )
            frames = list(self._frames)
            if len(frames) == count:
                return [frame.copy() for frame in frames]
            indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
            return [frames[index].copy() for index in indices]


class GenerationControl:
    def __init__(self, prompt: str) -> None:
        self.cancelled = asyncio.Event()
        self.resumed = asyncio.Event()
        self.resumed.set()
        self._prompt = prompt
        self._transition = TransitionMode.SMOOTH
        self._prompt_version = 0
        self._lock = asyncio.Lock()

    async def pause(self) -> None:
        self.resumed.clear()

    async def resume(self) -> None:
        self.resumed.set()

    async def cancel(self) -> None:
        self.cancelled.set()
        self.resumed.set()

    async def update_prompt(self, prompt: str, transition: TransitionMode) -> None:
        async with self._lock:
            self._prompt = prompt
            self._transition = transition
            self._prompt_version += 1

    async def prompt_state(self) -> tuple[str, TransitionMode, int]:
        async with self._lock:
            return self._prompt, self._transition, self._prompt_version

    async def checkpoint(self) -> bool:
        await self.resumed.wait()
        return not self.cancelled.is_set()


class VideoAdapter(ABC):
    name = "unknown"

    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def generate(
        self,
        spec: GenerationSpec,
        control: GenerationControl,
        inputs: InputFrameBuffer,
    ) -> AsyncIterator[FramePacket]:
        if False:
            yield FramePacket(np.empty((0, 0, 3), dtype=np.uint8))

    async def close(self) -> None:
        return None

    def telemetry(self) -> dict[str, float | bool | str]:
        return {"backend": self.name, "gpu_available": False, "vram_used_gb": 0.0}
