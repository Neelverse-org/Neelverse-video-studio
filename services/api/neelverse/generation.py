import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import UUID

import av
import numpy as np
from PIL import Image

from .adapters.base import (
    FramePacket,
    GenerationControl,
    GenerationSpec,
    InputFrameBuffer,
    VideoAdapter,
)
from .config import Settings
from .db import Database
from .schemas import SessionCreate, SessionStatus, TransitionMode

logger = logging.getLogger(__name__)


class FrameHub:
    def __init__(self) -> None:
        self.previous: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.sequence = 0
        self._condition = asyncio.Condition()

    async def publish(self, image: np.ndarray) -> None:
        async with self._condition:
            self.previous = self.current
            self.current = image
            self.sequence += 1
            self._condition.notify_all()

    async def snapshot(self) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        async with self._condition:
            current = self.current.copy() if self.current is not None else None
            previous = self.previous.copy() if self.previous is not None else None
            return previous, current, self.sequence

    async def wait_for_first(self, timeout: float = 30.0) -> None:
        async with self._condition:
            await asyncio.wait_for(self._condition.wait_for(lambda: self.current is not None), timeout)


class SessionRecorder:
    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self.fps = fps
        self.container: av.container.OutputContainer | None = None
        self.stream: av.video.stream.VideoStream | None = None

    def write(self, image: np.ndarray) -> None:
        if self.container is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.container = av.open(str(self.path), "w")
            self.stream = self.container.add_stream("libx264", rate=self.fps)
            self.stream.width = image.shape[1]
            self.stream.height = image.shape[0]
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "20", "preset": "veryfast"}
        assert self.stream is not None
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.container is None or self.stream is None:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.container = None
        self.stream = None


@dataclass(slots=True)
class RuntimeSession:
    id: UUID
    user_id: str
    request: SessionCreate
    spec: GenerationSpec
    control: GenerationControl
    inputs: InputFrameBuffer
    frames: FrameHub
    status: SessionStatus = SessionStatus.QUEUED
    queue_position: int | None = None
    native_fps: float = 0.0
    display_fps: float = 0.0
    latency_ms: float = 0.0
    vram_used_gb: float = 0.0
    frames_generated: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    peer_connections: set[object] = field(default_factory=set)


class GenerationManager:
    def __init__(self, database: Database, settings: Settings, adapter: VideoAdapter) -> None:
        self.database = database
        self.settings = settings
        self.adapter = adapter
        self.sessions: dict[UUID, RuntimeSession] = {}
        self._queue: asyncio.Queue[UUID] = asyncio.Queue(maxsize=settings.max_queue_size)
        self._worker_task: asyncio.Task[None] | None = None
        self._adapter_loaded = False
        self.active_session_id: UUID | None = None

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._worker(), name="generation-worker")

    async def close(self) -> None:
        for runtime in self.sessions.values():
            await runtime.control.cancel()
        if self._worker_task is not None:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
        for runtime in self.sessions.values():
            for peer in runtime.peer_connections:
                close = getattr(peer, "close", None)
                if close is not None:
                    await close()
        await self.adapter.close()

    def queued_count_for_user(self, user_id: str) -> int:
        return sum(
            runtime.user_id == user_id and runtime.status == SessionStatus.QUEUED
            for runtime in self.sessions.values()
        )

    async def enqueue(
        self,
        session_id: UUID,
        user_id: str,
        request: SessionCreate,
        source_path: Path | None,
    ) -> RuntimeSession:
        if self._queue.full():
            raise OverflowError("Generation queue is full")
        if self.queued_count_for_user(user_id) >= self.settings.max_queued_per_user:
            raise PermissionError("Per-user queue limit reached")
        spec = GenerationSpec(
            session_id=str(session_id),
            mode=request.mode,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            resolution=request.resolution,
            quality=request.quality,
            motion_strength=request.motion_strength,
            duration_seconds=request.duration_seconds,
            source_path=source_path,
            native_fps=self.settings.native_fps,
        )
        runtime = RuntimeSession(
            id=session_id,
            user_id=user_id,
            request=request,
            spec=spec,
            control=GenerationControl(request.prompt),
            inputs=InputFrameBuffer(),
            frames=FrameHub(),
            display_fps=float(self.settings.display_fps),
        )
        self.sessions[session_id] = runtime
        await self._queue.put(session_id)
        self._refresh_queue_positions()
        return runtime

    async def pause(self, session_id: UUID) -> None:
        runtime = self.require(session_id)
        if runtime.status != SessionStatus.RUNNING:
            raise ValueError("Only a running session can be paused")
        await runtime.control.pause()
        runtime.status = SessionStatus.PAUSED
        self.database.update_session(session_id, status=runtime.status.value)

    async def resume(self, session_id: UUID) -> None:
        runtime = self.require(session_id)
        if runtime.status != SessionStatus.PAUSED:
            raise ValueError("Only a paused session can be resumed")
        await runtime.control.resume()
        runtime.status = SessionStatus.RUNNING
        self.database.update_session(session_id, status=runtime.status.value)

    async def stop(self, session_id: UUID) -> None:
        runtime = self.require(session_id)
        await runtime.control.cancel()
        if runtime.status == SessionStatus.QUEUED:
            runtime.status = SessionStatus.STOPPED
            runtime.ended_at = datetime.now(UTC)
            self.database.update_session(
                session_id, status=runtime.status.value, ended_at=runtime.ended_at.isoformat()
            )

    async def update_prompt(
        self, session_id: UUID, prompt: str, transition: TransitionMode
    ) -> None:
        runtime = self.require(session_id)
        await runtime.control.update_prompt(prompt, transition)
        runtime.request.prompt = prompt
        runtime.request.transition = transition
        self.database.update_session(session_id, config_json=runtime.request.model_dump(mode="json"))

    def require(self, session_id: UUID) -> RuntimeSession:
        runtime = self.sessions.get(session_id)
        if runtime is None:
            raise KeyError("Session runtime is no longer available")
        return runtime

    def _refresh_queue_positions(self) -> None:
        position = 1
        for runtime in self.sessions.values():
            if runtime.status == SessionStatus.QUEUED:
                runtime.queue_position = position
                position += 1
            else:
                runtime.queue_position = None

    async def _worker(self) -> None:
        while True:
            session_id = await self._queue.get()
            runtime = self.sessions[session_id]
            self._refresh_queue_positions()
            try:
                if runtime.control.cancelled.is_set():
                    continue
                self.active_session_id = session_id
                if not self._adapter_loaded:
                    runtime.status = SessionStatus.LOADING
                    self.database.update_session(session_id, status=runtime.status.value)
                    await self.adapter.load()
                    self._adapter_loaded = True
                await self._run(runtime)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("Generation session %s failed", session_id)
                runtime.status = SessionStatus.FAILED
                runtime.error = str(error)
                runtime.ended_at = datetime.now(UTC)
                self.database.update_session(
                    session_id,
                    status=runtime.status.value,
                    error=runtime.error,
                    ended_at=runtime.ended_at.isoformat(),
                )
            finally:
                self.active_session_id = None
                self._queue.task_done()
                self._refresh_queue_positions()

    async def _run(self, runtime: RuntimeSession) -> None:
        runtime.status = SessionStatus.RUNNING
        runtime.started_at = datetime.now(UTC)
        runtime.queue_position = None
        self.database.update_session(
            runtime.id, status=runtime.status.value, started_at=runtime.started_at.isoformat()
        )
        started = monotonic()
        recent_times: deque[float] = deque(maxlen=30)
        recorder = (
            SessionRecorder(self.settings.outputs_dir / f"{runtime.id}.mp4", runtime.spec.native_fps)
            if runtime.request.record
            else None
        )
        try:
            async for packet in self.adapter.generate(runtime.spec, runtime.control, runtime.inputs):
                if runtime.control.cancelled.is_set():
                    break
                image = self._normalize_output(packet, runtime)
                await runtime.frames.publish(image)
                if recorder is not None:
                    await asyncio.to_thread(recorder.write, image)
                runtime.frames_generated += 1
                recent_times.append(monotonic())
                if len(recent_times) > 1:
                    runtime.native_fps = (len(recent_times) - 1) / (recent_times[-1] - recent_times[0])
                runtime.latency_ms = packet.inference_ms
                telemetry = self.adapter.telemetry()
                runtime.vram_used_gb = float(telemetry.get("vram_used_gb", 0))
                if runtime.frames_generated % 8 == 0:
                    self.database.update_session(runtime.id, stats_json=self._stats(runtime))
                if runtime.spec.duration_seconds and monotonic() - started >= runtime.spec.duration_seconds:
                    break
            runtime.status = (
                SessionStatus.STOPPED if runtime.control.cancelled.is_set() else SessionStatus.COMPLETED
            )
        finally:
            if recorder is not None:
                await asyncio.to_thread(recorder.close)
                if recorder.path.exists() and recorder.path.stat().st_size > 0:
                    runtime.output_path = recorder.path
            runtime.ended_at = datetime.now(UTC)
            self.database.update_session(
                runtime.id,
                status=runtime.status.value,
                stats_json=self._stats(runtime),
                output_path=str(runtime.output_path) if runtime.output_path else None,
                ended_at=runtime.ended_at.isoformat(),
            )

    @staticmethod
    def _normalize_output(packet: FramePacket, runtime: RuntimeSession) -> np.ndarray:
        image = packet.image
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError("Adapter returned an invalid RGB frame")
        image = image[..., :3].astype(np.uint8, copy=False)
        target = (1280, 720) if runtime.request.resolution.value == "720p" else (832, 480)
        if (image.shape[1], image.shape[0]) != target:
            image = np.asarray(Image.fromarray(image).resize(target, Image.Resampling.LANCZOS))
        return np.ascontiguousarray(image)

    @staticmethod
    def _stats(runtime: RuntimeSession) -> dict[str, float | int]:
        return {
            "native_fps": round(runtime.native_fps, 2),
            "display_fps": round(runtime.display_fps, 2),
            "latency_ms": round(runtime.latency_ms, 2),
            "vram_used_gb": round(runtime.vram_used_gb, 2),
            "frames_generated": runtime.frames_generated,
        }
