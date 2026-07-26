import asyncio
from fractions import Fraction
from time import monotonic
from uuid import UUID

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from .generation import FrameHub, GenerationManager
from .schemas import RTCAnswer, RTCOffer

VIDEO_CLOCK = 90_000


class GeneratedVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, hub: FrameHub, fps: int) -> None:
        super().__init__()
        self.hub = hub
        self.fps = fps
        self.started_at = monotonic()
        self.frame_index = 0
        self.last_sequence = -1
        self.interpolation_step = 2

    async def recv(self) -> av.VideoFrame:
        target_time = self.started_at + self.frame_index / self.fps
        await asyncio.sleep(max(0.0, target_time - monotonic()))
        previous, current, sequence = await self.hub.snapshot()
        if current is None:
            current = self._placeholder()
        elif sequence != self.last_sequence:
            self.last_sequence = sequence
            self.interpolation_step = 0

        if previous is not None and self.interpolation_step < 2:
            alpha = (self.interpolation_step + 1) / 3
            image = np.clip(previous * (1 - alpha) + current * alpha, 0, 255).astype(np.uint8)
            self.interpolation_step += 1
        else:
            image = current

        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = round(self.frame_index * VIDEO_CLOCK / self.fps)
        frame.time_base = Fraction(1, VIDEO_CLOCK)
        self.frame_index += 1
        return frame

    @staticmethod
    def _placeholder() -> np.ndarray:
        frame = np.zeros((480, 832, 3), dtype=np.uint8)
        frame[:, :, 0] = 10
        frame[:, :, 1] = 8
        frame[:, :, 2] = 18
        return frame


async def consume_camera(track: MediaStreamTrack, manager: GenerationManager, session_id: UUID) -> None:
    runtime = manager.require(session_id)
    try:
        while True:
            frame = await track.recv()
            image = frame.to_ndarray(format="rgb24")
            await runtime.inputs.push(image)
    except asyncio.CancelledError:
        raise
    except Exception:
        return


async def negotiate(
    manager: GenerationManager,
    session_id: UUID,
    offer: RTCOffer,
) -> RTCAnswer:
    runtime = manager.require(session_id)
    peer = RTCPeerConnection()
    runtime.peer_connections.add(peer)
    camera_task: asyncio.Task[None] | None = None

    @peer.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        nonlocal camera_task
        if track.kind == "video":
            camera_task = asyncio.create_task(consume_camera(track, manager, session_id))

    @peer.on("connectionstatechange")
    async def on_state_change() -> None:
        if peer.connectionState in {"failed", "closed", "disconnected"}:
            if camera_task is not None:
                camera_task.cancel()
            runtime.peer_connections.discard(peer)
            await peer.close()

    await peer.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
    peer.addTrack(GeneratedVideoTrack(runtime.frames, manager.settings.display_fps))
    answer = await peer.createAnswer()
    await peer.setLocalDescription(answer)
    assert peer.localDescription is not None
    return RTCAnswer(sdp=peer.localDescription.sdp, type=peer.localDescription.type)
