from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerationMode(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    CAMERA = "camera"


class SessionStatus(StrEnum):
    QUEUED = "queued"
    LOADING = "loading"
    WARMING = "warming"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class TransitionMode(StrEnum):
    SMOOTH = "smooth"
    RESTART = "restart"


class Resolution(StrEnum):
    SD_480 = "480p"
    HD_720 = "720p"


class QualityProfile(StrEnum):
    TURBO = "turbo"
    BALANCED = "balanced"
    QUALITY = "quality"


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class UserView(BaseModel):
    id: str
    username: str
    is_admin: bool


class AssetView(BaseModel):
    id: UUID
    filename: str
    content_type: str
    size_bytes: int


class SessionCreate(BaseModel):
    mode: GenerationMode
    prompt: str = Field(min_length=3, max_length=4000)
    negative_prompt: str = Field(default="", max_length=2000)
    transition: TransitionMode = TransitionMode.SMOOTH
    resolution: Resolution = Resolution.SD_480
    quality: QualityProfile = QualityProfile.TURBO
    motion_strength: float = Field(default=0.55, ge=0.0, le=1.0)
    duration_seconds: Literal[5, 10, 30, 60, 120, 180] | None = None
    asset_id: UUID | None = None
    record: bool = True

    @field_validator("asset_id")
    @classmethod
    def source_modes_require_asset(cls, value: UUID | None, info: object) -> UUID | None:
        mode = getattr(info, "data", {}).get("mode")
        if mode in {GenerationMode.IMAGE, GenerationMode.VIDEO} and value is None:
            raise ValueError("Image and video modes require an uploaded source asset")
        return value


class PromptUpdate(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    transition: TransitionMode


class SessionView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    mode: GenerationMode
    prompt: str
    negative_prompt: str
    status: SessionStatus
    transition: TransitionMode
    resolution: Resolution
    quality: QualityProfile
    motion_strength: float
    duration_seconds: int | None
    queue_position: int | None = None
    native_fps: float = 0
    display_fps: float = 0
    latency_ms: float = 0
    vram_used_gb: float = 0
    frames_generated: int = 0
    error: str | None = None
    output_available: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    ended_at: datetime | None = None


class RTCOffer(BaseModel):
    sdp: str
    type: str


class RTCAnswer(BaseModel):
    sdp: str
    type: str


class HealthView(BaseModel):
    status: str
    backend: str
    gpu_available: bool
    active_session_id: UUID | None = None
    queued_sessions: int = 0
