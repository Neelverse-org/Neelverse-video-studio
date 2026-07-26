from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NEELVERSE_",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Literal["development", "production"] = "development"
    secret_key: str = "change-this-development-secret-before-deployment"
    allowed_origin: str = "http://localhost:5173"
    cookie_secure: bool = False
    admin_username: str = "admin"
    admin_password: str = "change-me-now"

    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    inference_backend: Literal["mock", "krea", "self_forcing"] = "mock"
    max_queue_size: int = Field(default=32, ge=1, le=500)
    max_queued_per_user: int = Field(default=2, ge=1, le=20)
    session_stale_seconds: int = Field(default=30, ge=10, le=300)
    jwt_ttl_minutes: int = Field(default=720, ge=5, le=10080)

    krea_model_id: str = "krea/krea-realtime-video"
    krea_model_revision: str | None = None
    krea_enable_fp8: bool = True
    krea_enable_compile: bool = True
    hf_token: str | None = None

    self_forcing_url: str = "http://self-forcing-worker:5001"
    native_fps: int = Field(default=8, ge=1, le=24)
    display_fps: int = Field(default=24, ge=1, le=60)

    @field_validator("secret_key")
    @classmethod
    def secure_production_secret(cls, value: str, info: object) -> str:
        if len(value) < 32:
            raise ValueError("NEELVERSE_SECRET_KEY must contain at least 32 characters")
        return value

    @property
    def database_path(self) -> Path:
        return self.data_dir / "neelverse.db"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.assets_dir, self.outputs_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
