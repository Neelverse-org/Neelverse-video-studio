from ..config import Settings
from .base import VideoAdapter
from .krea import KreaRealtimeAdapter
from .mock import MockVideoAdapter
from .self_forcing import SelfForcingAdapter


def create_adapter(settings: Settings) -> VideoAdapter:
    if settings.inference_backend == "krea":
        return KreaRealtimeAdapter(settings)
    if settings.inference_backend == "self_forcing":
        return SelfForcingAdapter(settings)
    return MockVideoAdapter()
