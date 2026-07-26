import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .adapters.factory import create_adapter
from .auth import (
    CurrentUser,
    authenticate,
    clear_auth_cookie,
    create_token,
    initialize_admin,
    set_auth_cookie,
)
from .config import get_settings
from .db import Database
from .generation import GenerationManager, RuntimeSession
from .rtc import negotiate
from .schemas import (
    AssetView,
    HealthView,
    LoginRequest,
    PromptUpdate,
    RTCAnswer,
    RTCOffer,
    SessionCreate,
    SessionStatus,
    SessionView,
    UserView,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-matroska",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    initialize_admin(database, settings)
    adapter = create_adapter(settings)
    manager = GenerationManager(database, settings, adapter)
    await manager.start()
    app.state.settings = settings
    app.state.database = database
    app.state.manager = manager
    app.state.adapter = adapter
    try:
        yield
    finally:
        await manager.close()


app = FastAPI(
    title="Neelverse Video Studios API",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


def manager_from(request: Request) -> GenerationManager:
    return request.app.state.manager


def database_from(request: Request) -> Database:
    return request.app.state.database


def ensure_owner(runtime: RuntimeSession, user: UserView) -> None:
    if runtime.user_id != user.id and not user.is_admin:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")


def session_view(row: dict[str, Any], runtime: RuntimeSession | None = None) -> SessionView:
    config = row["config"]
    stats = row["stats"]
    if runtime is not None:
        stats = {
            "native_fps": runtime.native_fps,
            "display_fps": runtime.display_fps,
            "latency_ms": runtime.latency_ms,
            "vram_used_gb": runtime.vram_used_gb,
            "frames_generated": runtime.frames_generated,
        }
    return SessionView(
        id=UUID(row["id"]),
        mode=config["mode"],
        prompt=config["prompt"],
        negative_prompt=config.get("negative_prompt", ""),
        status=runtime.status if runtime else row["status"],
        transition=config.get("transition", "smooth"),
        resolution=config.get("resolution", "480p"),
        quality=config.get("quality", "turbo"),
        motion_strength=config.get("motion_strength", 0.55),
        duration_seconds=config.get("duration_seconds"),
        queue_position=runtime.queue_position if runtime else None,
        native_fps=stats.get("native_fps", 0),
        display_fps=stats.get("display_fps", 0),
        latency_ms=stats.get("latency_ms", 0),
        vram_used_gb=stats.get("vram_used_gb", 0),
        frames_generated=stats.get("frames_generated", 0),
        error=runtime.error if runtime else row.get("error"),
        output_available=bool(runtime.output_path if runtime else row.get("output_path")),
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row.get("started_at") else None,
        ended_at=datetime.fromisoformat(row["ended_at"]) if row.get("ended_at") else None,
    )


@app.get("/api/health", response_model=HealthView)
async def health(request: Request) -> HealthView:
    manager = manager_from(request)
    telemetry = request.app.state.adapter.telemetry()
    return HealthView(
        status="ok",
        backend=str(telemetry["backend"]),
        gpu_available=bool(telemetry["gpu_available"]),
        active_session_id=manager.active_session_id,
        queued_sessions=sum(
            runtime.status == SessionStatus.QUEUED for runtime in manager.sessions.values()
        ),
    )


@app.post("/api/auth/login", response_model=UserView)
async def login(credentials: LoginRequest, request: Request, response: Response) -> UserView:
    user = authenticate(database_from(request), credentials)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_token(user, request.app.state.settings)
    set_auth_cookie(response, token, request.app.state.settings)
    return UserView(id=user["id"], username=user["username"], is_admin=bool(user["is_admin"]))


@app.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    clear_auth_cookie(response)


@app.get("/api/auth/me", response_model=UserView)
async def me(user: CurrentUser) -> UserView:
    return user


@app.post("/api/assets", response_model=AssetView, status_code=status.HTTP_201_CREATED)
async def upload_asset(
    request: Request,
    user: CurrentUser,
    file: UploadFile = File(...),
) -> AssetView:
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image or video format")
    extension = Path(file.filename or "asset").suffix.lower()
    asset_id = uuid4()
    path = request.app.state.settings.assets_dir / f"{asset_id}{extension}"
    size = 0
    with path.open("wb") as destination:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                destination.close()
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Asset exceeds 250 MB")
            destination.write(chunk)
    filename = Path(file.filename or "asset").name
    database_from(request).create_asset(
        {
            "id": str(asset_id),
            "user_id": user.id,
            "filename": filename,
            "content_type": content_type,
            "size_bytes": size,
            "path": str(path),
            "created_at": datetime.now().astimezone().isoformat(),
        }
    )
    return AssetView(id=asset_id, filename=filename, content_type=content_type, size_bytes=size)


@app.post("/api/sessions", response_model=SessionView, status_code=status.HTTP_202_ACCEPTED)
async def create_session(payload: SessionCreate, request: Request, user: CurrentUser) -> SessionView:
    database = database_from(request)
    manager = manager_from(request)
    source_path: Path | None = None
    if payload.asset_id:
        asset = database.asset_for_user(payload.asset_id, user.id)
        if asset is None:
            raise HTTPException(status_code=404, detail="Source asset not found")
        source_path = Path(asset["path"])
    session_id = uuid4()
    database.create_session(session_id, user.id, payload.model_dump(mode="json"))
    try:
        runtime = await manager.enqueue(session_id, user.id, payload, source_path)
    except OverflowError as error:
        raise HTTPException(status_code=429, detail=str(error), headers={"Retry-After": "30"}) from error
    except PermissionError as error:
        raise HTTPException(status_code=429, detail=str(error), headers={"Retry-After": "20"}) from error
    row = database.session_for_user(session_id, user.id)
    assert row is not None
    return session_view(row, runtime)


@app.get("/api/sessions", response_model=list[SessionView])
async def list_sessions(request: Request, user: CurrentUser) -> list[SessionView]:
    database = database_from(request)
    manager = manager_from(request)
    rows = database.sessions_for_user(user.id)
    return [session_view(row, manager.sessions.get(UUID(row["id"]))) for row in rows]


@app.get("/api/sessions/{session_id}", response_model=SessionView)
async def get_session(session_id: UUID, request: Request, user: CurrentUser) -> SessionView:
    row = database_from(request).session_for_user(session_id, user.id)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_view(row, manager_from(request).sessions.get(session_id))


@app.post("/api/sessions/{session_id}/pause", response_model=SessionView)
async def pause_session(session_id: UUID, request: Request, user: CurrentUser) -> SessionView:
    manager = manager_from(request)
    runtime = manager.require(session_id)
    ensure_owner(runtime, user)
    try:
        await manager.pause(session_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    row = database_from(request).session_for_user(session_id, user.id)
    assert row is not None
    return session_view(row, runtime)


@app.post("/api/sessions/{session_id}/resume", response_model=SessionView)
async def resume_session(session_id: UUID, request: Request, user: CurrentUser) -> SessionView:
    manager = manager_from(request)
    runtime = manager.require(session_id)
    ensure_owner(runtime, user)
    try:
        await manager.resume(session_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    row = database_from(request).session_for_user(session_id, user.id)
    assert row is not None
    return session_view(row, runtime)


@app.post("/api/sessions/{session_id}/stop", response_model=SessionView)
async def stop_session(session_id: UUID, request: Request, user: CurrentUser) -> SessionView:
    manager = manager_from(request)
    runtime = manager.require(session_id)
    ensure_owner(runtime, user)
    await manager.stop(session_id)
    row = database_from(request).session_for_user(session_id, user.id)
    assert row is not None
    return session_view(row, runtime)


@app.patch("/api/sessions/{session_id}/prompt", response_model=SessionView)
async def update_prompt(
    session_id: UUID,
    payload: PromptUpdate,
    request: Request,
    user: CurrentUser,
) -> SessionView:
    manager = manager_from(request)
    runtime = manager.require(session_id)
    ensure_owner(runtime, user)
    await manager.update_prompt(session_id, payload.prompt, payload.transition)
    row = database_from(request).session_for_user(session_id, user.id)
    assert row is not None
    return session_view(row, runtime)


@app.post("/api/sessions/{session_id}/rtc/offer", response_model=RTCAnswer)
async def rtc_offer(
    session_id: UUID,
    offer: RTCOffer,
    request: Request,
    user: CurrentUser,
) -> RTCAnswer:
    manager = manager_from(request)
    runtime = manager.require(session_id)
    ensure_owner(runtime, user)
    return await negotiate(manager, session_id, offer)


@app.get("/api/sessions/{session_id}/download")
async def download_session(session_id: UUID, request: Request, user: CurrentUser) -> FileResponse:
    row = database_from(request).session_for_user(session_id, user.id)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    path_value = row.get("output_path")
    if not path_value or not Path(path_value).exists():
        raise HTTPException(status_code=409, detail="Recording is not ready")
    return FileResponse(path_value, media_type="video/mp4", filename=f"neelverse-{session_id}.mp4")
