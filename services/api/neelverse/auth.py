from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from uuid import uuid4

import jwt
from fastapi import Depends, HTTPException, Request, Response, status
from pwdlib import PasswordHash

from .config import Settings
from .db import Database
from .schemas import LoginRequest, UserView

password_hash = PasswordHash.recommended()
COOKIE_NAME = "neelverse_session"


def initialize_admin(database: Database, settings: Settings) -> None:
    database.create_admin(
        user_id=str(uuid4()),
        username=settings.admin_username,
        password_hash=password_hash.hash(settings.admin_password),
    )


def create_token(user: dict[str, Any], settings: Settings) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": user["id"],
        "username": user["username"],
        "admin": bool(user["is_admin"]),
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_ttl_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


def authenticate(database: Database, credentials: LoginRequest) -> dict[str, Any] | None:
    user = database.user_by_username(credentials.username)
    if user is None or not password_hash.verify(credentials.password, user["password_hash"]):
        return None
    return user


def set_auth_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=settings.jwt_ttl_minutes * 60,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


def current_user(request: Request) -> UserView:
    settings: Settings = request.app.state.settings
    database: Database = request.app.state.database
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    except jwt.PyJWTError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired") from error
    user = database.user_by_id(payload["sub"])
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown user")
    return UserView(id=user["id"], username=user["username"], is_admin=bool(user["is_admin"]))


CurrentUser = Annotated[UserView, Depends(current_user)]
