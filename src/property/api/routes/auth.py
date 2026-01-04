"""Authentication routes for the property API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import bcrypt
import jwt
from flask import Blueprint, current_app, jsonify, request
from sqlalchemy.exc import IntegrityError

from property.database import User, db

TOKEN_TYPE = "Bearer"

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def _token_lifetime_seconds() -> int:
    lifetime = current_app.config["token_lifetime"]
    return int(lifetime.total_seconds())


def _issue_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "iat": int(now.timestamp()),
        "exp": int((now + current_app.config["token_lifetime"]).timestamp()),
    }
    return jwt.encode(payload, current_app.config["SECRET_KEY"], algorithm=current_app.config["token_algorithm"])


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:  # pragma: no cover - corrupted hash guard
        return False


def _serialize_user(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "username": user.username,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _load_payload() -> Mapping[str, Any]:
    payload = request.get_json(silent=True) or {}
    return payload if isinstance(payload, Mapping) else {}


def _authentication_response(user: User) -> Any:
    token = _issue_token(user)
    return {
        "user": _serialize_user(user),
        "token": token,
        "token_type": TOKEN_TYPE,
        "expires_in": _token_lifetime_seconds(),
    }


@auth_bp.post("/register")
def register_user() -> Any:
    payload = _load_payload()
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "password must be at least 8 characters"}), 400

    user = User(username=username, password_hash=_hash_password(password))
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "username already exists"}), 409

    return jsonify(_authentication_response(user)), 201


@auth_bp.post("/login")
def login_user() -> Any:
    payload = _load_payload()
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    user = User.query.filter_by(username=username).first()
    if user is None or not _verify_password(password, user.password_hash):
        return jsonify({"error": "invalid credentials"}), 401

    return jsonify({"message": "login successful", **_authentication_response(user)})


__all__ = ["auth_bp"]
