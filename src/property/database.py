"""Database helpers and models for the property API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

if TYPE_CHECKING:  # pragma: no cover - hints only
    from property.config import Settings


db = SQLAlchemy()


class User(db.Model):  # type: ignore[misc]
    """Simple user entity used for registration and login."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now(), nullable=False)


def configure_database(app: Flask, settings: "Settings") -> None:
    """Configure SQLAlchemy for the Flask app and ensure tables exist."""

    app.config.setdefault("SQLALCHEMY_DATABASE_URI", settings.database.url)
    app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)

    engine_options: Dict[str, Any] = {
        "pool_pre_ping": True,
        "pool_recycle": settings.database.pool_recycle,
        "pool_timeout": settings.database.pool_timeout,
        "pool_size": settings.database.pool_size,
    }
    existing = app.config.get("SQLALCHEMY_ENGINE_OPTIONS", {})
    merged_options = {**engine_options, **existing}
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = merged_options

    db.init_app(app)
    with app.app_context():
        db.create_all()


__all__ = ["configure_database", "db", "User"]
