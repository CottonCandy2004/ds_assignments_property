"""Flask API for serving Melbourne property price predictions and auth."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

import bcrypt
import jwt
from flask import Flask, jsonify, request
from sqlalchemy.exc import IntegrityError

from property.features import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_TARGET,
    FEATURE_ARGUMENTS,
    apply_feature_overrides,
    load_feature_defaults,
    parse_custom_features,
    resolve_path,
)
from property.config import load_settings
from property.database import User, configure_database, db
from property.melb_price_model import load_trained_pipeline, predict_price


@runtime_checkable
class _SupportsGetlist(Protocol):
    def getlist(self, key: str) -> list[str]:
        ...


class PricePredictionService:
    """Encapsulates model loading and feature preparation for inference."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        data_path: str | os.PathLike[str] | None = None,
        target_column: str = DEFAULT_TARGET,
    ) -> None:
        self.model_path = resolve_path(
            model_path or os.getenv("PROPERTY_MODEL_PATH", DEFAULT_MODEL), must_exist=True
        )
        self.data_path = resolve_path(
            data_path or os.getenv("PROPERTY_DATA_PATH", DEFAULT_DATA), must_exist=True
        )
        self.target_column = target_column or os.getenv("PROPERTY_TARGET", DEFAULT_TARGET)

        self.pipeline = load_trained_pipeline(self.model_path)
        self.defaults = load_feature_defaults(self.data_path, self.target_column)

    def predict(self, overrides: Mapping[str, Any]) -> Dict[str, Any]:
        feature_row = apply_feature_overrides(self.defaults, dict(overrides))
        price = predict_price(self.pipeline, feature_row)
        return {
            "currency": "AUD",
            "prediction": price,
            "features": feature_row,
        }


def _coerce_value(raw: str, caster: type, *, feature_name: str) -> Any:
    if caster is str or caster is Any:
        return raw
    try:
        return caster(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Invalid value for '{feature_name}': expected {caster.__name__}") from exc


def _collect_overrides(args: Mapping[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for cli_name, column_name, caster, _ in FEATURE_ARGUMENTS:
        raw_value = args.get(cli_name)
        if raw_value is None:
            raw_value = args.get(column_name)
        if raw_value is None:
            continue
        overrides[column_name] = _coerce_value(str(raw_value), caster, feature_name=column_name)

    if isinstance(args, _SupportsGetlist):
        feature_pairs = args.getlist("feature")
        if feature_pairs:
            overrides.update(parse_custom_features(feature_pairs))
    return overrides


def create_app(
    service: Optional[PricePredictionService] = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> Flask:
    app = Flask(__name__)
    settings = load_settings(config_path)
    app.config["SECRET_KEY"] = settings.app.secret_key
    configure_database(app, settings)

    token_lifetime = timedelta(minutes=settings.security.token_exp_minutes)
    token_algorithm = settings.security.token_algorithm

    prediction_service = service or PricePredictionService()

    def issue_token(user: User) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "iat": int(now.timestamp()),
            "exp": int((now + token_lifetime).timestamp()),
        }
        return jwt.encode(payload, app.config["SECRET_KEY"], algorithm=token_algorithm)

    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def verify_password(password: str, password_hash: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except ValueError:  # pragma: no cover - corrupted hash guard
            return False

    def serialize_user(user: User) -> Dict[str, Any]:
        return {
            "id": user.id,
            "username": user.username,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }

    @app.get("/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "model_path": str(prediction_service.model_path),
                "data_path": str(prediction_service.data_path),
                "target": prediction_service.target_column,
                "feature_count": len(prediction_service.defaults),
            }
        )

    @app.get("/predict")
    def predict_endpoint() -> Any:
        try:
            overrides = _collect_overrides(request.args)
            result = prediction_service.predict(overrides)
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - defensive path
            return jsonify({"error": f"Unexpected error: {exc}"}), 500

        response = {
            "prediction": round(result["prediction"], 2),
            "currency": result["currency"],
            "features": result["features"],
            "overrides": overrides,
        }
        return jsonify(response)

    @app.post("/auth/register")
    def register_user() -> Any:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, Mapping):
            payload = {}

        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400
        if len(password) < 8:
            return jsonify({"error": "password must be at least 8 characters"}), 400

        user = User(username=username, password_hash=hash_password(password))
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return jsonify({"error": "username already exists"}), 409

        token = issue_token(user)
        return (
            jsonify(
                {
                    "user": serialize_user(user),
                    "token": token,
                    "token_type": "Bearer",
                    "expires_in": int(token_lifetime.total_seconds()),
                }
            ),
            201,
        )

    @app.post("/auth/login")
    def login_user() -> Any:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, Mapping):
            payload = {}

        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        user = User.query.filter_by(username=username).first()
        if user is None or not verify_password(password, user.password_hash):
            return jsonify({"error": "invalid credentials"}), 401

        token = issue_token(user)
        return jsonify(
            {
                "message": "login successful",
                "user": serialize_user(user),
                "token": token,
                "token_type": "Bearer",
                "expires_in": int(token_lifetime.total_seconds()),
            }
        )

    return app


app = create_app()


def main() -> None:  # pragma: no cover - convenience entry point
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=os.getenv("FLASK_DEBUG") == "1")


if __name__ == "__main__":  # pragma: no cover - script mode
    main()
