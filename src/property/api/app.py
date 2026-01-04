"""Flask application factory and entry points for the property API."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Optional

from flask import Flask
from flask_cors import CORS

from property.api.routes.auth import auth_bp
from property.api.routes.prediction import prediction_bp
from property.api.services import PricePredictionService
from property.config import load_settings
from property.database import configure_database


def create_app(
    service: Optional[PricePredictionService] = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> Flask:
    app = Flask(__name__)
    cors_origins_env = os.getenv("PROPERTY_CORS_ORIGINS", "*")
    cors_origins = (
        [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
        if cors_origins_env != "*"
        else cors_origins_env
    )
    CORS(app, resources={r"/*": {"origins": cors_origins}}, supports_credentials=False)
    settings = load_settings(config_path)
    app.config["SECRET_KEY"] = settings.app.secret_key
    app.config["token_lifetime"] = timedelta(minutes=settings.security.token_exp_minutes)
    app.config["token_algorithm"] = settings.security.token_algorithm

    configure_database(app, settings)

    prediction_service = service or PricePredictionService()
    app.config["prediction_service"] = prediction_service

    app.register_blueprint(prediction_bp)
    app.register_blueprint(auth_bp)

    return app


def main() -> None:  # pragma: no cover - convenience entry point
    app = create_app()
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=os.getenv("FLASK_DEBUG") == "1")


app = create_app()


__all__ = ["create_app", "app", "main", "PricePredictionService"]
