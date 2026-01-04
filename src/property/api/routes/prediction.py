"""Prediction-related Flask routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, current_app, jsonify, request

from property.api.services import PricePredictionService, collect_overrides

prediction_bp = Blueprint("prediction", __name__)


def _service() -> PricePredictionService:
    return current_app.config["prediction_service"]


@prediction_bp.get("/health")
def health() -> Any:
    service = _service()
    return jsonify(
        {
            "status": "ok",
            "model_path": str(service.model_path),
            "data_path": str(service.data_path),
            "target": service.target_column,
            "feature_count": len(service.defaults),
        }
    )


@prediction_bp.get("/predict")
def predict_endpoint() -> Any:
    service = _service()
    try:
        overrides = collect_overrides(request.args)
        result = service.predict(overrides)
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


__all__ = ["prediction_bp"]
