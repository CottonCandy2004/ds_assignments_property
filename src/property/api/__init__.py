"""Property API package exposing the Flask application factory."""

from property.api.app import PricePredictionService, app, create_app, main

__all__ = ["app", "create_app", "main", "PricePredictionService"]
