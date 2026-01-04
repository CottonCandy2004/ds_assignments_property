"""Core services and helpers for the property API."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Protocol, runtime_checkable

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


def coerce_value(raw: str, caster: type, *, feature_name: str) -> Any:
    if caster is str or caster is Any:
        return raw
    try:
        return caster(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Invalid value for '{feature_name}': expected {caster.__name__}") from exc


def collect_overrides(args: Mapping[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for cli_name, column_name, caster, _ in FEATURE_ARGUMENTS:
        raw_value = args.get(cli_name)
        if raw_value is None:
            raw_value = args.get(column_name)
        if raw_value is None:
            continue
        overrides[column_name] = coerce_value(str(raw_value), caster, feature_name=column_name)

    if isinstance(args, _SupportsGetlist):
        feature_pairs = args.getlist("feature")
        if feature_pairs:
            overrides.update(parse_custom_features(feature_pairs))
    return overrides


__all__ = ["PricePredictionService", "collect_overrides"]
