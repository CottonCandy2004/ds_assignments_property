"""Shared feature metadata and helper utilities for the property project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "melb_data.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "melb_gbr_pipeline.joblib"
DEFAULT_TARGET = "Price"

FeatureArgument = Tuple[str, str, type, str]

FEATURE_ARGUMENTS: Tuple[FeatureArgument, ...] = (
	("suburb", "Suburb", str, "Suburb name"),
	("address", "Address", str, "Street address"),
	("rooms", "Rooms", int, "Number of rooms"),
	("type", "Type", str, "Property type code (h, u, t)"),
	("method", "Method", str, "Sale method code"),
	("seller", "SellerG", str, "Selling agency"),
	("date", "Date", str, "Sale date (e.g., 4/03/2017)"),
	("distance", "Distance", float, "Distance to CBD in km"),
	("postcode", "Postcode", float, "Postcode"),
	("bedroom2", "Bedroom2", float, "Secondary bedroom count"),
	("bathroom", "Bathroom", float, "Bathroom count"),
	("car", "Car", float, "Car spots"),
	("landsize", "Landsize", float, "Land size in m^2"),
	("building-area", "BuildingArea", float, "Building area in m^2"),
	("year-built", "YearBuilt", float, "Year built"),
	("council-area", "CouncilArea", str, "Council area"),
	("lattitude", "Lattitude", float, "Latitude"),
	("longtitude", "Longtitude", float, "Longitude"),
	("region", "Regionname", str, "Region name"),
	("property-count", "Propertycount", float, "Number of properties in the suburb"),
)


def resolve_path(path_like: str | os.PathLike[str] | Path, *, must_exist: bool = False) -> Path:
	path = Path(path_like)
	path = path if path.is_absolute() else (PROJECT_ROOT / path)
	if must_exist and not path.exists():
		raise FileNotFoundError(path)
	return path


def load_feature_defaults(data_path: Path, target_column: str) -> Dict[str, Any]:
	df = pd.read_csv(data_path)
	if target_column not in df.columns:
		raise ValueError(f"Target column '{target_column}' not found in dataset {data_path}")

	df = df.dropna(subset=[target_column])
	if df.empty:
		raise ValueError("Dataset has no rows after dropping missing targets.")

	feature_frame = df.drop(columns=[target_column])
	defaults: Dict[str, Any] = {}

	numeric_cols = feature_frame.select_dtypes(include=["number"]).columns
	if len(numeric_cols) > 0:
		defaults.update(feature_frame[numeric_cols].median(numeric_only=True).to_dict())

	categorical_cols = feature_frame.select_dtypes(exclude=["number"]).columns
	if len(categorical_cols) > 0:
		defaults.update(feature_frame[categorical_cols].mode(dropna=True).iloc[0].to_dict())

	return defaults


def apply_feature_overrides(
	base_features: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
	combined = base_features.copy()
	combined.update({k: v for k, v in overrides.items() if v is not None})
	return combined


def parse_custom_features(feature_pairs: Iterable[str]) -> Dict[str, Any]:
	overrides: Dict[str, Any] = {}
	for pair in feature_pairs:
		if "=" not in pair:
			raise ValueError(
				f"Invalid --feature '{pair}'. Expected 'COLUMN=VALUE' (e.g., --feature Rooms=3)."
			)
		key, value = pair.split("=", maxsplit=1)
		key = key.strip()
		value = value.strip()
		if not key:
			raise ValueError("Feature key cannot be empty.")
		overrides[key] = maybe_cast_value(value)
	return overrides


def maybe_cast_value(value: str) -> Any:
	for caster in (int, float):
		try:
			return caster(value)
		except ValueError:
			continue
	return value


__all__ = [
	"DEFAULT_DATA",
	"DEFAULT_MODEL",
	"DEFAULT_TARGET",
	"FEATURE_ARGUMENTS",
	"apply_feature_overrides",
	"load_feature_defaults",
	"maybe_cast_value",
	"parse_custom_features",
	"resolve_path",
]
