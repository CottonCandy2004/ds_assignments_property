"""CLI utilities for training and predicting Melbourne property prices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

from database_ml.melb_price_model import (
    TrainingConfig,
    load_trained_pipeline,
    predict_price,
    train_gradient_boosting,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "melb_data.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "melb_gbr_pipeline.joblib"
DEFAULT_TARGET = "Price"

FEATURE_ARGUMENTS: Tuple[Tuple[str, str, Any, str], ...] = (
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


def _resolve_path(path_like: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(path_like)
    path = path if path.is_absolute() else (PROJECT_ROOT / path)
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_feature_defaults(data_path: Path, target_column: str) -> Dict[str, Any]:
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


def _apply_feature_overrides(
    base_features: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    combined = base_features.copy()
    combined.update({k: v for k, v in overrides.items() if v is not None})
    return combined


def _parse_custom_features(feature_pairs: Iterable[str]) -> Dict[str, Any]:
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
        overrides[key] = _maybe_cast_value(value)
    return overrides


def _maybe_cast_value(value: str) -> Any:
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def handle_train(args: argparse.Namespace) -> None:
    data_path = _resolve_path(args.data, must_exist=True)
    model_path = _resolve_path(args.model)
    config = TrainingConfig(
        data_path=data_path,
        model_output_path=model_path,
        test_size=args.test_size,
        random_state=args.random_state,
        target_column=args.target,
        use_hist_gradient_boosting=not args.disable_hist,
        n_threads=args.n_threads,
    )

    result = train_gradient_boosting(config)

    print("Training complete. Metrics:")
    for name, value in result.metrics.items():
        if name == "r2_score":
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value:,.2f}")

    if result.model_path:
        print(f"Model saved to {result.model_path}")


def handle_calc(args: argparse.Namespace) -> None:
    model_path = _resolve_path(args.model, must_exist=True)
    data_path = _resolve_path(args.data, must_exist=True)

    pipeline = load_trained_pipeline(model_path)
    defaults = _load_feature_defaults(data_path, args.target)

    overrides: Dict[str, Any] = {}
    for cli_name, column_name, _type, _help in FEATURE_ARGUMENTS:
        attr_name = cli_name.replace("-", "_")
        overrides[column_name] = getattr(args, attr_name)

    if args.feature:
        overrides.update(_parse_custom_features(args.feature))

    feature_row = _apply_feature_overrides(defaults, overrides)
    prediction = predict_price(pipeline, feature_row)

    print("Prediction inputs (overrides applied):")
    for key in sorted(overrides):
        value = overrides[key]
        if value is not None:
            print(f"  {key}: {value}")

    print(f"\nPredicted price: ${prediction:,.0f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="property",
        description="Train and query a Gradient Boosting model for Melbourne property prices.",
    )
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Gradient Boosting model")
    train_parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to melb_data.csv")
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Output model path")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out ratio")
    train_parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    train_parser.add_argument(
        "--target", default=DEFAULT_TARGET, help="Target column name in the dataset"
    )
    train_parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Limit CPU threads for training (default: use all available cores)",
    )
    train_parser.add_argument(
        "--disable-hist",
        action="store_true",
        help="Fallback to classic GradientBoostingRegressor (single-core).",
    )
    train_parser.set_defaults(func=handle_train)

    calc_parser = subparsers.add_parser(
        "calc", help="Load a trained model and predict price for a custom property"
    )
    calc_parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to saved model")
    calc_parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset for defaults")
    calc_parser.add_argument(
        "--target", default=DEFAULT_TARGET, help="Target column used during training"
    )

    for arg_name, _, arg_type, help_text in FEATURE_ARGUMENTS:
        dest = arg_name.replace("-", "_")
        calc_parser.add_argument(
            f"--{arg_name}",
            dest=dest,
            type=arg_type,
            help=help_text,
        )

    calc_parser.add_argument(
        "--feature",
        action="append",
        default=None,
        help=(
            "Custom COLUMN=VALUE overrides (repeatable). Useful for columns without dedicated flags."
        ),
    )
    calc_parser.set_defaults(func=handle_calc)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func") or args.func is None:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


__all__ = ["main", "build_parser"]
