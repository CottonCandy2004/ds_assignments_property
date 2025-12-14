"""CLI utilities for training and predicting Melbourne property prices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

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
from property.melb_price_model import (
    TrainingConfig,
    load_trained_pipeline,
    predict_price,
    train_gradient_boosting,
)


def handle_train(args: argparse.Namespace) -> None:
    data_path = resolve_path(args.data, must_exist=True)
    model_path = resolve_path(args.model)
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
    model_path = resolve_path(args.model, must_exist=True)
    data_path = resolve_path(args.data, must_exist=True)

    pipeline = load_trained_pipeline(model_path)
    defaults = load_feature_defaults(data_path, args.target)

    overrides: Dict[str, Any] = {}
    for cli_name, column_name, _type, _help in FEATURE_ARGUMENTS:
        attr_name = cli_name.replace("-", "_")
        overrides[column_name] = getattr(args, attr_name)

    if args.feature:
        overrides.update(parse_custom_features(args.feature))

    feature_row = apply_feature_overrides(defaults, overrides)
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
