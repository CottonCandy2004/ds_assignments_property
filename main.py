from pathlib import Path

import pandas as pd

from database_ml.melb_price_model import (
    TrainingConfig,
    predict_price,
    train_gradient_boosting,
)


def _get_project_paths() -> tuple[Path, Path]:
    """Resolve key filesystem paths used by the training workflow."""

    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "melb_data.csv"
    model_path = project_root / "models" / "melb_gbr_pipeline.joblib"
    return data_path, model_path


def main() -> None:
    data_path, model_path = _get_project_paths()

    config = TrainingConfig(data_path=data_path, model_output_path=model_path)
    result = train_gradient_boosting(config)

    print("\n=== Gradient Boosting Regression Results ===")
    print(f"R^2 Score : {result.metrics['r2_score']:.4f}")
    print(f"MAE        : {result.metrics['mae']:.2f}")
    print(f"RMSE       : {result.metrics['rmse']:.2f}")

    sample_row = (
        pd.read_csv(data_path)
        .dropna(subset=[config.target_column])
        .drop(columns=[config.target_column])
        .iloc[0]
        .to_dict()
    )
    predicted_price = predict_price(result.pipeline, sample_row)
    print("\nSample prediction (first row of dataset):")
    print(f"Predicted Price: ${predicted_price:,.0f}")

    if result.model_path:
        print(f"\nPersisted trained pipeline to: {result.model_path}")


if __name__ == "__main__":
    main()
