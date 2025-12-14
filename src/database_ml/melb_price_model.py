from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainingConfig:
    """Configuration values for training a Gradient Boosting regressor."""

    data_path: Path
    target_column: str = "Price"
    test_size: float = 0.2
    random_state: int = 42
    model_output_path: Optional[Path] = None
    use_hist_gradient_boosting: bool = True
    n_threads: Optional[int] = None


@dataclass
class TrainingResult:
    """Encapsulates artifacts produced during training."""

    pipeline: Pipeline
    metrics: Dict[str, float]
    feature_names: List[str]
    model_path: Optional[Path] = None


def load_dataset(data_path: Path, target_column: str) -> pd.DataFrame:
    """Load the Melbourne dataset and drop rows without the target."""

    df = pd.read_csv(data_path)
    df = df.dropna(subset=[target_column])
    df = df.drop_duplicates()
    return df


def _build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Create the preprocessing pipeline for numeric and categorical features."""

    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No features available to build the preprocessing pipeline.")

    return ColumnTransformer(transformers=transformers)


def _configure_threading(n_threads: Optional[int]) -> None:
    """Optionally constrain the number of CPU threads used by numeric backends."""

    if n_threads is None or n_threads <= 0:
        return

    thread_env_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )

    for env_var in thread_env_vars:
        os.environ[env_var] = str(n_threads)


def _build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
    use_hist_gradient_boosting: bool,
) -> Pipeline:
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    if use_hist_gradient_boosting:
        regressor = HistGradientBoostingRegressor(random_state=random_state)
    else:
        regressor = GradientBoostingRegressor(random_state=random_state)
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def train_gradient_boosting(config: TrainingConfig) -> TrainingResult:
    """Train a Gradient Boosting Regressor and optionally persist the fitted pipeline."""

    df = load_dataset(config.data_path, config.target_column)
    feature_frame = df.drop(columns=[config.target_column])
    target = df[config.target_column]

    numeric_features = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_frame.select_dtypes(exclude=["number"]).columns.tolist()

    pipeline = _build_pipeline(
        numeric_features,
        categorical_features,
        config.random_state,
        config.use_hist_gradient_boosting,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    _configure_threading(config.n_threads)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "r2_score": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mse**0.5,
    }

    model_path = None
    if config.model_output_path:
        config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, config.model_output_path)
        model_path = config.model_output_path

    return TrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        feature_names=list(feature_frame.columns),
        model_path=model_path,
    )


def load_trained_pipeline(model_path: Path) -> Pipeline:
    """Load a previously persisted pipeline from disk."""

    return joblib.load(model_path)


def predict_price(pipeline: Pipeline, feature_row: Dict[str, Any]) -> float:
    """Generate a single price prediction from raw feature values."""

    single_frame = pd.DataFrame([feature_row])
    prediction = pipeline.predict(single_frame)[0]
    return float(prediction)
