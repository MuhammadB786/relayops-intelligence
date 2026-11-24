"""Baseline RUL model: histogram gradient boosting (scikit-learn).

Chosen as the M1 baseline because it is strong on tabular features, trains
in seconds on C-MAPSS-scale data, and needs no GPU or extra dependencies.
Sequence models (LSTM/DCNN) are the published state of the art on FD001;
this baseline exists to anchor the pipeline and the evaluation honestly.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_SEED = 42


def build_model(random_state: int = RANDOM_SEED) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=random_state,
    )


def train_model(
    features: pd.DataFrame, target: pd.Series, random_state: int = RANDOM_SEED
) -> HistGradientBoostingRegressor:
    model = build_model(random_state=random_state)
    model.fit(features, target)
    return model


def predict_rul(
    model: HistGradientBoostingRegressor, features: pd.DataFrame
) -> np.ndarray:
    """Predict RUL, clipped at 0 (negative remaining life is meaningless)."""
    return np.clip(model.predict(features), 0.0, None)


def save_model(model: HistGradientBoostingRegressor, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> HistGradientBoostingRegressor:
    return joblib.load(path)
