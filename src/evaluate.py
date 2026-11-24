"""Evaluation metrics for RUL prediction.

Headline evaluation follows the standard C-MAPSS protocol: one prediction
per test engine, made at its last observed cycle, compared with the true
RUL from the RUL file (uncapped ground truth).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root mean squared error, in cycles."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def nasa_score(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> float:
    """NASA PHM'08 asymmetric scoring function (lower is better).

    Per engine, with d = predicted - true RUL:
        d < 0  (early prediction):  exp(-d / 13) - 1
        d >= 0 (late prediction):   exp( d / 10) - 1
    Late predictions are penalised more heavily: in maintenance, predicting
    failure later than it occurs is the dangerous direction.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    terms = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(terms))


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: its final observed cycle, sorted by unit."""
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit").reset_index(drop=True)
