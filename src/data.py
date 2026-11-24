"""C-MAPSS data loading, RUL labelling, engine-wise splits, and features.

Design rules (see README "Evaluation methodology"):
- Splits are by engine (unit id), never by row, so cycles from one engine
  can never appear on both sides of a split.
- Rolling features are trailing windows computed per engine, so a row's
  features only ever depend on that engine's past and present cycles.
- RUL labels use a piecewise-linear cap (default 125 cycles): an engine in
  early life is effectively healthy, so its exact RUL (e.g. 310 vs 260) is
  mostly label noise; capping focuses the model on the degradation regime.
  The cap follows the convention used by most published C-MAPSS work
  (R_early = 125), which also keeps our metrics comparable to theirs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SETTING_COLS: list[str] = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS: list[str] = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLS: list[str] = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS

#: Piecewise RUL cap, in cycles (see module docstring).
RUL_CAP = 125

#: Default trailing rolling windows, in cycles.
DEFAULT_WINDOWS: tuple[int, ...] = (5, 10)


def load_cmapss(path: str | Path) -> pd.DataFrame:
    """Load a C-MAPSS text file (train/test) into a named DataFrame.

    The files are whitespace-separated with 26 columns and no header:
    unit number, cycle, 3 operating settings, 21 sensor measurements.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLS, engine="python")
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_rul(path: str | Path) -> pd.Series:
    """Load a RUL_<subset>.txt file: true RUL at each test engine's last cycle.

    Returns a Series indexed by unit id (1-based, matching the data files).
    """
    values = pd.read_csv(path, header=None, names=["rul"], engine="python")["rul"]
    return pd.Series(values.to_numpy(dtype=float), index=np.arange(1, len(values) + 1),
                     name="rul_true")


def add_rul(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Add RUL labels to run-to-failure (training) data.

    True RUL at a row is (engine's final cycle - current cycle). The capped
    label ``rul`` is what the model trains on; ``rul_raw`` is kept for
    reporting.
    """
    out = df.copy()
    final_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul_raw"] = (final_cycle - out["cycle"]).astype(float)
    out["rul"] = out["rul_raw"].clip(upper=cap) if cap is not None else out["rul_raw"]
    return out


def add_test_rul(
    df: pd.DataFrame, rul_at_end: pd.Series, cap: int | None = RUL_CAP
) -> pd.DataFrame:
    """Add RUL labels to (truncated) test data.

    Test engines are cut off before failure; the RUL file gives the true RUL
    at each engine's last observed cycle, so a row k cycles before the end
    has true RUL = rul_at_end + k.
    """
    out = df.copy()
    final_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul_raw"] = (
        out["unit"].map(rul_at_end) + (final_cycle - out["cycle"])
    ).astype(float)
    out["rul"] = out["rul_raw"].clip(upper=cap) if cap is not None else out["rul_raw"]
    return out


def split_by_engine(
    df: pd.DataFrame, val_units: set[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into (train, validation) by whole engines."""
    is_val = df["unit"].isin(val_units)
    return df.loc[~is_val].copy(), df.loc[is_val].copy()


def constant_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Columns with at most one distinct value (e.g. dead sensors in FD001)."""
    return [c for c in cols if df[c].nunique(dropna=False) <= 1]


def build_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add trailing rolling mean/std features per engine.

    Returns the augmented frame (sorted by unit, cycle) and the ordered list
    of model feature columns: ``cycle`` itself (engine age, known at predict
    time), the raw settings/sensors minus dropped columns, and for each base
    column and window a ``<col>_rmean<w>`` and ``<col>_rstd<w>`` column.

    Windows are trailing and include the current cycle only, so features at
    cycle t never see cycles > t (no future leakage).
    """
    drop = set(drop_cols or [])
    base_cols = [c for c in SETTING_COLS + SENSOR_COLS if c not in drop]
    out = df.sort_values(["unit", "cycle"]).reset_index(drop=True).copy()

    feature_cols: list[str] = ["cycle"] + base_cols
    grouped = out.groupby("unit", sort=False)
    for w in windows:
        for col in base_cols:
            mean_col, std_col = f"{col}_rmean{w}", f"{col}_rstd{w}"
            series = grouped[col]
            out[mean_col] = series.transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            out[std_col] = series.transform(
                lambda s: s.rolling(w, min_periods=1).std()
            )
            feature_cols += [mean_col, std_col]

    std_cols = [c for c in feature_cols if "_rstd" in c]
    out[std_cols] = out[std_cols].fillna(0.0)  # std of a single obs is NaN
    return out, feature_cols
