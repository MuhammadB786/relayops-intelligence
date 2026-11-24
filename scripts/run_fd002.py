"""Generalization stretch: the M1 pipeline on C-MAPSS FD002.

FD002 is the materially harder sibling of FD001: 260 training engines
under SIX operating conditions (vs one), 259 test engines. The same
pipeline (cap-125 labels, trailing rolling features, gradient boosting)
is run two ways:

  baseline               — exactly as M1 ran FD001;
  condition_normalized   — the standard remedy for multi-condition data:
                           the 3 operating settings are clustered into
                           conditions (KMeans, k=6, fit on training data
                           only), and each sensor channel is z-scored
                           within its condition using training-only
                           per-condition statistics, BEFORE feature
                           building. All fitted objects (scaler, KMeans,
                           per-condition stats) see training rows only;
                           validation/test rows are transformed, never
                           fit on — the remedy stays leak-free.

Protocol mirrors M1: engine-wise validation split (ids divisible by 5),
checkpoint scoring on held-out engines, then a final refit on all
training engines scored on the official test set (last cycle, uncapped
truth). Writes results/fd002_metrics.json. M1 artifacts are not touched.

Run:  python scripts/run_fd002.py [--data-dir data] [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as d
from src import evaluate as ev
from src import model as m

SUBSET = "FD002"
VAL_MODULO = 5
VAL_CHECKPOINTS = (100, 75, 50, 25, 10)
N_CONDITIONS = 6


def evaluate_checkpoints(model, frame: pd.DataFrame, feature_cols: list[str]) -> dict:
    """M1's validation protocol: one point per engine per true-RUL horizon."""
    rows = frame[frame["rul_raw"].isin(VAL_CHECKPOINTS)]
    pred = m.predict_rul(model, rows[feature_cols])
    return {
        "protocol": f"predictions at true RUL in {list(VAL_CHECKPOINTS)}",
        "points": int(len(rows)),
        "engines": int(rows["unit"].nunique()),
        "rmse": float(ev.rmse(rows["rul_raw"], pred)),
        "nasa_score": float(ev.nasa_score(rows["rul_raw"], pred)),
    }


def evaluate_last_cycle(model, frame: pd.DataFrame, feature_cols: list[str]) -> dict:
    last = ev.last_cycle_rows(frame)
    pred = m.predict_rul(model, last[feature_cols])
    return {
        "engines": int(len(last)),
        "rmse": float(ev.rmse(last["rul_raw"], pred)),
        "nasa_score": float(ev.nasa_score(last["rul_raw"], pred)),
    }


# --- operating-condition normalization (fit on training rows only) ------

def fit_condition_normalizer(train: pd.DataFrame) -> dict:
    """Cluster settings into conditions; per-condition sensor statistics."""
    scaler = StandardScaler().fit(train[d.SETTING_COLS])
    kmeans = KMeans(n_clusters=N_CONDITIONS, random_state=m.RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(scaler.transform(train[d.SETTING_COLS]))
    work = train[d.SENSOR_COLS].copy()
    work["condition"] = labels
    stats = {}
    for cond, grp in work.groupby("condition"):
        stats[int(cond)] = {
            "mean": grp[d.SENSOR_COLS].mean(),
            "std": grp[d.SENSOR_COLS].std().replace(0.0, 1.0),
        }
    return {"scaler": scaler, "kmeans": kmeans, "stats": stats}


def apply_condition_normalizer(df: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """Z-score each sensor within its (predicted) operating condition."""
    out = df.copy()
    out[d.SENSOR_COLS] = out[d.SENSOR_COLS].astype(float)  # some load as int64
    labels = fitted["kmeans"].predict(fitted["scaler"].transform(df[d.SETTING_COLS]))
    out["condition"] = labels
    for cond, s in fitted["stats"].items():
        mask = out["condition"] == cond
        out.loc[mask, d.SENSOR_COLS] = (
            out.loc[mask, d.SENSOR_COLS] - s["mean"]) / s["std"]
    return out.drop(columns=["condition"])


# --- one full M1-style run (validation split + final refit) -------------

def run_variant(train_lab: pd.DataFrame, test_lab: pd.DataFrame,
                normalize: bool) -> dict:
    val_units = {u for u in train_lab["unit"].unique() if u % VAL_MODULO == 0}
    results: dict = {}

    # Stage 1: validation split. A normalizer, if used, is fit ONLY on the
    # split's training engines so held-out engines stay untouched.
    tr_raw, va_raw = d.split_by_engine(train_lab, val_units)
    if normalize:
        fitted = fit_condition_normalizer(tr_raw)
        tr_raw = apply_condition_normalizer(tr_raw, fitted)
        va_raw = apply_condition_normalizer(va_raw, fitted)
    dead = d.constant_columns(tr_raw, d.SETTING_COLS + d.SENSOR_COLS)
    tr_feat, feature_cols = d.build_features(tr_raw, drop_cols=dead)
    va_feat, _ = d.build_features(va_raw, drop_cols=dead)
    val_model = m.train_model(tr_feat[feature_cols], tr_feat["rul"])
    results["validation"] = {
        "train_engines": int(tr_raw["unit"].nunique()),
        "val_engines": int(va_raw["unit"].nunique()),
        **evaluate_checkpoints(val_model, va_feat, feature_cols),
    }

    # Stage 2: refit on all training engines -> official test set.
    full_raw, test_raw = train_lab, test_lab
    if normalize:
        fitted = fit_condition_normalizer(full_raw)
        full_raw = apply_condition_normalizer(full_raw, fitted)
        test_raw = apply_condition_normalizer(test_raw, fitted)
    dead = d.constant_columns(full_raw, d.SETTING_COLS + d.SENSOR_COLS)
    full_feat, feature_cols = d.build_features(full_raw, drop_cols=dead)
    test_feat, _ = d.build_features(test_raw, drop_cols=dead)
    final_model = m.train_model(full_feat[feature_cols], full_feat["rul"])
    results["test_official_last_cycle"] = {
        "train_engines": int(full_raw["unit"].nunique()),
        "dropped_constant_columns": dead,
        "n_features": len(feature_cols),
        **evaluate_last_cycle(final_model, test_feat, feature_cols),
    }
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    train_raw = d.load_cmapss(data_dir / f"train_{SUBSET}.txt")
    test_raw = d.load_cmapss(data_dir / f"test_{SUBSET}.txt")
    rul_true = d.load_rul(data_dir / f"RUL_{SUBSET}.txt")
    train_lab = d.add_rul(train_raw, cap=d.RUL_CAP)
    test_lab = d.add_test_rul(test_raw, rul_true, cap=d.RUL_CAP)

    metrics: dict = {
        "subset": SUBSET,
        "seed": m.RANDOM_SEED,
        "rul_cap": d.RUL_CAP,
        "dataset": {
            "train_rows": int(len(train_raw)),
            "train_engines": int(train_raw["unit"].nunique()),
            "test_rows": int(len(test_raw)),
            "test_engines": int(test_raw["unit"].nunique()),
        },
        "fd001_reference_test": {"rmse": 19.61, "nasa_score": 966.678},
    }
    for name, normalize in (("baseline", False), ("condition_normalized", True)):
        t = time.perf_counter()
        metrics[name] = run_variant(train_lab, test_lab, normalize)
        metrics[name]["seconds"] = round(time.perf_counter() - t, 2)
        print(f"{name}: {json.dumps(metrics[name]['test_official_last_cycle'])}")
    metrics["total_seconds"] = round(time.perf_counter() - t0, 2)

    (results_dir / "fd002_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
