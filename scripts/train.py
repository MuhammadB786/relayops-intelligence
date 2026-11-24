"""End-to-end M1 training + evaluation for the RelayOps ML baseline.

Pipeline:
  1. Load FD001 train/test + RUL files from data/.
  2. Label RUL (piecewise cap 125), build trailing rolling features.
  3. Sanity stage: train on 80 engines, validate on the held-out 20
     (split strictly by engine id).
  4. Final stage: refit the same configuration on all 100 training
     engines and evaluate on the official test set — one prediction per
     test engine at its last observed cycle, against the uncapped true
     RUL from RUL_FD001.txt (the standard published protocol).
  5. Write results/metrics.json, results/test_predictions.csv and
     models/model.joblib.

Run:  python scripts/train.py [--data-dir data] [--out results]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as d
from src import evaluate as ev
from src import model as m

SUBSET = "FD001"
VAL_UNITS = {u for u in range(1, 101) if u % 5 == 0}  # 20 held-out engines


def evaluate_last_cycle(model, frame: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Metrics on one prediction per engine at its last cycle (raw truth)."""
    last = ev.last_cycle_rows(frame)
    pred = m.predict_rul(model, last[feature_cols])
    return {
        "engines": int(len(last)),
        "rmse": ev.rmse(last["rul_raw"], pred),
        "nasa_score": ev.nasa_score(last["rul_raw"], pred),
    }, last, pred


#: True-RUL horizons at which held-out run-to-failure engines are scored.
#: (Their last cycle has RUL 0 by construction, so last-cycle scoring would
#: be meaningless there; the official test engines are genuinely truncated,
#: which is why last-cycle scoring is correct for them.)
VAL_CHECKPOINTS = (100, 75, 50, 25, 10)


def evaluate_checkpoints(model, frame: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Metrics at fixed true-RUL checkpoints on held-out engines.

    Each held-out engine contributes exactly one row per checkpoint (its
    RUL decreases by 1 per cycle), so every engine is weighted equally.
    """
    rows = frame[frame["rul_raw"].isin(VAL_CHECKPOINTS)]
    pred = m.predict_rul(model, rows[feature_cols])
    return {
        "protocol": f"predictions at true RUL in {list(VAL_CHECKPOINTS)}",
        "points": int(len(rows)),
        "engines": int(rows["unit"].nunique()),
        "rmse": float(ev.rmse(rows["rul_raw"], pred)),
        "nasa_score": float(ev.nasa_score(rows["rul_raw"], pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()
    data_dir, results_dir, models_dir = (
        Path(args.data_dir), Path(args.results_dir), Path(args.models_dir))
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    train_raw = d.load_cmapss(data_dir / f"train_{SUBSET}.txt")
    test_raw = d.load_cmapss(data_dir / f"test_{SUBSET}.txt")
    rul_true = d.load_rul(data_dir / f"RUL_{SUBSET}.txt")

    train_lab = d.add_rul(train_raw, cap=d.RUL_CAP)
    test_lab = d.add_test_rul(test_raw, rul_true, cap=d.RUL_CAP)

    # Drop sensors/settings that never vary in the training data (FD001 has
    # several constant sensor channels); they carry no signal.
    dead = d.constant_columns(train_lab, d.SETTING_COLS + d.SENSOR_COLS)

    train_feat, feature_cols = d.build_features(train_lab, drop_cols=dead)
    test_feat, _ = d.build_features(test_lab, drop_cols=dead)

    metrics: dict = {
        "subset": SUBSET,
        "seed": m.RANDOM_SEED,
        "rul_cap": d.RUL_CAP,
        "windows": list(d.DEFAULT_WINDOWS),
        "dropped_constant_columns": dead,
        "n_features": len(feature_cols),
        "dataset": {
            "train_rows": int(len(train_raw)),
            "train_engines": int(train_raw["unit"].nunique()),
            "test_rows": int(len(test_raw)),
            "test_engines": int(test_raw["unit"].nunique()),
        },
    }

    # --- sanity stage: 80 train engines / 20 held-out engines -------------
    tr, va = d.split_by_engine(train_feat, VAL_UNITS)
    t = time.perf_counter()
    val_model = m.train_model(tr[feature_cols], tr["rul"])
    val_train_s = time.perf_counter() - t
    val_metrics = evaluate_checkpoints(val_model, va, feature_cols)
    metrics["validation_80_20_by_engine"] = {
        "train_engines": int(tr["unit"].nunique()),
        "val_engines": int(va["unit"].nunique()),
        "train_seconds": round(val_train_s, 2),
        **{k: round(v, 3) if isinstance(v, float) else v
           for k, v in val_metrics.items()},
    }

    # --- final stage: all 100 training engines -> official test set -------
    t = time.perf_counter()
    final_model = m.train_model(train_feat[feature_cols], train_feat["rul"])
    final_train_s = time.perf_counter() - t
    test_metrics, last, pred = evaluate_last_cycle(final_model, test_feat, feature_cols)
    metrics["test_official_last_cycle"] = {
        "train_engines": 100,
        "train_seconds": round(final_train_s, 2),
        **{k: round(v, 3) if isinstance(v, float) else v
           for k, v in test_metrics.items()},
    }
    metrics["total_seconds"] = round(time.perf_counter() - t0, 2)

    m.save_model(final_model, models_dir / "model.joblib")
    (results_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    preds = pd.DataFrame({
        "unit": last["unit"].to_numpy(),
        "cycle": last["cycle"].to_numpy(),
        "rul_true": last["rul_raw"].to_numpy(),
        "rul_pred": np.round(pred, 2),
        "error": np.round(pred - last["rul_raw"].to_numpy(), 2),
    })
    preds.to_csv(results_dir / "test_predictions.csv", index=False)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    worst = preds.reindex(preds["error"].abs().sort_values(ascending=False).index)
    print("\nWorst 5 test engines by |error|:")
    print(worst.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
