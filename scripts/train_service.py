"""Train the deployable M3 service artifacts, end to end, in one command.

Trains exactly the components the API serves, on all 100 FD001 training
engines (the M1/M2 "final stage" protocol — no validation split here;
model selection already happened in M1/M2):

  1. M1 RUL model (HistGradientBoostingRegressor, seed 42), then scored
     on the official test protocol so the model card's headline metrics
     describe *this* artifact, not a sibling run.
  2. M2 anomaly detector (IsolationForest) on healthy early life only
     (first 30% of each engine, warm-up cycles excluded), plus the
     healthy reference scores the health index's percentile needs.
  3. The bundle is written to models/ via api.artifacts.save_artifacts:
     rul_model.joblib, anomaly_detector.joblib, reference_scores.npy and
     model_card.json (features, training summary, metrics, the M2 alert
     operating point, component versions, SHA-256 per artifact file).

Deterministic (seed 42 throughout). The M2 alert operating point and its
evaluation are *copied* from results/m2_metrics.json when present — the
decision was made there, on validation, and is not re-derived here.

Run:  python scripts/train_service.py [--data-dir data] [--models-dir models]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.artifacts import save_artifacts
from src import anomaly as an
from src import data as d
from src import evaluate as ev
from src import model as m

#: Service release trained by this script.
MODEL_VERSION = "0.3.0"

#: Alert levels served by the API. The critical line is the M2 decision
#: (health <= 50, chosen on validation under a 50:1 cost model). The watch
#: band above it is a presentation tier only — it was not part of the M2
#: precision/recall evaluation, and the card says so.
CRITICAL_MAX_HEALTH = 50.0
WATCH_MAX_HEALTH = 70.0

WARMUP = max(d.DEFAULT_WINDOWS)  # same exclusion as M2


def alert_card_section(results_dir: Path) -> dict:
    """The alert block for the model card, with M2 provenance attached."""
    section: dict = {
        "signal": "health_index",
        "critical_max_health": CRITICAL_MAX_HEALTH,
        "watch_max_health": WATCH_MAX_HEALTH,
        "levels": {
            "critical": f"health <= {CRITICAL_MAX_HEALTH:g}",
            "watch": f"{CRITICAL_MAX_HEALTH:g} < health <= {WATCH_MAX_HEALTH:g}",
            "none": f"health > {WATCH_MAX_HEALTH:g}",
        },
        "watch_band_note": (
            "The watch band is a presentation tier for operators; the "
            "evaluated M2 decision is the critical rule (health <= 50) only."
        ),
        "operating_point": {
            "chosen_in": "M2 (scripts/run_m2.py)",
            "chosen_on": "validation engines only, 50:1 missed-failure : "
                         "false-alarm cost model",
            "decision": "alert when health index <= 50",
        },
    }
    m2_path = results_dir / "m2_metrics.json"
    if m2_path.is_file():
        m2 = json.loads(m2_path.read_text())
        points = m2["alert_operating_points"]
        section["operating_point"]["validation"] = points[
            "chosen_on_validation"]["health_index"]
        section["operating_point"]["test_evaluated_unchanged"] = points[
            "chosen_evaluated_on_test"]["health_index"]
    else:
        section["operating_point"]["metrics_note"] = (
            "results/m2_metrics.json not found at training time; run "
            "scripts/run_m2.py to attach the operating point's evaluation."
        )
    return section


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    data_dir, models_dir, results_dir = (
        Path(args.data_dir), Path(args.models_dir), Path(args.results_dir))

    t0 = time.perf_counter()
    train_raw = d.load_cmapss(data_dir / "train_FD001.txt")
    test_raw = d.load_cmapss(data_dir / "test_FD001.txt")
    rul_true = d.load_rul(data_dir / "RUL_FD001.txt")
    train_lab = d.add_rul(train_raw, cap=d.RUL_CAP)
    test_lab = d.add_test_rul(test_raw, rul_true, cap=d.RUL_CAP)
    dead = d.constant_columns(train_lab, d.SETTING_COLS + d.SENSOR_COLS)
    train_feat, feature_cols = d.build_features(train_lab, drop_cols=dead)
    test_feat, _ = d.build_features(test_lab, drop_cols=dead)
    anomaly_cols = an.anomaly_feature_columns(feature_cols)

    # --- M1 RUL model: all 100 engines, official test protocol ----------
    rul_model = m.train_model(train_feat[feature_cols], train_feat["rul"])
    last = ev.last_cycle_rows(test_feat)
    test_pred = m.predict_rul(rul_model, last[feature_cols])
    test_rmse = ev.rmse(last["rul_raw"], test_pred)
    test_nasa = ev.nasa_score(last["rul_raw"], test_pred)

    # Cross-check against M1's recorded metrics: same data, seed and
    # configuration must reproduce them.
    m1_path = results_dir / "metrics.json"
    if m1_path.is_file():
        recorded = json.loads(m1_path.read_text())["test_official_last_cycle"]
        assert abs(test_rmse - recorded["rmse"]) < 0.01, (
            f"service model test RMSE {test_rmse:.3f} != M1 recorded "
            f"{recorded['rmse']} — training configuration has drifted")
        assert abs(test_nasa - recorded["nasa_score"]) < 1.0, (
            f"service model NASA score {test_nasa:.1f} != M1 recorded "
            f"{recorded['nasa_score']} — training configuration has drifted")

    # --- M2 anomaly detector: healthy early life only --------------------
    healthy = train_feat[an.healthy_mask(train_feat) & (train_feat["cycle"] > WARMUP)]
    detector = an.train_detector(healthy[anomaly_cols])
    reference_scores = an.anomaly_scores(detector, healthy[anomaly_cols])

    total_s = round(time.perf_counter() - t0, 2)

    card_fields = {
        "model_version": MODEL_VERSION,
        "milestone": "M3",
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_seconds": total_s,
        "seed": m.RANDOM_SEED,
        "feature_columns": feature_cols,
        "anomaly_feature_columns": anomaly_cols,
        "dropped_constant_columns": dead,
        "windows": list(d.DEFAULT_WINDOWS),
        "rul_cap": d.RUL_CAP,
        "warmup_cycles": WARMUP,
        "min_history_cycles": WARMUP + 1,
        "training_data": {
            "subset": "FD001",
            "train_rows": int(len(train_raw)),
            "train_engines": int(train_raw["unit"].nunique()),
            "healthy_reference_rows": int(len(healthy)),
            "healthy_fraction": an.HEALTHY_FRACTION,
        },
        "metrics": {
            "protocol": "official FD001: one prediction per test engine at "
                        "its last observed cycle vs uncapped true RUL",
            "test_rmse": round(test_rmse, 3),
            "test_nasa_score": round(test_nasa, 3),
            "test_engines": int(len(last)),
        },
        "alert": alert_card_section(results_dir),
        "components": {
            "rul_model": {
                "type": "HistGradientBoostingRegressor",
                "params": {
                    k: rul_model.get_params()[k]
                    for k in ("max_iter", "learning_rate", "max_leaf_nodes",
                              "min_samples_leaf", "l2_regularization",
                              "random_state")
                },
            },
            "anomaly_detector": {
                "type": "IsolationForest",
                "params": {
                    k: detector.get_params()[k]
                    for k in ("n_estimators", "contamination", "random_state")
                },
                "trained_on": "first 30% of each training engine's life, "
                              "cycles > warm-up only",
            },
            "health_index": {
                "formula": "0.6 * (100 * clip(pred_rul / 125, 0, 1)) + "
                           "0.4 * (100 - anomaly_percentile)",
                "w_rul": 0.6,
            },
            "libraries": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
        "validity": "FD001 regime only (one operating condition, one fault "
                    "mode); FD002 degraded materially in M2 — see README.",
    }

    card = save_artifacts(
        models_dir,
        rul_model=rul_model,
        detector=detector,
        reference_scores=reference_scores,
        card_fields=card_fields,
    )
    print(f"service artifacts written to {models_dir}/ in {total_s:.1f}s")
    print(f"  model_version   {card['model_version']}")
    print(f"  features        {len(feature_cols)} "
          f"({len(anomaly_cols)} anomaly)")
    print(f"  test RMSE       {test_rmse:.2f}   NASA score {test_nasa:.1f}")
    print(f"  bundle checksum {card['bundle_checksum'][:16]}…")


if __name__ == "__main__":
    main()
