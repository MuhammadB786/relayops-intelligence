"""M6: FD002 regime models — the follow-up M2 named but did not run.

M2's condition normalization improved validation and failed on the
official test (27.72 / 13,666 vs the global baseline's 27.60 / 8,658),
and its write-up named the next experiments: per-condition models, and
condition identity as an explicit model feature. This script runs both,
against the unchanged global baseline, under strict selection
discipline:

  Candidates (all use the exact M1/M2 feature pipeline — cap-125
  labels, trailing rolling features, engine-id splits):
    global            — the M2 baseline, rerun as the control;
    global_plus_regime (A) — global model + regime id and per-centroid
                        distance features (src.regimes);
    per_regime        (B) — one model per operating regime, global
                        fallback for regimes with < 10 training
                        engines (src.regimes.RegimeEnsemble).

  Selection rule, fixed before any test number exists: the candidate
  with the lowest VALIDATION RMSE under M1's fixed-horizon checkpoint
  protocol (engine ids divisible by 5 held out; predictions at true
  RUL in [100, 75, 50, 25, 10]) is refit on all 260 training engines
  and scored ONCE on the official test set (last cycle, uncapped
  truth). The other candidates never touch the test set. Validation
  NASA scores are reported for all three but do not select — the same
  primary metric M1/M2 headlined.

Regime clustering (KMeans k=6 on the 3 settings, standardised) is fit
on training rows only at each stage — on the split's training engines
for validation, on all training engines for the final refit — and
applied to held-out/test rows by nearest-centroid assignment. Nothing
fitted sees a held-out row.

Writes results/fd002_regime_metrics.json. M1/M2 artifacts are not
touched; the reference numbers are read from results/fd002_metrics.json.

Run:  python scripts/run_fd002_regimes.py [--data-dir data] [--results-dir results]
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
from src import regimes as rg

SUBSET = "FD002"
VAL_MODULO = 5
VAL_CHECKPOINTS = (100, 75, 50, 25, 10)


# --- evaluation helpers (same protocols as run_fd002.py) ----------------

def checkpoint_metrics(predict_fn, frame: pd.DataFrame) -> dict:
    rows = frame[frame["rul_raw"].isin(VAL_CHECKPOINTS)]
    pred = predict_fn(rows)
    return {
        "protocol": f"predictions at true RUL in {list(VAL_CHECKPOINTS)}",
        "points": int(len(rows)),
        "engines": int(rows["unit"].nunique()),
        "rmse": float(ev.rmse(rows["rul_raw"], pred)),
        "nasa_score": float(ev.nasa_score(rows["rul_raw"], pred)),
    }


def tail_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-engine error tail, in the form M2's write-up used."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    terms = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    total = float(terms.sum())
    top5 = float(np.sort(terms)[::-1][:5].sum()) if len(terms) >= 5 else total
    return {
        "engines_late_over_30": int((err > 30).sum()),
        "engines_early_over_30": int((err < -30).sum()),
        "max_late_error": float(err.max()),
        "max_early_error": float(err.min()),
        "top5_engine_score_share": top5 / total if total > 0 else 0.0,
    }


def last_cycle_eval(predict_fn, frame: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    last = ev.last_cycle_rows(frame)
    pred = predict_fn(last)
    metrics = {
        "engines": int(len(last)),
        "rmse": float(ev.rmse(last["rul_raw"], pred)),
        "nasa_score": float(ev.nasa_score(last["rul_raw"], pred)),
        **tail_analysis(last["rul_raw"].to_numpy(), pred),
    }
    return metrics, last, pred


# --- candidate plumbing ---------------------------------------------------

def prepare(train_raw: pd.DataFrame, other_raw: pd.DataFrame):
    """Fit regimes on train rows only; build features for both frames.

    Returns (assigner, train_feat, other_feat, feature_cols, extra_cols)
    where extra_cols are the regime-feature columns for approach A.
    """
    assigner = rg.fit_regimes(train_raw)
    tr = rg.add_regime_column(train_raw, assigner)
    ot = rg.add_regime_column(other_raw, assigner)
    dead = d.constant_columns(tr, d.SETTING_COLS + d.SENSOR_COLS)
    tr_feat, feature_cols = d.build_features(tr, drop_cols=dead)
    ot_feat, _ = d.build_features(ot, drop_cols=dead)
    extra_tr = rg.regime_feature_frame(tr_feat, assigner)
    extra_ot = rg.regime_feature_frame(ot_feat, assigner)
    for col in extra_tr.columns:
        tr_feat[col] = extra_tr[col].to_numpy()
        ot_feat[col] = extra_ot[col].to_numpy()
    extra_cols = list(extra_tr.columns)
    return assigner, tr_feat, ot_feat, feature_cols, extra_cols


def fit_candidate(name: str, tr_feat: pd.DataFrame, feature_cols: list[str],
                  extra_cols: list[str]):
    """Return a predict function over a feature-built frame."""
    if name == "global":
        model_g = m.train_model(tr_feat[feature_cols], tr_feat["rul"])
        return lambda frame: m.predict_rul(model_g, frame[feature_cols]), {}
    if name == "global_plus_regime":
        cols = feature_cols + extra_cols
        model_a = m.train_model(tr_feat[cols], tr_feat["rul"])
        return lambda frame: m.predict_rul(model_a, frame[cols]), {}
    if name == "per_regime":
        ens = rg.RegimeEnsemble(min_engines=rg.MIN_REGIME_ENGINES)
        ens.fit(tr_feat, feature_cols, n_regimes=rg.N_REGIMES)
        info = {
            "regime_stats": ens.regime_stats_,
            "fallback_regimes": sorted(ens.fallback_regimes_),
        }
        return lambda frame: ens.predict(frame, feature_cols), info
    raise ValueError(name)


CANDIDATES = ("global", "global_plus_regime", "per_regime")


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

    ref = json.loads((results_dir / "fd002_metrics.json").read_text())
    metrics: dict = {
        "subset": SUBSET,
        "seed": m.RANDOM_SEED,
        "rul_cap": d.RUL_CAP,
        "n_regimes": rg.N_REGIMES,
        "min_regime_engines": rg.MIN_REGIME_ENGINES,
        "selection_rule": (
            "lowest validation RMSE (fixed-horizon checkpoint protocol); "
            "winner refit on all training engines and scored once on test"
        ),
        "references": {
            "global_baseline_test": ref["baseline"]["test_official_last_cycle"],
            "condition_normalized_test": ref["condition_normalized"]["test_official_last_cycle"],
            "global_baseline_validation": ref["baseline"]["validation"],
        },
        "dataset": {
            "train_rows": int(len(train_raw)),
            "train_engines": int(train_raw["unit"].nunique()),
            "test_rows": int(len(test_raw)),
            "test_engines": int(test_raw["unit"].nunique()),
        },
    }

    # Stage 1: validation — all three candidates, test set untouched.
    val_units = {u for u in train_lab["unit"].unique() if u % VAL_MODULO == 0}
    tr_raw, va_raw = d.split_by_engine(train_lab, val_units)
    _, tr_feat, va_feat, feature_cols, extra_cols = prepare(tr_raw, va_raw)
    metrics["validation_regime_stats"] = rg.regime_stats(tr_feat)
    metrics["candidates"] = {}
    predict_fns = {}
    for name in CANDIDATES:
        t = time.perf_counter()
        predict_fns[name], info = fit_candidate(name, tr_feat, feature_cols, extra_cols)
        entry = {
            "train_engines": int(tr_raw["unit"].nunique()),
            "val_engines": int(va_raw["unit"].nunique()),
            **checkpoint_metrics(predict_fns[name], va_feat),
            "seconds": round(time.perf_counter() - t, 2),
        }
        entry.update(info)
        metrics["candidates"][name] = {"validation": entry}
        print(f"validation {name}: rmse={entry['rmse']:.4f} "
              f"nasa={entry['nasa_score']:.1f}")

    # Selection — by validation RMSE only, per the predeclared rule.
    selected = min(CANDIDATES,
                   key=lambda n: metrics["candidates"][n]["validation"]["rmse"])
    metrics["selected"] = selected
    print(f"selected by validation RMSE: {selected}")

    # Stage 2: refit the winner on all training engines; score test ONCE.
    t = time.perf_counter()
    _, full_feat, test_feat, feature_cols, extra_cols = prepare(train_lab, test_lab)
    predict_fn, info = fit_candidate(selected, full_feat, feature_cols, extra_cols)
    test_metrics, last_rows, _ = last_cycle_eval(predict_fn, test_feat)
    test_block = {
        "train_engines": int(train_lab["unit"].nunique()),
        "n_features": len(feature_cols) + (len(extra_cols) if selected == "global_plus_regime" else 0),
        **test_metrics,
        "seconds": round(time.perf_counter() - t, 2),
    }
    test_block.update(info)
    if selected == "per_regime":
        test_block["test_last_cycle_rows_served_by_fallback"] = int(
            last_rows["regime"].isin(info["fallback_regimes"]).sum())
    metrics["candidates"][selected]["test_official_last_cycle"] = test_block
    metrics["final_regime_stats"] = rg.regime_stats(full_feat)
    print(f"test {selected}: {json.dumps(test_block)}")

    metrics["total_seconds"] = round(time.perf_counter() - t0, 2)
    (results_dir / "fd002_regime_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote results/fd002_regime_metrics.json in {metrics['total_seconds']}s")


if __name__ == "__main__":
    main()
