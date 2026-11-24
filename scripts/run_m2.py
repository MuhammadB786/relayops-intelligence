"""Milestone 2 end-to-end: anomaly detection, health index, alert thresholds.

Mirrors the M1 protocol exactly (same engine split, same models):
  Stage A (validation): RUL model + anomaly detector trained on 80
    engines; anomaly lead times, health-index sanity and the alert
    threshold sweep are measured on the 20 held-out engines. Operating
    points are CHOSEN here, by the stated cost model.
  Stage B (test): final models refit on all 100 training engines; the
    same measurements are repeated on the official test set, and the
    operating points chosen in Stage A are evaluated unchanged — no
    re-tuning on test.

All anomaly/health/alert evaluation excludes the first WARMUP cycles of
each engine (rolling-feature warm-up artifact; see src/anomaly.py).

Writes results/m2_metrics.json, results/alert_thresholds.csv and
results/anomaly_lead_times.csv. M1 artifacts are not touched.

Run:  python scripts/run_m2.py [--data-dir data] [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import alerts as al
from src import anomaly as an
from src import data as d
from src import health as hl
from src import model as m

VAL_UNITS = {u for u in range(1, 101) if u % 5 == 0}  # same split as M1
#: Cycles excluded everywhere (detector training, reference, evaluation):
#: rolling features are partially filled until the longest window fills,
#: which makes warm-up rows spuriously "anomalous" (see src/anomaly.py).
WARMUP = max(d.DEFAULT_WINDOWS)
THRESHOLD_PERCENTILES = (90.0, 95.0, 97.5, 99.0, 99.5)
HEALTHY_RUL_MIN = 60.0  # cycles with true RUL above this count as "healthy segment"
RUL_THRESHOLDS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110]
HEALTH_THRESHOLDS = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
COST_RATIOS = (10.0, 50.0, 100.0)


def round_floats(obj, ndigits: int = 4):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def anomaly_evaluation(scored: pd.DataFrame,
                       thresholds: dict[float, float]) -> list[dict]:
    """Lead-time / false-alarm tradeoff per threshold on a scored frame."""
    scores = scored["anomaly_score"].to_numpy()
    healthy_seg = (scored["rul_raw"] > HEALTHY_RUL_MIN).to_numpy()
    n_engines = scored["unit"].nunique()
    rows = []
    for pct, thr in thresholds.items():
        lead = an.first_crossing_lead_times(scored, scores, thr)
        fired_healthy = healthy_seg & (scores >= thr)
        rows.append({
            "reference_percentile": pct,
            "threshold": float(thr),
            "detection_rate": float(lead["detected"].mean()),
            "median_lead_time_detected": float(
                lead.loc[lead["detected"], "lead_time"].median()),
            "mean_lead_time_detected": float(
                lead.loc[lead["detected"], "lead_time"].mean()),
            "false_alarm_rate_healthy_cycles": float(np.mean(fired_healthy[healthy_seg])),
            "frac_engines_any_false_alarm": float(
                scored.loc[fired_healthy, "unit"].nunique() / n_engines),
        })
    return rows


def health_sanity(frame: pd.DataFrame, values: np.ndarray) -> dict:
    """Per-engine trend sanity for a per-cycle health-like series."""
    work = frame[["unit", "cycle"]].copy()
    work["value"] = np.asarray(values, dtype=float)
    rhos, slopes, end_below = [], [], 0
    for _, grp in work.groupby("unit", sort=True):
        grp = grp.sort_values("cycle")
        rho = stats.spearmanr(grp["cycle"], grp["value"]).statistic
        rhos.append(float(rho))
        slopes.append(float(np.polyfit(grp["cycle"], grp["value"], 1)[0]))
        if grp["value"].tail(10).mean() < grp["value"].head(10).mean():
            end_below += 1
    rhos, slopes = np.array(rhos), np.array(slopes)
    return {
        "engines": int(len(rhos)),
        "median_spearman_vs_cycle": float(np.median(rhos)),
        "mean_spearman_vs_cycle": float(np.mean(rhos)),
        "frac_engines_spearman_negative": float(np.mean(rhos < 0)),
        "frac_engines_slope_negative": float(np.mean(slopes < 0)),
        "frac_engines_end_below_start": float(end_below / len(rhos)),
    }


def score_frame(rul_model, detector, ref_scores: np.ndarray,
                frame: pd.DataFrame, feature_cols: list[str],
                anomaly_cols: list[str]) -> pd.DataFrame:
    """Add pred_rul / anomaly_score / anomaly_percentile / health to a frame."""
    out = frame[["unit", "cycle", "rul_raw"]].copy()
    out["pred_rul"] = m.predict_rul(rul_model, frame[feature_cols])
    out["anomaly_score"] = an.anomaly_scores(detector, frame[anomaly_cols])
    out["anomaly_percentile"] = an.score_percentiles(ref_scores, out["anomaly_score"])
    out["health"] = hl.health_index(out["pred_rul"], out["anomaly_percentile"])
    return out


def alert_rows(scored: pd.DataFrame, eval_set: str) -> list[dict]:
    """Threshold sweep rows for both alert signals on one scored frame."""
    truth = (scored["rul_raw"] <= al.HORIZON).to_numpy()
    rows = []
    for signal, values, thresholds in (
        ("predicted_rul", scored["pred_rul"].to_numpy(), RUL_THRESHOLDS),
        ("health_index", scored["health"].to_numpy(), HEALTH_THRESHOLDS),
    ):
        for row in al.sweep_thresholds(values, truth, thresholds, direction="below"):
            rows.append({"eval_set": eval_set, "signal": signal, **row})
    return rows


def choose_operating_point(rows: list[dict], signal: str,
                           cost_fn: float = al.COST_MISSED_FAILURE) -> dict:
    """Min-cost row for one signal (cost recomputed at the given FN cost)."""
    cands = [r for r in rows if r["signal"] == signal]
    best = min(cands, key=lambda r: r["fp"] * al.COST_FALSE_ALARM + r["fn"] * cost_fn)
    return {**best, "selection_cost_fn": cost_fn,
            "selection_expected_cost": best["fp"] * al.COST_FALSE_ALARM
            + best["fn"] * cost_fn}


def evaluate_fixed_point(scored: pd.DataFrame, signal_col: str, threshold: float) -> dict:
    """Precision/recall/cost of one fixed threshold on one scored frame."""
    truth = (scored["rul_raw"] <= al.HORIZON).to_numpy()
    conf = al.alert_confusion(truth, scored[signal_col].to_numpy() <= threshold)
    precision, recall = al.precision_recall(conf)
    return {"threshold": threshold, **conf, "precision": precision,
            "recall": recall, "expected_cost": al.expected_cost(conf)}


def engine_level_at_point(scored: pd.DataFrame, signal_col: str,
                          threshold: float) -> dict:
    """Per-engine view of one operating point (complements per-cycle metrics).

    An operator experiences alerts per asset, not per cycle: this reports
    in how many engines the alert fired at least once inside the failure
    window (detection) and in how many it fired at least once while the
    engine was still healthy (false-alarm engines).
    """
    fired = scored[signal_col].to_numpy() <= threshold
    in_window = (scored["rul_raw"] <= al.HORIZON).to_numpy()
    healthy = (scored["rul_raw"] > HEALTHY_RUL_MIN).to_numpy()
    n = scored["unit"].nunique()
    reached = scored.loc[in_window, "unit"].nunique()
    alerted_in_window = scored.loc[fired & in_window, "unit"].nunique()
    return {
        "engines_reaching_window": int(reached),
        "frac_engines_alerted_in_window_of_all": float(alerted_in_window / n),
        "frac_engines_alerted_in_window_of_reached": float(
            alerted_in_window / reached) if reached else 0.0,
        "frac_engines_false_alarm_healthy": float(
            scored.loc[fired & healthy, "unit"].nunique() / n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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

    metrics: dict = {
        "milestone": "M2",
        "seed": m.RANDOM_SEED,
        "healthy_fraction": an.HEALTHY_FRACTION,
        "warmup_cycles_excluded": WARMUP,
        "healthy_segment_rul_min": HEALTHY_RUL_MIN,
        "alert_horizon": al.HORIZON,
        "cost_model": {"false_alarm": al.COST_FALSE_ALARM,
                       "missed_failure": al.COST_MISSED_FAILURE},
        "threshold_percentiles": list(THRESHOLD_PERCENTILES),
    }

    # ---------------- Stage A: validation (80 train / 20 held-out) --------
    tr, va = d.split_by_engine(train_feat, VAL_UNITS)
    val_rul_model = m.train_model(tr[feature_cols], tr["rul"])
    tr_healthy = tr[an.healthy_mask(tr) & (tr["cycle"] > WARMUP)]
    val_detector = an.train_detector(tr_healthy[anomaly_cols])
    val_ref_scores = an.anomaly_scores(val_detector, tr_healthy[anomaly_cols])
    metrics["validation_reference_healthy_rows"] = int(len(tr_healthy))

    va_scored = score_frame(val_rul_model, val_detector, val_ref_scores,
                            va, feature_cols, anomaly_cols)
    va_eval = va_scored[va_scored["cycle"] > WARMUP]
    thresholds_a = an.percentile_thresholds(val_ref_scores, THRESHOLD_PERCENTILES)
    metrics["anomaly_validation"] = anomaly_evaluation(va_eval, thresholds_a)
    metrics["health_validation"] = {
        "composite": health_sanity(va_eval, va_eval["health"].to_numpy()),
        "rul_only": health_sanity(va_eval, hl.rul_component(va_eval["pred_rul"])),
        "anomaly_only": health_sanity(
            va_eval, hl.anomaly_component(va_eval["anomaly_percentile"])),
    }
    sweep_rows = alert_rows(va_eval, "validation")

    chosen = {
        signal: choose_operating_point(sweep_rows, signal)
        for signal in ("predicted_rul", "health_index")
    }
    metrics["alert_operating_points"] = {
        "chosen_on_validation": chosen,
        "cost_sensitivity": {
            str(int(ratio)): {
                signal: choose_operating_point(sweep_rows, signal, cost_fn=ratio)
                for signal in ("predicted_rul", "health_index")
            } for ratio in COST_RATIOS
        },
    }

    # ---------------- Stage B: final models -> official test set ----------
    final_rul_model = m.train_model(train_feat[feature_cols], train_feat["rul"])
    full_healthy = train_feat[
        an.healthy_mask(train_feat) & (train_feat["cycle"] > WARMUP)]
    final_detector = an.train_detector(full_healthy[anomaly_cols])
    test_ref_scores = an.anomaly_scores(final_detector, full_healthy[anomaly_cols])
    metrics["test_reference_healthy_rows"] = int(len(full_healthy))

    test_scored = score_frame(final_rul_model, final_detector, test_ref_scores,
                              test_feat, feature_cols, anomaly_cols)
    # Warm-up diagnostic (pre-filter): the artifact that motivated WARMUP.
    cyc = test_scored["cycle"]
    metrics["warmup_diagnostic"] = {
        "mean_anomaly_score_cycle_1": float(
            test_scored.loc[cyc == 1, "anomaly_score"].mean()),
        "mean_anomaly_score_cycles_20_30": float(
            test_scored.loc[(cyc >= 20) & (cyc <= 30), "anomaly_score"].mean()),
        "frac_cycle_1_above_ref_97_5th_pct": float(np.mean(
            test_scored.loc[cyc == 1, "anomaly_score"].to_numpy()
            >= np.percentile(test_ref_scores, 97.5))),
    }
    test_eval = test_scored[test_scored["cycle"] > WARMUP]
    thresholds_b = an.percentile_thresholds(test_ref_scores, THRESHOLD_PERCENTILES)
    metrics["anomaly_test"] = anomaly_evaluation(test_eval, thresholds_b)
    metrics["health_test"] = {
        "composite": health_sanity(test_eval, test_eval["health"].to_numpy()),
        "rul_only": health_sanity(test_eval, hl.rul_component(test_eval["pred_rul"])),
        "anomaly_only": health_sanity(
            test_eval, hl.anomaly_component(test_eval["anomaly_percentile"])),
    }
    sweep_rows += alert_rows(test_eval, "test")

    metrics["alert_operating_points"]["chosen_evaluated_on_test"] = {
        "predicted_rul": evaluate_fixed_point(
            test_eval, "pred_rul", chosen["predicted_rul"]["threshold"]),
        "health_index": evaluate_fixed_point(
            test_eval, "health", chosen["health_index"]["threshold"]),
    }
    metrics["alert_operating_points"]["engine_level_on_test"] = {
        "predicted_rul": engine_level_at_point(
            test_eval, "pred_rul", chosen["predicted_rul"]["threshold"]),
        "health_index": engine_level_at_point(
            test_eval, "health", chosen["health_index"]["threshold"]),
    }

    # M1's documented dangerous overestimates, checked at the chosen points.
    check = {}
    for unit in (41, 37):
        eng = test_eval[test_eval["unit"] == unit].sort_values("cycle")
        last = eng.iloc[-1]
        in_window = eng[eng["rul_raw"] <= al.HORIZON]
        check[f"engine_{unit}"] = {
            "true_rul_at_last_cycle": float(last["rul_raw"]),
            "pred_rul_at_last_cycle": float(last["pred_rul"]),
            "health_at_last_cycle": float(last["health"]),
            "rul_alert_at_last_cycle": bool(
                last["pred_rul"] <= chosen["predicted_rul"]["threshold"]),
            "health_alert_at_last_cycle": bool(
                last["health"] <= chosen["health_index"]["threshold"]),
            "rul_alert_any_cycle_in_window": bool(
                (in_window["pred_rul"] <= chosen["predicted_rul"]["threshold"]).any()),
            "health_alert_any_cycle_in_window": bool(
                (in_window["health"] <= chosen["health_index"]["threshold"]).any()),
        }
    metrics["dangerous_overestimate_check"] = check

    metrics["total_seconds"] = round(time.perf_counter() - t0, 2)
    metrics = round_floats(metrics)

    (results_dir / "m2_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(sweep_rows).to_csv(results_dir / "alert_thresholds.csv", index=False)
    lead_rows = (
        [dict(r, eval_set="validation") for r in metrics["anomaly_validation"]]
        + [dict(r, eval_set="test") for r in metrics["anomaly_test"]]
    )
    pd.DataFrame(lead_rows).to_csv(results_dir / "anomaly_lead_times.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
