"""M4 drift evaluation: does the PSI retraining trigger behave?

Populations are **cycle-matched**: every engine contributes cycles
11-38 — past the 10-cycle warm-up, and inside the first 30% of life
for every FD001 training engine (the shortest life is 128 cycles), so
the rows are healthy-population by the project's own definition.
Cycle-matching matters because FD001 sensors trend with absolute
cycle: windows defined as "first 30% of life" mix different absolute
cycle ranges for different engines, and their marginals differ by
construction. (Measured: with fraction-of-life windows, the unshifted
control below false-fires the trigger.)

Reference: cycles 11-38 of the 80 FD001 *training-split* engines. The
20 validation-split engines (ids divisible by 5) are held out of the
reference so the control is an honest held-out comparison. Cases, each
evaluated as two consecutive monitored windows (validation engines
split in half: ids ending 5, then ids ending 0):

- **Control** — the held-out cycles, unshifted. The trigger must NOT
  fire: same regime, same population definition, different engines.
- **Injected shift** — a bias of ``m`` reference-sigmas added to
  sensors 2, 4 and 11 in both windows (a sensor-recalibration /
  regime-change simulation), swept over m in {0.25, 0.5, 1.0}. The
  unit is the *reference distribution's* standard deviation for that
  sensor — the spread PSI actually compares against. (Within-engine
  noise sigmas would understate it: the pooled marginal also carries
  between-engine baseline variation, and a first sweep in those units
  produced window-dependent near-misses at the same physical shift.)
  How large a shift the trigger needs is reported as measured,
  including misses.

PSI is also computed on the health index itself (reference health vs
pooled-window health) as a secondary signal. For the control, a
variance decomposition is reported for the channels that flag:
FD001's corrected-speed sensors (9, 14) carry most of their pooled
variance *between* engines, so a window pooled from a different
engine subset has a genuinely different marginal on them with zero
drift in any engine — the decomposition shows that split directly.

Results are merged into ``results/m4_metrics.json``.

Run:  .venv/bin/python scripts/run_drift_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.artifacts import load_artifacts
from src import anomaly as an
from src import data as d
from src import drift as dr
from src import health as hl
from src import model as m

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

#: Cycle-matched window: past warm-up, inside the first 30% of life
#: for every FD001 engine (min training life = 128 cycles).
CYCLE_LO, CYCLE_HI = 11, 38

SHIFTED_SENSORS = ["sensor_2", "sensor_4", "sensor_11"]
SHIFT_MAGNITUDES_SIGMA = [0.25, 0.5, 1.0]


def bulk_health(artifacts, frame: pd.DataFrame) -> pd.DataFrame:
    """Health index for every row of a multi-engine frame.

    Uses the shipped bundle exactly as the API does (same features,
    same formula), computed in bulk instead of per HTTP call. Returns
    a frame with ``unit``, ``cycle`` and ``health`` columns (sorted by
    unit/cycle, as :func:`src.data.build_features` emits). Trailing
    features are the true ones only for cycles whose engine history
    back to cycle 1 is present in ``frame``.
    """
    features, _ = d.build_features(
        frame,
        windows=tuple(artifacts.card["windows"]),
        drop_cols=list(artifacts.card["dropped_constant_columns"]),
    )
    rul = m.predict_rul(artifacts.rul_model, features[artifacts.feature_columns])
    scores = an.anomaly_scores(
        artifacts.detector, features[artifacts.anomaly_feature_columns])
    pct = an.score_percentiles(artifacts.reference_scores, scores)
    out = features[["unit", "cycle"]].copy()
    out["health"] = hl.health_index(rul, pct)
    return out


def trigger_summary(result: dict) -> dict:
    return {
        "retraining_recommended": result["retraining_recommended"],
        "flagged_sensors_per_window": result["flagged_sensors_per_window"],
        "sustained_sensors": result["sustained_sensors"],
        "reasons": result["reasons"],
    }


def psi_summary(psis: dict[str, float]) -> dict:
    return {
        "max_psi": round(max(psis.values()), 4),
        "sensors_over_0.10": sorted(c for c, v in psis.items() if v >= 0.10),
        "sensors_over_0.25": sorted(c for c, v in psis.items() if v > 0.25),
    }


def main() -> None:
    started = time.time()
    artifacts = load_artifacts(ROOT / "models")
    train = d.load_cmapss(ROOT / "data" / "train_FD001.txt")

    is_val = train["unit"] % 5 == 0
    in_window = train["cycle"].between(CYCLE_LO, CYCLE_HI)
    reference = train[~is_val & in_window]
    val = train[is_val]
    # Segments start at cycle 1 so trailing features (and therefore
    # health) are the true ones; the PSI rows are the 11-38 slice.
    seg_a = val[(val["unit"] % 10 == 5) & (val["cycle"] <= CYCLE_HI)]
    seg_b = val[(val["unit"] % 10 == 0) & (val["cycle"] <= CYCLE_HI)]
    assert len(reference) and len(seg_a) and len(seg_b)

    ref_scored = bulk_health(artifacts, train[~is_val])
    ref_health = ref_scored.loc[
        ref_scored["cycle"].between(CYCLE_LO, CYCLE_HI), "health"]

    def evaluate(segments: list[pd.DataFrame]) -> tuple[dict, dict, float]:
        raw_windows = [s[s["cycle"] >= CYCLE_LO] for s in segments]
        result = dr.evaluate_windows(reference, raw_windows, d.SENSOR_COLS)
        health_values = pd.concat([
            bulk_health(artifacts, s).loc[lambda f: f["cycle"] >= CYCLE_LO,
                                           "health"]
            for s in segments
        ])
        health_psi = dr.population_stability_index(
            ref_health.to_numpy(), health_values.to_numpy())
        return result, result["window_psis"][-1], float(health_psi)

    report: dict = {
        "rule": {
            "psi_threshold": dr.PSI_THRESHOLD,
            "min_sensors": dr.MIN_SENSORS,
            "consecutive_windows": dr.CONSECUTIVE_WINDOWS,
            "bands": {"stable": "< 0.10", "moderate": "0.10-0.25",
                      "significant": "> 0.25"},
        },
        "reference": {
            "population": f"cycles {CYCLE_LO}-{CYCLE_HI} of the 80 "
                          "training-split FD001 engines (past warm-up; "
                          "inside the first 30% of life for every engine)",
            "rows": int(len(reference)),
        },
        "windows": {
            "definition": "same cycle range, held-out validation engines; "
                          "window A = engine ids ending 5, window B = ids "
                          "ending 0 (10 engines each)",
            "rows_a": int((seg_a["cycle"] >= CYCLE_LO).sum()),
            "rows_b": int((seg_b["cycle"] >= CYCLE_LO).sum()),
        },
    }

    # ---- control: unshifted held-out data must NOT fire
    result, last_psis, health_psi = evaluate([seg_a, seg_b])

    def variance_decomposition(frame: pd.DataFrame, col: str) -> dict:
        between = float(frame.groupby("unit")[col].mean().std(ddof=0))
        within = float(frame.groupby("unit")[col].std().mean())
        pooled_var = float(frame[col].var(ddof=0))
        return {
            "between_engine_std": round(between, 3),
            "within_engine_std": round(within, 3),
            "between_share_of_pooled_variance": round(
                between ** 2 / pooled_var, 3),
        }

    report["control"] = {
        "expected": "no retraining recommended",
        "trigger": trigger_summary(result),
        "last_window_psi": psi_summary(last_psis),
        "last_window_psi_by_sensor": {
            c: round(v, 4) for c, v in sorted(last_psis.items())
        },
        "health_index_psi": round(health_psi, 4),
        "variance_decomposition_reference": {
            c: variance_decomposition(reference, c)
            for c in ("sensor_9", "sensor_14", "sensor_3")
        },
    }

    # ---- injected shifts: how big must a shift be before it fires?
    ref_std = reference[d.SENSOR_COLS].std(ddof=0)
    shift_rows = []
    for mag in SHIFT_MAGNITUDES_SIGMA:
        shifted = []
        for seg in (seg_a, seg_b):
            s2 = seg.copy()
            for s in SHIFTED_SENSORS:
                s2[s] = s2[s] + mag * float(ref_std[s])
            shifted.append(s2)
        result, last_psis, health_psi = evaluate(shifted)
        shift_rows.append({
            "magnitude_in_reference_sigmas": mag,
            "shifted_sensors": SHIFTED_SENSORS,
            "bias_applied": {
                s: round(mag * float(ref_std[s]), 4)
                for s in SHIFTED_SENSORS
            },
            "psi_of_shifted_sensors_last_window": {
                s: round(last_psis[s], 4) for s in SHIFTED_SENSORS
            },
            "last_window_psi": psi_summary(last_psis),
            "health_index_psi": round(health_psi, 4),
            "trigger": trigger_summary(result),
        })
    report["injected_shift_sweep"] = shift_rows
    report["drift_eval_seconds"] = round(time.time() - started, 2)

    path = RESULTS / "m4_metrics.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing["drift_evaluation"] = report
    path.write_text(json.dumps(existing, indent=2))

    c = report["control"]
    print(f"control: trigger={c['trigger']['retraining_recommended']} "
          f"(expected False), max sensor PSI "
          f"{c['last_window_psi']['max_psi']}, health PSI "
          f"{c['health_index_psi']}")
    for row in shift_rows:
        print(f"shift {row['magnitude_in_reference_sigmas']} ref-sigma: "
              f"trigger={row['trigger']['retraining_recommended']}, "
              f"shifted-sensor PSIs "
              f"{row['psi_of_shifted_sensors_last_window']}, "
              f"health PSI {row['health_index_psi']}")
    print(f"merged drift_evaluation into results/m4_metrics.json "
          f"in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
