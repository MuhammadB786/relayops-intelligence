"""Anomaly detection: IsolationForest trained on healthy engine life only.

Design (see README "Milestone 2"):
- The detector is trained ONLY on the first ``HEALTHY_FRACTION`` of each
  training engine's life. Early life is the one period we can label
  "healthy" without hindsight about the fault: the degradation process in
  C-MAPSS develops over the whole trajectory, so the opening fraction of
  every run-to-failure engine is as close to a clean reference population
  as this dataset offers. 0.30 keeps even the shortest-lived FD001 engine
  (128 cycles) contributing ~38 reference cycles, while staying well clear
  of the late-life regime where the fault dominates the sensors.
- Features are the M1 trailing rolling features minus ``cycle``. Engine
  age is excluded on purpose: a detector that can see age learns "old =
  anomalous" and its lead times become meaningless. This one must react
  to sensor behaviour, not to the clock.
- Scores are ``-score_samples``: higher means more anomalous. Alert
  thresholds are never taken from the fitted ``contamination`` guess;
  they are set afterwards from percentiles of the healthy reference
  scores, which makes the false-alarm rate an explicit, measured choice.
- Everything is causal: rolling features are trailing (M1), and scoring a
  cycle uses only that cycle's feature row.
- Warm-up: during the first ``max(window)`` cycles of an engine's life the
  rolling features are computed from partially filled windows (at cycle 1
  every rolling std is exactly 0), which lands those rows in a corner of
  feature space the detector scores as highly anomalous regardless of
  sensor health — measured in M2: mean cycle-1 score 0.61 vs a settled
  baseline of ~0.44, with 100% of engines above the reference 97.5th
  percentile at cycle 1. Scores from warm-up cycles are therefore not
  meaningful; the M2 evaluation excludes them everywhere (reference,
  training and scoring) and a serving deployment should report "warming
  up" for an asset's first 10 cycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.model import RANDOM_SEED

#: Fraction of each training engine's life treated as the healthy reference.
HEALTHY_FRACTION = 0.30

#: Features never given to the detector (see module docstring).
EXCLUDED_FEATURES = ("cycle",)


def healthy_mask(df: pd.DataFrame, fraction: float = HEALTHY_FRACTION) -> pd.Series:
    """Boolean mask: rows in the first ``fraction`` of their engine's life.

    Life length is the engine's maximum observed cycle, so the mask is
    ``cycle <= fraction * max_cycle`` per engine. For truncated (test)
    engines this uses the *observed* length; the mask is only used to
    select training reference data, where engines run to failure.
    """
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return df["cycle"] <= fraction * max_cycle


def anomaly_feature_columns(feature_cols: list[str]) -> list[str]:
    """The M1 feature columns the detector uses (``cycle`` removed)."""
    return [c for c in feature_cols if c not in EXCLUDED_FEATURES]


def train_detector(
    features: pd.DataFrame, random_state: int = RANDOM_SEED
) -> IsolationForest:
    """Fit an IsolationForest on (healthy) feature rows.

    ``contamination`` is set to a small nominal value only because the
    estimator requires one; no threshold is ever derived from it.
    """
    detector = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=0.01,
        random_state=random_state,
    )
    detector.fit(features)
    return detector


def anomaly_scores(detector: IsolationForest, features: pd.DataFrame) -> np.ndarray:
    """Anomaly score per row: higher = more anomalous."""
    return -detector.score_samples(features)


def score_percentiles(reference_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Percentile (0-100) of each score within a reference distribution.

    Empirical CDF against the healthy reference scores: a score at the
    99th percentile is more anomalous than 99% of healthy reference
    cycles. Ties resolve to the upper rank (``side="right"``), so a score
    equal to the reference maximum maps to 100.
    """
    ref = np.sort(np.asarray(reference_scores, dtype=float))
    ranks = np.searchsorted(ref, np.asarray(scores, dtype=float), side="right")
    return 100.0 * ranks / len(ref)


def percentile_thresholds(
    reference_scores: np.ndarray, percentiles: tuple[float, ...]
) -> dict[float, float]:
    """Score thresholds at the given percentiles of the reference scores."""
    return {p: float(np.percentile(reference_scores, p)) for p in percentiles}


def first_crossing_lead_times(
    frame: pd.DataFrame, scores: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Per engine: lead time of the first threshold crossing.

    ``frame`` must carry ``unit`` and ``rul_raw`` and be row-aligned with
    ``scores``. Lead time is the true RUL at the first cycle whose score
    reaches ``threshold`` — i.e. how many cycles before failure the
    detector first fired. Engines that never cross get ``detected=False``
    and a NaN lead time (they are misses, not zero-lead detections).
    """
    work = frame[["unit", "rul_raw"]].copy()
    work["score"] = np.asarray(scores, dtype=float)
    work["crossed"] = work["score"] >= threshold
    rows = []
    for unit, grp in work.groupby("unit", sort=True):
        hits = grp[grp["crossed"]]
        if len(hits):
            rows.append({
                "unit": int(unit),
                "detected": True,
                "lead_time": float(hits["rul_raw"].iloc[0]),
            })
        else:
            rows.append({"unit": int(unit), "detected": False, "lead_time": np.nan})
    return pd.DataFrame(rows)
