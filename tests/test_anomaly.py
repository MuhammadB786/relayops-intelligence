"""Tests for src.anomaly: healthy mask, detector training/scoring, lead times."""

import numpy as np
import pandas as pd
import pytest

from src import anomaly as an


def test_healthy_mask_takes_first_fraction_of_each_engine():
    df = pd.DataFrame({
        "unit": [1] * 10 + [2] * 4,
        "cycle": list(range(1, 11)) + list(range(1, 5)),
    })
    mask = an.healthy_mask(df, fraction=0.3)
    assert mask[df["unit"] == 1].tolist() == [True] * 3 + [False] * 7
    assert mask[df["unit"] == 2].tolist() == [True] + [False] * 3


def test_anomaly_feature_columns_drops_cycle_only():
    cols = ["cycle", "sensor_2", "sensor_2_rmean5"]
    assert an.anomaly_feature_columns(cols) == ["sensor_2", "sensor_2_rmean5"]


def _cluster_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sensor_2": rng.normal(0.0, 1.0, n),
        "sensor_3": rng.normal(10.0, 0.5, n),
    })


def test_detector_scores_shape_and_determinism():
    feats = _cluster_frame()
    det_a = an.train_detector(feats, random_state=42)
    det_b = an.train_detector(feats, random_state=42)
    scores_a = an.anomaly_scores(det_a, feats)
    scores_b = an.anomaly_scores(det_b, feats)
    assert scores_a.shape == (len(feats),)
    assert np.all(np.isfinite(scores_a))
    np.testing.assert_allclose(scores_a, scores_b)


def test_detector_scores_outliers_as_more_anomalous():
    feats = _cluster_frame()
    det = an.train_detector(feats)
    inliers = pd.DataFrame({"sensor_2": [0.1, -0.2], "sensor_3": [10.1, 9.9]})
    outliers = pd.DataFrame({"sensor_2": [25.0, -30.0], "sensor_3": [40.0, -5.0]})
    assert an.anomaly_scores(det, outliers).min() > an.anomaly_scores(det, inliers).max()


def test_score_percentiles_bounds_and_ordering():
    ref = np.arange(100, dtype=float)          # reference scores 0..99
    pct = an.score_percentiles(ref, np.array([-5.0, 49.5, 99.0, 500.0]))
    assert pct[0] == 0.0                        # below all reference scores
    assert pct[1] == pytest.approx(50.0)        # 50 reference scores <= 49.5
    assert pct[2] == 100.0                      # ties resolve to upper rank
    assert pct[3] == 100.0                      # above all reference scores


def test_percentile_thresholds_match_numpy():
    ref = np.arange(101, dtype=float)
    th = an.percentile_thresholds(ref, (50.0, 99.0))
    assert th[50.0] == pytest.approx(50.0)
    assert th[99.0] == pytest.approx(99.0)


def test_first_crossing_lead_times_detects_and_misses():
    frame = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2],
        "rul_raw": [30.0, 20.0, 10.0, 15.0, 5.0],
    })
    scores = np.array([0.1, 0.9, 0.95, 0.2, 0.3])   # engine 2 never crosses 0.5
    out = an.first_crossing_lead_times(frame, scores, threshold=0.5)
    e1 = out[out["unit"] == 1].iloc[0]
    e2 = out[out["unit"] == 2].iloc[0]
    assert bool(e1["detected"]) and e1["lead_time"] == 20.0
    assert not bool(e2["detected"]) and np.isnan(e2["lead_time"])
