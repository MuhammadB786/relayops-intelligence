"""Tests for src.regimes: regime assignment and the regime ensemble.

All frames are tiny synthetics whose settings fall in three widely
separated clusters, so clustering outcomes are unambiguous and the
suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src import data as d
from src import model as m
from src import regimes as rg

CENTERS = np.array([
    [0.0, 0.0, 100.0],
    [30.0, 0.2, 80.0],
    [60.0, 0.4, 60.0],
])


def synthetic_raw(n_engines: int = 12, seed: int = 0) -> pd.DataFrame:
    """Run-to-failure-shaped frame; engines drift through setting clusters."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_engines + 1):
        life = int(rng.integers(25, 40))
        for cycle in range(1, life + 1):
            centre = CENTERS[(unit + cycle // 10) % len(CENTERS)]
            settings = centre + rng.normal(0, 0.5, size=3)
            sensors = rng.normal(50.0, 5.0, size=21) + cycle * 0.1
            rows.append([unit, cycle, *settings, *sensors])
    return pd.DataFrame(rows, columns=d.RAW_COLS)


def featured(seed: int = 0, n_engines: int = 12):
    lab = d.add_rul(synthetic_raw(n_engines=n_engines, seed=seed))
    assigner = rg.fit_regimes(lab, n_regimes=3)
    lab = rg.add_regime_column(lab, assigner)
    feat, feature_cols = d.build_features(lab)
    return lab, feat, feature_cols, assigner


# --- assignment -----------------------------------------------------------

def test_assignment_is_deterministic():
    lab = d.add_rul(synthetic_raw())
    a1 = rg.fit_regimes(lab, n_regimes=3)
    a2 = rg.fit_regimes(lab, n_regimes=3)
    assert np.allclose(a1.kmeans.cluster_centers_, a2.kmeans.cluster_centers_)
    assert np.array_equal(a1.assign(lab), a2.assign(lab))


def test_clustering_is_fit_on_train_rows_only():
    train = synthetic_raw(n_engines=8, seed=1)
    other = synthetic_raw(n_engines=8, seed=2) * 10.0  # wildly different scale
    assigner = rg.fit_regimes(train, n_regimes=3)

    # Centroids equal a by-hand KMeans fit on the training settings alone.
    scaler = StandardScaler().fit(train[d.SETTING_COLS])
    km = KMeans(n_clusters=3, random_state=m.RANDOM_SEED, n_init=10)
    km.fit(scaler.transform(train[d.SETTING_COLS]))
    assert np.allclose(np.sort(assigner.kmeans.cluster_centers_, axis=0),
                       np.sort(km.cluster_centers_, axis=0))

    # Assigning other rows never refits: centroids are bit-identical after.
    before = assigner.kmeans.cluster_centers_.copy()
    labels = assigner.assign(other)
    assert np.array_equal(assigner.kmeans.cluster_centers_, before)
    assert set(np.unique(labels)) <= set(range(3))


def test_unseen_rows_get_nearest_centroid():
    train = synthetic_raw(n_engines=8, seed=3)
    assigner = rg.fit_regimes(train, n_regimes=3)
    far = train.iloc[:1].copy()
    far[d.SETTING_COLS] = [[10_000.0, 10_000.0, 10_000.0]]  # outside the envelope
    (label,) = assigner.assign(far)
    assert 0 <= label < 3
    dist = assigner.distances(far)[0]
    assert label == int(np.argmin(dist))  # nearest centroid, by construction


def test_regime_feature_frame_matches_assignment():
    lab, feat, _, assigner = featured()
    extra = rg.regime_feature_frame(feat, assigner)
    assert list(extra.columns) == ["regime"] + [f"regime_dist_{k}" for k in range(3)]
    assert len(extra) == len(feat)
    assert np.array_equal(extra["regime"].to_numpy(), feat["regime"].to_numpy())
    # The assigned regime is the nearest centroid for every row.
    dist_cols = [f"regime_dist_{k}" for k in range(3)]
    assert (extra[dist_cols].to_numpy().argmin(axis=1) == extra["regime"]).all()


def test_regime_stats_counts_rows_and_engines():
    lab, _, _, _ = featured()
    stats = rg.regime_stats(lab)
    assert sum(s["rows"] for s in stats.values()) == len(lab)
    for s in stats.values():
        assert 0 < s["engines"] <= lab["unit"].nunique()


# --- ensemble ---------------------------------------------------------------

def test_ensemble_falls_back_for_sparse_regime():
    _, feat, feature_cols, _ = featured(n_engines=12)
    frame = feat.copy()
    # Regime 1 keeps rows from only 2 engines; regime 0 keeps the rest.
    sparse_units = frame["unit"].unique()[:2]
    frame["regime"] = np.where(frame["unit"].isin(sparse_units), 1, 0)

    ens = rg.RegimeEnsemble(min_engines=5).fit(frame, feature_cols, n_regimes=2)
    assert ens.fallback_regimes_ == {1}
    assert set(ens.regime_models_) == {0}
    assert ens.regime_stats_[1]["engines"] == 2

    # Fallback rows are served by exactly the global model's predictions.
    sparse_rows = frame[frame["regime"] == 1]
    global_only = m.train_model(frame[feature_cols], frame["rul"])
    assert np.allclose(ens.predict(sparse_rows, feature_cols),
                       m.predict_rul(global_only, sparse_rows[feature_cols]))
    assert ens.fallback_row_count(frame) == len(sparse_rows)


def test_ensemble_end_to_end_tiny():
    _, feat, feature_cols, _ = featured(n_engines=12)
    ens = rg.RegimeEnsemble(min_engines=3).fit(feat, feature_cols, n_regimes=3)
    pred = ens.predict(feat, feature_cols)
    assert pred.shape == (len(feat),)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()  # RUL predictions are clipped at 0
    # Every regime is accounted for: dedicated model or named fallback.
    assert set(ens.regime_models_) | ens.fallback_regimes_ == {0, 1, 2}
    assert not (set(ens.regime_models_) & ens.fallback_regimes_)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
