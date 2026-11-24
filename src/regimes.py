"""Operating-regime identification and regime-conditional RUL models (M6).

FD002's engines run under six operating conditions. M2's remedy —
z-scoring sensors within KMeans conditions — improved validation but
failed on the official test, and the M2 write-up named the follow-up
experiments: per-condition models, and condition identity as an
explicit model feature. This module implements the machinery for both:

- ``RegimeAssigner`` clusters the 3 operating-setting columns into
  regimes (KMeans, k=6 by default, the documented FD002 condition
  count). It is fit on training rows ONLY; assignment of any later row
  is a nearest-centroid lookup in standardised setting space, so rows
  outside the training envelope — an unseen combination, an edge
  operating point — still get the nearest regime. Assignment never
  refits anything.
- ``regime_feature_frame`` exposes regime identity to a single global
  model: the regime id plus the row's distance to every centroid (a
  soft membership profile — a row near a boundary looks different from
  one deep inside a regime, and a tree can use that).
- ``RegimeEnsemble`` trains one gradient-boosting model per regime on
  that regime's rows, behind a global model trained on all rows.
  Regimes with too few distinct training engines fall back to the
  global model: a per-regime model fit on a handful of engines would
  mostly memorise those engines. Which regimes fell back, and how many
  engines/rows each regime had, is recorded on the fitted ensemble —
  the fallback is part of the result, not a silent detail.

Features fed to either approach are the standard ``src.data``
pipeline, computed on each engine's full trajectory exactly as in the
baseline: only the regressor (or its extra inputs) is
regime-conditional. Rolling windows therefore still span regime
changes — a stated limitation, shared with the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src import data as d
from src import model as m

#: FD002 is documented to have six operating conditions.
N_REGIMES = 6

#: A regime needs at least this many distinct training engines before
#: it gets its own model; below that, the global model serves it.
MIN_REGIME_ENGINES = 10


@dataclass
class RegimeAssigner:
    """Fitted regime clustering over the operating-setting space."""

    scaler: StandardScaler
    kmeans: KMeans

    @property
    def n_regimes(self) -> int:
        return int(self.kmeans.n_clusters)

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """Regime id per row: nearest centroid in scaled setting space."""
        scaled = self.scaler.transform(df[d.SETTING_COLS])
        return self.kmeans.predict(scaled).astype(int)

    def distances(self, df: pd.DataFrame) -> np.ndarray:
        """Euclidean distance from each row to every centroid (scaled space)."""
        scaled = self.scaler.transform(df[d.SETTING_COLS])
        return self.kmeans.transform(scaled)

    def centers_in_setting_units(self) -> pd.DataFrame:
        """Cluster centroids back in raw setting units, for reporting."""
        centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        return pd.DataFrame(centers, columns=d.SETTING_COLS)


def fit_regimes(
    train: pd.DataFrame,
    n_regimes: int = N_REGIMES,
    seed: int = m.RANDOM_SEED,
) -> RegimeAssigner:
    """Fit the regime clustering on training rows only.

    Deterministic for a fixed seed (KMeans with ``n_init=10``, the same
    configuration M2's condition clustering used, so M6 regimes are
    comparable to M2 conditions).
    """
    scaler = StandardScaler().fit(train[d.SETTING_COLS])
    kmeans = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
    kmeans.fit(scaler.transform(train[d.SETTING_COLS]))
    return RegimeAssigner(scaler=scaler, kmeans=kmeans)


def add_regime_column(df: pd.DataFrame, assigner: RegimeAssigner) -> pd.DataFrame:
    """Copy of ``df`` with an integer ``regime`` column added."""
    out = df.copy()
    out["regime"] = assigner.assign(out)
    return out


def regime_feature_frame(df: pd.DataFrame, assigner: RegimeAssigner) -> pd.DataFrame:
    """Regime features for the global-model approach, aligned to ``df``.

    Columns: ``regime`` (integer id) and ``regime_dist_<k>`` for every
    regime — the row's distance to each centroid in scaled setting
    space. Computed from the frame's own setting columns, so it works
    identically on raw and feature-built frames (feature building
    retains the raw settings).
    """
    dist = assigner.distances(df)
    cols = {f"regime_dist_{k}": dist[:, k] for k in range(assigner.n_regimes)}
    out = pd.DataFrame(cols, index=df.index)
    out.insert(0, "regime", assigner.assign(df))
    return out


def regime_stats(df: pd.DataFrame) -> dict[int, dict[str, int]]:
    """Per-regime row and distinct-engine counts for a frame with ``regime``."""
    stats: dict[int, dict[str, int]] = {}
    for regime, grp in df.groupby("regime"):
        stats[int(regime)] = {
            "rows": int(len(grp)),
            "engines": int(grp["unit"].nunique()),
        }
    return stats


class RegimeEnsemble:
    """One RUL model per regime, with a global model as fallback.

    ``fit`` trains the global model on every row, then a dedicated
    model for each regime whose distinct training-engine count reaches
    ``min_engines``. Regimes below the threshold — including regimes
    with no training rows at all — are recorded in ``fallback_regimes``
    and served by the global model at predict time.
    """

    def __init__(
        self,
        min_engines: int = MIN_REGIME_ENGINES,
        random_state: int = m.RANDOM_SEED,
    ) -> None:
        self.min_engines = min_engines
        self.random_state = random_state

    def fit(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        n_regimes: int,
        target_col: str = "rul",
        regime_col: str = "regime",
        unit_col: str = "unit",
    ) -> "RegimeEnsemble":
        self.global_model_ = m.train_model(
            frame[feature_cols], frame[target_col],
            random_state=self.random_state,
        )
        self.regime_models_: dict[int, object] = {}
        self.fallback_regimes_: set[int] = set()
        self.regime_stats_: dict[int, dict[str, int]] = {}
        for regime in range(n_regimes):
            sub = frame[frame[regime_col] == regime]
            engines = int(sub[unit_col].nunique())
            self.regime_stats_[regime] = {
                "rows": int(len(sub)),
                "engines": engines,
            }
            if engines >= self.min_engines and len(sub) > 0:
                self.regime_models_[regime] = m.train_model(
                    sub[feature_cols], sub[target_col],
                    random_state=self.random_state,
                )
            else:
                self.fallback_regimes_.add(regime)
        return self

    def predict(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        regime_col: str = "regime",
    ) -> np.ndarray:
        """Global predictions, overwritten by per-regime models where fitted."""
        pred = m.predict_rul(self.global_model_, frame[feature_cols])
        regimes = frame[regime_col].to_numpy()
        for regime, model_r in self.regime_models_.items():
            mask = regimes == regime
            if mask.any():
                pred[mask] = m.predict_rul(model_r, frame.loc[mask, feature_cols])
        return pred

    def fallback_row_count(self, frame: pd.DataFrame,
                           regime_col: str = "regime") -> int:
        """How many of ``frame``'s rows the global fallback would serve."""
        return int(frame[regime_col].isin(self.fallback_regimes_).sum())
