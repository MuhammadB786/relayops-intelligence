"""Live sensor simulator: synthetic degrading assets + recorded replay.

Two feed modes for the M4 live demo:

- **Synthetic** — a new asset generated cycle by cycle from parameters
  *derived from the FD001 training data* (see :func:`derive_params`):
  healthy baselines are the per-sensor means over the first
  ``EARLY_CYCLES`` cycles of every training engine; each simulated asset
  draws its own baseline offset from the observed between-engine spread
  of those early means (real engines start at slightly different
  baselines); per-cycle noise uses the observed within-engine healthy
  standard deviation; and the degradation applied to each sensor is the
  drift actually measured in the data — per engine, mean of the last
  10% of life minus mean of the first 10%, averaged over engines. No
  sensor's drift is guessed: the constant FD001 channels (sensors 1, 5,
  10, 16, 18, 19) measure zero drift and stay constant, sensor_6
  measures ~0.3 healthy-sigmas over a full life and barely moves, and
  the strong movers are the ones the data names (sensor_9 +8.4 sigma,
  sensor_14 +7.9, sensor_11 +6.8, sensor_12 -6.0, sensor_4 +5.7,
  sensor_7 -5.3 over a full life).
- **Replay** — a recorded FD001 test engine streamed back verbatim, so
  the demo also runs on data nobody synthesized.

Degradation trajectory: drift is applied along a convex exponential
wear curve ``w(u) = (exp(k*u) - 1) / (exp(k) - 1)`` over life fraction
``u`` — slow early, accelerating late, the qualitative shape of the
HPC degradation C-MAPSS simulates. Operating settings are sampled per
cycle from their observed FD001 distribution (FD001 is a single
operating condition: setting_1/setting_2 jitter around 0, setting_3 is
a constant 100), which is the only "regime variation" FD001 contains.

Everything is deterministic: one ``numpy.random.default_rng(seed)``
drives a simulation, so the same seed reproduces the same asset exactly.

This is a *model* of degradation, not physics: drift is linear in the
wear curve per sensor, noise is independent Gaussian per channel, and
cross-sensor correlations of the real simulator are not reproduced.
See README (M4) limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data import RAW_COLS, SENSOR_COLS, SETTING_COLS

#: Cycles at the start of each training engine's life treated as the
#: healthy reference window for baselines and noise levels.
EARLY_CYCLES = 30

#: Fraction of life averaged at each end when measuring a sensor's
#: full-life drift (first vs last slice, per engine).
EDGE_FRACTION = 0.10

#: Default wear-curve steepness (see module docstring).
DEFAULT_CURVE_K = 2.5


@dataclass
class SimulatorParams:
    """Everything the simulator needs, all measured from training data."""

    setting_mean: dict[str, float]
    setting_std: dict[str, float]
    sensor_baseline: dict[str, float]      # global healthy mean
    sensor_between_std: dict[str, float]   # spread of per-engine healthy means
    sensor_noise_std: dict[str, float]     # mean within-engine healthy std
    sensor_drift: dict[str, float]         # full-life drift (last - first slice)
    n_engines: int = 0
    early_cycles: int = EARLY_CYCLES
    edge_fraction: float = EDGE_FRACTION
    provenance: dict = field(default_factory=dict)

    def drift_in_sigmas(self, sensor: str) -> float:
        """Full-life drift in units of that sensor's healthy noise std."""
        noise = self.sensor_noise_std[sensor]
        if noise == 0.0:
            return 0.0
        return self.sensor_drift[sensor] / noise

    def moving_sensors(self, min_sigmas: float = 1.0) -> list[str]:
        """Sensors whose measured full-life drift exceeds ``min_sigmas``."""
        return [s for s in SENSOR_COLS if abs(self.drift_in_sigmas(s)) >= min_sigmas]


def derive_params(
    train: pd.DataFrame,
    early_cycles: int = EARLY_CYCLES,
    edge_fraction: float = EDGE_FRACTION,
) -> SimulatorParams:
    """Measure simulator parameters from run-to-failure training data.

    - baselines / noise: cycles ``<= early_cycles`` of every engine —
      global mean, std of per-engine means (between), and mean of
      per-engine stds (within, used as the per-cycle noise level).
    - drift: per engine, mean of the last ``edge_fraction`` of cycles
      minus mean of the first ``edge_fraction``, averaged over engines.
    """
    early = train[train["cycle"] <= early_cycles]

    setting_mean = {c: float(early[c].mean()) for c in SETTING_COLS}
    setting_std = {c: float(early[c].std(ddof=0)) for c in SETTING_COLS}

    sensor_baseline: dict[str, float] = {}
    sensor_between: dict[str, float] = {}
    sensor_noise: dict[str, float] = {}
    sensor_drift: dict[str, float] = {}
    for c in SENSOR_COLS:
        per_engine = early.groupby("unit")[c]
        sensor_baseline[c] = float(early[c].mean())
        sensor_between[c] = float(per_engine.mean().std(ddof=0))
        sensor_noise[c] = float(per_engine.std().mean())

        drifts = []
        for _, eng in train.groupby("unit"):
            k = max(1, int(edge_fraction * len(eng)))
            drifts.append(float(eng[c].iloc[-k:].mean() - eng[c].iloc[:k].mean()))
        sensor_drift[c] = float(np.mean(drifts))

    return SimulatorParams(
        setting_mean=setting_mean,
        setting_std=setting_std,
        sensor_baseline=sensor_baseline,
        sensor_between_std=sensor_between,
        sensor_noise_std=sensor_noise,
        sensor_drift=sensor_drift,
        n_engines=int(train["unit"].nunique()),
        early_cycles=early_cycles,
        edge_fraction=edge_fraction,
        provenance={
            "source": "FD001 training engines (run to failure)",
            "baseline_window": f"cycles <= {early_cycles}",
            "drift_window": f"first vs last {edge_fraction:.0%} of each engine's life",
        },
    )


def wear_curve(u: np.ndarray, k: float = DEFAULT_CURVE_K) -> np.ndarray:
    """Convex degradation profile over life fraction ``u`` in [0, 1].

    ``w(0) = 0``, ``w(1) = 1``; larger ``k`` keeps the asset healthy
    longer and degrades it faster at the end.
    """
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return (np.exp(k * u) - 1.0) / (np.exp(k) - 1.0)


def simulate_asset(
    params: SimulatorParams,
    n_cycles: int,
    seed: int,
    unit: int = 1,
    degradation_scale: float = 1.0,
    curve_k: float = DEFAULT_CURVE_K,
) -> pd.DataFrame:
    """Generate one synthetic asset's full life, CMAPSS-shaped.

    The asset's designed end of life is cycle ``n_cycles``: the wear
    curve reaches the full measured drift there. ``degradation_scale``
    multiplies every sensor's measured drift (1.0 = exactly the drift
    FD001 engines show over their lives).
    """
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)
    u = cycles / n_cycles
    wear = wear_curve(u, k=curve_k)

    data: dict[str, np.ndarray] = {
        "unit": np.full(n_cycles, unit, dtype=int),
        "cycle": cycles,
    }
    for c in SETTING_COLS:
        std = params.setting_std[c]
        data[c] = (
            np.full(n_cycles, params.setting_mean[c])
            if std == 0.0
            else rng.normal(params.setting_mean[c], std, n_cycles)
        )
    for c in SENSOR_COLS:
        offset = rng.normal(0.0, params.sensor_between_std[c])
        noise = rng.normal(0.0, params.sensor_noise_std[c], n_cycles)
        data[c] = (
            params.sensor_baseline[c]
            + offset
            + degradation_scale * params.sensor_drift[c] * wear
            + noise
        )
    return pd.DataFrame(data, columns=RAW_COLS)


def replay_engine(test: pd.DataFrame, unit: int) -> pd.DataFrame:
    """One recorded engine's cycles from a test frame, in cycle order."""
    eng = test[test["unit"] == unit].sort_values("cycle").reset_index(drop=True)
    if eng.empty:
        raise ValueError(f"unit {unit} not present in the given frame")
    return eng[RAW_COLS]
