"""Tests for the M4 sensor simulator (src/simulator.py).

Synthetic parameters only — no dataset downloads, per suite convention.
The parameters are hand-built to mimic FD001's structure: a couple of
constant channels, small setting jitter, and sensors with known drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data as d
from src.simulator import (
    SimulatorParams,
    derive_params,
    replay_engine,
    simulate_asset,
    wear_curve,
)


def hand_params() -> SimulatorParams:
    """FD001-shaped parameters with known values."""
    baseline = {c: 100.0 * (i + 1) for i, c in enumerate(d.SENSOR_COLS)}
    noise = {c: 1.0 for c in d.SENSOR_COLS}
    drift = {c: 0.0 for c in d.SENSOR_COLS}
    drift["sensor_2"] = 20.0    # strong riser
    drift["sensor_7"] = -10.0   # faller
    for c in ("sensor_1", "sensor_5"):  # constant channels, as in FD001
        noise[c] = 0.0
    return SimulatorParams(
        setting_mean={"setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0},
        setting_std={"setting_1": 0.002, "setting_2": 0.0003, "setting_3": 0.0},
        sensor_baseline=baseline,
        sensor_between_std={c: 0.5 for c in d.SENSOR_COLS},
        sensor_noise_std=noise,
        sensor_drift=drift,
        n_engines=3,
    )


def test_simulation_is_deterministic_per_seed():
    params = hand_params()
    a = simulate_asset(params, n_cycles=60, seed=123)
    b = simulate_asset(params, n_cycles=60, seed=123)
    pd.testing.assert_frame_equal(a, b)
    c = simulate_asset(params, n_cycles=60, seed=124)
    assert not a.equals(c)


def test_output_shape_and_semantics():
    params = hand_params()
    frame = simulate_asset(params, n_cycles=50, seed=1, unit=9)
    assert list(frame.columns) == d.RAW_COLS
    assert len(frame) == 50
    assert (frame["unit"] == 9).all()
    assert frame["cycle"].tolist() == list(range(1, 51))
    # 3 settings + 21 sensors present in every row, all finite
    assert frame[d.SETTING_COLS + d.SENSOR_COLS].notna().all().all()
    # zero-variance setting stays exactly constant (FD001's setting_3)
    assert (frame["setting_3"] == 100.0).all()
    # zero-noise, zero-drift sensors stay exactly constant (at the
    # asset's own drawn baseline, which need not equal the global mean)
    assert frame["sensor_1"].nunique() == 1


def test_degradation_moves_targeted_sensors_in_measured_direction():
    params = hand_params()
    frame = simulate_asset(params, n_cycles=200, seed=5)
    k = 20
    first = frame.iloc[:k]
    last = frame.iloc[-k:]
    # riser rises, faller falls, untouched sensor shows no trend
    assert last["sensor_2"].mean() - first["sensor_2"].mean() > 15.0
    assert last["sensor_7"].mean() - first["sensor_7"].mean() < -7.0
    assert abs(last["sensor_3"].mean() - first["sensor_3"].mean()) < 1.0


def test_wear_curve_endpoints_and_convexity():
    u = np.linspace(0, 1, 11)
    w = wear_curve(u, k=2.5)
    assert w[0] == 0.0 and w[-1] == pytest.approx(1.0)
    assert np.all(np.diff(w) > 0)
    # convex: less than half the drift by mid-life
    assert w[5] < 0.5


def test_derive_params_recovers_known_structure():
    # Tiny run-to-failure fleet: sensor_2 drifts +1/cycle, rest flat.
    frames = []
    for unit in (1, 2, 3):
        life = 100
        frame = pd.DataFrame({
            c: np.zeros(life) for c in d.RAW_COLS})
        frame["unit"] = unit
        frame["cycle"] = np.arange(1, life + 1)
        frame["setting_3"] = 100.0
        frame["sensor_1"] = 518.67
        frame["sensor_2"] = 640.0 + np.arange(life)  # +1 per cycle
        frames.append(frame)
    train = pd.concat(frames, ignore_index=True)
    params = derive_params(train, early_cycles=10, edge_fraction=0.10)
    assert params.sensor_baseline["sensor_1"] == pytest.approx(518.67)
    assert params.sensor_noise_std["sensor_1"] == 0.0
    # first-slice mean ~ cycle 5.5, last-slice ~ cycle 95.5 -> drift ~90
    assert params.sensor_drift["sensor_2"] == pytest.approx(90.0, abs=2.0)
    assert params.sensor_drift["sensor_1"] == 0.0
    assert "sensor_2" in params.moving_sensors()
    assert "sensor_1" not in params.moving_sensors()


def test_replay_returns_recorded_rows_in_order():
    frame = pd.DataFrame({c: np.arange(6, dtype=float) for c in d.RAW_COLS})
    frame["unit"] = [1, 1, 1, 2, 2, 2]
    frame["cycle"] = [2, 1, 3, 1, 2, 3]
    out = replay_engine(frame, 1)
    assert out["cycle"].tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        replay_engine(frame, 99)
