"""Tests for src.health: bounds, components, monotonicity on synthetic decay."""

import numpy as np

from src import health as h


def test_components_hit_their_endpoints():
    assert h.rul_component(np.array([0.0]))[0] == 0.0
    assert h.rul_component(np.array([125.0]))[0] == 100.0
    assert h.rul_component(np.array([999.0]))[0] == 100.0   # above cap saturates
    assert h.anomaly_component(np.array([0.0]))[0] == 100.0
    assert h.anomaly_component(np.array([100.0]))[0] == 0.0


def test_health_index_stays_within_bounds():
    rng = np.random.default_rng(1)
    health = h.health_index(rng.uniform(-50, 300, 500), rng.uniform(-10, 150, 500))
    assert health.min() >= 0.0 and health.max() <= 100.0


def test_health_index_weighting_is_exact_at_extremes():
    # Perfect RUL, worst anomaly: 0.6*100 + 0.4*0 = 60.
    assert h.health_index(np.array([125.0]), np.array([100.0]))[0] == 60.0
    # Zero RUL, perfect anomaly: 0.6*0 + 0.4*100 = 40.
    assert h.health_index(np.array([0.0]), np.array([0.0]))[0] == 40.0


def test_health_decreases_monotonically_on_synthetic_degradation():
    cycles = np.arange(200)
    pred_rul = np.clip(180 - cycles, 0, None).astype(float)      # steady decay
    anomaly_pct = np.clip(cycles * 0.6, 0, 100)                  # steadily worse
    health = h.health_index(pred_rul, anomaly_pct)
    assert np.all(np.diff(health) <= 0)                          # non-increasing
    assert health[0] == 100.0                                    # starts healthy
    assert health[-1] < 5.0                                      # ends near failure
