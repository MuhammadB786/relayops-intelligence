"""Tests for M4 drift monitoring (src/drift.py).

The PSI correctness case is hand-computable: with explicit bin edges
[-0.5, 0.5, 1.5], a reference split 50/50 and a current split 25/75
give PSI = (0.25-0.5)*ln(0.25/0.5) + (0.75-0.5)*ln(0.75/0.5)
         = 0.25*ln(2) + 0.25*ln(1.5) ~= 0.2747.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import drift as dr


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 2.0, 2000)
    assert dr.population_stability_index(x, x.copy()) == pytest.approx(0.0)


def test_psi_known_value_with_explicit_bins():
    reference = np.array([0.0] * 50 + [1.0] * 50)
    current = np.array([0.0] * 25 + [1.0] * 75)
    edges = np.array([-0.5, 0.5, 1.5])
    expected = 0.25 * math.log(2.0) + 0.25 * math.log(1.5)
    got = dr.population_stability_index(reference, current, bin_edges=edges)
    assert got == pytest.approx(expected, abs=1e-9)


def test_psi_grows_with_shift_size():
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, 4000)
    small = rng.normal(0.2, 1.0, 4000)
    large = rng.normal(1.0, 1.0, 4000)
    assert dr.population_stability_index(ref, small) < \
        dr.population_stability_index(ref, large)


def test_psi_constant_reference():
    ref = np.full(100, 3.5)
    assert dr.population_stability_index(ref, np.full(50, 3.5)) == 0.0
    assert dr.population_stability_index(ref, np.full(50, 4.0)) == float("inf")


def test_psi_quantized_reference_uses_value_proportions():
    # A two-valued channel (FD001's sensor_6 in healthy data): decile
    # binning degenerates, so PSI falls back to the value proportions.
    ref = np.array([21.60] * 4 + [21.61] * 96)
    same = np.array([21.60] * 2 + [21.61] * 48)
    assert dr.population_stability_index(ref, same) == pytest.approx(0.0)
    flipped = np.array([21.60] * 48 + [21.61] * 52)  # proportions .48/.52
    expected = ((0.48 - 0.04) * math.log(0.48 / 0.04)
                + (0.52 - 0.96) * math.log(0.52 / 0.96))
    got = dr.population_stability_index(ref, flipped)
    assert got == pytest.approx(expected, rel=1e-6)


def test_psi_bands():
    assert dr.psi_band(0.05) == "stable"
    assert dr.psi_band(0.10) == "moderate"
    assert dr.psi_band(0.24) == "moderate"
    assert dr.psi_band(0.25) == "significant"


def _frame(values: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(values)


def test_sensor_psi_skips_constant_columns():
    rng = np.random.default_rng(2)
    ref = _frame({"a": rng.normal(0, 1, 500), "b": np.full(500, 7.0)})
    win = _frame({"a": rng.normal(0, 1, 500), "b": np.full(500, 7.0)})
    psis = dr.sensor_psi(ref, win, ["a", "b"])
    assert set(psis) == {"a"}
    assert psis["a"] < 0.10  # same distribution, sampling noise only


def _psis_frame(base: float, shifted: dict[str, float], n: int = 800,
                seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {f"s{i}": rng.normal(base, 1.0, n) for i in range(5)}
    for c, delta in shifted.items():
        cols[c] = cols[c] + delta
    return pd.DataFrame(cols)


def test_trigger_fires_on_sustained_multisensor_shift():
    ref = _psis_frame(0.0, {})
    w1 = _psis_frame(0.0, {"s0": 1.2, "s1": 1.2, "s2": 1.2}, seed=4)
    w2 = _psis_frame(0.0, {"s0": 1.2, "s1": 1.2, "s2": 1.2}, seed=5)
    result = dr.evaluate_windows(ref, [w1, w2], [f"s{i}" for i in range(5)])
    assert result["retraining_recommended"] is True
    assert set(result["sustained_sensors"]) == {"s0", "s1", "s2"}


def test_trigger_does_not_fire_on_unshifted_windows():
    ref = _psis_frame(0.0, {})
    w1 = _psis_frame(0.0, {}, seed=6)
    w2 = _psis_frame(0.0, {}, seed=7)
    result = dr.evaluate_windows(ref, [w1, w2], [f"s{i}" for i in range(5)])
    assert result["retraining_recommended"] is False


def test_trigger_needs_both_windows_and_enough_sensors():
    ref = _psis_frame(0.0, {})
    hot = _psis_frame(0.0, {"s0": 1.2, "s1": 1.2, "s2": 1.2}, seed=8)
    calm = _psis_frame(0.0, {}, seed=9)
    cols = [f"s{i}" for i in range(5)]
    # one hot window is an incident, not a retraining decision
    assert dr.evaluate_windows(ref, [hot, calm], cols)[
        "retraining_recommended"] is False
    # only two sensors shifted: below the min-sensors count
    two = _psis_frame(0.0, {"s0": 1.2, "s1": 1.2}, seed=10)
    two2 = _psis_frame(0.0, {"s0": 1.2, "s1": 1.2}, seed=11)
    assert dr.evaluate_windows(ref, [two, two2], cols)[
        "retraining_recommended"] is False
