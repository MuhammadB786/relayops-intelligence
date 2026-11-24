"""Tests for src.alerts: confusion counts, precision/recall, cost, sweeps."""

import numpy as np
import pytest

from src import alerts as al


def test_confusion_counts_hand_computable():
    # truth:  P P N N ; alerts: P N P N  -> one of each outcome.
    conf = al.alert_confusion([1, 1, 0, 0], [True, False, True, False])
    assert conf == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_precision_recall_hand_computable():
    conf = {"tp": 3, "fp": 1, "fn": 2, "tn": 94}
    precision, recall = al.precision_recall(conf)
    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.6)


def test_precision_recall_degenerate_conventions():
    # No alerts fired -> precision 0 by the stated convention.
    assert al.precision_recall({"tp": 0, "fp": 0, "fn": 5, "tn": 5}) == (0.0, 0.0)
    # No positive cycles -> recall 0 by the stated convention.
    assert al.precision_recall({"tp": 0, "fp": 2, "fn": 0, "tn": 8}) == (0.0, 0.0)
    # Perfect alerting.
    assert al.precision_recall({"tp": 4, "fp": 0, "fn": 0, "tn": 6}) == (1.0, 1.0)


def test_expected_cost_weights_missed_failures():
    conf = {"tp": 0, "fp": 2, "fn": 3, "tn": 0}
    assert al.expected_cost(conf) == 2 * 1.0 + 3 * 50.0
    assert al.expected_cost(conf, cost_fp=1.0, cost_fn=10.0) == 32.0


def test_sweep_thresholds_below_direction():
    signal = np.array([10.0, 25.0, 60.0, 90.0])   # e.g. predicted RUL
    truth = np.array([1, 1, 0, 0])                # failure within horizon
    rows = al.sweep_thresholds(signal, truth, [30.0, 5.0], direction="below")
    first, second = rows
    assert (first["tp"], first["fp"], first["fn"], first["tn"]) == (2, 0, 0, 2)
    assert first["precision"] == 1.0 and first["recall"] == 1.0
    assert (second["tp"], second["fp"], second["fn"], second["tn"]) == (0, 0, 2, 2)


def test_sweep_thresholds_above_direction():
    signal = np.array([0.9, 0.4, 0.8, 0.1])       # e.g. anomaly score
    truth = np.array([1, 1, 0, 0])
    (row,) = al.sweep_thresholds(signal, truth, [0.5], direction="above")
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 1, 1, 1)
    assert row["precision"] == 0.5 and row["recall"] == 0.5
