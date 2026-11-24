"""Tests for src.evaluate: RMSE, NASA score, last-cycle selection."""

import numpy as np
import pytest

from src import data as d
from src import evaluate as ev
from tests.test_data import make_frame


def test_rmse_known_value():
    assert ev.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(np.sqrt(4 / 3))
    assert ev.rmse([5.0], [5.0]) == 0.0


def test_nasa_score_zero_for_perfect_predictions():
    assert ev.nasa_score([10.0, 20.0], [10.0, 20.0]) == 0.0


def test_nasa_score_known_values():
    # d = +10 (late) -> exp(10/10) - 1; d = -13 (early) -> exp(13/13) - 1.
    assert ev.nasa_score([0.0], [10.0]) == pytest.approx(np.e - 1)
    assert ev.nasa_score([13.0], [0.0]) == pytest.approx(np.e - 1)


def test_nasa_score_penalises_late_more_than_early():
    late = ev.nasa_score([100.0], [120.0])   # d = +20
    early = ev.nasa_score([100.0], [80.0])   # d = -20
    assert late > early


def test_last_cycle_rows_picks_final_cycle_per_engine():
    df = d.add_rul(make_frame({1: 5, 2: 3}))
    last = ev.last_cycle_rows(df)
    assert last["unit"].tolist() == [1, 2]
    assert last["cycle"].tolist() == [5, 3]
    assert last["rul_raw"].tolist() == [0.0, 0.0]
