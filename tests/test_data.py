"""Tests for src.data: parsing, RUL labels, splits, feature windows."""

import numpy as np
import pandas as pd
import pytest

from src import data as d


def make_frame(lives: dict[int, int]) -> pd.DataFrame:
    """Tiny run-to-failure frame: unit -> number of cycles."""
    rows = []
    for unit, life in lives.items():
        for cycle in range(1, life + 1):
            row = {"unit": unit, "cycle": cycle}
            row.update({c: 0.5 for c in d.SETTING_COLS})
            row.update({c: float(cycle) for c in d.SENSOR_COLS})
            rows.append(row)
    return pd.DataFrame(rows, columns=d.RAW_COLS)


def test_load_cmapss_parses_named_columns(tmp_path):
    path = tmp_path / "mini.txt"
    line = " ".join(["1", "1"] + ["0.1"] * 24)
    path.write_text(line + "\n" + line.replace("1 1", "1 2", 1) + "\n")
    df = d.load_cmapss(path)
    assert list(df.columns) == d.RAW_COLS
    assert len(df) == 2
    assert df["cycle"].tolist() == [1, 2]
    assert df["sensor_21"].dtype == float


def test_load_rul_indexes_units_from_one(tmp_path):
    path = tmp_path / "RUL.txt"
    path.write_text("112\n98\n")
    s = d.load_rul(path)
    assert s.index.tolist() == [1, 2]
    assert s.loc[2] == 98.0


def test_add_rul_counts_down_to_zero_and_caps():
    df = d.add_rul(make_frame({1: 10, 2: 4}), cap=6)
    u1 = df[df["unit"] == 1].sort_values("cycle")
    assert u1["rul_raw"].tolist() == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert u1["rul"].tolist() == [6, 6, 6, 6, 5, 4, 3, 2, 1, 0]
    assert df[df["unit"] == 2]["rul_raw"].max() == 3


def test_add_rul_no_cap_when_none():
    df = d.add_rul(make_frame({1: 10}), cap=None)
    assert df["rul"].equals(df["rul_raw"])


def test_add_test_rul_offsets_from_last_cycle():
    frame = make_frame({1: 5, 2: 3})
    rul_at_end = pd.Series({1: 40.0, 2: 7.0})
    df = d.add_test_rul(frame, rul_at_end, cap=None)
    u1 = df[df["unit"] == 1].sort_values("cycle")
    assert u1["rul_raw"].tolist() == [44, 43, 42, 41, 40]
    assert df[df["unit"] == 2]["rul_raw"].tolist() == [9, 8, 7]


def test_split_by_engine_is_disjoint_and_complete():
    df = make_frame({1: 3, 2: 3, 3: 3, 4: 3})
    tr, va = d.split_by_engine(df, {2, 4})
    assert set(tr["unit"]) == {1, 3}
    assert set(va["unit"]) == {2, 4}
    assert len(tr) + len(va) == len(df)


def test_constant_columns_detects_dead_sensors():
    df = make_frame({1: 5})
    assert d.constant_columns(df, d.SETTING_COLS) == d.SETTING_COLS
    assert d.constant_columns(df, d.SENSOR_COLS) == []  # sensors vary by cycle


def test_rolling_features_are_trailing_only():
    df = make_frame({1: 6})
    feats, cols = d.build_features(df, windows=(3,))
    s2 = feats.sort_values("cycle")["sensor_2"].tolist()  # sensor_2 == cycle
    rmean = feats.sort_values("cycle")["sensor_2_rmean3"].tolist()
    assert s2 == [1, 2, 3, 4, 5, 6]
    assert rmean == pytest.approx([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    assert "sensor_2_rstd3" in cols


def test_rolling_features_do_not_leak_from_the_future():
    df = make_frame({1: 6, 2: 6})
    feats_before, _ = d.build_features(df, windows=(3,))
    mutated = df.copy()
    mutated.loc[mutated["cycle"] >= 5, d.SENSOR_COLS] = 9999.0
    feats_after, _ = d.build_features(mutated, windows=(3,))
    early_before = feats_before[feats_before["cycle"] <= 4]
    early_after = feats_after[feats_after["cycle"] <= 4]
    pd.testing.assert_frame_equal(early_before, early_after)


def test_build_features_respects_drop_cols():
    df = make_frame({1: 4})
    feats, cols = d.build_features(df, windows=(5,), drop_cols=["sensor_1"])
    assert "sensor_1" not in cols
    assert "sensor_1_rmean5" not in feats.columns
