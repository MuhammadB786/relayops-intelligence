"""Tests for src.model: training, prediction clipping, persistence."""

import numpy as np
import pandas as pd

from src import data as d
from src import model as m
from tests.test_data import make_frame


def _toy_training_set() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    frame = d.add_rul(make_frame({1: 40, 2: 40, 3: 40}))
    feats, cols = d.build_features(frame, windows=(5,))
    return feats, feats["rul"], cols


def test_train_and_predict_shapes():
    feats, target, cols = _toy_training_set()
    model = m.train_model(feats[cols], target)
    pred = m.predict_rul(model, feats[cols])
    assert pred.shape == (len(feats),)
    assert np.all(np.isfinite(pred))


def test_predict_clips_negative_rul():
    feats, target, cols = _toy_training_set()
    model = m.train_model(feats[cols], target)
    extreme = feats[cols].copy()
    extreme.loc[:, :] = -1e6  # far outside the training distribution
    assert np.all(m.predict_rul(model, extreme) >= 0.0)


def test_save_load_roundtrip(tmp_path):
    feats, target, cols = _toy_training_set()
    model = m.train_model(feats[cols], target)
    path = tmp_path / "model.joblib"
    m.save_model(model, path)
    loaded = m.load_model(path)
    np.testing.assert_allclose(
        m.predict_rul(loaded, feats[cols]), m.predict_rul(model, feats[cols])
    )
