"""Tests for the M3 scoring API (api/) and its artifact loader.

The fixture trains a miniature artifact bundle on synthetic data in a
temp directory (same convention as the rest of the suite: synthetic
only, no downloads, and never the real models/ bundle) using the same
save path as scripts/train_service.py.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.artifacts import ArtifactError, load_artifacts, save_artifacts
from api.main import MIN_HISTORY_CYCLES, create_app
from src import anomaly as an
from src import data as d
from src import model as m

WARMUP = max(d.DEFAULT_WINDOWS)


def synthetic_engines(n_units: int = 14, seed: int = 7) -> pd.DataFrame:
    """Run-to-failure engines whose sensors drift with wear (cf. smoke.py)."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(90, 170))
        wear = np.linspace(0, 1, life)
        row = {c: np.zeros(life) for c in d.RAW_COLS}
        row["unit"] = np.full(life, unit)
        row["cycle"] = np.arange(1, life + 1)
        for c in d.SETTING_COLS:
            row[c] = rng.normal(0, 0.01, life)
        for i, c in enumerate(d.SENSOR_COLS):
            row[c] = 10 * i + 5 * wear + rng.normal(0, 0.1, life)
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def window_payload(frame: pd.DataFrame, asset_id: str) -> dict:
    """One engine slice (already cycle-ordered) as a /score request body."""
    cycles = [
        {
            "cycle": int(r["cycle"]),
            "settings": [float(r[c]) for c in d.SETTING_COLS],
            "sensors": [float(r[c]) for c in d.SENSOR_COLS],
        }
        for _, r in frame.iterrows()
    ]
    return {"asset_id": asset_id, "cycles": cycles}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    """A trained synthetic artifact bundle in a temp models directory."""
    models_dir = tmp_path_factory.mktemp("models")
    raw = synthetic_engines()
    lab = d.add_rul(raw)
    dead = d.constant_columns(lab, d.SETTING_COLS + d.SENSOR_COLS)
    feats, feature_cols = d.build_features(lab, drop_cols=dead)
    anomaly_cols = an.anomaly_feature_columns(feature_cols)

    rul_model = m.train_model(feats[feature_cols], feats["rul"])
    healthy = feats[an.healthy_mask(feats) & (feats["cycle"] > WARMUP)]
    detector = an.train_detector(healthy[anomaly_cols])
    reference_scores = an.anomaly_scores(detector, healthy[anomaly_cols])

    save_artifacts(
        models_dir,
        rul_model=rul_model,
        detector=detector,
        reference_scores=reference_scores,
        card_fields={
            "model_version": "0.0.0-test",
            "milestone": "M3-test",
            "seed": m.RANDOM_SEED,
            "feature_columns": feature_cols,
            "anomaly_feature_columns": anomaly_cols,
            "dropped_constant_columns": dead,
            "windows": list(d.DEFAULT_WINDOWS),
            "rul_cap": d.RUL_CAP,
            "training_data": {"subset": "SYNTHETIC", "train_engines": 14},
            "alert": {"critical_max_health": 50.0, "watch_max_health": 70.0},
        },
    )
    return models_dir


@pytest.fixture(scope="module")
def engines() -> pd.DataFrame:
    return synthetic_engines()


@pytest.fixture()
def client(bundle):
    with TestClient(create_app(bundle)) as c:
        yield c


def engine_slice(engines: pd.DataFrame, unit: int, tail: int | None = None,
                 head: int | None = None) -> pd.DataFrame:
    eng = engines[engines["unit"] == unit].sort_values("cycle")
    if tail is not None:
        return eng.tail(tail)
    return eng.head(head)


# ---------------------------------------------------------------- endpoints

def test_healthz_shape(client, bundle):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "0.0.0-test"
    assert body["feature_count"] == 121  # nothing constant in synthetic data
    card = json.loads((bundle / "model_card.json").read_text())
    assert body["artifact_checksum"] == card["bundle_checksum"]


def test_degrading_window_scores_lower_health_than_healthy(client, engines):
    healthy = window_payload(engine_slice(engines, 1, head=40), "eng-early")
    degraded = window_payload(engine_slice(engines, 1, tail=40), "eng-late")
    h = client.post("/score", json=healthy)
    g = client.post("/score", json=degraded)
    assert h.status_code == 200 and g.status_code == 200
    assert g.json()["health_index"] < h.json()["health_index"]
    assert g.json()["predicted_rul"] < h.json()["predicted_rul"]
    assert g.json()["anomaly_percentile"] > h.json()["anomaly_percentile"]


def test_alert_fires_critical_on_strongly_degrading_window(client, engines):
    payload = window_payload(engine_slice(engines, 2, tail=30), "eng-dying")
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["health_index"] <= 50.0
    assert body["alert_level"] == "critical"
    assert body["model_version"] == "0.0.0-test"
    assert body["scored_at"].endswith(("Z", "+00:00"))


def test_insufficient_history_is_a_clear_422(client, engines):
    payload = window_payload(engine_slice(engines, 1, head=MIN_HISTORY_CYCLES - 1),
                             "eng-short")
    resp = client.post("/score", json=payload)
    assert resp.status_code == 422
    assert "insufficient history" in resp.json()["detail"]


def test_exactly_min_history_scores(client, engines):
    payload = window_payload(engine_slice(engines, 1, tail=MIN_HISTORY_CYCLES),
                             "eng-min")
    assert client.post("/score", json=payload).status_code == 200


@pytest.mark.parametrize("mutate", [
    lambda p: p["cycles"][0].update(sensors=p["cycles"][0]["sensors"][:20]),
    lambda p: p["cycles"][0].update(sensors=["hot"] * 21),
    lambda p: p["cycles"][0].update(settings=[1.0, 2.0]),
    lambda p: p.update(cycles=[]),
    lambda p: p.pop("asset_id"),
    lambda p: p["cycles"].reverse(),
])
def test_malformed_payloads_are_422(client, engines, mutate):
    payload = window_payload(engine_slice(engines, 1, head=20), "eng-bad")
    mutate(payload)
    assert client.post("/score", json=payload).status_code == 422


def test_batch_one_bad_asset_does_not_sink_the_rest(client, engines):
    good = window_payload(engine_slice(engines, 3, tail=40), "good")
    short = window_payload(engine_slice(engines, 3, head=5), "short")
    bad = window_payload(engine_slice(engines, 3, head=20), "bad")
    bad["cycles"][0]["sensors"] = bad["cycles"][0]["sensors"][:19]
    resp = client.post("/score/batch", json={"assets": [good, short, bad]})
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 3
    assert results[0]["asset_id"] == "good" and "health_index" in results[0]
    assert results[1]["asset_id"] == "short"
    assert "insufficient history" in results[1]["error"]
    assert results[2]["asset_id"] == "bad" and "error" in results[2]


# ------------------------------------------------------------- artifact loading

def _tampered_copy(bundle: Path, tmp_path: Path, edit) -> Path:
    target = tmp_path / "tampered"
    shutil.copytree(bundle, target)
    card_path = target / "model_card.json"
    card = json.loads(card_path.read_text())
    edit(card, target)
    card_path.write_text(json.dumps(card, indent=2))
    return target


def test_loader_rejects_feature_list_mismatch(bundle, tmp_path):
    def drop_a_feature(card, _target):
        card["feature_columns"] = card["feature_columns"][:-1]

    target = _tampered_copy(bundle, tmp_path, drop_a_feature)
    with pytest.raises(ArtifactError, match="feature list"):
        load_artifacts(target)


def test_loader_rejects_checksum_mismatch(bundle, tmp_path):
    def leave_card(card, target):
        # Edit an artifact file without updating the card's checksums.
        (target / "reference_scores.npy").write_bytes(b"tampered")

    target = _tampered_copy(bundle, tmp_path, leave_card)
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_artifacts(target)


def test_loader_rejects_missing_directory(tmp_path):
    with pytest.raises(ArtifactError, match="not found"):
        load_artifacts(tmp_path / "does-not-exist")


def test_startup_fails_fast_on_tampered_card(bundle, tmp_path):
    def drop_a_feature(card, _target):
        card["feature_columns"] = card["feature_columns"][:-1]

    target = _tampered_copy(bundle, tmp_path, drop_a_feature)
    app = create_app(target)
    with pytest.raises(ArtifactError):
        with TestClient(app):
            pass


# ------------------------------------------------------------------ auth stub

def test_api_key_stub(client, engines, monkeypatch):
    payload = window_payload(engine_slice(engines, 1, tail=20), "eng-key")
    monkeypatch.setenv("RELAYOPS_ML_API_KEY", "test-secret")
    assert client.post("/score", json=payload).status_code == 401
    ok = client.post("/score", json=payload, headers={"X-API-Key": "test-secret"})
    assert ok.status_code == 200
    # /healthz stays open: health checks must not need the secret.
    assert client.get("/healthz").status_code == 200
