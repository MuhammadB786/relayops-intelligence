"""Service artifact format: persistence, integrity checks, loading.

``scripts/train_service.py`` trains the M1 RUL model and the M2 anomaly
detector and writes one artifact bundle into ``models/``:

    rul_model.joblib          M1 HistGradientBoostingRegressor (all engines)
    anomaly_detector.joblib   M2 IsolationForest (healthy-life reference)
    reference_scores.npy      anomaly scores of the healthy reference rows
                              (the distribution anomaly percentiles are
                              computed against at serving time)
    model_card.json           everything else the service needs to know:
                              feature columns, training summary, headline
                              metrics, the alert operating point, component
                              versions, and a SHA-256 of every artifact
                              file above (plus a bundle checksum over them)

The loader is deliberately strict — it fails fast with a specific message
rather than serving from a half-broken bundle:

- missing directory / files            -> ArtifactError
- checksum mismatch (file edited or    -> ArtifactError
  truncated after training)
- the card's feature list disagrees    -> ArtifactError
  with what the current code builds
  from the card's own windows/drop list

Checksums are verified *before* any pickle is loaded: integrity first,
unpickling second.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import anomaly as an
from src import data as d

RUL_MODEL_FILE = "rul_model.joblib"
DETECTOR_FILE = "anomaly_detector.joblib"
REFERENCE_FILE = "reference_scores.npy"
CARD_FILE = "model_card.json"
ARTIFACT_FILES = (RUL_MODEL_FILE, DETECTOR_FILE, REFERENCE_FILE)


class ArtifactError(RuntimeError):
    """Raised when the artifact bundle is missing, corrupt, or stale."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bundle_checksum(file_hashes: dict[str, str]) -> str:
    """One checksum over the whole bundle, from the per-file hashes."""
    h = hashlib.sha256()
    for name in sorted(file_hashes):
        h.update(f"{name}:{file_hashes[name]}\n".encode())
    return h.hexdigest()


def expected_feature_columns(windows: list[int], drop_cols: list[str]) -> list[str]:
    """Feature columns the *current code* builds for a bundle's config.

    Recomputed from a one-row stub frame so the check compares the card
    against the code as it exists now, not as it existed at training time.
    """
    stub = pd.DataFrame([{c: 0.0 for c in d.RAW_COLS}])
    stub["unit"] = 1
    stub["cycle"] = 1
    _, cols = d.build_features(stub, windows=tuple(windows), drop_cols=drop_cols)
    return cols


def save_artifacts(
    models_dir: str | Path,
    *,
    rul_model,
    detector,
    reference_scores: np.ndarray,
    card_fields: dict,
) -> dict:
    """Write a complete artifact bundle + model card; returns the card.

    ``card_fields`` carries everything except the checksum block, which is
    computed here from the files actually written.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rul_model, models_dir / RUL_MODEL_FILE)
    joblib.dump(detector, models_dir / DETECTOR_FILE)
    np.save(models_dir / REFERENCE_FILE, np.asarray(reference_scores, dtype=float))

    file_hashes = {name: _sha256(models_dir / name) for name in ARTIFACT_FILES}
    card = {
        **card_fields,
        "artifacts": file_hashes,
        "bundle_checksum": _bundle_checksum(file_hashes),
    }
    (models_dir / CARD_FILE).write_text(json.dumps(card, indent=2))
    return card


@dataclass
class LoadedArtifacts:
    """A validated artifact bundle, ready to score with."""

    rul_model: object
    detector: object
    reference_scores: np.ndarray
    card: dict

    @property
    def model_version(self) -> str:
        return str(self.card["model_version"])

    @property
    def bundle_checksum(self) -> str:
        return str(self.card["bundle_checksum"])

    @property
    def feature_columns(self) -> list[str]:
        return list(self.card["feature_columns"])

    @property
    def anomaly_feature_columns(self) -> list[str]:
        return list(self.card["anomaly_feature_columns"])

    @property
    def critical_max_health(self) -> float:
        return float(self.card["alert"]["critical_max_health"])

    @property
    def watch_max_health(self) -> float:
        return float(self.card["alert"]["watch_max_health"])


def load_artifacts(models_dir: str | Path) -> LoadedArtifacts:
    """Load and validate an artifact bundle, or raise ArtifactError."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise ArtifactError(
            f"models directory not found: {models_dir} — "
            "run scripts/train_service.py first"
        )
    card_path = models_dir / CARD_FILE
    if not card_path.is_file():
        raise ArtifactError(
            f"model card not found: {card_path} — "
            "run scripts/train_service.py first"
        )
    try:
        card = json.loads(card_path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"model card is not valid JSON: {exc}") from exc

    # Integrity first: every artifact file present and unmodified.
    recorded = card.get("artifacts")
    if not isinstance(recorded, dict) or set(recorded) != set(ARTIFACT_FILES):
        raise ArtifactError(
            "model card 'artifacts' block must list exactly: "
            + ", ".join(ARTIFACT_FILES)
        )
    for name, expected_hash in recorded.items():
        path = models_dir / name
        if not path.is_file():
            raise ArtifactError(f"artifact file missing: {path}")
        actual = _sha256(path)
        if actual != expected_hash:
            raise ArtifactError(
                f"checksum mismatch for {name}: card records "
                f"{expected_hash[:12]}…, file hashes to {actual[:12]}… — "
                "the bundle was modified after training; retrain with "
                "scripts/train_service.py"
            )
    if card.get("bundle_checksum") != _bundle_checksum(recorded):
        raise ArtifactError("model card bundle checksum does not match its "
                            "per-file checksums — the card was edited")

    # The card's feature list must be what the current code would build.
    try:
        expected_cols = expected_feature_columns(
            card["windows"], card["dropped_constant_columns"])
    except KeyError as exc:
        raise ArtifactError(f"model card is missing field {exc}") from exc
    if card.get("feature_columns") != expected_cols:
        raise ArtifactError(
            "model card feature list does not match the features the "
            "current code builds from the card's windows/drop list — the "
            "bundle was trained by different code; retrain with "
            "scripts/train_service.py"
        )
    if card.get("anomaly_feature_columns") != an.anomaly_feature_columns(
        expected_cols
    ):
        raise ArtifactError(
            "model card anomaly feature list is inconsistent with its "
            "feature list (it must be the feature list minus 'cycle')"
        )

    return LoadedArtifacts(
        rul_model=joblib.load(models_dir / RUL_MODEL_FILE),
        detector=joblib.load(models_dir / DETECTOR_FILE),
        reference_scores=np.load(models_dir / REFERENCE_FILE),
        card=card,
    )


def alert_level(health: float, critical_max: float, watch_max: float) -> str:
    """Map a health index to an alert level.

    The evaluated M2 decision is the critical line (health <= 50). The
    watch band above it is a presentation tier only — it was not part of
    the M2 precision/recall evaluation.
    """
    if health <= critical_max:
        return "critical"
    if health <= watch_max:
        return "watch"
    return "none"
