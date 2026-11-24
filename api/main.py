"""FastAPI scoring service (M3).

Endpoints:
    GET  /healthz       liveness + model version + artifact checksum
    POST /score         score one asset's recent cycle window
    POST /score/batch   score many assets; per-item errors, never all-or-nothing

Artifacts are loaded once at startup (lifespan) and validated by
``api.artifacts.load_artifacts``; if the bundle is missing, corrupt, or
stale the application fails to start, with the loader's message as the
error. The models directory resolves as: ``create_app(models_dir)``
argument > ``RELAYOPS_ML_MODELS_DIR`` env var > ``./models``.

Auth: none by default (demo scope — see README). If the
``RELAYOPS_ML_API_KEY`` env var is set, the scoring endpoints require a
matching ``X-API-Key`` header; ``/healthz`` stays open so health checks
work without the secret. This is a stub for the RelayOps worker
integration, not a real auth system.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import ValidationError

from api.artifacts import LoadedArtifacts, alert_level, load_artifacts
from api.schemas import (
    BatchRequest,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from src import anomaly as an
from src import data as d
from src import health as hl
from src import model as m

#: Rolling-feature warm-up (M2 finding): features from the first
#: ``max(window)`` cycles of a window are partially filled and score
#: spuriously anomalous, so a window must hold at least one cycle beyond
#: the warm-up before its last cycle can be scored honestly.
WARMUP_CYCLES = max(d.DEFAULT_WINDOWS)
MIN_HISTORY_CYCLES = WARMUP_CYCLES + 1

MODELS_DIR_ENV = "RELAYOPS_ML_MODELS_DIR"
API_KEY_ENV = "RELAYOPS_ML_API_KEY"


class InsufficientHistoryError(ValueError):
    """The window is too short to score (warm-up cycles only)."""


def resolve_models_dir(models_dir: str | Path | None) -> Path:
    if models_dir is not None:
        return Path(models_dir)
    return Path(os.environ.get(MODELS_DIR_ENV, "models"))


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Shared-secret stub, active only when the env var is set."""
    expected = os.environ.get(API_KEY_ENV)
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


def window_to_frame(request: ScoreRequest) -> pd.DataFrame:
    """Raw request window -> a one-asset CMAPSS-shaped DataFrame."""
    rows = []
    for c in request.cycles:
        row = {"unit": 1, "cycle": c.cycle}
        row.update(dict(zip(d.SETTING_COLS, c.settings)))
        row.update(dict(zip(d.SENSOR_COLS, c.sensors)))
        rows.append(row)
    return pd.DataFrame(rows, columns=d.RAW_COLS)


def score_asset(artifacts: LoadedArtifacts, request: ScoreRequest) -> ScoreResponse:
    """Score the last cycle of one asset's window.

    Trailing rolling features at the last cycle depend only on the last
    ``max(window)`` cycles, so scoring a window reproduces exactly the
    values the offline pipeline computes for the same engine at the same
    cycle (verified against M1/M2 outputs for FD001 test engine 41).
    """
    if len(request.cycles) < MIN_HISTORY_CYCLES:
        raise InsufficientHistoryError(
            f"insufficient history: {len(request.cycles)} cycles supplied, "
            f"at least {MIN_HISTORY_CYCLES} required — the first "
            f"{WARMUP_CYCLES} cycles of a window are rolling-feature "
            "warm-up and cannot be scored (see README, M2 warm-up finding)"
        )
    frame = window_to_frame(request)
    features, _ = d.build_features(
        frame,
        windows=tuple(artifacts.card["windows"]),
        drop_cols=list(artifacts.card["dropped_constant_columns"]),
    )
    last = features.iloc[[-1]]

    predicted_rul = float(m.predict_rul(artifacts.rul_model, last[artifacts.feature_columns])[0])
    anomaly_score = float(
        an.anomaly_scores(artifacts.detector, last[artifacts.anomaly_feature_columns])[0]
    )
    anomaly_percentile = float(
        an.score_percentiles(artifacts.reference_scores, [anomaly_score])[0]
    )
    health = float(hl.health_index([predicted_rul], [anomaly_percentile])[0])

    return ScoreResponse(
        asset_id=request.asset_id,
        predicted_rul=round(predicted_rul, 3),
        health_index=round(health, 3),
        anomaly_score=round(anomaly_score, 3),
        anomaly_percentile=round(anomaly_percentile, 3),
        alert_level=alert_level(
            health, artifacts.critical_max_health, artifacts.watch_max_health
        ),
        model_version=artifacts.model_version,
        scored_at=datetime.now(timezone.utc),
    )


def _first_validation_message(exc: ValidationError) -> str:
    err = exc.errors()[0]
    loc = ".".join(str(part) for part in err["loc"]) or "body"
    return f"{loc}: {err['msg']}"


def create_app(models_dir: str | Path | None = None) -> FastAPI:
    resolved_dir = resolve_models_dir(models_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast: a missing/corrupt/stale bundle stops startup here,
        # with the loader's specific message, instead of serving garbage.
        app.state.artifacts = load_artifacts(resolved_dir)
        yield

    app = FastAPI(title="RelayOps ML scoring service", lifespan=lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        artifacts: LoadedArtifacts = request.app.state.artifacts
        return HealthResponse(
            status="ok",
            model_version=artifacts.model_version,
            artifact_checksum=artifacts.bundle_checksum,
            feature_count=len(artifacts.feature_columns),
        )

    @app.post("/score", response_model=ScoreResponse,
              dependencies=[Depends(require_api_key)])
    async def score(request: Request, payload: ScoreRequest) -> ScoreResponse:
        try:
            return score_asset(request.app.state.artifacts, payload)
        except InsufficientHistoryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/score/batch", dependencies=[Depends(require_api_key)])
    async def score_batch(request: Request, payload: BatchRequest) -> list[dict]:
        artifacts: LoadedArtifacts = request.app.state.artifacts
        results: list[dict] = []
        for item in payload.assets:
            asset_id = item.get("asset_id") if isinstance(item, dict) else None
            try:
                parsed = ScoreRequest.model_validate(item)
            except ValidationError as exc:
                results.append({
                    "asset_id": asset_id,
                    "error": f"invalid request: {_first_validation_message(exc)}",
                })
                continue
            try:
                results.append(score_asset(artifacts, parsed).model_dump(mode="json"))
            except InsufficientHistoryError as exc:
                results.append({"asset_id": parsed.asset_id, "error": str(exc)})
        return results

    return app


app = create_app()
