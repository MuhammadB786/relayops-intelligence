"""Request/response schemas for the scoring API (pydantic).

Validation policy: reject, never repair. Wrong sensor counts, non-numeric
or non-finite values, empty windows and non-increasing cycle numbers are
all 422s with readable messages — inputs are scored exactly as given, so
nothing is silently clipped, reordered, or filled in.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field, model_validator

from src import data as d

N_SETTINGS = len(d.SETTING_COLS)  # 3
N_SENSORS = len(d.SENSOR_COLS)  # 21


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be a finite number")
    return value


FiniteFloat = Annotated[float, AfterValidator(_finite)]


class CycleInput(BaseModel):
    """One cycle of one asset: CMAPSS-shaped raw readings."""

    cycle: int = Field(ge=1, description="Asset cycle counter (its true age)")
    settings: list[FiniteFloat] = Field(min_length=N_SETTINGS, max_length=N_SETTINGS)
    sensors: list[FiniteFloat] = Field(min_length=N_SENSORS, max_length=N_SENSORS)


class ScoreRequest(BaseModel):
    """A window of recent cycles for one asset, oldest first."""

    asset_id: str = Field(min_length=1)
    cycles: list[CycleInput] = Field(min_length=1)

    @model_validator(mode="after")
    def _cycles_strictly_increasing(self) -> "ScoreRequest":
        numbers = [c.cycle for c in self.cycles]
        if any(b <= a for a, b in zip(numbers, numbers[1:])):
            raise ValueError(
                "cycles must be strictly increasing (ordered oldest first, "
                "no duplicates)"
            )
        return self


class ScoreResponse(BaseModel):
    asset_id: str
    predicted_rul: float
    health_index: float
    anomaly_score: float
    anomaly_percentile: float
    alert_level: Literal["none", "watch", "critical"]
    model_version: str
    scored_at: datetime


class BatchRequest(BaseModel):
    """Batch scoring input.

    Items are accepted as raw objects and validated one by one inside the
    handler, so a single malformed asset produces a per-item error entry
    instead of rejecting the whole batch.
    """

    assets: list[Any] = Field(min_length=1)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    artifact_checksum: str
    feature_count: int
