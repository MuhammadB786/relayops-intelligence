"""Synthetic-data smoke test: exercises the full pipeline without the
C-MAPSS download (used as the Docker image's default command)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as d
from src import evaluate as ev
from src import model as m


def synthetic_frame(n_units: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(60, 120))
        wear = np.linspace(0, 1, life)
        row = {c: np.zeros(life) for c in d.RAW_COLS}
        row["unit"] = np.full(life, unit)
        row["cycle"] = np.arange(1, life + 1)
        for c in d.SETTING_COLS:
            row[c] = rng.normal(0, 0.01, life)
        for i, c in enumerate(d.SENSOR_COLS):
            row[c] = 10 * i + 5 * wear + rng.normal(0, 0.1, life)
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    frame = d.add_rul(synthetic_frame())
    feats, cols = d.build_features(frame)
    model = m.train_model(feats[cols], feats["rul"])
    last = ev.last_cycle_rows(feats)
    pred = m.predict_rul(model, last[cols])
    score = ev.nasa_score(last["rul_raw"], pred)
    assert np.isfinite(score)
    print(f"smoke ok: {len(cols)} features, last-cycle NASA score={score:.2f}")


if __name__ == "__main__":
    main()
