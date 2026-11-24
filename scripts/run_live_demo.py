"""M4 live demo: stream a synthetic asset + a replayed engine through
the M3 scoring service and log every scored cycle.

Feeds, both scored in-process through the real HTTP layer
(``fastapi.testclient.TestClient`` against ``api.main.create_app`` —
the same routes, validation and artifact loader a deployed server
runs; only the transport is in-process):

- ``sim-pump-01``: a synthetic degrading asset from
  :mod:`src.simulator`, parameters derived from the FD001 training
  data. Its designed end of life is its last simulated cycle.
- ``fd001-test-engine-41``: the recorded FD001 test engine 41 replayed
  verbatim — the M1/M2 dangerous-overestimate case (true RUL 18 at its
  last observed cycle, 123; true end of life at cycle 141).

Each cycle from the minimum scoreable history onward is POSTed as the
window ``cycles 1..t`` (the service scores the window's last cycle),
and one log row per scored cycle is written to
``results/live_demo_log.csv``. A health-trajectory chart goes to
``results/live_demo_health.png`` and the run summary is merged into
``results/m4_metrics.json``.

Run:  .venv/bin/python scripts/run_live_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import MIN_HISTORY_CYCLES, create_app
from src import data as d
from src import simulator as sim

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS = ROOT / "results"

REPLAY_UNIT = 41

#: Synthetic asset configuration. degradation_scale 1.0 = exactly the
#: drift FD001 engines show over their lives; the asset's life length
#: (190) sits inside the FD001 training life range (128-362).
SYNTHETIC = {
    "asset_id": "sim-pump-01",
    "n_cycles": 190,
    "seed": 7,
    "degradation_scale": 1.0,
    "curve_k": 2.5,
}


def window_payload(frame: pd.DataFrame, upto: int, asset_id: str) -> dict:
    """Cycles 1..``upto`` of a frame as a /score request body."""
    part = frame.iloc[:upto]
    return {
        "asset_id": asset_id,
        "cycles": [
            {
                "cycle": int(r["cycle"]),
                "settings": [float(r[c]) for c in d.SETTING_COLS],
                "sensors": [float(r[c]) for c in d.SENSOR_COLS],
            }
            for _, r in part.iterrows()
        ],
    }


def stream_asset(client: TestClient, frame: pd.DataFrame, asset_id: str,
                 source: str, end_of_life: int) -> pd.DataFrame:
    """Score every cycle of one asset through the service, live-style."""
    rows = []
    for t in range(MIN_HISTORY_CYCLES, len(frame) + 1):
        resp = client.post("/score", json=window_payload(frame, t, asset_id))
        if resp.status_code != 200:
            raise RuntimeError(f"{asset_id} cycle {t}: {resp.status_code} "
                               f"{resp.text[:200]}")
        body = resp.json()
        rows.append({
            "asset_id": asset_id,
            "source": source,
            "cycle": t,
            "predicted_rul": body["predicted_rul"],
            "health_index": body["health_index"],
            "anomaly_score": body["anomaly_score"],
            "anomaly_percentile": body["anomaly_percentile"],
            "alert_level": body["alert_level"],
            "cycles_remaining_true": end_of_life - t,
        })
    return pd.DataFrame(rows)


#: An alert crossing counts as *confirmed* when health stays at or
#: below the band for this many consecutive scored cycles. Raw first
#: crossings are also reported (they are the M2 lead-time definition);
#: the confirmed crossing is the operator-meaningful one, because the
#: raw series flaps around the band edges — including a two-cycle
#: "watch" blip at cycles 11-12 of both demo assets, a residual of the
#: M2 warm-up boundary in the anomaly component.
CONFIRM_RUN = 5


def _confirmed_crossing(log: pd.DataFrame, threshold: float) -> int | None:
    below = (log["health_index"] <= threshold).to_numpy()
    cycles = log["cycle"].to_numpy()
    for i in range(len(below) - CONFIRM_RUN + 1):
        if below[i:i + CONFIRM_RUN].all():
            return int(cycles[i])
    return None


def summarise(log: pd.DataFrame, end_of_life: int) -> dict:
    """Alert trajectory of one asset from its live log."""
    def first_cycle(level: str) -> int | None:
        hits = log.loc[log["alert_level"] == level, "cycle"]
        return int(hits.iloc[0]) if len(hits) else None

    first_critical = first_cycle("critical")
    confirmed_watch = _confirmed_crossing(log, 70.0)
    confirmed_critical = _confirmed_crossing(log, 50.0)
    last = log.iloc[-1]
    return {
        "cycles_scored": int(len(log)),
        "end_of_life_cycle": int(end_of_life),
        "first_cycle_per_level": {
            lvl: first_cycle(lvl) for lvl in ("none", "watch", "critical")
        },
        "critical_lead_time_cycles": (
            int(end_of_life - first_critical)
            if first_critical is not None else None
        ),
        "confirmed_watch_cycle": confirmed_watch,
        "confirmed_critical_cycle": confirmed_critical,
        "confirmed_critical_lead_time_cycles": (
            int(end_of_life - confirmed_critical)
            if confirmed_critical is not None else None
        ),
        "confirmation_rule": f"health at/below the band for "
                             f"{CONFIRM_RUN} consecutive scored cycles",
        "final_cycle": int(last["cycle"]),
        "final_predicted_rul": float(last["predicted_rul"]),
        "final_health_index": float(last["health_index"]),
        "final_anomaly_percentile": float(last["anomaly_percentile"]),
        "final_alert_level": str(last["alert_level"]),
    }


def chart(log: pd.DataFrame, summaries: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharey=True)
    titles = {
        "sim-pump-01": "Synthetic asset (sim-pump-01) — simulated degradation",
        "fd001-test-engine-41": "Replayed FD001 test engine 41 — recorded data",
    }
    for ax, (asset_id, sub) in zip(axes, log.groupby("asset_id", sort=False)):
        s = summaries[asset_id]
        ax.axhspan(0, 50, color="#d64541", alpha=0.10)
        ax.axhspan(50, 70, color="#e8a33d", alpha=0.12)
        ax.axhspan(70, 100, color="#3d9970", alpha=0.08)
        ax.axhline(50, color="#d64541", lw=0.8, ls="--")
        ax.axhline(70, color="#e8a33d", lw=0.8, ls="--")
        ax.plot(sub["cycle"], sub["health_index"], color="#205c9c", lw=1.4)
        fc = s["confirmed_critical_cycle"]
        if fc is not None:
            row = sub[sub["cycle"] == fc].iloc[0]
            ax.plot([fc], [row["health_index"]], "o", color="#d64541",
                    ms=7, zorder=5)
            ax.annotate(
                f"sustained critical from cycle {fc}\n"
                f"({s['confirmed_critical_lead_time_cycles']} cycles "
                f"before end of life)",
                xy=(fc, row["health_index"]),
                xytext=(8, 18), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#d64541", lw=0.8),
            )
        ax.axvline(s["end_of_life_cycle"], color="black", lw=0.9, ls=":")
        ax.annotate("end of life", xy=(s["end_of_life_cycle"], 96),
                    xytext=(-4, 0), textcoords="offset points", fontsize=8,
                    ha="right")
        ax.set_title(titles.get(asset_id, asset_id), fontsize=10)
        ax.set_ylabel("health index")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("cycle")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def merge_metrics(section: dict) -> None:
    path = RESULTS / "m4_metrics.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(section)
    path.write_text(json.dumps(existing, indent=2))


def main() -> None:
    started = time.time()
    train = d.load_cmapss(ROOT / "data" / "train_FD001.txt")
    test = d.load_cmapss(ROOT / "data" / "test_FD001.txt")
    rul_at_end = d.load_rul(ROOT / "data" / "RUL_FD001.txt")

    params = sim.derive_params(train)

    synth_frame = sim.simulate_asset(
        params,
        n_cycles=SYNTHETIC["n_cycles"],
        seed=SYNTHETIC["seed"],
        degradation_scale=SYNTHETIC["degradation_scale"],
        curve_k=SYNTHETIC["curve_k"],
    )
    replay_frame = sim.replay_engine(test, REPLAY_UNIT)
    replay_eol = int(replay_frame["cycle"].max() + rul_at_end.loc[REPLAY_UNIT])

    app = create_app(MODELS_DIR)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        logs = [
            stream_asset(client, synth_frame, SYNTHETIC["asset_id"],
                         "synthetic", SYNTHETIC["n_cycles"]),
            stream_asset(client, replay_frame, f"fd001-test-engine-{REPLAY_UNIT}",
                         "replay", replay_eol),
        ]
    log = pd.concat(logs, ignore_index=True)
    log.to_csv(RESULTS / "live_demo_log.csv", index=False)

    summaries = {
        SYNTHETIC["asset_id"]: summarise(
            log[log["asset_id"] == SYNTHETIC["asset_id"]],
            SYNTHETIC["n_cycles"]),
        f"fd001-test-engine-{REPLAY_UNIT}": summarise(
            log[log["asset_id"] == f"fd001-test-engine-{REPLAY_UNIT}"],
            replay_eol),
    }
    chart(log, summaries, RESULTS / "live_demo_health.png")

    moving = {
        s: round(params.drift_in_sigmas(s), 2)
        for s in params.moving_sensors(min_sigmas=1.0)
    }
    merge_metrics({
        "milestone": "M4",
        "seed": SYNTHETIC["seed"],
        "scoring_path": "in-process TestClient against api.main.create_app "
                        "(real HTTP layer, artifact loader and validation)",
        "simulator": {
            "provenance": params.provenance,
            "n_training_engines": params.n_engines,
            "moving_sensors_drift_in_healthy_sigmas": moving,
            "constant_sensors": [
                s for s in d.SENSOR_COLS if params.sensor_noise_std[s] == 0.0
            ],
            "synthetic_asset_config": SYNTHETIC,
        },
        "live_demo": summaries,
        "live_demo_seconds": round(time.time() - started, 2),
    })

    for asset_id, s in summaries.items():
        print(f"{asset_id}: scored {s['cycles_scored']} cycles, "
              f"first levels {s['first_cycle_per_level']}, "
              f"confirmed watch @ {s['confirmed_watch_cycle']}, "
              f"confirmed critical @ {s['confirmed_critical_cycle']} "
              f"(lead {s['confirmed_critical_lead_time_cycles']} cycles), "
              f"final health {s['final_health_index']}, "
              f"final RUL {s['final_predicted_rul']}, "
              f"final level {s['final_alert_level']}")
    print(f"wrote results/live_demo_log.csv, results/live_demo_health.png, "
          f"results/m4_metrics.json in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
