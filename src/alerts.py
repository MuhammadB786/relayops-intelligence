"""Alert-threshold analysis: turning scores into maintenance decisions.

Framing: at every cycle the service either raises an alert or it doesn't.
The alert means "this asset will fail within ``HORIZON`` cycles — schedule
intervention now." Ground truth per cycle is therefore binary:
``rul_raw <= HORIZON``. Metrics are computed per cycle (the streaming
view an operator actually experiences), and thresholds are chosen with an
explicit cost model, not by maximising F1:

    expected_cost = FP * COST_FALSE_ALARM + FN * COST_MISSED_FAILURE

A missed failure means unplanned downtime; a false alarm means an
inspection and possibly an early, planned part swap. The README's problem
framing puts unplanned downtime on the order of $125k/hour against an
inspection cost orders of magnitude smaller, so the analysis uses a
50:1 ratio (one missed failure costs as much as fifty false alarms). The
ratio is a stated assumption, sensitivity to it is reported in the M2
results, and the chosen operating point is a written decision in the
README — not a number buried in code.
"""

from __future__ import annotations

import numpy as np

#: Alert horizon, in cycles: an alert claims failure within this window.
HORIZON = 30

#: Cost model (normalised units): a missed failure costs 50 false alarms.
COST_FALSE_ALARM = 1.0
COST_MISSED_FAILURE = 50.0


def alert_confusion(y_true: np.ndarray, alerts: np.ndarray) -> dict[str, int]:
    """Confusion counts for binary truth vs boolean alerts."""
    truth = np.asarray(y_true, dtype=bool)
    fired = np.asarray(alerts, dtype=bool)
    return {
        "tp": int(np.sum(truth & fired)),
        "fp": int(np.sum(~truth & fired)),
        "fn": int(np.sum(truth & ~fired)),
        "tn": int(np.sum(~truth & ~fired)),
    }


def precision_recall(conf: dict[str, int]) -> tuple[float, float]:
    """(precision, recall) from confusion counts.

    Degenerate conventions, stated rather than hidden: with no alerts
    fired, precision is 0.0 (the system claimed nothing, correctly or
    otherwise); with no positive cycles, recall is 0.0.
    """
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def expected_cost(
    conf: dict[str, int],
    cost_fp: float = COST_FALSE_ALARM,
    cost_fn: float = COST_MISSED_FAILURE,
) -> float:
    """Total cost of a confusion outcome under the stated cost model."""
    return conf["fp"] * cost_fp + conf["fn"] * cost_fn


def sweep_thresholds(
    signal: np.ndarray,
    y_true: np.ndarray,
    thresholds: list[float],
    direction: str,
) -> list[dict]:
    """Precision/recall/cost at each candidate threshold.

    ``direction="below"`` alerts when ``signal <= threshold`` (RUL,
    health); ``direction="above"`` alerts when ``signal >= threshold``
    (anomaly score). Rows are returned in the order thresholds are given.
    """
    sig = np.asarray(signal, dtype=float)
    rows = []
    for t in thresholds:
        fired = sig <= t if direction == "below" else sig >= t
        conf = alert_confusion(y_true, fired)
        precision, recall = precision_recall(conf)
        rows.append({
            "threshold": float(t),
            **conf,
            "precision": precision,
            "recall": recall,
            "expected_cost": expected_cost(conf),
        })
    return rows
