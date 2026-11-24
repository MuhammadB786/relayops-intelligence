"""Composite health index: one 0-100 number per engine cycle.

Formula (documented, deliberately simple):

    rul_health      = 100 * clip(predicted_RUL / RUL_CAP, 0, 1)
    anomaly_health  = 100 - anomaly_percentile        (vs healthy reference)
    health          = W_RUL * rul_health + (1 - W_RUL) * anomaly_health

- ``rul_health`` reads the M1 model's capped scale: a predicted RUL at or
  above the 125-cycle cap is "as healthy as the model can express" (100);
  a predicted RUL of 0 is end of life (0).
- ``anomaly_health`` inverts the anomaly percentile from
  :mod:`src.anomaly`: a cycle that looks exactly like the healthy
  reference population sits near 100; one beyond the 99th percentile of
  healthy scores sits near 0.
- The weighting (0.6 RUL / 0.4 anomaly) favours the supervised signal,
  which carries the actual failure-time information, while letting the
  unsupervised signal pull health down when sensors behave abnormally in
  ways the RUL model's average behaviour would miss — the combination
  matters because M1's known failure mode is *overestimating* RUL on a
  few engines (see README, failure mode 2), exactly the case where an
  independent anomaly signal can compensate.

No leakage: predicted RUL comes from trailing features only (M1), the
anomaly percentile is computed against a fixed training reference, and
both are per-cycle quantities known at that cycle.
"""

from __future__ import annotations

import numpy as np

from src.data import RUL_CAP

#: Weight on the RUL component; the anomaly component gets ``1 - W_RUL``.
W_RUL = 0.6


def rul_component(predicted_rul: np.ndarray, cap: int = RUL_CAP) -> np.ndarray:
    """Map predicted RUL onto 0-100 on the capped scale."""
    return 100.0 * np.clip(np.asarray(predicted_rul, dtype=float) / cap, 0.0, 1.0)


def anomaly_component(anomaly_percentile: np.ndarray) -> np.ndarray:
    """Map an anomaly percentile (0-100) onto 0-100 health (inverted)."""
    return 100.0 - np.clip(np.asarray(anomaly_percentile, dtype=float), 0.0, 100.0)


def health_index(
    predicted_rul: np.ndarray,
    anomaly_percentile: np.ndarray,
    w_rul: float = W_RUL,
    cap: int = RUL_CAP,
) -> np.ndarray:
    """Composite health index in [0, 100]; higher = healthier."""
    health = w_rul * rul_component(predicted_rul, cap=cap) + (
        1.0 - w_rul
    ) * anomaly_component(anomaly_percentile)
    return np.clip(health, 0.0, 100.0)
