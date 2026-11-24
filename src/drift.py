"""Drift monitoring: Population Stability Index + a retraining trigger.

PSI compares a monitored window's distribution against a reference
distribution (here: FD001 training healthy cycles) per sensor:

    PSI = sum_bins (cur% - ref%) * ln(cur% / ref%)

Bins are the reference distribution's deciles, so each reference bin
holds ~10% by construction. Conventional reading (the bands used
across credit-risk and ML-monitoring practice):

    PSI < 0.10          stable — no action
    0.10 <= PSI < 0.25  moderate shift — watch
    PSI >= 0.25         significant shift — act

**Retraining trigger (the stated rule):** retraining is recommended
when **at least ``MIN_SENSORS`` (3) sensors show PSI >
``PSI_THRESHOLD`` (0.25) in each of ``CONSECUTIVE_WINDOWS`` (2)
consecutive monitored windows**. One hot window is an incident to look
at, not a retraining decision; a shift that persists across two
windows on several independent channels is a population change.
:func:`evaluate_windows` returns the flag plus the evidence (per-window
PSI values and the sensors flagged in every window).

Two honest caveats, by design:

- PSI is **univariate**: it sees each sensor alone, so a pure
  correlation change with unchanged marginals is invisible to it.
- PSI compares *populations*. A window of degrading assets shifts
  sensor distributions legitimately — drift monitoring must compare
  like-for-like (the reference here is healthy cycles, so monitored
  windows should be healthy-population cycles too), or degradation
  itself masquerades as drift. The M4 evaluation's control case is
  built exactly that way; see README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: PSI at/above which a single sensor counts as significantly shifted.
PSI_THRESHOLD = 0.25

#: Number of simultaneously shifted sensors the trigger requires.
MIN_SENSORS = 3

#: Consecutive windows the shift must persist across.
CONSECUTIVE_WINDOWS = 2

#: Proportion floor: empty bins get this share instead of 0 (log guard).
_EPS = 1e-4


def psi_band(psi: float) -> str:
    """Conventional PSI band: stable / moderate / significant."""
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    bin_edges: np.ndarray | None = None,
) -> float:
    """PSI of ``current`` against ``reference`` (0 = identical).

    Bin edges default to the reference's quantiles. A constant
    reference has no distribution to compare: PSI is 0 when the current
    values equal that constant and inf when they do not. A heavily
    quantized reference (fewer than 3 distinct decile edges — FD001's
    sensor_6 takes only two values in healthy data) is binned on its
    distinct values instead: PSI over the value proportions, which is
    the only distribution such a channel has.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI needs non-empty reference and current samples")

    if bin_edges is None:
        if np.all(ref == ref[0]):
            return 0.0 if np.all(cur == ref[0]) else float("inf")
        bin_edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0,
                                                            n_bins + 1)))
        if len(bin_edges) < 3:
            uniq = np.unique(ref)
            midpoints = (uniq[:-1] + uniq[1:]) / 2.0
            bin_edges = np.concatenate(([uniq[0]], midpoints, [uniq[-1]]))
    edges = np.asarray(bin_edges, dtype=float)
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf  # out-of-range values land in end bins

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), _EPS, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def sensor_psi(
    reference: pd.DataFrame,
    window: pd.DataFrame,
    columns: list[str],
) -> dict[str, float]:
    """Per-column PSI of a monitored window against the reference.

    Columns with zero reference variance (the constant FD001 channels)
    are skipped — they carry no distribution to drift; a change there
    is a schema event, not drift, and PSI would be degenerate.
    """
    out: dict[str, float] = {}
    for c in columns:
        if float(reference[c].std(ddof=0)) == 0.0:
            continue
        out[c] = population_stability_index(
            reference[c].to_numpy(), window[c].to_numpy()
        )
    return out


def evaluate_windows(
    reference: pd.DataFrame,
    windows: list[pd.DataFrame],
    columns: list[str],
    psi_threshold: float = PSI_THRESHOLD,
    min_sensors: int = MIN_SENSORS,
    consecutive: int = CONSECUTIVE_WINDOWS,
) -> dict:
    """Apply the retraining-trigger rule to a sequence of windows.

    Returns the flag and its evidence: per-window PSI tables, the
    sensors over threshold in each window, the sensors flagged in
    *every* one of the last ``consecutive`` windows, and readable
    reasons. The rule fires only when each of the last ``consecutive``
    windows has at least ``min_sensors`` sensors over threshold.
    """
    window_psis = [sensor_psi(reference, w, columns) for w in windows]
    flagged = [
        sorted([c for c, v in psis.items() if v > psi_threshold])
        for psis in window_psis
    ]

    reasons: list[str] = []
    sustained: list[str] = []
    fired = False
    if len(windows) < consecutive:
        reasons.append(
            f"only {len(windows)} window(s) supplied; the rule needs "
            f"{consecutive} consecutive windows — not evaluated"
        )
    else:
        recent = flagged[-consecutive:]
        sustained = sorted(set.intersection(*(set(f) for f in recent)))
        counts_ok = all(len(f) >= min_sensors for f in recent)
        if counts_ok:
            fired = True
            detail = (
                "sustained on: " + ", ".join(sustained)
                if sustained
                else "no single sensor flagged in every window, "
                     "but the per-window count rule is met"
            )
            reasons.append(
                f"PSI > {psi_threshold} on "
                f"{min(len(f) for f in recent)}+ sensors in each of the "
                f"last {consecutive} windows ({detail})"
            )
        else:
            worst = min(len(f) for f in recent)
            reasons.append(
                f"at most {worst} sensor(s) over PSI {psi_threshold} in one "
                f"of the last {consecutive} windows; the rule needs "
                f">= {min_sensors} in every one — no retraining recommended"
            )

    return {
        "retraining_recommended": fired,
        "reasons": reasons,
        "window_psis": window_psis,
        "flagged_sensors_per_window": flagged,
        "sustained_sensors": sustained,
        "rule": {
            "psi_threshold": psi_threshold,
            "min_sensors": min_sensors,
            "consecutive_windows": consecutive,
        },
    }
