# RelayOps ML — Predictive Maintenance Intelligence

A standalone Python ML service that will score equipment health and
remaining useful life (RUL) for [RelayOps](https://github.com/MuhammadB786/RelayOps)
(a maintenance & reliability platform) and feed predictions back to it over
an API. This repository is deliberately separate from the RelayOps web app:
the model pipeline is a portfolio-grade, independently deployable artifact,
and integration is API-only (Milestone 3).

**Status: Milestone 2 complete** — M1 (data pipeline, baseline RUL model,
honest published evaluation) plus anomaly detection, a composite health
index, a costed alert-threshold decision, and a generalization run on the
harder FD002 subset. FastAPI serving and RelayOps integration (M3), a
live sensor simulator with drift monitoring (M4), and the full case
study (M5) follow.

## Problem

Unplanned equipment downtime is one of the most expensive failure modes in
asset-intensive operations (industry estimates put it on the order of
$125k/hour median across heavy industries, ~$1T/year globally). Maintenance
teams mostly still react: something breaks, then they scramble. The
alternative — predicting remaining useful life from sensor history early
enough to schedule the repair — is what this service builds, honestly
evaluated against the public benchmark the field uses for exactly this.

## Data

NASA **C-MAPSS** turbofan engine degradation simulation (FD001 subset):
100 run-to-failure training engines and 100 truncated test engines, each a
multivariate time series of 3 operating settings and 21 sensor channels
sampled per cycle, with one fault mode (HPC degradation) under one
operating condition. Test ground truth is the true RUL at each engine's
last observed cycle (`RUL_FD001.txt`).

Source used: the official NASA distribution mirrored by the PHM Society
data repository (`phm-datasets.s3.amazonaws.com/NASA/6. Turbofan Engine
Degradation Simulation Data Set.zip`). The raw text files live in `data/`
(gitignored, ~9 MB). M2 additionally uses FD002 from the same
distribution (260 run-to-failure training engines, 259 test engines, six
operating conditions, one fault mode).

## Approach (M1)

- **Labels** — piecewise-linear RUL with a cap of **125 cycles**: in early
  life an engine is effectively healthy, so its exact RUL (310 vs 260) is
  label noise; the cap focuses learning on the degradation regime. This is
  the convention used by most published C-MAPSS work (R_early = 125), which
  keeps our numbers comparable to theirs.
- **Features** — per engine, trailing rolling mean and standard deviation
  of every setting/sensor channel over the last 5 and 10 cycles, plus the
  raw channels and cycle number (engine age): 86 features after dropping
  the 7 channels that are constant in FD001 (`setting_3`, `sensor_1`,
  `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`).
- **Model** — `HistGradientBoostingRegressor` (scikit-learn), fixed seed
  42, 400 iterations. A tabular gradient-boosting baseline on purpose:
  fast, dependency-light, and a fair anchor before sequence models.

## Evaluation methodology

- **Split by engine, never by row.** Internal validation holds out 20 whole
  engines (ids divisible by 5); no cycle of a validation engine is ever
  trained on. Rolling features are trailing windows, so a row's features
  depend only on that engine's past and present — a unit test mutates
  future rows and asserts earlier features are unchanged.
- **Validation protocol.** Run-to-failure engines end at RUL 0 by
  construction, so last-cycle scoring would be meaningless for them.
  Validation instead scores predictions at fixed true-RUL horizons
  (100 / 75 / 50 / 25 / 10 cycles before failure), one point per engine
  per horizon.
- **Headline protocol (official test).** The final model is refit on all
  100 training engines with identical hyperparameters, then scored exactly
  as the literature scores FD001: one prediction per test engine at its
  last observed cycle, against the **uncapped** true RUL. Metrics: RMSE
  (cycles) and the NASA PHM'08 asymmetric score, which penalises late
  predictions (exp(d/10)) more than early ones (exp(−d/13)) — late is the
  dangerous direction in maintenance.

## Results (FD001, official test protocol)

| Method | RMSE ↓ | NASA score ↓ | Source |
|---|---|---|---|
| SVR (Babu et al., 2016) | 20.96 | — | Zhang et al., *Sensors* 2020, comparison table |
| CNN (Babu et al., 2016) | 18.45 | 1287 | IJPHM comparison table (below) |
| LSTM (Zheng et al., 2017) | 16.10 | 338 | IJPHM comparison table |
| MODBNE (Zhang et al., 2016) | 15.00 | 334 | IJPHM comparison table |
| DCNN (Li et al., 2018) | 12.61 | 274 | IJPHM comparison table |
| **This baseline (GBM, M1)** | **19.61** | **967** | `results/metrics.json` |

Internal validation (20 held-out engines, checkpoint protocol):
**RMSE 18.69, NASA score 1051** — consistent with the test result, i.e.
the validation protocol is not flattering the model.

**Where it wins:** it beats the classical SVR baseline on RMSE, and beats
the CNN on the NASA score (967 vs 1287) despite a worse RMSE — its errors
skew early (safe) rather than late (dangerous) more often than the CNN's.

**Where it loses — plainly:** every sequence model beats it. The LSTM is
3.5 RMSE points better and its NASA score is ~3× better; DCNN is in
another class (12.61 / 274). Two failure modes dominate, both visible in
`results/test_predictions.csv`:

1. **Long-life underestimation.** Engines truncated with true RUL
   114–136 are predicted at 64–89 (worst: engine 74, true 126 → 63).
   A model trained with labels capped at 125 rarely predicts above ~90,
   so healthy engines look closer to failure than they are. Operationally
   this means premature maintenance — wasteful, not dangerous.
2. **A few dangerous overestimates.** Engines 41 and 37 (true RUL 18 and
   21) are predicted at 75 and 64. Those late predictions are exactly what
   the NASA score punishes, and in a plant they are the failures that
   matter. Any alerting threshold built on this model (M2) must be set
   with this tail behaviour in mind, not the average error.

## Milestone 2 — anomaly detection, health index, alert decision

### Anomaly detection

An IsolationForest (300 trees, seed 42) trained **only on healthy engine
life**: the first 30% of each training engine's trajectory, the one
period labelable "healthy" without hindsight about the fault. Features
are the M1 trailing rolling features **minus `cycle`** — a detector that
can see engine age learns "old = anomalous" and its lead times become
meaningless; this one must react to sensor behaviour. Scores are
`-score_samples` (higher = more anomalous); thresholds are set from
percentiles of the healthy reference scores, never from the fitted
contamination guess, so the false-alarm rate is an explicit choice.

**Warm-up finding (a bug-class artifact, caught by evaluation).** The
first M2 run reported a median detection "lead time" of ~200 cycles at
every threshold — the detector appeared to fire at birth. Diagnosis:
during the first 10 cycles the rolling features come from partially
filled windows (at cycle 1 every rolling std is exactly 0), landing
those rows in a feature-space corner that scores as anomalous regardless
of health — mean cycle-1 score 0.59 vs a settled ~0.45, with 100% of
engines above the reference 97.5th percentile at cycle 1. Cycles 1–10
are therefore excluded everywhere (detector training, reference, and all
evaluation), and a deployment should report "warming up" for an asset's
first 10 cycles. The underlying signal is real: on training engines the
mean score rises monotonically as true RUL falls (0.44 at RUL 100–150 →
0.59 at RUL < 10).

**Lead time vs false alarms** (threshold = percentile of healthy
reference scores; "healthy segment" = cycles with true RUL > 60;
detection = score crosses the threshold at least once, lead time = true
RUL at first crossing):

Held-out validation engines (20, run to failure):

| Ref. percentile | Detection rate | Median lead (cycles) | False-alarm rate, healthy cycles | Engines with ≥1 false alarm |
|---|---|---|---|---|
| 90 | 1.00 | 140 | 16.0% | 95% |
| 95 | 1.00 | 106 | 10.9% | 70% |
| 97.5 | 1.00 | 78 | 8.0% | 65% |
| 99 | 1.00 | 72 | 6.1% | 65% |
| 99.5 | 1.00 | 64 | 4.8% | 55% |

Official test engines (100, truncated mid-life):

| Ref. percentile | Detection rate | Median lead (cycles) | False-alarm rate, healthy cycles | Engines with ≥1 false alarm |
|---|---|---|---|---|
| 90 | 0.76 | 158 | 12.8% | 73% |
| 95 | 0.67 | 148 | 8.3% | 60% |
| 97.5 | 0.60 | 124 | 5.5% | 46% |
| 99 | 0.53 | 112 | 3.5% | 39% |
| 99.5 | 0.45 | 81 | 2.7% | 27% |

The two tables measure different things and are not in conflict:
validation engines are observed to failure, so detection means "fires
before end of life" (it always does, eventually); test engines are cut
off mid-life, so a non-detection mostly means observation ended while
the engine was still far from failure — not a missed failure. The
tradeoff itself is the honest headline: at the 99.5th-percentile
threshold the detector still gives a median 64 cycles of warning on
run-to-failure engines while false-alarming on under 5% of healthy
cycles; loosening to the 90th percentile buys ~2× the lead time at ~3×
the false-alarm rate. There is no free point on this curve.

### Health index

A 0–100 composite per cycle, deliberately simple and documented in
`src/health.py`:

    rul_health     = 100 * clip(predicted_RUL / 125, 0, 1)
    anomaly_health = 100 - anomaly_percentile   (vs healthy reference)
    health         = 0.6 * rul_health + 0.4 * anomaly_health

Trend sanity (Spearman correlation of the series against cycle, per
engine; a health index should fall as the engine ages):

| Series | Validation median ρ | Test median ρ | Engines ending below start (val / test) |
|---|---|---|---|
| Composite | −0.93 | −0.64 | 100% / 83% |
| RUL component only | −0.97 | −0.92 | 100% / 100% |
| Anomaly component only | −0.82 | −0.29 | 100% / 72% |

Honest read: on run-to-failure trajectories the composite trends
correctly for every engine. On truncated test engines the anomaly
component wanders mid-life (median ρ −0.29) and dilutes the composite
below the RUL-only trend. The composite does **not** win on trend
smoothness — it earns its place in alerting, below, where the anomaly
component's independence from the RUL model is exactly what catches the
RUL model's worst mistakes.

### Alert thresholds — and the operating-point decision

Alerts are framed as **"failure within 30 cycles"**, scored per cycle
(the streaming view an operator experiences). Thresholds were swept on
both candidate signals (`results/alert_thresholds.csv`, full table) and
chosen on the **validation engines only**, under an explicit cost model:
a missed failure costs **50×** a false alarm (unplanned downtime is
priced in the problem framing at ~$125k/hour; an inspection is orders of
magnitude cheaper). Chosen points, then evaluated **unchanged** on the
test set:

| Signal / rule | Validation P / R | Validation cost | Test P / R | Test cost |
|---|---|---|---|---|
| **Health ≤ 50 (chosen)** | 0.43 / 1.000 | **824** | 0.165 / 0.994 | **1770** |
| Predicted RUL ≤ 90 (best RUL-only) | 0.38 / 0.997 | 1111 | 0.114 / 0.985 | 2796 |

Per-engine view on test: only 25 of the 100 engines are ever observed
inside the 30-cycle window; both rules alert inside the window for
**25 of 25**. Sensitivity: at a 10:1 cost ratio the optimum moves
conservative (health ≤ 30: P 0.60 / R 0.97, and health still wins,
612 vs 636); at 100:1 the health point is unchanged and still wins
(824 vs 1463). The recommendation is not an artifact of the exact ratio.

> **Decision: alert when the health index ≤ 50.**
> Chosen on validation by the stated 50:1 cost model; confirmed on the
> test set without re-tuning (recall 0.994 — 2 missed window-cycles of
> 332 — at the lowest cost of any candidate on both sets).
>
> The deciding evidence is M1's documented dangerous-overestimate mode.
> Test engine 41 (true RUL 18 at last observation) is predicted at
> **75** by the RUL model: its RUL component alone reads 60/100, and a
> RUL-only alert rule needs a threshold of ≥76 to catch it at all — the
> best RUL-only rule (≤90) does catch it, but pays 2,546 false-alarm
> cycles on test for that bluntness. Engine 41's anomaly percentile, by
> contrast, sits at ~100 — the sensors are screaming even though the
> RUL model is calm — pulling its health index to **36**, comfortably
> under the alert line. Engine 37 (true 21, predicted 64) behaves the
> same way (health 31). The composite exists for exactly these engines.
>
> The accepted price, stated plainly: per-cycle precision on test is
> 0.165 — about five false-alarm cycles per true alert cycle — and 52%
> of engines see at least one alert while still healthy (RUL > 60).
> Two mitigations keep this honest rather than alarming: per-cycle
> costing overstates operator burden, because an alert that persists
> across consecutive cycles is one inspection episode, not fifty; and
> the cost model says the asymmetry is correct — the failure mode being
> bought against is the unplanned outage, not the inspection. If a
> deployment's true cost ratio is closer to 10:1, the sensitivity
> analysis says to move the line to health ≤ 30, not to abandon the
> signal.

### Generalization: FD002 (six operating conditions)

The M1 pipeline run unchanged on FD002 (260 train / 259 test engines),
plus the standard remedy — operating-condition normalization (KMeans
k=6 on the settings, per-condition sensor z-scoring, all statistics fit
on training rows only, applied before feature building):

| Variant | Validation RMSE / score | Test RMSE / score |
|---|---|---|
| FD001 reference (M1) | 18.69 / 1051 | 19.61 / 967 |
| FD002 baseline | 17.70 / 2152 | **27.60 / 8658** |
| FD002 condition-normalized | 16.16 / 1655 | 27.72 / 13,666 |

Published FD002 context (official protocol): DCNN 22.36 / 10,412;
LSTM 24.49; CNN 30.29. The baseline GBM (27.60) lands between the
published CNN and LSTM on RMSE, and its NASA score (8,658) is lower
than the DCNN's published score despite the worse RMSE — its late
errors are less catastrophic. Cross-paper comparisons are indicative,
not exact.

What the numbers say, plainly:

- **FD002 is materially harder for this pipeline**: +8 RMSE points and
  ~9× the NASA score vs FD001. Six operating conditions move the sensor
  baselines, so the same reading means different health under different
  conditions, and rolling features entangle condition changes with
  degradation. Published models degrade the same way (DCNN: 12.61 →
  22.36); ours degrades more.
- **The remedy did not transfer.** Condition normalization improved
  validation (RMSE 17.70 → 16.16) but not the official test (27.60 →
  27.72, score 8,658 → 13,666). Error-tail analysis explains it:
  normalization *reduced* the count of >30-cycle late predictions
  (16 → 11 engines) but made the worst ones worse (max +49 → +51; the
  five worst engines account for 68% of its score vs 55% for the
  baseline), and the NASA score's exponential late penalty is decided
  by exactly that tail. Validation on run-to-failure engines flattered
  the remedy because checkpoint scoring samples trajectories that
  resemble training; the truncated test endpoints are where the shift
  bites. Verdict: the remedy as implemented is **not validated** by the
  official protocol. Named next experiments, not claims: per-condition
  models, and condition identity as an explicit model feature.

## Limitations

- FD002 attempted (above): the pipeline degrades materially under six
  operating conditions and the standard normalization remedy did not
  transfer to the official test protocol. FD004 (six conditions, two
  fault modes) not attempted.
- The first 10 cycles of any engine's life are unscoreable by design
  (rolling-feature warm-up; see the M2 anomaly section).
- Alert precision/recall are per-cycle; a deployment would alert on
  episodes, and episode-level metrics remain future work. The chosen
  operating point inherits the 50:1 cost-ratio assumption — sensitivity
  to 10:1 and 100:1 is reported, but the true ratio is a business
  decision, not a measurable constant.
- The cap-125 label convention structurally biases long-RUL predictions
  downward (failure mode 1); a two-stage or uncapped-with-sample-weighting
  variant is a known next experiment, not yet run.
- No per-engine uncertainty estimate; point predictions only.
- The Dockerfile is written to spec but has **not been build-verified**
  (no Docker daemon on the build machine). Its default command runs the
  test suite plus a synthetic-data smoke train, which do pass locally.

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# place train_FD001.txt / test_FD001.txt / RUL_FD001.txt in data/
# (NASA C-MAPSS distribution; see "Data" above; FD002 files likewise)
.venv/bin/python -m pytest tests/ -q     # 35 tests
.venv/bin/python scripts/train.py        # M1: writes results/ + models/
.venv/bin/python scripts/run_m2.py       # M2: anomaly, health, alerts
.venv/bin/python scripts/run_fd002.py    # M2 stretch: FD002 run
.venv/bin/python scripts/smoke.py        # synthetic-data pipeline check
```

All scripts are deterministic (seed 42). `scripts/train.py` takes ~20 s
end-to-end on a laptop CPU; `run_m2.py` ~30 s; `run_fd002.py` ~2 min.
CI (GitHub Actions, Python 3.12) runs the test suite on push and pull
request; the tests are synthetic-data only and need no downloads.

## Layout

```
src/data.py       loading, RUL labels, engine splits, rolling features
src/model.py      gradient-boosting baseline, persistence (joblib)
src/evaluate.py   RMSE, NASA asymmetric score, last-cycle selection
src/anomaly.py    IsolationForest on healthy life, scores, lead times
src/health.py     composite 0-100 health index (documented formula)
src/alerts.py     alert confusion/precision/recall, costed threshold sweep
scripts/train.py  M1 end-to-end train + evaluate -> results/metrics.json
scripts/run_m2.py M2: anomaly + health + alert decision -> m2_metrics.json,
                  alert_thresholds.csv, anomaly_lead_times.csv
scripts/run_fd002.py  FD002 baseline + condition-normalized -> fd002_metrics.json
scripts/smoke.py  synthetic-data smoke run (Docker default command)
tests/            parsing, labels, leakage, metrics, model, anomaly,
                  health, alerts (35 tests)
results/          metrics.json (M1), m2_metrics.json, fd002_metrics.json,
                  alert_thresholds.csv, anomaly_lead_times.csv,
                  test_predictions.csv, feature_columns.json
```

## References

- Saxena, A. et al. *Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation* (C-MAPSS dataset + NASA scoring function),
  PHM 2008.
- Comparison values: *Remaining Useful Life Prediction Using
  Attention-LSTM Neural Network of Aircraft Engines*, IJPHM 2025
  (Table 3) — https://papers.phmsociety.org/index.php/ijphm/article/download/4274/2620
- SVR value: Zhang, C. et al., *A Double-Channel Hybrid Deep Neural
  Network Based on CNN and BiLSTM for Remaining Useful Life Prediction*,
  Sensors 2020 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7764061/
- FD002 comparison values: *Remaining Useful Life Estimation of Aircraft
  Engines Using Differentiable Architecture Search*, Mathematics 2022
  (Table 8) — https://www.mdpi.com/2227-7390/10/3/352/pdf — corroborated
  by Muneer, A. et al., *Data-Driven Deep Learning-Based Attention
  Mechanism for Remaining Useful Life Prediction*, Electronics 2021.
