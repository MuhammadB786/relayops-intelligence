# RelayOps ML — Predictive Maintenance Intelligence

A standalone Python ML service that will score equipment health and
remaining useful life (RUL) for [RelayOps](https://github.com/MuhammadB786/RelayOps)
(a maintenance & reliability platform) and feed predictions back to it over
an API. This repository is deliberately separate from the RelayOps web app:
the model pipeline is a portfolio-grade, independently deployable artifact,
and integration is API-only (Milestone 3).

**Status: Milestone 1 complete** — data pipeline, feature engineering,
baseline RUL model, and honest published evaluation on the standard
C-MAPSS FD001 benchmark. Anomaly/health scoring (M2), FastAPI serving and
RelayOps integration (M3), a live sensor simulator with drift monitoring
(M4), and the full case study (M5) follow.

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
(gitignored, ~9 MB).

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

## Limitations

- FD001 only: one operating condition, one fault mode. FD002/FD004 (six
  conditions) are materially harder and not yet attempted.
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
# (NASA C-MAPSS distribution; see "Data" above)
.venv/bin/python -m pytest tests/ -q     # 18 tests
.venv/bin/python scripts/train.py        # writes results/ + models/
.venv/bin/python scripts/smoke.py        # synthetic-data pipeline check
```

`scripts/train.py` is fully deterministic (seed 42) and takes ~20 s
end-to-end on a laptop CPU: feature build + two trainings + evaluation.

## Layout

```
src/data.py       loading, RUL labels, engine splits, rolling features
src/model.py      gradient-boosting baseline, persistence (joblib)
src/evaluate.py   RMSE, NASA asymmetric score, last-cycle selection
scripts/train.py  end-to-end train + evaluate, writes results/metrics.json
scripts/smoke.py  synthetic-data smoke run (Docker default command)
tests/            parsing, labels, leakage, metrics, model roundtrip
results/          metrics.json, test_predictions.csv, feature_columns.json
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
