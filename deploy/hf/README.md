---
title: RelayOps ML — Predictive Maintenance Intelligence
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# RelayOps ML — scoring service (live demo)

The deployed FastAPI scoring service from the **relayops-intelligence**
project: remaining-useful-life regression, anomaly detection, a 0–100
health index, and a cost-weighted alert rule (health ≤ 50 → critical),
trained on NASA's C-MAPSS FD001 benchmark and evaluated honestly against
published baselines.

- Full case study, results, and code: https://github.com/MuhammadB786/relayops-intelligence
- Health check: `GET /healthz`
- Score an asset: `POST /score` with `{asset_id, cycles: [{cycle, settings[3], sensors[21]}, …]}`
  (minimum 11 cycles of history; shorter windows get a readable 422, not a guess)
- Batch scoring: `POST /score/batch`

The model bundle in this Space was trained during the image build by the
project's own `scripts/train_service.py`, which asserts its test metrics
reproduce the recorded M1 evaluation (RMSE 19.61, NASA score 966.678)
before the build is allowed to succeed.
