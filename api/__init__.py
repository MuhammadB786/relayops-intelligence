"""FastAPI scoring service for the RelayOps ML pipeline (M3).

Loads the artifacts produced by ``scripts/train_service.py`` from
``models/`` and serves RUL / health / anomaly scores over HTTP. See
``api.artifacts`` for the artifact format and its validation rules.
"""
