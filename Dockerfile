# RelayOps ML — pipeline + scoring-service image (M3).
# The C-MAPSS data and the trained artifacts (models/) are gitignored and
# intentionally NOT baked into the image, so the default command runs the
# test suite plus a synthetic-data smoke train.
#
# Train for real:   docker run -v $PWD/data:/app/data -v $PWD/models:/app/models \
#                     <img> python scripts/train_service.py
# Serve the API:    docker run -v $PWD/models:/app/models -p 8000:8000 <img> \
#                     uvicorn api.main:app --host 0.0.0.0 --port 8000
# (arguments to `docker run <img> ...` replace the default CMD below; the
# API fails fast at startup if the mounted models/ bundle is missing or
# stale — see api/artifacts.py.)
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY scripts ./scripts
COPY tests ./tests
COPY conftest.py .

CMD ["sh", "-c", "python -m pytest tests/ -q && python scripts/smoke.py"]
