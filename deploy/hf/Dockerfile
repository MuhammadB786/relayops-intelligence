# RelayOps ML — Hugging Face Spaces deployment image.
# Unlike the repo's main Dockerfile (which keeps data/models out of the
# image), this image bakes the trained bundle in at build time: the build
# fetches the C-MAPSS data and runs the full service training, whose
# built-in cross-check asserts the reproduced metrics match the recorded
# M1 results — a failed assertion fails the build, loudly.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY scripts ./scripts
COPY results ./results

RUN python scripts/fetch_data.py && python scripts/train_service.py

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
