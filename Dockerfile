# RelayOps ML — M1 baseline image.
# The C-MAPSS data is gitignored and intentionally NOT baked into the image,
# so the default command runs the test suite plus a synthetic-data smoke
# train. To train for real:  docker run -v $PWD/data:/app/data <img> \
#   python scripts/train.py
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY conftest.py .

CMD ["sh", "-c", "python -m pytest tests/ -q && python scripts/smoke.py"]
