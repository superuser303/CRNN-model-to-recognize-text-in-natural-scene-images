FROM python:3.10-slim

WORKDIR /app
COPY . .

# Install with dataset caching
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt && \
    python scripts/download_data.py

CMD ["python", "scripts/train.py"]