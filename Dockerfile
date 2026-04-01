# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime stage ----
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
