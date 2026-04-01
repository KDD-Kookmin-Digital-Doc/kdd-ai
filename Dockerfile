# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime stage ----
FROM python:3.11-slim

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /install /usr/local
COPY app/ ./app/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
