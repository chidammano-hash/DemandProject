FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY api/ api/
COPY common/ common/
COPY scripts/ scripts/
COPY config/ config/
COPY sql/ sql/

EXPOSE 8000

# Default to 4 workers — override via GUNICORN_WORKERS env var. Single-worker
# uvicorn was the dominant scalability bottleneck: 13 concurrent CA endpoints
# all share one event loop, so any sync DB call blocks every other request.
# Multi-worker requires Redis as the cache backend (otherwise per-process caches
# can't share state). See common/services/cache.py.
ENV GUNICORN_WORKERS=4
ENV GUNICORN_TIMEOUT=60

# sh -c so we can interpolate ${GUNICORN_WORKERS}; exec keeps gunicorn as PID 1
# so SIGTERM from `docker stop` reaches it (graceful shutdown).
CMD ["sh", "-c", "exec uv run gunicorn api.main:app --bind 0.0.0.0:8000 --workers ${GUNICORN_WORKERS} --worker-class uvicorn.workers.UvicornWorker --timeout ${GUNICORN_TIMEOUT} --access-logfile - --error-logfile -"]
