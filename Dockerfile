FROM python:3.12-slim

WORKDIR /app

# LightGBM needs GNU OpenMP. Durable job ownership uses ``ps`` start/command
# markers to reject PID reuse, so the runtime image also needs procps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 procps \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management.
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev \
    --extra foundation \
    --extra dl \
    --extra statistical

# Copy application code
COPY api/ api/
COPY common/ common/
COPY scripts/ scripts/
COPY config/ config/
COPY sql/ sql/

EXPOSE 8000

# JobManager queues, group locks, recovery, and APScheduler ownership are
# process-local. Keep one API worker until execution moves to a durable worker;
# multiple workers can duplicate schedules and overlap model jobs.
ENV GUNICORN_WORKERS=1
ENV GUNICORN_TIMEOUT=60
# Every optional forecasting runtime is baked above. JobManager still invokes
# scripts through `uv run`; forbid it from pruning extras or reaching the
# network when a job starts.
ENV UV_NO_SYNC=1 UV_FROZEN=1

# Invoke the frozen environment directly so startup never resolves or installs
# dependencies. sh expands the worker settings; exec keeps gunicorn as PID 1 so
# SIGTERM from `docker stop` reaches it for graceful shutdown.
CMD ["sh", "-c", "exec /app/.venv/bin/gunicorn api.main:app --bind 0.0.0.0:8000 --workers ${GUNICORN_WORKERS} --worker-class uvicorn.workers.UvicornWorker --timeout ${GUNICORN_TIMEOUT} --access-logfile - --error-logfile -"]
