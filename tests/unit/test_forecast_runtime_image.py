"""Deployment-image guards for the canonical forecasting runtime."""

import yaml

from common.core.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def test_docker_image_installs_all_five_model_runtimes():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "libgomp1" in dockerfile
    assert "procps" in dockerfile
    assert "--extra foundation" in dockerfile
    assert "--extra dl" in dockerfile
    assert "--extra statistical" in dockerfile
    assert "GUNICORN_WORKERS=1" in dockerfile
    assert "UV_NO_SYNC=1" in dockerfile
    assert "UV_FROZEN=1" in dockerfile


def test_docker_runtime_does_not_resolve_dependencies_at_startup():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "/app/.venv/bin/gunicorn" in dockerfile
    assert "uv run gunicorn" not in dockerfile
    assert "ENV UV_NO_SYNC=1 UV_FROZEN=1" in dockerfile


def test_compose_enforces_single_api_worker():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())

    assert compose["services"]["api"]["environment"]["GUNICORN_WORKERS"] == "1"


def test_compose_routes_api_mlflow_over_the_docker_network():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())

    assert compose["services"]["api"]["environment"]["MLFLOW_TRACKING_URI"] == (
        "http://mlflow:5000"
    )


def test_compose_api_can_read_and_persist_forecasting_artifacts():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())

    volumes = compose["services"]["api"]["volumes"]
    assert "./data:/app/data" in volumes
    assert "./config:/app/config" in volumes


def test_docker_context_excludes_local_data_secrets_and_build_outputs():
    ignored = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {".env", ".git", "data", "frontend/node_modules"} <= ignored
