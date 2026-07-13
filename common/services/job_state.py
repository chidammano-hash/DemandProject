"""Pure state management for the job engine (no APScheduler, no psycopg at module level).

Contains:
- DB connection helper (_get_conn)
- JobTypeDef dataclass
- Job callable wrappers (_run_*)
- job-row serialization helper
- _SCRIPTS_DIR / _UV constants

Deliberately free of APScheduler and psycopg imports at the module level so
that this module can be imported from tests without starting the full API or
requiring a running scheduler.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

from common.core.sql_helpers import row_to_dict_from_cols

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _get_conn():
    """Open a single psycopg connection using get_db_params()."""
    import psycopg  # imported here to keep module-level imports APScheduler-free

    from common.core.db import get_db_params

    return psycopg.connect(**get_db_params(), autocommit=True)


# ---------------------------------------------------------------------------
# Job type definition
# ---------------------------------------------------------------------------


@dataclass
class JobTypeDef:
    """Metadata for a registered job type."""

    type_id: str
    label: str
    description: str
    group: str  # concurrency group — one active job per group
    callable: Callable[..., dict[str, Any]]  # (params, progress_cb) -> result dict
    params_schema: dict[str, Any] = field(default_factory=dict)
    default_max_retries: int = 0


# ---------------------------------------------------------------------------
# Job type callables — thin wrappers around existing scripts
# ---------------------------------------------------------------------------

from common.core.paths import DATA_DIR as _DATA_DIR  # noqa: E402
from common.core.paths import SCRIPTS_DIR as _SCRIPTS_DIR  # noqa: E402

_UV = "uv"


_SUBPROCESS_TIMEOUT = 7200  # 2 hours — prevents hung jobs from blocking the executor thread
_SUBPROCESS_POLL_INTERVAL = 0.1
_PROCESS_IDENTITY_CAPTURE_TIMEOUT = 2
_PROCESS_IDENTITY_PARAM = "__process_identity"
_LOG_FLUSH_INTERVAL = 5  # seconds between DB log flushes
_LOG_FLUSH_LINES = 20  # flush after this many buffered lines
_CURRENT_ATTEMPT_TOKEN: ContextVar[str | None] = ContextVar(
    "job_attempt_token",
    default=None,
)

_ATTEMPT_WRAPPER = r"""
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

attempt_token, result_name, gate_name, *command = sys.argv[1:]
command_digest = hashlib.sha256(
    json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()
gate = pathlib.Path(gate_name)
deadline = time.monotonic() + 30
while not gate.exists():
    if time.monotonic() >= deadline:
        exit_code = 125
        break
    time.sleep(0.02)
else:
    try:
        output_path = pathlib.Path(result_name).with_suffix(".log")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as durable_output:
            child = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert child.stdout is not None
            stream_to_parent = True
            for line in child.stdout:
                durable_output.write(line)
                durable_output.flush()
                if stream_to_parent:
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except (BrokenPipeError, OSError):
                        stream_to_parent = False
                        sys.stdout = open(os.devnull, "w", encoding="utf-8")
            exit_code = child.wait()
    except OSError:
        exit_code = 127

payload = {
    "attempt_token": attempt_token,
    "exit_code": int(exit_code),
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command_digest": command_digest,
}
result_path = pathlib.Path(result_name)
result_path.parent.mkdir(parents=True, exist_ok=True)
temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
os.replace(temporary, result_path)
raise SystemExit(exit_code)
"""


class JobCancelledError(RuntimeError):
    """Raised when a managed job is cancelled by its user."""


# LightGBM membership gate + output-dir lookup for the tuning job
# callables (used as `model in MODEL_OUTPUT_DIRS` in a few places — do NOT add
# other models here, it would widen those gates).
MODEL_OUTPUT_DIRS: dict[str, str] = {
    "lgbm": "lgbm_cluster",
}

# Backtest model key → output directory under data/backtest/, for the FULL
# roster. Single source of truth shared by _update_backtest_run_on_completion
# and _auto_load_backtest so the metadata read and the auto-load read the same
# directory. Tree aliases map to the _cluster dir; everything else is identity.
_BACKTEST_OUTPUT_DIRS: dict[str, str] = {
    "lgbm": "lgbm_cluster",
    "lgbm_cluster": "lgbm_cluster",
    "chronos2_enriched": "chronos2_enriched",
    "mstl": "mstl",
    "nbeats": "nbeats",
    "nhits": "nhits",
}


# ---------------------------------------------------------------------------
# PID + log DB helpers (used by _run_subprocess for resilient jobs)
# ---------------------------------------------------------------------------


def bind_job_attempt(attempt_token: str) -> Token[str | None]:
    """Bind one durable attempt token to the current worker context."""
    if not attempt_token:
        raise ValueError("Attempt token must be non-empty")
    return _CURRENT_ATTEMPT_TOKEN.set(attempt_token)


def reset_job_attempt(context_token: Token[str | None]) -> None:
    """Restore the worker context after an attempt exits."""
    _CURRENT_ATTEMPT_TOKEN.reset(context_token)


def _command_digest(cmd: list[str]) -> str:
    """Return the wrapper-compatible digest for an exact command vector."""
    payload = json.dumps(
        cmd,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _attempt_file_stem(job_id: str, attempt_token: str) -> str:
    return hashlib.sha256(f"{job_id}:{attempt_token}".encode()).hexdigest()


def _prepare_attempt_files(job_id: str, attempt_token: str) -> tuple[Path, Path]:
    """Create a clean durable result slot and a closed execution gate."""
    attempt_dir = _DATA_DIR / "job_attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stem = _attempt_file_stem(job_id, attempt_token)
    result_path = attempt_dir / f"{stem}.json"
    gate_path = attempt_dir / f"{stem}.gate"
    output_path = attempt_dir / f"{stem}.log"
    result_path.unlink(missing_ok=True)
    gate_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
    return result_path, gate_path


def _attempt_result_path(job_id: str, attempt_token: str) -> Path:
    stem = _attempt_file_stem(job_id, attempt_token)
    return _DATA_DIR / "job_attempts" / f"{stem}.json"


def _validate_attempt_result(
    payload: object,
    attempt_token: str,
    expected_command_digest: str | None = None,
) -> dict[str, Any] | None:
    """Validate an exact wrapper exit record, returning None on ambiguity."""
    if not isinstance(payload, dict):
        return None
    if payload.get("attempt_token") != attempt_token:
        return None
    exit_code = payload.get("exit_code")
    completed_at = payload.get("completed_at")
    command_digest = payload.get("command_digest")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return None
    if not isinstance(completed_at, str) or not completed_at:
        return None
    if not isinstance(command_digest, str) or not command_digest:
        return None
    if expected_command_digest is not None and command_digest != expected_command_digest:
        return None
    return {
        "attempt_token": attempt_token,
        "exit_code": exit_code,
        "completed_at": completed_at,
        "command_digest": command_digest,
    }


def load_attempt_result(
    job_id: str,
    attempt_token: str | None,
    *,
    expected_command_digest: str | None = None,
) -> dict[str, Any] | None:
    """Load a wrapper result only when token and command identity are exact."""
    if not attempt_token:
        return None
    result_path = _attempt_result_path(job_id, attempt_token)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "Ignored malformed attempt result for job %s token %s",
            job_id,
            attempt_token,
            exc_info=True,
        )
        return None
    return _validate_attempt_result(
        payload,
        attempt_token,
        expected_command_digest,
    )


def _store_pid(job_id: str | None, pid: int) -> None:
    """Store the subprocess PID in job_history for kill/recovery."""
    if not job_id:
        return
    try:
        with _get_conn() as conn:
            conn.execute("UPDATE job_history SET pid = %s WHERE job_id = %s", (pid, job_id))
    except Exception:
        logger.warning("Failed to store PID %d for job %s", pid, job_id)


def _store_attempt_process(
    job_id: str,
    pid: int,
    identity: dict[str, str],
    attempt_token: str,
    command_digest: str,
) -> bool:
    """Atomically attach a child identity to its exact running attempt."""
    try:
        with _get_conn() as conn:
            result = conn.execute(
                """UPDATE job_history
                   SET pid = %s,
                       params = jsonb_set(
                           jsonb_set(
                               COALESCE(params, '{}'::jsonb),
                               '{__process_identity}',
                               %s::jsonb,
                               true
                           ),
                           '{__attempt_command_digest}',
                           %s::jsonb,
                           true
                       ),
                       attempt_result = NULL
                   WHERE job_id = %s
                     AND status = 'running'
                     AND attempt_token = %s""",
                (
                    pid,
                    json.dumps(identity),
                    json.dumps(command_digest),
                    job_id,
                    attempt_token,
                ),
            )
        return int(result.rowcount or 0) == 1
    except Exception:  # noqa: BLE001,RUF100 — caller kills the child and fails closed.
        logger.exception(
            "Failed to persist child identity for job %s attempt %s",
            job_id,
            attempt_token,
        )
        return False


def _store_attempt_result(
    job_id: str,
    attempt_token: str,
    attempt_result: dict[str, Any],
) -> bool:
    """Persist a wrapper exit record without touching a newer attempt."""
    validated = _validate_attempt_result(attempt_result, attempt_token)
    if validated is None:
        raise ValueError("Attempt result does not match its token")
    try:
        with _get_conn() as conn:
            result = conn.execute(
                """UPDATE job_history
                   SET attempt_result = %s::jsonb
                   WHERE job_id = %s
                     AND attempt_token = %s""",
                (json.dumps(validated), job_id, attempt_token),
            )
        return int(result.rowcount or 0) == 1
    except Exception:  # noqa: BLE001,RUF100 — recovery keeps the job quarantined.
        logger.exception(
            "Failed to persist exit result for job %s attempt %s",
            job_id,
            attempt_token,
        )
        return False


def _store_process_identity(
    job_id: str | None,
    pid: int,
    identity: dict[str, str],
) -> None:
    """Persist the OS start/command markers used to reject recycled PIDs."""
    if not job_id:
        return
    try:
        with _get_conn() as conn:
            conn.execute(
                """UPDATE job_history
                   SET params = jsonb_set(
                       COALESCE(params, '{}'::jsonb),
                       '{__process_identity}',
                       %s::jsonb,
                       true
                   )
                   WHERE job_id = %s AND pid = %s""",
                (json.dumps(identity), job_id, pid),
            )
    except Exception:  # noqa: BLE001,RUF100 — identity persistence failure is handled fail-closed.
        logger.warning(
            "Failed to store process identity for PID %d (job %s)",
            pid,
            job_id,
            exc_info=True,
        )


def _clear_pid(job_id: str | None, attempt_token: str | None = None) -> None:
    """Clear the PID column after subprocess exits."""
    if not job_id:
        return
    try:
        with _get_conn() as conn:
            if attempt_token is None:
                conn.execute(
                    """UPDATE job_history
                       SET pid = NULL,
                           params = COALESCE(params, '{}'::jsonb) - %s
                       WHERE job_id = %s""",
                    (
                        _PROCESS_IDENTITY_PARAM,
                        job_id,
                    ),
                )
            else:
                conn.execute(
                    """UPDATE job_history
                       SET pid = NULL,
                           params = COALESCE(params, '{}'::jsonb) - %s
                       WHERE job_id = %s
                         AND attempt_token = %s""",
                    (
                        _PROCESS_IDENTITY_PARAM,
                        job_id,
                        attempt_token,
                    ),
                )
    except Exception:
        logger.warning("Failed to clear PID for job %s", job_id)


def _append_log(job_id: str | None, text: str) -> None:
    """Append text to the persistent log column in job_history."""
    if not job_id or not text:
        return
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE job_history SET log = COALESCE(log, '') || %s WHERE job_id = %s",
                (text, job_id),
            )
    except Exception:
        logger.warning("Failed to append log for job %s", job_id)


def get_job_log(job_id: str) -> str:
    """Read the persistent log for a job. Returns empty string if not found."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(log, '') FROM job_history WHERE job_id = %s", (job_id,)
            ).fetchone()
        return row[0] if row else ""
    except Exception:
        logger.warning("Failed to read log for job %s", job_id)
        return ""


def get_job_pid(job_id: str) -> int | None:
    """Read the PID for a running job. Returns None if not found or cleared."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT pid FROM job_history WHERE job_id = %s", (job_id,)
            ).fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        logger.warning("Failed to read PID for job %s", job_id)
        return None


def get_job_process_identity(job_id: str) -> dict[str, str] | None:
    """Read the persisted identity for the process currently owned by a job."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT params -> %s FROM job_history WHERE job_id = %s",
                (_PROCESS_IDENTITY_PARAM, job_id),
            ).fetchone()
        identity = row[0] if row else None
        if isinstance(identity, str):
            identity = json.loads(identity)
        if not isinstance(identity, dict):
            return None
        start_marker = identity.get("start_marker")
        command_marker = identity.get("command_marker")
        if not isinstance(start_marker, str) or not isinstance(command_marker, str):
            return None
        return {
            "start_marker": start_marker,
            "command_marker": command_marker,
        }
    except Exception:  # noqa: BLE001,RUF100 — callers fail closed when identity cannot be read.
        logger.warning("Failed to read process identity for job %s", job_id, exc_info=True)
        return None


def _read_process_marker(pid: int, field: str) -> str | None:
    """Read one stable process field through portable ``ps`` output."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", f"{field}="],
            capture_output=True,
            text=True,
            timeout=_PROCESS_IDENTITY_CAPTURE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("Failed to inspect process %d field %s", pid, field, exc_info=True)
        return None
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        return None
    return value


def _read_linux_process_identity(pid: int) -> dict[str, str] | None:
    """Read clock-stable process identity from Linux procfs.

    ``ps lstart`` is wall-clock based and can change after host sleep or clock
    correction. Linux boot id plus the kernel start-time ticks cannot.
    """
    proc_dir = Path("/proc") / str(pid)
    if not proc_dir.exists():
        return None
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        stat_text = (proc_dir / "stat").read_text()
        command = (proc_dir / "cmdline").read_bytes()
        closing_paren = stat_text.rfind(")")
        if closing_paren < 0:
            return None
        fields_after_command = stat_text[closing_paren + 2 :].split()
        start_ticks = fields_after_command[19]
    except (OSError, IndexError, UnicodeError):
        logger.warning("Failed to inspect Linux process identity for PID %d", pid)
        return None
    if not boot_id or not start_ticks or not command:
        return None
    return {
        "start_marker": f"linux:{boot_id}:{start_ticks}",
        "command_marker": hashlib.sha256(command).hexdigest(),
    }


def capture_process_identity(pid: int) -> dict[str, str] | None:
    """Capture markers that distinguish a live child from later PID reuse."""
    linux_identity = _read_linux_process_identity(pid)
    if linux_identity is not None:
        return linux_identity
    start_marker = _read_process_marker(pid, "lstart")
    command = _read_process_marker(pid, "command")
    if start_marker is None or command is None:
        return None
    return {
        "start_marker": start_marker,
        "command_marker": hashlib.sha256(command.encode("utf-8")).hexdigest(),
    }


def process_identity_matches(
    pid: int,
    expected: dict[str, str] | None,
) -> bool | None:
    """Return True for the same process, False for PID reuse, None if unknown.

    A different OS start marker proves PID reuse. A command-only mismatch can
    result from a legitimate ``exec``, so it is treated as unverifiable and
    callers fail closed rather than signalling or duplicating the process.
    """
    if expected is None:
        return None
    current = capture_process_identity(pid)
    if current is None:
        return None
    if current["start_marker"] != expected.get("start_marker"):
        return False
    if current["command_marker"] != expected.get("command_marker"):
        return None
    return True


def _terminate_subprocess(proc: subprocess.Popen[str]) -> None:
    """Terminate a managed process group, escalating when SIGTERM is ignored."""
    try:
        process_group = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
        proc.wait(timeout=5)
        return
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=5)


def _read_subprocess_output(
    stream: Any,
    output_queue: Queue[tuple[str, str | BaseException | None]],
) -> None:
    """Read a blocking stdout stream without blocking cancellation polling."""
    try:
        for line in stream:
            output_queue.put(("line", line))
    except (OSError, ValueError) as exc:
        output_queue.put(("error", exc))
    finally:
        output_queue.put(("eof", None))


def _run_subprocess(
    cmd: list[str],
    progress_cb: Callable | None = None,
    step_msg: str = "",
    cancel_event: Event | None = None,
    job_id: str | None = None,
    timeout_seconds: float | None = None,
    env_overrides: dict[str, str] | None = None,
) -> str:
    """Run a subprocess command with PID tracking, cancellation, and log streaming.

    - Subprocess runs in its own process group (start_new_session=True) so it
      survives API restarts.
    - PID is stored in job_history for real kill and startup recovery.
    - stdout is streamed to DB log column periodically.
    - cancel_event and timeout are polled independently of stdout activity.

    Returns the full stdout as a string. Raises on failure, timeout, or cancel.
    """
    if progress_cb and step_msg:
        progress_cb(msg=step_msg)

    attempt_token = _CURRENT_ATTEMPT_TOKEN.get() if job_id else None
    original_cmd = list(cmd)
    expected_command_digest: str | None = None
    result_path: Path | None = None
    gate_path: Path | None = None
    if job_id and attempt_token:
        expected_command_digest = _command_digest(original_cmd)
        result_path, gate_path = _prepare_attempt_files(job_id, attempt_token)
        cmd = [
            sys.executable,
            "-u",
            "-c",
            _ATTEMPT_WRAPPER,
            attempt_token,
            str(result_path),
            str(gate_path),
            *original_cmd,
        ]

    # Force line-buffered stdout so log streaming is real-time, not block-buffered
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        **(env_overrides or {}),
    }
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(_SCRIPTS_DIR.parent),
        start_new_session=True,
        env=env,
    )

    # A managed command is wrapped behind a file gate. The domain process cannot
    # start until this exact attempt owns the persisted PID + OS identity.
    if job_id and attempt_token:
        identity = capture_process_identity(proc.pid)
        persisted = bool(
            identity
            and expected_command_digest
            and _store_attempt_process(
                job_id,
                proc.pid,
                identity,
                attempt_token,
                expected_command_digest,
            )
        )
        if not persisted:
            _terminate_subprocess(proc)
            _clear_pid(job_id, attempt_token)
            raise RuntimeError("Failed to persist attempt identity before child execution")
        assert gate_path is not None
        try:
            gate_path.write_text(attempt_token, encoding="utf-8")
        except OSError as exc:
            _terminate_subprocess(proc)
            _clear_pid(job_id, attempt_token)
            raise RuntimeError("Failed to release persisted child attempt") from exc
    else:
        # Direct library callers do not have a durable job attempt. Retain
        # legacy PID visibility without pretending it is restart-safe.
        _store_pid(job_id, proc.pid)
        identity = capture_process_identity(proc.pid)
        if identity is None:
            logger.error(
                "Could not capture identity for PID %d (job %s)",
                proc.pid,
                job_id,
            )
        else:
            _store_process_identity(job_id, proc.pid, identity)

    stdout_lines: list[str] = []
    log_buffer: list[str] = []
    last_flush = time.monotonic()
    output_queue: Queue[tuple[str, str | BaseException | None]] = Queue()
    assert proc.stdout is not None
    reader = Thread(
        target=_read_subprocess_output,
        args=(proc.stdout, output_queue),
        name=f"job-output-{proc.pid}",
        daemon=True,
    )
    reader.start()

    try:
        timeout = _SUBPROCESS_TIMEOUT if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("Subprocess timeout must be positive")
        start = time.monotonic()
        output_complete = False
        while True:
            if cancel_event and cancel_event.is_set():
                _terminate_subprocess(proc)
                raise JobCancelledError("Job cancelled by user")

            if time.monotonic() - start > timeout:
                _terminate_subprocess(proc)
                raise RuntimeError("Subprocess timed out")

            if output_complete and proc.poll() is not None:
                break

            try:
                event, payload = output_queue.get(timeout=_SUBPROCESS_POLL_INTERVAL)
            except Empty:
                continue

            if event == "eof":
                output_complete = True
                continue
            if event == "error":
                _terminate_subprocess(proc)
                raise RuntimeError("Failed to read subprocess output") from payload
            if not isinstance(payload, str):
                _terminate_subprocess(proc)
                raise RuntimeError("Subprocess output reader returned invalid data")

            stripped = payload.rstrip("\n")
            stdout_lines.append(stripped)
            log_buffer.append(payload)
            if progress_cb and stripped:
                try:
                    progress_cb(msg=stripped)
                except JobCancelledError:
                    _terminate_subprocess(proc)
                    raise

            if log_buffer and (
                len(log_buffer) >= _LOG_FLUSH_LINES
                or time.monotonic() - last_flush > _LOG_FLUSH_INTERVAL
            ):
                _append_log(job_id, "".join(log_buffer))
                log_buffer.clear()
                last_flush = time.monotonic()

        proc.wait(timeout=30)

        exit_code = proc.returncode
        if job_id and attempt_token:
            assert expected_command_digest is not None
            attempt_result = load_attempt_result(
                job_id,
                attempt_token,
                expected_command_digest=expected_command_digest,
            )
            if attempt_result is None:
                raise RuntimeError("Managed subprocess exited without an exact attempt result")
            if not _store_attempt_result(job_id, attempt_token, attempt_result):
                raise RuntimeError("Failed to persist managed subprocess exit result")
            exit_code = int(attempt_result["exit_code"])

        if exit_code != 0:
            error_msg = "\n".join(stdout_lines[-20:]) or "Unknown error"
            raise RuntimeError(f"Command failed: {' '.join(original_cmd)}\n{error_msg.strip()}")

        return "\n".join(stdout_lines)
    finally:
        try:
            if proc.poll() is None:
                _terminate_subprocess(proc)
        finally:
            if log_buffer:
                _append_log(job_id, "".join(log_buffer))
            reader.join(timeout=1)
            if gate_path is not None:
                gate_path.unlink(missing_ok=True)
            if attempt_token is None:
                _clear_pid(job_id)
            else:
                _clear_pid(job_id, attempt_token)


def _run_cluster_scenario(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a what-if clustering scenario (delegates to run_clustering_scenario.py).

    When params contains ``experiment_id``, the cluster_experiment row is updated
    to ``status='running'`` before starting, and the experiment_id is forwarded to
    ``run_scenario()`` so it can write results on completion/failure.
    """
    from scripts.ml.run_clustering_scenario import (
        generate_scenario_id,
        get_scenario_result,
        run_scenario,
    )

    scenario_id = params.get("scenario_id") or generate_scenario_id()
    experiment_id: int | None = params.get("experiment_id")

    if progress_cb:
        progress_cb(pct=5, msg="Starting clustering scenario")

    # If this is a cluster experiment, mark it as running
    if experiment_id is not None:
        try:
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE cluster_experiment SET status = 'running', started_at = NOW() "
                    "WHERE experiment_id = %s",
                    (experiment_id,),
                )
        except Exception:
            logger.warning("Failed to update cluster_experiment %d to running", experiment_id)

    try:
        run_scenario(
            scenario_id=scenario_id,
            feature_params=params.get("feature_params"),
            model_params=params.get("model_params"),
            label_params=params.get("label_params"),
            relabel_only=params.get("relabel_only", False),
            previous_scenario_id=params.get("previous_scenario_id"),
            experiment_id=experiment_id,
        )
    except Exception:
        # run_scenario already handles updating experiment status to 'failed'
        # internally, but if it raises before that (unlikely), mark failed here
        if experiment_id is not None:
            try:
                with _get_conn() as conn:
                    conn.execute(
                        "UPDATE cluster_experiment SET status = 'failed', completed_at = NOW() "
                        "WHERE experiment_id = %s AND status = 'running'",
                        (experiment_id,),
                    )
            except Exception:
                logger.warning("Failed to mark cluster_experiment %d as failed", experiment_id)
        raise

    result = get_scenario_result(scenario_id) or {}
    return {"scenario_id": scenario_id, **result}


def _cluster_pipeline_payload(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize the public job parameters into the exact child CLI payload."""
    supported = {
        "feature_params",
        "model_params",
        "label_params",
        "label",
        "auto_promote",
        "time_window_months",
        "k_range",
    }
    unsupported = sorted(set(params) - supported)
    if unsupported:
        raise ValueError("Unsupported cluster pipeline parameter(s): " + ", ".join(unsupported))

    payload: dict[str, Any] = {
        "label": params.get("label", "Job Pipeline Run"),
        "auto_promote": params.get("auto_promote", True),
    }
    for key in ("feature_params", "model_params", "label_params"):
        value = params.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be an object")
        payload[key] = dict(value)

    if "time_window_months" in params:
        feature_params = dict(payload.get("feature_params") or {})
        feature_params["time_window_months"] = params["time_window_months"]
        payload["feature_params"] = feature_params
    if "k_range" in params:
        model_params = dict(payload.get("model_params") or {})
        model_params["k_range"] = params["k_range"]
        payload["model_params"] = model_params
    return payload


def verify_cluster_pipeline_completion(
    job_id: str,
    *,
    require_promoted: bool,
) -> dict[str, Any]:
    """Verify that one exact managed experiment finished its full lifecycle."""
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT experiment.experiment_id,
                   experiment.scenario_id,
                   experiment.status,
                   experiment.is_promoted,
                   experiment.total_dfus,
                   experiment.artifacts_path
            FROM job_history AS job
            JOIN cluster_experiment AS experiment
              ON experiment.job_id = job.job_id
             AND experiment.experiment_id =
                   (job.params ->> 'cluster_experiment_id')::bigint
             AND experiment.scenario_id =
                   job.params ->> 'cluster_scenario_id'
            WHERE job.job_id = %s
            """,
            (job_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError("Managed cluster pipeline has no exact experiment lineage")
    experiment_id, scenario_id, status, is_promoted, total_dfus, artifacts_path = row
    if status != "completed":
        raise RuntimeError(f"Cluster experiment {experiment_id} has unexpected status {status}")
    if int(total_dfus or 0) <= 0 or not artifacts_path:
        raise RuntimeError(f"Cluster experiment {experiment_id} has incomplete artifacts")
    if require_promoted and not bool(is_promoted):
        raise RuntimeError(f"Cluster experiment {experiment_id} was not promoted")
    return {
        "experiment_id": int(experiment_id),
        "scenario_id": str(scenario_id),
        "status": str(status),
        "is_promoted": bool(is_promoted),
        "total_dfus": int(total_dfus),
        "artifacts_path": str(artifacts_path),
    }


def reconcile_cluster_pipeline_experiment(
    job_id: str,
    terminal_status: str,
) -> bool:
    """Move the current exact managed experiment to failed or cancelled."""
    if terminal_status not in {"failed", "cancelled"}:
        raise ValueError("Cluster terminal status must be failed or cancelled")
    with _get_conn() as conn:
        result = conn.execute(
            """
            UPDATE cluster_experiment AS experiment
            SET status = %s,
                completed_at = NOW()
            FROM job_history AS job
            WHERE job.job_id = %s
              AND experiment.job_id = job.job_id
              AND experiment.experiment_id =
                    (job.params ->> 'cluster_experiment_id')::bigint
              AND experiment.scenario_id =
                    job.params ->> 'cluster_scenario_id'
              AND experiment.status IN ('queued', 'running')
            """,
            (terminal_status, job_id),
        )
    return int(result.rowcount or 0) == 1


def reconcile_backtest_run(job_id: str, terminal_status: str) -> bool:
    """Move the exact managed backtest run to a terminal state.

    Recovery can quarantine a subprocess after the normal backtest callable is
    no longer present to execute its exception handler.  Reconcile through the
    durable job-to-run lineage so an acknowledged quarantine cannot leave a
    phantom running backtest behind.
    """
    if terminal_status not in {"failed", "cancelled"}:
        raise ValueError("Backtest terminal status must be failed or cancelled")
    with _get_conn() as conn:
        result = conn.execute(
            """
            UPDATE backtest_run AS run
            SET status = %s,
                completed_at = NOW()
            FROM job_history AS job
            WHERE job.job_id = %s
              AND run.job_id = job.job_id
              AND run.id = NULLIF(job.params ->> 'backtest_run_id', '')::integer
              AND run.status IN ('queued', 'running')
            """,
            (terminal_status, job_id),
        )
    return int(result.rowcount or 0) == 1


def _run_cluster_pipeline(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the entire clustering lifecycle in one recoverable subprocess."""
    if not job_id:
        raise RuntimeError("Managed cluster pipeline requires a job id")
    attempt_token = _CURRENT_ATTEMPT_TOKEN.get()
    if not attempt_token:
        raise RuntimeError("Managed cluster pipeline requires an attempt token")

    payload = _cluster_pipeline_payload(params)
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    cmd = [
        _UV,
        "run",
        "python",
        str(_SCRIPTS_DIR / "ml" / "run_cluster_pipeline.py"),
        "--job-id",
        job_id,
        "--attempt-token",
        attempt_token,
        "--params-json",
        payload_json,
    ]

    if progress_cb:
        progress_cb(pct=5, msg="Starting unified clustering pipeline")
    output = _run_subprocess(
        cmd,
        progress_cb,
        "Running unified clustering pipeline",
        cancel_event=cancel_event,
        job_id=job_id,
        timeout_seconds=_subprocess_timeout_seconds("cluster_pipeline"),
    )
    result = verify_cluster_pipeline_completion(
        job_id,
        require_promoted=bool(payload["auto_promote"]),
    )

    if progress_cb:
        progress_cb(pct=100, msg="Pipeline completed")

    return {
        **result,
        "output_log": output or "Cluster pipeline completed",
    }


def _run_seasonality(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Legacy seasonality pipeline — delegates to unified SKU features."""
    return _run_compute_sku_features(params, progress_cb, cancel_event, job_id)


def _reserve_backtest_run(
    model_id: str,
    job_id: str | None,
    *,
    get_conn: Callable[[], Any] | None = None,
) -> int:
    """Create the tracking row required for a managed backtest execution.

    API-triggered runs reserve their row before submitting the job. Named
    pipelines submit backtest jobs directly, so they arrive here without a
    ``backtest_run_id``. Reserving one at execution time keeps both entry
    points on the same auto-load and champion-selection contract.
    """
    connection_factory = get_conn or _get_conn
    with connection_factory() as conn:
        row = conn.execute(
            """INSERT INTO backtest_run (model_id, job_id, status)
               VALUES (%s, %s, 'queued')
               RETURNING id""",
            (model_id, job_id),
        ).fetchone()
    if row is None:
        raise RuntimeError("Failed to create backtest tracking run")
    return int(row[0])


def _backtest_metadata_path(model: str) -> Path:
    model_dir = _BACKTEST_OUTPUT_DIRS.get(model, model)
    return _DATA_DIR / "backtest" / model_dir / "backtest_metadata.json"


def _mark_backtest_run_running(run_id: int) -> None:
    """Start or resume a non-terminal tracking row without reopening evidence."""
    with _get_conn() as conn:
        row = conn.execute(
            """UPDATE backtest_run
               SET status = 'running', started_at = NOW()
               WHERE id = %s
                 AND status IN ('queued', 'failed', 'running')
               RETURNING id""",
            (run_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError(
            f"Backtest run {run_id} is not eligible to start; completed evidence is immutable"
        )


def _load_governed_backtest_lineage() -> dict[str, Any]:
    """Capture the immutable sales and promoted-cluster inputs for a backtest."""
    from common.services.cluster_lineage import load_promoted_cluster_population
    from common.services.sales_lineage import load_completed_sales_lineage

    with _get_conn() as conn, conn.transaction():
        sales = load_completed_sales_lineage(conn)
        clusters = load_promoted_cluster_population(conn)
    if sales.batch_id <= 0 or clusters.experiment_id <= 0:
        raise RuntimeError("Governed backtest lineage identifiers must be positive")
    if clusters.assignment_count <= 0:
        raise RuntimeError("Governed backtests require promoted cluster assignments")
    return {
        "source_sales_batch_id": sales.batch_id,
        "data_checksum": sales.source_hash,
        "cluster_experiment_id": clusters.experiment_id,
        "cluster_assignment_count": clusters.assignment_count,
        "cluster_assignment_checksum": clusters.assignment_checksum,
    }


def record_backtest_artifact_identity(
    model: str,
    run_id: int,
    job_id: str | None,
    *,
    governed_lineage: dict[str, Any] | None = None,
) -> None:
    """Atomically bind generated metadata to one managed tracking run."""
    metadata_path = _backtest_metadata_path(model)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Backtest metadata is missing for {model}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Backtest metadata is invalid for {model}") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Backtest metadata is not an object for {model}")
    metadata["managed_execution"] = {
        "backtest_run_id": int(run_id),
        "job_id": job_id,
        "model_id": _BACKTEST_OUTPUT_DIRS.get(model, model),
    }
    if governed_lineage is None:
        metadata.pop("governed_lineage", None)
    else:
        metadata["governed_lineage"] = governed_lineage
    temporary_path = metadata_path.with_name(f".{metadata_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_path, metadata_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to bind backtest metadata to run {run_id}") from exc


def verify_backtest_artifact_identity(
    model: str,
    run_id: int,
    job_id: str | None,
) -> None:
    """Reject stale artifacts before loading them into shared forecast tables."""
    metadata_path = _backtest_metadata_path(model)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Backtest metadata is missing for {model}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Backtest metadata is invalid for {model}") from exc
    managed = metadata.get("managed_execution") if isinstance(metadata, dict) else None
    if not isinstance(managed, dict):
        raise RuntimeError(f"Backtest metadata has no managed identity for {model}")
    artifact_run_id = managed.get("backtest_run_id")
    artifact_model_id = managed.get("model_id")
    artifact_job_id = managed.get("job_id")
    expected_model_id = _BACKTEST_OUTPUT_DIRS.get(model, model)
    if artifact_run_id != int(run_id):
        raise RuntimeError(f"Backtest artifact belongs to run {artifact_run_id}, expected {run_id}")
    if artifact_model_id != expected_model_id:
        raise RuntimeError(
            f"Backtest artifact belongs to model {artifact_model_id}, expected {expected_model_id}"
        )
    if job_id is not None and artifact_job_id != job_id:
        raise RuntimeError(f"Backtest artifact belongs to job {artifact_job_id}, expected {job_id}")


def verify_backtest_tracking_identity(
    model: str,
    run_id: int,
    job_id: str,
) -> None:
    """Verify that a recovered artifact still owns its tracking row."""
    with _get_conn() as conn:
        row = conn.execute(
            """SELECT model_id, job_id, status
               FROM backtest_run
               WHERE id = %s""",
            (run_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"Backtest tracking run {run_id} no longer exists")
    tracked_model, tracked_job_id, status = row
    expected_model = _BACKTEST_OUTPUT_DIRS.get(model, model)
    actual_model = _BACKTEST_OUTPUT_DIRS.get(str(tracked_model), str(tracked_model))
    if actual_model != expected_model or tracked_job_id != job_id:
        raise RuntimeError(
            f"Backtest tracking run {run_id} is owned by "
            f"{tracked_model}/{tracked_job_id}, expected {expected_model}/{job_id}"
        )
    if status != "running":
        raise RuntimeError(f"Backtest tracking run {run_id} has unexpected status {status}")


def _update_backtest_run_on_completion(run_id: int, model: str) -> None:
    """Update a backtest_run row with results from the completed backtest metadata."""
    # Backtest key → output directory (shared with _auto_load_backtest).
    meta_path = _backtest_metadata_path(model)

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot complete backtest run {run_id} without valid metadata") from exc
    if not isinstance(meta, dict):
        raise RuntimeError(f"Backtest run {run_id} metadata is not an object")
    acc = meta.get("accuracy_at_execution_lag", {})
    with _get_conn() as conn:
        result = conn.execute(
            """UPDATE backtest_run SET
                status = 'completed', completed_at = NOW(),
                accuracy_pct = %s, wape = %s, bias = %s,
                n_predictions = %s, n_dfus = %s,
                metadata = %s::jsonb
            WHERE id = %s AND status = 'running'""",
            (
                acc.get("accuracy_pct"),
                acc.get("wape"),
                acc.get("bias"),
                meta.get("n_predictions"),
                meta.get("n_dfus"),
                json.dumps(meta),
                run_id,
            ),
        )
    if int(result.rowcount or 0) != 1:
        raise RuntimeError(f"Backtest run {run_id} was not in the running state at completion")


def _auto_load_backtest(
    model: str,
    run_id: int,
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> None:
    """Load a completed backtest's predictions into the DB.

    Runs immediately after a successful backtest so results show up in accuracy
    views and the Item Analysis ``forecast_<model>`` line without a separate
    "Load" click. Resolves the output directory via ``_BACKTEST_OUTPUT_DIRS``
    (the same map :func:`_update_backtest_run_on_completion` uses to read
    metadata) and reuses :func:`_run_load_backtest_model`, which writes
    ``fact_external_forecast_monthly`` + refreshes ``agg_forecast_monthly``.

    Loading is part of successful backtest completion. Missing predictions or
    loader errors propagate so the managed job and ``backtest_run`` both fail;
    the UI must never present an unloaded run as successfully completed.

    Note: the load refreshes the ``agg_forecast_monthly`` MV. Under a parallel
    "Run all", concurrent refreshes serialize on the MV lock (slower, not a
    deadlock); sequential runs (the default) avoid this.
    """
    from common.core.paths import PROJECT_ROOT as ROOT

    model_dir = _BACKTEST_OUTPUT_DIRS.get(model, model)
    pred_path = ROOT / "data" / "backtest" / model_dir / "backtest_predictions.csv"
    if not pred_path.exists():
        raise RuntimeError(f"Backtest predictions file is missing for {model}")
    _run_load_backtest_model(
        {"model_id": model_dir, "run_id": run_id},
        progress_cb,
        cancel_event,
        job_id,
    )


def _mark_backtest_run_failed(run_id: int) -> None:
    """Best-effort status update for any generation or auto-load failure."""
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE backtest_run SET status = 'failed', completed_at = NOW() WHERE id = %s",
                (run_id,),
            )
    except Exception:
        logger.warning("Failed to mark backtest_run %d as failed", run_id, exc_info=True)


def _subprocess_timeout_seconds(workload: str) -> float:
    """Return the configured wall-clock allowance for one managed workload."""
    from common.core.utils import load_config

    config = load_config("pipelines.yaml") or {}
    timeout_config = config.get("job_timeouts_seconds") or {}
    if not isinstance(timeout_config, dict):
        raise ValueError("pipelines.job_timeouts_seconds must be a mapping")
    raw_timeout = timeout_config.get(
        workload,
        timeout_config.get("default", _SUBPROCESS_TIMEOUT),
    )
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid subprocess timeout for {workload}") from exc
    if timeout <= 0:
        raise ValueError(f"Subprocess timeout for {workload} must be positive")
    return timeout


def _run_backtest(
    model: str,
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a backtest for a given model type.

    Supports LightGBM (direct script), foundation models (module invocation),
    deep learning (--model flag), and statistical baselines.
    """
    # LightGBM uses the direct backtest script.
    tree_scripts = {
        "lgbm": "scripts/ml/run_backtest.py",
    }
    # Foundation models: python -m invocation
    foundation_modules = {
        "chronos2_enriched": "scripts.ml.run_backtest_chronos2_enriched",
    }
    # Models whose backtest scripts need a heavy optional dependency, mapped to the
    # extra that provides it. The Chronos 2 Enriched foundation model imports
    # chronos-forecasting (the `foundation` extra); nbeats/nhits import
    # neuralforecast (the `dl` extra). Both pull torch.
    _MODEL_EXTRAS = {
        "chronos2_enriched": "foundation",
        "nhits": "dl",
        "nbeats": "dl",
        "mstl": "statistical",
    }
    # Special scripts: direct file path
    special_scripts = {
        "mstl": "scripts/ml/run_backtest_mstl.py",
        "nhits": ("scripts/ml/run_backtest_dl.py", ["--model", "nhits"]),
        "nbeats": ("scripts/ml/run_backtest_dl.py", ["--model", "nbeats"]),
    }

    supported_models = set(tree_scripts) | set(foundation_modules) | set(special_scripts)
    if model not in supported_models:
        raise ValueError(f"Unknown backtest model: {model}")

    output_model = str(params.get("model_id") or model)
    tracking_model_id = _BACKTEST_OUTPUT_DIRS.get(output_model, output_model)
    backtest_run_id = params.get("backtest_run_id")
    if backtest_run_id is None:
        backtest_run_id = _reserve_backtest_run(tracking_model_id, job_id)
    if progress_cb:
        progress_cb(pct=0, msg=f"Running {model} backtest")

    # Mark as running without ever reopening completed governed evidence.
    if backtest_run_id:
        _mark_backtest_run_running(int(backtest_run_id))

    # Pass --extra for models that need a heavy optional dep, so `uv run` ensures
    # it's present even after a plain `uv sync` would otherwise strip it (the
    # recurring "was working, then not" failure).
    run_prefix = [_UV, "run"]
    extra = _MODEL_EXTRAS.get(model)
    if extra:
        run_prefix += ["--extra", extra]

    if model in tree_scripts:
        cmd = [*run_prefix, "python", tree_scripts[model]]
        model_id = params.get("model_id")
        if model_id:
            cmd.extend(["--model-id", str(model_id)])
    elif model in foundation_modules:
        cmd = [*run_prefix, "python", "-m", foundation_modules[model]]
    elif model in special_scripts:
        entry = special_scripts[model]
        if isinstance(entry, tuple):
            script, extra_args = entry
            cmd = [*run_prefix, "python", script, *extra_args]
        else:
            cmd = [*run_prefix, "python", entry]
    else:
        raise ValueError(f"Unknown backtest model: {model}")

    config_path = params.get("config")
    if config_path:
        cmd.extend(["--config", config_path])
    if params.get("resume"):
        cmd.append("--resume")

    # Final artifacts are valid only for this invocation. Checkpoints live in a
    # separate directory and remain available to --resume.
    model_dir = _BACKTEST_OUTPUT_DIRS.get(output_model, output_model)
    artifact_dir = _SCRIPTS_DIR.parent / "data" / "backtest" / model_dir
    for filename in (
        "backtest_predictions.csv",
        "backtest_predictions_all_lags.csv",
        "backtest_metadata.json",
    ):
        (artifact_dir / filename).unlink(missing_ok=True)

    governed_lineage: dict[str, Any] | None = None
    try:
        if params.get("governed"):
            governed_lineage = _load_governed_backtest_lineage()
        output = _run_subprocess(
            cmd,
            progress_cb,
            cancel_event=cancel_event,
            job_id=job_id,
            timeout_seconds=_subprocess_timeout_seconds(model),
        )
        if governed_lineage is not None:
            ending_lineage = _load_governed_backtest_lineage()
            if ending_lineage != governed_lineage:
                raise RuntimeError("Governed backtest source lineage changed during execution")
    except Exception:
        if backtest_run_id:
            _mark_backtest_run_failed(backtest_run_id)
        raise

    # Auto-load predictions into the DB so backtest results appear without a
    # separate "Load" click. Run it BEFORE the completion update so, in the happy
    # path, is_loaded_to_db is set before the UI sees status='completed' (avoids a
    # poll landing in the gap and showing completed-but-unloaded). The completion
    # update happens only after a successful load. A load failure is a
    # failed managed run because execution-lag accuracy is not usable until both
    # database destinations are populated.
    try:
        record_backtest_artifact_identity(
            output_model,
            int(backtest_run_id),
            job_id,
            governed_lineage=governed_lineage,
        )
        verify_backtest_artifact_identity(
            output_model,
            int(backtest_run_id),
            job_id,
        )
        _auto_load_backtest(output_model, backtest_run_id, progress_cb, cancel_event, job_id)
        _update_backtest_run_on_completion(backtest_run_id, output_model)
    except Exception:
        _mark_backtest_run_failed(backtest_run_id)
        raise

    if progress_cb:
        progress_cb(pct=100, msg=f"{model} backtest complete (results loaded)")

    return {
        "model": model,
        "backtest_run_id": backtest_run_id,
        "output_log": output if output else "Completed",
    }


def _make_backtest_runner(model: str):
    """Factory to create a backtest runner for a specific model."""

    def runner(
        params: dict[str, Any],
        progress_cb: Callable | None = None,
        cancel_event: Event | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        return _run_backtest(model, params, progress_cb, cancel_event=cancel_event, job_id=job_id)

    runner.__name__ = f"_run_backtest_{model}"
    return runner


_run_backtest_lgbm = _make_backtest_runner("lgbm")
_run_backtest_chronos2_enriched = _make_backtest_runner("chronos2_enriched")
_run_backtest_mstl = _make_backtest_runner("mstl")
_run_backtest_nhits = _make_backtest_runner("nhits")
_run_backtest_nbeats = _make_backtest_runner("nbeats")


def _run_champion_experiment(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a champion selection strategy experiment."""
    experiment_id = params["experiment_id"]
    if progress_cb:
        progress_cb(pct=5, msg=f"Starting champion experiment #{experiment_id}")

    # Store job_id on experiment record
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE champion_experiment SET job_id = %s WHERE experiment_id = %s",
                (job_id, experiment_id),
            )
    except Exception:
        logger.warning("Failed to store job_id on champion experiment %d", experiment_id)

    cmd = [
        _UV,
        "run",
        "python",
        "scripts/ml/run_champion_experiment.py",
        "--experiment-id",
        str(experiment_id),
    ]
    output = _run_subprocess(
        cmd, progress_cb, "Running champion experiment", cancel_event=cancel_event, job_id=job_id
    )
    if progress_cb:
        progress_cb(pct=100, msg=f"Champion experiment #{experiment_id} completed")
    return {"experiment_id": experiment_id, "output_log": output or "Champion experiment completed"}


def _finalize_champion_results_lineage(
    experiment_id: int,
    job_id: str | None,
    winners_csv: Path,
) -> dict[str, Any]:
    """Idempotently bind loaded champion rows to one exact experiment."""
    from common.services.forecast_lineage import (
        compute_champion_results_stats,
        sha256_file,
    )

    routing_checksum = sha256_file(winners_csv)
    with _get_conn() as conn, conn.transaction(), conn.cursor() as cur:
        results_stats = compute_champion_results_stats(cur, experiment_id)
        if results_stats.row_count <= 0:
            raise ValueError("Champion results load produced no stamped rows")
        cur.execute(
            "UPDATE champion_experiment SET is_results_promoted = FALSE "
            "WHERE experiment_id != %s AND is_results_promoted = TRUE",
            (experiment_id,),
        )
        cur.execute(
            "UPDATE champion_experiment SET is_results_promoted = TRUE, "
            "results_promoted_at = NOW(), results_promote_job_id = %s, "
            "results_artifact_checksum = %s, results_forecast_checksum = %s, "
            "results_forecast_row_count = %s "
            "WHERE experiment_id = %s",
            (
                job_id,
                routing_checksum,
                results_stats.checksum,
                results_stats.row_count,
                experiment_id,
            ),
        )
        if cur.rowcount != 1:
            raise ValueError("Champion results audit row was not updated")
    return {
        "routing_artifact_checksum": routing_checksum,
        "results_forecast_checksum": results_stats.checksum,
        "results_forecast_row_count": results_stats.row_count,
    }


def _run_champion_results_load(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Load the exact cached champion experiment winners and persist lineage."""
    experiment_id = params["experiment_id"]
    if progress_cb:
        progress_cb(pct=5, msg="Loading champion results into forecast tables")

    # Recomputing here could load different results under the chosen experiment.
    winners_csv = (
        _SCRIPTS_DIR.parent / "data" / "champion" / f"experiment_{experiment_id}_winners.csv"
    )
    if not winners_csv.exists():
        raise FileNotFoundError(f"Champion experiment {experiment_id} has no winners artifact")
    cmd = [
        _UV,
        "run",
        "python",
        "scripts/ml/run_champion_selection.py",
        "--load-winners-from",
        str(winners_csv),
        "--champion-experiment-id",
        str(experiment_id),
    ]
    logger.info("Using cached winners from experiment %d: %s", experiment_id, winners_csv)
    if progress_cb:
        progress_cb(pct=10, msg="Loading the exact cached experiment winners")

    output = _run_subprocess(
        cmd, progress_cb, "Loading champion results", cancel_event=cancel_event, job_id=job_id
    )

    lineage = _finalize_champion_results_lineage(
        experiment_id,
        job_id,
        winners_csv,
    )

    if progress_cb:
        progress_cb(pct=100, msg="Champion results loaded successfully")
    return {
        "experiment_id": experiment_id,
        **lineage,
        "output_log": output or "Champion results loaded",
    }


def _run_champion_sweep(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a champion strategy sweep (tournament).

    Fans out a grid of candidate champion configs (each a real champion_experiment),
    ranks them globally + per demand segment, assembles a per-segment composite, and
    writes the recommendation back to champion_sweep. See spec 30.
    """
    import psycopg  # lazy — module keeps psycopg off the top-level import path

    sweep_id = params["sweep_id"]
    if progress_cb:
        progress_cb(pct=5, msg=f"Starting champion sweep #{sweep_id}")

    # Store job_id on the sweep record for log/cancel correlation.
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE champion_sweep SET job_id = %s WHERE sweep_id = %s",
                (job_id, sweep_id),
            )
    except psycopg.Error:
        logger.warning("Failed to store job_id on champion sweep %d", sweep_id)

    cmd = [_UV, "run", "python", "scripts/ml/run_champion_sweep.py", "--sweep-id", str(sweep_id)]
    output = _run_subprocess(
        cmd, progress_cb, "Running champion sweep", cancel_event=cancel_event, job_id=job_id
    )
    if progress_cb:
        progress_cb(pct=100, msg=f"Champion sweep #{sweep_id} completed")
    return {"sweep_id": sweep_id, "output_log": output or "Champion sweep completed"}


def _run_train_production_model(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Train a production model on full history for forecasting.

    Invokes ``scripts/ml/train_production_models.py`` as a subprocess.
    Supports one persisted model (``--model``) or the complete persisted-model
    roster (``--all``): LightGBM, N-HiTS, and N-BEATS.
    """
    from common.core.paths import PROJECT_ROOT as ROOT

    model_id = params.get("model_id")
    # An empty request is the safe one-click path for all three persisted
    # artifacts. A supplied model keeps single-model diagnostics available.
    all_models = bool(params.get("all_models", not model_id))

    if not all_models and not model_id:
        raise ValueError("Production training requires model_id or all_models=true")

    if progress_cb:
        progress_cb(pct=0, msg=f"Starting production training: {model_id or 'all models'}")

    cmd = [_UV, "run", "python", str(ROOT / "scripts" / "ml" / "train_production_models.py")]
    if all_models:
        cmd.append("--all")
    elif model_id:
        cmd.extend(["--model", model_id])

    start = time.time()
    if progress_cb:
        progress_cb(pct=5, msg=f"Training {'all models' if all_models else model_id}")

    _run_subprocess(
        cmd,
        progress_cb,
        f"Training production model: {model_id or 'all'}",
        cancel_event=cancel_event,
        job_id=job_id,
        timeout_seconds=_subprocess_timeout_seconds("production_training"),
    )
    duration = time.time() - start

    if progress_cb:
        progress_cb(pct=100, msg=f"Training completed in {duration:.0f}s")

    return {
        "model_id": model_id,
        "all_models": all_models,
        "duration_s": round(duration, 1),
        "status": "trained",
    }


def _run_generate_production_forecast(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the production forecast generation pipeline (F1.1)."""
    horizon = params.get("horizon")
    model_id = params.get("model_id")
    run_id = str(params.get("run_id") or uuid.uuid4())
    generation_purpose = str(params.get("generation_purpose") or "release_candidate")
    # Tri-state: None → use the script/config default; True/False → force on/off.
    confidence_intervals = params.get("confidence_intervals")
    if progress_cb:
        horizon_label = horizon if horizon is not None else "config default"
        progress_cb(pct=5, msg=f"Starting production forecast generation (horizon={horizon_label})")
    cmd = [_UV, "run", "python", "scripts/forecasting/generate_production_forecasts.py"]
    if horizon is not None:
        cmd.extend(["--horizon", str(horizon)])
    if model_id:
        cmd.extend(["--model-id", str(model_id)])
    cmd.extend(["--run-id", run_id])
    cmd.extend(["--generation-purpose", generation_purpose])
    if confidence_intervals is True:
        cmd.append("--confidence-intervals")
    elif confidence_intervals is False:
        cmd.append("--no-confidence-intervals")
    output = _run_subprocess(
        cmd,
        progress_cb,
        "Generating production forecasts",
        cancel_event=cancel_event,
        job_id=job_id,
        timeout_seconds=_subprocess_timeout_seconds("production_generation"),
        # LightGBM and PyTorch ship separate OpenMP runtimes on macOS. A
        # parallel tensor copy can deadlock after LightGBM is loaded; mixed
        # inference stays safe while the separately trained neural models
        # retain full MPS acceleration.
        env_overrides={"OMP_NUM_THREADS": "1"},
    )
    if progress_cb:
        progress_cb(pct=100, msg="Production forecast generation complete")
    return {
        "horizon": horizon,
        "run_id": run_id,
        "generation_purpose": generation_purpose,
        "confidence_intervals": confidence_intervals,
        "output_log": output if output else "Production forecast generation completed",
    }


def _set_customer_forecast_job_status(
    run_id: str,
    status: str,
    error_summary: str,
) -> None:
    """Reconcile a managed-job terminal state to its customer forecast run."""
    import psycopg

    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE customer_forecast_run SET run_status = %s, error_summary = %s, "
                "completed_at = NOW() WHERE run_id = %s::uuid "
                "AND run_status IN ('queued', 'generating', 'failed')",
                (status, error_summary[:500], run_id),
            )
    except (OSError, RuntimeError, ValueError, psycopg.Error):
        logger.exception("Reconciling customer forecast job status failed")


def _run_generate_customer_forecast(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run generation-only customer Chronos inference as a durable subprocess."""
    run_id = str(params.get("run_id") or "")
    if not run_id:
        raise ValueError("Customer forecast generation requires run_id")
    if progress_cb:
        progress_cb(pct=5, msg="Starting customer forecast generation")
    cmd = [
        _UV,
        "run",
        "python",
        "scripts/forecasting/generate_customer_forecasts.py",
        "--run-id",
        run_id,
    ]
    try:
        output = _run_subprocess(
            cmd,
            progress_cb,
            "Generating customer forecasts",
            cancel_event=cancel_event,
            job_id=job_id,
            timeout_seconds=_subprocess_timeout_seconds("customer_forecast"),
            env_overrides={"OMP_NUM_THREADS": "1"},
        )
    except JobCancelledError:
        _set_customer_forecast_job_status(run_id, "cancelled", "job cancelled")
        raise
    except (OSError, RuntimeError, ValueError):
        _set_customer_forecast_job_status(run_id, "failed", "managed job failed")
        raise
    if progress_cb:
        progress_cb(pct=100, msg="Customer forecast generation complete")
    return {"run_id": run_id, "output_log": output or "Customer forecast generation completed"}


def _run_prepare_forecast_snapshot_contenders(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Freeze the top-three live-FVA roster and generate its six-lag forecasts."""
    cmd = [
        _UV,
        "run",
        "python",
        str(_SCRIPTS_DIR / "forecasting" / "prepare_forecast_snapshot_contenders.py"),
    ]
    if params.get("record_month"):
        cmd.extend(["--record-month", str(params["record_month"])])
    if params.get("dry_run"):
        cmd.append("--dry-run")
    if params.get("from_existing_staging"):
        cmd.append("--from-existing-staging")
    output = _run_subprocess(
        cmd,
        progress_cb,
        "Preparing forecast snapshot contenders",
        cancel_event=cancel_event,
        job_id=job_id,
        timeout_seconds=_subprocess_timeout_seconds("snapshot_contenders"),
    )
    return {"output_log": output or "Forecast snapshot contenders prepared"}


def _run_archive_forecast_snapshot(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Archive the frozen champion-plus-three snapshot before staging overwrite."""
    cmd = [_UV, "run", "python", str(_SCRIPTS_DIR / "forecasting" / "archive_forecast_snapshot.py")]
    if params.get("record_month"):
        cmd.extend(["--record-month", str(params["record_month"])])
    if params.get("dry_run"):
        cmd.append("--dry-run")
    if params.get("overwrite"):
        cmd.append("--overwrite")
    output = _run_subprocess(
        cmd,
        progress_cb,
        "Archiving forecast snapshot",
        cancel_event=cancel_event,
        job_id=job_id,
    )
    return {"output_log": output or "Forecast snapshot archived"}


def _run_refresh_forecast_snapshot_kpis(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Score newly closed live-snapshot lags after monthly actuals load."""
    from common.core.mv_refresh import refresh_materialized_views
    from common.services.cache import get_cache

    result = refresh_materialized_views(
        ["agg_accuracy_snapshot"],
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )
    if result["failed"] or result["missing"]:
        raise RuntimeError("Forecast snapshot KPI refresh did not complete")
    get_cache().invalidate("ds:fva_snapshot*")
    return result


def _run_period_roll(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the canonical Period Roll preset inline for recurring schedules.

    Manual workflow launches use the named pipeline and expose each step as a
    separate job. Recurring schedules can only target one registered job type,
    so this wrapper reads and executes the same preset without duplicating its
    ordering in Python.
    """
    from common.services.job_registry import JOB_TYPE_REGISTRY
    from common.services.pipeline_presets import get_pipeline_preset, preset_steps

    steps = preset_steps(get_pipeline_preset("period-roll"))
    output_logs: list[str] = []
    total_steps = len(steps)
    for index, step in enumerate(steps, start=1):
        if cancel_event and cancel_event.is_set():
            raise JobCancelledError("Period Roll cancelled by user")
        job_type = step["job_type"]
        step_params = dict(step["params"])
        if params.get("record_month") and job_type in {
            "prepare_forecast_snapshot_contenders",
            "archive_forecast_snapshot",
        }:
            step_params["record_month"] = params["record_month"]

        def step_progress(*, pct: int | None = None, msg: str, step_number: int = index) -> None:
            if progress_cb:
                completed_share = (step_number - 1) / total_steps
                step_pct = 5 if pct is None else pct
                current_share = max(0, min(step_pct, 100)) / 100 / total_steps
                progress_cb(
                    pct=int((completed_share + current_share) * 100),
                    msg=f"Period Roll {step_number}/{total_steps}: {msg}",
                )

        result = JOB_TYPE_REGISTRY[job_type].callable(
            step_params,
            progress_cb=step_progress,
            cancel_event=cancel_event,
            job_id=job_id,
        )
        output_logs.append(str(result.get("output_log") or result))

    return {
        "steps_completed": total_steps,
        "output_log": "\n".join(output_logs),
    }


def _run_cleanup_forecast_staging(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Delete staging generations only after their bounded snapshot reconciles."""
    cmd = [_UV, "run", "python", str(_SCRIPTS_DIR / "forecasting" / "cleanup_forecast_staging.py")]
    if params.get("generation"):
        cmd.extend(["--generation", str(params["generation"])])
    if params.get("dry_run", True):
        cmd.append("--dry-run")
    output = _run_subprocess(
        cmd,
        progress_cb,
        "Cleaning forecast staging",
        cancel_event=cancel_event,
        job_id=job_id,
    )
    return {"output_log": output or "Forecast staging cleaned"}


def _run_compute_replenishment_plan(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the forward-looking replenishment plan computation (CI Bands + Repl. Plan)."""
    if progress_cb:
        progress_cb(pct=10, msg="Starting replenishment plan computation")
    cmd = [_UV, "run", "python", "scripts/inventory/compute_replenishment_plan.py"]
    output = _run_subprocess(
        cmd,
        progress_cb,
        "Computing replenishment plan from production forecast",
        cancel_event=cancel_event,
        job_id=job_id,
    )
    if progress_cb:
        progress_cb(pct=100, msg="Replenishment plan computation complete")
    return {"output_log": output if output else "Replenishment plan computation completed"}


def _run_generate_ai_insights(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run AI Planning Agent portfolio scan to generate insights."""
    if progress_cb:
        progress_cb(pct=5, msg="Starting AI insights generation")
    cmd = [_UV, "run", "python", "scripts/ai/generate_ai_insights.py", "--portfolio"]
    output = _run_subprocess(
        cmd,
        progress_cb,
        "Scanning portfolio for exceptions",
        cancel_event=cancel_event,
        job_id=job_id,
    )
    if progress_cb:
        progress_cb(pct=100, msg="AI insights generation complete")
    return {"output_log": output if output else "AI insights generation completed"}


def _run_generate_storyboard(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Generate storyboard exceptions for all DFUs."""
    if progress_cb:
        progress_cb(pct=10, msg="Generating storyboard exceptions")
    cmd = [_UV, "run", "python", "scripts/ops/generate_storyboard_exceptions.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Storyboard exceptions generated"}


def _run_inventory_backtest(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run inventory backtest simulation."""
    cmd = [_UV, "run", "python", "scripts/inventory/run_inventory_backtest.py"]
    if params.get("models"):
        cmd.extend(["--models", params["models"]])
    if params.get("months"):
        cmd.extend(["--months", str(params["months"])])
    output = _run_subprocess(
        cmd,
        progress_cb,
        "Inventory backtest",
        cancel_event=cancel_event,
        job_id=job_id,
    )
    return {"output_log": output or "Inventory backtest completed"}


def _run_inventory_planning_pipeline(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the end-to-end inventory planning pipeline."""
    steps = params.get("steps")
    cmd = [_UV, "run", "python", "scripts/inventory/run_inventory_planning_pipeline.py"]
    if steps:
        cmd.extend(["--steps", steps])
    if progress_cb:
        progress_cb(pct=0, msg="Starting inventory planning pipeline")
    output = _run_subprocess(
        cmd,
        progress_cb,
        "Inventory planning pipeline",
        cancel_event=cancel_event,
        job_id=job_id,
    )
    if progress_cb:
        progress_cb(pct=100, msg="Inventory planning pipeline complete")
    return {"output_log": output or "Pipeline completed"}


def _run_compute_safety_stock(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute safety stock targets for all DFUs."""
    if progress_cb:
        progress_cb(pct=10, msg="Computing safety stock targets")
    cmd = [
        _UV,
        "run",
        "python",
        "scripts/inventory/compute_safety_stock.py",
        "--config",
        "config/inventory/safety_stock_config.yaml",
    ]
    if params.get("forecast_source"):
        cmd.extend(["--forecast-source", params["forecast_source"]])
    if params.get("model_id"):
        cmd.extend(["--model-id", params["model_id"]])
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Safety stock computation completed"}


def _run_compute_eoq(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute EOQ cycle stock targets."""
    if progress_cb:
        progress_cb(pct=10, msg="Computing EOQ targets")
    cmd = [
        _UV,
        "run",
        "python",
        "scripts/inventory/compute_eoq.py",
        "--config",
        "config/inventory/eoq_config.yaml",
    ]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "EOQ computation completed"}


def _run_compare_inventory_algorithms(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compare SS/EOQ/ROP across forecast algorithms."""
    if progress_cb:
        progress_cb(pct=10, msg="Comparing inventory algorithms")
    cmd = [_UV, "run", "python", "scripts/inventory/compare_inventory_algorithms.py"]
    models = params.get("models")
    if models:
        cmd.extend(["--models", models])
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output or "Algorithm comparison completed"}


def _run_assign_policies(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Upsert replenishment policies and auto-assign DFUs by segment."""
    if progress_cb:
        progress_cb(pct=10, msg="Assigning replenishment policies")
    cmd = [
        _UV,
        "run",
        "python",
        "scripts/inventory/assign_replenishment_policies.py",
        "--config",
        "config/inventory/replenishment_policy_config.yaml",
    ]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Policy assignment completed"}


def _run_generate_exceptions(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Detect replenishment exceptions and write to queue."""
    if progress_cb:
        progress_cb(pct=10, msg="Detecting replenishment exceptions")
    cmd = [_UV, "run", "python", "scripts/inventory/generate_replenishment_exceptions.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Exception detection completed"}


def _run_classify_abc_xyz(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run ABC-XYZ classification and write to dim_sku."""
    if progress_cb:
        progress_cb(pct=10, msg="Running ABC-XYZ classification")
    cmd = [_UV, "run", "python", "scripts/inventory/classify_abc_xyz.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "ABC-XYZ classification completed"}


def _run_compute_variability(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Legacy variability pipeline — delegates to unified SKU features.

    Variability metrics (CV, MAD, intermittency, classification) are now
    produced as part of ``compute_sku_features``; the standalone
    ``compute_demand_variability.py`` script has been removed.
    """
    return _run_compute_sku_features(params, progress_cb, cancel_event, job_id)


def _run_compute_sku_features(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute all time-series features (volume, trend, seasonality, variability, lifecycle) for all SKUs."""
    from scripts.ml.compute_sku_features import run_pipeline

    if progress_cb:
        progress_cb(pct=10, msg="Computing SKU features")
    # The job/params schema uses "time_window_months"; run_pipeline's kwarg is "time_window".
    result = run_pipeline(time_window=params.get("time_window_months", 36))
    if progress_cb:
        progress_cb(pct=100, msg="Complete")
    return result


def _run_compute_demand_signals(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute short-horizon demand signals from sales velocity."""
    if progress_cb:
        progress_cb(pct=10, msg="Computing demand signals")
    cmd = [_UV, "run", "python", "scripts/inventory/compute_demand_signals.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Demand signals computation completed"}


def _run_compute_investment(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute efficient frontier and capital investment allocation."""
    if progress_cb:
        progress_cb(pct=10, msg="Computing investment plan")
    cmd = [_UV, "run", "python", "scripts/inventory/compute_investment_plan.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Investment plan computation completed"}


def _run_refresh_health_scores(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Refresh the inventory health score materialized view."""
    if progress_cb:
        progress_cb(pct=10, msg="Refreshing inventory health scores")
    cmd = [_UV, "run", "python", "scripts/inventory/refresh_health_scores.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Health scores refreshed"}


def _run_refresh_intramonth(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Refresh the intramonth stockout materialized view."""
    if progress_cb:
        progress_cb(pct=10, msg="Refreshing intramonth stockout view")
    cmd = [_UV, "run", "python", "scripts/inventory/refresh_intramonth_stockout.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Intramonth stockout view refreshed"}


def _run_refresh_forecast_views(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Refresh every MV that reads the forecast/backtest fact tables.

    Moved out of the API request thread (was inline in
    ``api/routers/forecasting/competition.py``). At 40x scale the synchronous
    refresh exceeds the 30s ``statement_timeout`` and holds ACCESS EXCLUSIVE
    locks blocking other readers — running it as a background job avoids both
    issues. The MV set comes from the central dependency map
    (common/core/mv_refresh.py), not a hand-picked list.

    Returns a dict with ``refreshed``, ``failed`` and ``missing`` MV lists.
    """
    # Imported here to keep module-level imports psycopg-free (see module docstring).
    from common.core.mv_refresh import refresh_for_tables

    return refresh_for_tables(
        ["fact_external_forecast_monthly", "backtest_lag_archive"],
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )


def _run_refresh_customer_analytics(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Refresh the materialized views backing the Customer Analytics tab.

    Backs the "Recalculate" button on that tab. The CA panels read from these
    MVs instead of aggregating fact tables per-request; this job runs the same
    refresh off the request thread. The MV set comes from the central
    dependency map keyed on the customer-demand fact + customer dimension.

    Returns a dict with ``refreshed``, ``failed`` and ``missing`` MV lists.
    """
    from common.core.mv_refresh import refresh_for_tables

    return refresh_for_tables(
        ["fact_customer_demand_monthly", "dim_customer"],
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )


def _run_refresh_all_mvs(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Refresh ALL materialized views in dependency order (staleness safety net).

    Scheduled recurring by default (config/platform/jobs_config.yaml) so no MV
    stays stale longer than the schedule interval even if a writer path forgot
    to refresh it. ``skip_heavy: true`` in params skips HEAVY_MVS
    (mv_intramonth_stockout has its own dedicated job/cadence).
    """
    from common.core.mv_refresh import HEAVY_MVS, all_mvs, refresh_materialized_views

    mvs = all_mvs()
    if params.get("skip_heavy"):
        mvs = [mv for mv in mvs if mv not in HEAVY_MVS]
    return refresh_materialized_views(mvs, progress_cb=progress_cb, cancel_event=cancel_event)


def _run_ss_simulation(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run Monte Carlo safety stock simulation."""
    if progress_cb:
        progress_cb(pct=10, msg="Running Monte Carlo SS simulation")
    cmd = [_UV, "run", "python", "scripts/inventory/run_ss_simulation.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "SS simulation completed"}


def _run_data_quality(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run all data quality checks via DQEngine."""
    from common.engines.dq_engine import DQEngine

    if progress_cb:
        progress_cb(pct=10, msg="Running data quality checks")
    domain = params.get("domain")
    engine = DQEngine()
    results = engine.run_all_checks(domain=domain)
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "pass")
    failed = sum(1 for r in results if r.get("status") == "fail")
    if progress_cb:
        progress_cb(pct=100, msg="Data quality checks complete")
    return {
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "output_log": f"Data quality checks completed: {passed}/{total} passed, {failed} failed",
    }


def _run_tune_stale_clusters(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Re-tune per-cluster hyperparameters for clusters flagged stale.

    Runs ``tune_cluster_hyperparams.py --stale-only``, which reads
    ``cluster_tuning_profile_state`` (rows marked stale by cluster promotion),
    tunes only those clusters, merges the results into
    ``cluster_tuning_profiles.yaml``, and clears the flags it covered.
    Submitted by ``POST /admin/tuning/invalidate-stale?retune=true``.
    """
    from common.core.utils import reset_config

    model = params.get("model") or "lgbm"
    if model not in MODEL_OUTPUT_DIRS:
        raise ValueError(
            f"Unsupported tuning model '{model}' (expected one of {sorted(MODEL_OUTPUT_DIRS)})"
        )
    if progress_cb:
        progress_cb(pct=5, msg=f"Re-tuning stale clusters ({model})")
    cmd = [
        _UV,
        "run",
        "python",
        "scripts/ml/tune_cluster_hyperparams.py",
        "--model",
        model,
        "--stale-only",
    ]
    if params.get("trials"):
        cmd.extend(["--trials", str(int(params["trials"]))])
    output = _run_subprocess(
        cmd,
        progress_cb,
        "Tuning stale clusters",
        cancel_event=cancel_event,
        job_id=job_id,
        timeout_seconds=_subprocess_timeout_seconds("stale_cluster_tuning"),
    )
    reset_config("cluster_tuning_profiles.yaml")
    return {"output_log": output if output else "Stale-cluster tuning completed"}


def _run_sampled_backtest(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a sampled-SKU LGBM backtest (Sampled Backtest panel).

    Replaces the bare ``subprocess.Popen`` the endpoint used to fire —
    running under the JobManager restores PID tracking, cancellation,
    log streaming, and restart recovery.
    """
    sku_file = params.get("sku_file")
    if not sku_file:
        raise ValueError("sampled_backtest requires a 'sku_file' param")
    if progress_cb:
        progress_cb(pct=5, msg="Running sampled backtest")
    cmd = [
        _UV,
        "run",
        "python",
        str(_SCRIPTS_DIR / "ml" / "run_backtest.py"),
        "--sampled-skus",
        str(sku_file),
    ]
    if params.get("param_overrides"):
        cmd.extend(["--param-overrides", json.dumps(params["param_overrides"])])
    output = _run_subprocess(
        cmd, progress_cb, "Running sampled backtest", cancel_event=cancel_event, job_id=job_id
    )
    return {
        "output_log": output if output else "Sampled backtest completed",
        "run_id": params.get("run_id"),
    }


def _run_tuning_backtest(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a tuning chat backtest: build temp config, run backtest, register results, insert chat message."""
    import copy
    import tempfile

    import yaml

    from common.core.paths import PROJECT_ROOT as ROOT

    run_id = params["run_id"]
    session_id = params["session_id"]
    overrides = params.get("overrides", {})
    strategy_label = params.get("strategy_label", "chat_experiment")

    if progress_cb:
        progress_cb(pct=5, msg=f"Starting tuning backtest #{run_id} ({strategy_label})")

    # 1. Build temp config with overrides (pipeline config format)
    from common.core.utils import get_pipeline_config_path

    pipeline_path = get_pipeline_config_path()
    with open(pipeline_path) as f:
        base_config = yaml.safe_load(f)

    cfg = copy.deepcopy(base_config)
    # Apply overrides to lgbm_cluster params in pipeline config
    lgbm_entry = cfg.get("algorithms", {}).get("lgbm_cluster", {})
    if "params" in lgbm_entry:
        lgbm_entry["params"].update(overrides)
    else:
        lgbm_entry.update(overrides)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tuning_chat_"))
    tmp_path = tmp_dir / f"pipeline_config_{strategy_label}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # 2. Run backtest via _run_subprocess (gets PID tracking + cancel + log streaming)
    cmd = [
        _UV,
        "run",
        "python",
        str(ROOT / "scripts" / "ml" / "run_backtest.py"),
        "--model",
        "lgbm",
        "--config",
        str(tmp_path),
    ]
    start = time.time()
    output = _run_subprocess(
        cmd, progress_cb, "Running LGBM backtest", cancel_event=cancel_event, job_id=job_id
    )
    duration = time.time() - start

    # 3. Complete run via tracker
    from common.ml.tuning_tracker import (
        complete_run,
        register_cluster_month_breakdowns,
        register_lag_breakdowns,
        register_timeframes,
    )

    meta_path = ROOT / "data" / "backtest" / "lgbm_cluster" / "backtest_metadata.json"
    complete_run(run_id, meta_path)
    register_timeframes(run_id, meta_path)

    all_lags_path = (
        ROOT / "data" / "backtest" / "lgbm_cluster" / "backtest_predictions_all_lags.csv"
    )
    register_lag_breakdowns(run_id, all_lags_path)

    predictions_path = ROOT / "data" / "backtest" / "lgbm_cluster" / "backtest_predictions.csv"
    if predictions_path.exists():
        register_cluster_month_breakdowns(run_id, predictions_path)

    # 4. Insert run_completed chat message
    _insert_tuning_chat_message(session_id, run_id, "run_completed", None)

    if progress_cb:
        progress_cb(pct=100, msg=f"Tuning run #{run_id} completed in {duration:.0f}s")
    return {
        "run_id": run_id,
        "strategy_label": strategy_label,
        "duration_seconds": round(duration),
        "output_log": output[:5000] if output else "Tuning backtest completed",
    }


def _run_model_tuning_experiment(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a model tuning experiment: update run status, run backtest, register results.

    Supports the canonical LightGBM tree model. The caller provides a pre-built
    temp config file and the run_id of the lgbm_tuning_run record.
    """
    from common.core.paths import PROJECT_ROOT as ROOT

    run_id = params["run_id"]
    model = params["model"]
    config_path = params["config_path"]
    run_label = params.get("run_label", "tuning_experiment")

    if model not in MODEL_OUTPUT_DIRS:
        raise ValueError(f"Unknown model type: {model}")

    if progress_cb:
        progress_cb(
            pct=0, msg=f"Starting {model.upper()} tuning experiment #{run_id} ({run_label})"
        )

    # 1. Update lgbm_tuning_run: set job_id, status=running, started_at
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE lgbm_tuning_run SET job_id = %s, status = 'running', started_at = NOW() "
                "WHERE run_id = %s",
                (job_id, run_id),
            )
    except Exception:
        logger.warning("Failed to update run %d status to running", run_id)

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Job cancelled by user")

    # 2. Build backtest command — optionally include cluster override
    cmd = [
        _UV,
        "run",
        "python",
        str(ROOT / "scripts" / "ml" / "run_backtest.py"),
        "--model",
        model,
        "--config",
        config_path,
    ]

    # If a cluster experiment is referenced, look up its artifacts and add --cluster-override
    cluster_experiment_id: int | None = params.get("cluster_experiment_id")
    if cluster_experiment_id is not None:
        try:
            with _get_conn() as conn:
                row = conn.execute(
                    "SELECT artifacts_path, scenario_id, status FROM cluster_experiment "
                    "WHERE experiment_id = %s",
                    (cluster_experiment_id,),
                ).fetchone()
            if row is None:
                raise ValueError(f"Cluster experiment {cluster_experiment_id} not found")
            ce_artifacts_path, ce_scenario_id, ce_status = row
            if ce_status != "completed":
                raise ValueError(
                    f"Cluster experiment {cluster_experiment_id} is not completed "
                    f"(status={ce_status})"
                )
            if not ce_artifacts_path:
                raise ValueError(
                    f"Cluster experiment {cluster_experiment_id} has no artifacts_path"
                )
            from common.core.paths import DATA_DIR

            safe_path = Path(ce_artifacts_path).resolve()
            allowed_base = DATA_DIR
            if not str(safe_path).startswith(str(allowed_base)):
                raise ValueError(
                    f"artifacts_path {ce_artifacts_path!r} is outside the allowed data directory"
                )
            cluster_override_csv = str(safe_path / "cluster_labels.csv")
            cmd.extend(["--cluster-override", cluster_override_csv])
            if progress_cb:
                progress_cb(
                    pct=3,
                    msg=f"Using clusters from experiment #{cluster_experiment_id} "
                    f"(scenario {ce_scenario_id})",
                )
        except ValueError:
            raise
        except Exception:
            logger.warning(
                "Failed to look up cluster experiment %d — proceeding with production clusters",
                cluster_experiment_id,
            )

    start = time.time()
    try:
        if progress_cb:
            progress_cb(pct=5, msg=f"Running {model.upper()} backtest")
        output = _run_subprocess(
            cmd,
            progress_cb,
            f"Running {model.upper()} backtest",
            cancel_event=cancel_event,
            job_id=job_id,
        )
        duration = time.time() - start
    except (RuntimeError, OSError) as exc:
        # 5. On failure: update lgbm_tuning_run with status=failed
        duration = time.time() - start
        error_msg = str(exc)[:2000]
        try:
            from common.ml.tuning_tracker import fail_run

            fail_run(run_id, error_msg)
        except ImportError:
            logger.warning(
                "tuning_tracker not available — marking run %d failed via direct SQL", run_id
            )
            try:
                with _get_conn() as conn:
                    conn.execute(
                        "UPDATE lgbm_tuning_run SET status = 'failed', completed_at = NOW(), "
                        "notes = COALESCE(notes || E'\\n', '') || %s WHERE run_id = %s",
                        (error_msg, run_id),
                    )
            except Exception:
                logger.warning("Failed to mark run %d as failed in DB", run_id)
        # 6. Clean up temp config file
        _cleanup_temp_config(config_path)
        raise

    # 3. On success: complete run via tracker
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Job cancelled by user")

    if progress_cb:
        progress_cb(pct=90, msg="Registering results")

    from common.ml.tuning_tracker import (
        complete_run,
        register_cluster_month_breakdowns,
        register_lag_breakdowns,
        register_timeframes,
    )

    output_dir_name = MODEL_OUTPUT_DIRS[model]
    meta_path = ROOT / "data" / "backtest" / output_dir_name / "backtest_metadata.json"
    complete_run(run_id, meta_path)

    if progress_cb:
        progress_cb(pct=93, msg="Registering timeframe breakdowns")
    register_timeframes(run_id, meta_path)

    all_lags_path = (
        ROOT / "data" / "backtest" / output_dir_name / "backtest_predictions_all_lags.csv"
    )
    register_lag_breakdowns(run_id, all_lags_path)

    predictions_path = ROOT / "data" / "backtest" / output_dir_name / "backtest_predictions.csv"
    if predictions_path.exists():
        if progress_cb:
            progress_cb(pct=96, msg="Registering cluster/month breakdowns")
        register_cluster_month_breakdowns(run_id, predictions_path)

    # 6. Clean up temp config file
    _cleanup_temp_config(config_path)

    if progress_cb:
        progress_cb(pct=100, msg=f"Tuning experiment #{run_id} completed in {duration:.0f}s")
    return {
        "run_id": run_id,
        "model": model,
        "run_label": run_label,
        "duration_seconds": round(duration),
        "output_log": output[:5000] if output else "Model tuning experiment completed",
    }


def _run_load_backtest_results(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Load backtest predictions into DB and refresh materialized views.

    Invokes ``scripts/load_backtest_forecasts.py --model <model_id> --replace``
    as a subprocess. On success, marks the tuning run as results-promoted.
    """
    from common.core.paths import PROJECT_ROOT as ROOT

    run_id = params["run_id"]
    model = params["model"]
    model_id = MODEL_OUTPUT_DIRS.get(model, params.get("model_id", ""))

    if progress_cb:
        progress_cb(pct=0, msg=f"Starting results load for {model.upper()}")

    pred_path = ROOT / "data" / "backtest" / model_id / "backtest_predictions.csv"
    if not pred_path.exists():
        raise RuntimeError(f"Prediction file not found: {pred_path}")

    cmd = [
        _UV,
        "run",
        "python",
        str(ROOT / "scripts" / "etl" / "load_backtest_forecasts.py"),
        "--model",
        model_id,
        "--replace",
    ]
    start = time.time()
    if progress_cb:
        progress_cb(pct=5, msg=f"Loading {model.upper()} predictions into database")

    _run_subprocess(
        cmd,
        progress_cb,
        f"Loading {model.upper()} results",
        cancel_event=cancel_event,
        job_id=job_id,
    )
    duration = time.time() - start

    # Mark tuning run as results-promoted
    if progress_cb:
        progress_cb(pct=95, msg="Updating promotion status")
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE lgbm_tuning_run "
                "SET is_results_promoted = TRUE, results_promoted_at = NOW() "
                "WHERE run_id = %s",
                (run_id,),
            )
            conn.execute(
                "INSERT INTO tuning_promotion_log "
                "(run_id, model_id, promoted_by, params_written, promotion_type) "
                "VALUES (%s, %s, %s, %s::jsonb, %s)",
                (run_id, model_id, "manual", "{}", "results"),
            )
    except Exception:
        logger.warning("Failed to update results promotion status for run %d", run_id)

    if progress_cb:
        progress_cb(pct=100, msg=f"Results loaded in {duration:.0f}s")

    return {
        "run_id": run_id,
        "model": model,
        "duration_seconds": round(duration),
        "status": "loaded",
    }


def _run_load_backtest_model(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Load backtest predictions for a specific model into Postgres.

    Invokes ``scripts/load_backtest_forecasts.py --model <model_id> --replace``
    as a subprocess.  If ``run_id`` is provided, marks the corresponding
    ``backtest_run`` row as loaded on success.
    """
    from common.core.paths import PROJECT_ROOT as ROOT

    model_id = params["model_id"]
    run_id = params.get("run_id")

    if progress_cb:
        progress_cb(pct=0, msg=f"Starting results load for {model_id}")

    pred_path = ROOT / "data" / "backtest" / model_id / "backtest_predictions.csv"
    if not pred_path.exists():
        raise RuntimeError(f"Prediction file not found: {pred_path}")

    cmd = [
        _UV,
        "run",
        "python",
        str(ROOT / "scripts" / "etl" / "load_backtest_forecasts.py"),
        "--model",
        model_id,
        "--replace",
    ]
    start = time.time()
    if progress_cb:
        progress_cb(pct=5, msg=f"Loading {model_id} predictions into database")

    _run_subprocess(
        cmd,
        progress_cb,
        f"Loading {model_id} results",
        cancel_event=cancel_event,
        job_id=job_id,
    )
    duration = time.time() - start

    # Mark backtest_run as loaded if run_id provided
    if run_id is not None:
        if progress_cb:
            progress_cb(pct=95, msg="Updating load status in backtest_run")
        try:
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE backtest_run "
                    "SET is_loaded_to_db = TRUE, loaded_at = NOW() "
                    "WHERE id = %s",
                    (run_id,),
                )
        except Exception:
            logger.warning("Failed to update backtest_run %s as loaded", run_id)

    if progress_cb:
        progress_cb(pct=100, msg=f"Results loaded in {duration:.0f}s")

    return {
        "model_id": model_id,
        "run_id": run_id,
        "duration_seconds": round(duration),
        "status": "loaded",
    }


def _cleanup_temp_config(config_path: str) -> None:
    """Remove a temporary config file and its parent directory if empty."""
    try:
        p = Path(config_path)
        if p.exists():
            p.unlink()
        parent = p.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        logger.warning("Failed to clean up temp config: %s", config_path)


def _insert_tuning_chat_message(
    session_id: str,
    run_id: int,
    msg_type: str,
    error: str | None,
) -> None:
    """Insert a run_completed or run_failed chat message (called from background callable)."""
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                if msg_type == "run_completed":
                    cur.execute(
                        "SELECT accuracy_pct, wape, bias, n_predictions, n_dfus "
                        "FROM lgbm_tuning_run WHERE run_id = %s",
                        (run_id,),
                    )
                    row = cur.fetchone()
                    metadata: dict[str, Any] = {"run_id": run_id}
                    if row:
                        metadata.update(
                            {
                                "accuracy_pct": row[0],
                                "wape": row[1],
                                "bias": row[2],
                                "n_predictions": row[3],
                                "n_dfus": row[4],
                            }
                        )
                    content = (
                        f"Run #{run_id} completed — "
                        f"accuracy {row[0]:.2f}%, WAPE {row[1]:.2f}, bias {row[2]:.4f}"
                        if row and row[0] is not None
                        else f"Run #{run_id} completed"
                    )
                else:
                    metadata = {"run_id": run_id, "error": error}
                    content = f"Run #{run_id} failed: {error or 'unknown error'}"

                cur.execute(
                    """INSERT INTO tuning_chat_message
                        (session_id, role, content, message_type, metadata)
                    VALUES (%s::uuid, 'system', %s, %s, %s)""",
                    (session_id, content, msg_type, json.dumps(metadata, default=str)),
                )
    except Exception:
        logger.warning("Failed to insert %s message for run %d", msg_type, run_id)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _serialize_job_row(cols: tuple[str, ...], row: tuple) -> dict[str, Any]:
    """Convert a job-table row to a dict with JSON/datetime/default handling.

    Wraps the canonical :func:`row_to_dict_from_cols` helper and then applies
    job-domain-specific coercions (deserialise params/result JSON,
    normalise ``logs`` to a list, ISO-format timestamps, default
    ``progress_pct`` to 0).
    """
    d = row_to_dict_from_cols(cols, row)
    for col in cols:
        val = d[col]
        if col in ("params", "result", "attempt_result"):
            if isinstance(val, dict):
                continue
            if val:
                d[col] = json.loads(val)
            else:
                d[col] = {} if col == "params" else None
        elif col == "logs":
            if isinstance(val, list):
                continue
            if val:
                d[col] = json.loads(val)
            else:
                d[col] = []
        elif col in ("submitted_at", "started_at", "completed_at"):
            d[col] = val.isoformat() if val else None
        elif col == "progress_pct":
            d[col] = val or 0
        # ``pid`` and other columns pass through unchanged from the helper.
    return d


def _run_etl_pipeline(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the data-ingestion pipeline as a managed job (US16).

    params:
        mode:     "full" | "refresh" (default "refresh")
        domains:  optional list of domain names to restrict to (default: all)
        parallel: bool — parallelize normalize/load/MV refresh (full mode)

    NOTE: a ``full`` reload of large fact tables can exceed the APScheduler
    comfort window; for very large datasets prefer routing full loads to the
    pg-queue worker (common/services/pg_queue.py). ``refresh`` is incremental
    and short, so it is well-suited to APScheduler.
    """
    from scripts.etl import run_pipeline as rp

    mode = params.get("mode", "refresh")
    if mode not in ("full", "refresh"):
        raise ValueError(f"invalid etl_pipeline mode: {mode!r}")
    requested = params.get("domains") or None
    parallel = bool(params.get("parallel", False))

    cfg = rp._cfg()
    domain_order = cfg.get("domain_order", rp.ALL_DOMAINS)
    if requested:
        domains = [d for d in domain_order if d in requested]
    else:
        domains = list(domain_order)
    source_dir = rp.ROOT / cfg.get("source_data_dir", "data/input")

    if progress_cb:
        progress_cb(pct=5, msg=f"Starting {mode} pipeline ({len(domains)} domains)")

    if mode == "full":
        results = rp.run_full(domains, source_dir, parallel=parallel)
    else:
        results = rp.run_refresh(domains, source_dir)

    loaded = sum(1 for r in results if r.get("loaded") or r.get("rows_loaded") is not None)
    skipped = sum(1 for r in results if r.get("skipped"))
    if progress_cb:
        progress_cb(pct=100, msg=f"{mode} pipeline complete: {loaded} loaded, {skipped} skipped")

    return {
        "mode": mode,
        "domains": domains,
        "loaded": loaded,
        "skipped": skipped,
        "results": results,
        "output_log": f"{mode} pipeline: {loaded} loaded, {skipped} skipped "
        f"across {len(domains)} domain(s)",
    }


def _run_load_domain(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a single-domain ETL load as a managed job (US17c).

    Shells out to the one unified load engine (``scripts.etl.load``) — the same
    path the legacy IntegrationRunner used — and records parsed row metrics in
    the job result so the ``integration_job_unified`` view (US17b) can surface
    them. Exit codes mirror the legacy runner: 0 -> loaded, 2 -> skipped (no
    work), any other -> failure (raise, so JobManager marks the job failed).

    params:
        domain:  domain name (required)
        mode:    onetime | delta | file (default "delta")
        slice:   optional partition slice (e.g. "2026-04")
        file:    optional explicit file path
        reindex: optional bool — REINDEX after a bulk upsert
    """
    from common.services.etl_job_output import parse_final_json

    domain = params.get("domain")
    if not domain:
        raise ValueError("load_domain requires a 'domain' param")
    mode = params.get("mode", "delta")
    if mode not in ("onetime", "delta", "file"):
        raise ValueError(f"invalid load_domain mode: {mode!r}")
    slice_ = params.get("slice")
    file = params.get("file")
    reindex = bool(params.get("reindex", False))

    cmd = [_UV, "run", "python", "-m", "scripts.etl.load", "--domain", domain, "--mode", mode]
    if slice_:
        cmd += ["--slice", slice_]
    if file:
        cmd += ["--file", file]
    if reindex:
        cmd.append("--reindex")

    if progress_cb:
        progress_cb(pct=5, msg=f"Loading {domain} ({mode})")

    # argv is built from the validated domain/mode/slice/file inputs (never shell).
    proc = subprocess.run(
        cmd,
        cwd=str(_SCRIPTS_DIR.parent),
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=False,
    )
    metrics = parse_final_json(proc.stdout)

    if proc.returncode == 0:
        skipped = False
    elif proc.returncode == 2:
        skipped = True  # dispatcher had no work to do (e.g. delta, no new files)
    else:
        err = (
            metrics["error"]
            or (proc.stderr.strip()[-2000:] if proc.stderr else None)
            or f"exit code {proc.returncode}"
        )
        raise RuntimeError(f"load {domain} failed: {err}")

    rows_loaded = metrics["rows_loaded"]
    if progress_cb:
        progress_cb(
            pct=100,
            msg=f"{domain} {'skipped' if skipped else 'loaded'}: {rows_loaded} rows",
        )
    return {
        "domain": domain,
        "mode": mode,
        "slice": slice_,
        "file": file,
        "skipped": skipped,
        "rows_loaded": rows_loaded,
        "rows_inserted": metrics["rows_inserted"],
        "rows_updated": metrics["rows_updated"],
        "rows_deleted": metrics["rows_deleted"],
        "output_log": f"load {domain} ({mode}): {rows_loaded} rows"
        f"{' [skipped]' if skipped else ''}",
    }
