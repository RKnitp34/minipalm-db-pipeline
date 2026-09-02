"""Pipeline run logger — writes one row per task to pipeline_run_log.

Mirrors the observability pattern used in production inference pipelines.
Each pipeline script calls PipelineRunLogger at start and finish so every
run (success, failure, or skip) is recorded and queryable.

Usage (in any P1-P7 script)::

    from palm.common.run_logger import PipelineRunLogger

    logger = PipelineRunLogger(
        spark         = spark,
        log_table     = tables["pipeline_run_log"],
        task_key      = "p1_load_population",
        dataset_date  = str(date.today()),
        experiment_uuid = args.experiment_uuid,
        env           = args.env,
    )

    # --- your pipeline logic here ---
    df = ...
    write_or_create(df, spark, ...)
    n = df.count()

    logger.success(rows_written=n)   # on success
    # OR
    logger.skip()                    # when check-if-exists fires
    # OR  (in except block)
    logger.fail(error=str(e))        # on exception

Schema of pipeline_run_log
--------------------------
    run_id           STRING    UUID — same for all tasks in one Databricks job run
    job_run_id       STRING    Databricks job run ID (from env var, null for notebook runs)
    task_key         STRING    e.g. "p1_load_population"
    dataset_date     DATE      partition date being processed
    experiment_uuid  STRING    which experiment
    status           STRING    "success" | "failed" | "skipped"
    started_at       TIMESTAMP when this task started
    completed_at     TIMESTAMP when this task finished
    duration_seconds INT       wall-clock seconds
    rows_written     INT       rows written to output table (null if skipped/failed)
    error_message    STRING    truncated exception message (null if success/skipped)
    env              STRING    "dev" | "prod"
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# Databricks injects this env var when running inside a job task.
# It is None when running from a notebook interactively.
_DATABRICKS_JOB_RUN_ID_ENV = "DATABRICKS_JOB_RUN_ID"


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class PipelineRunLogger:
    """Context-aware logger that writes one row to pipeline_run_log per call.

    Instantiate at the top of main(), call .success() / .skip() / .fail()
    at the end.  All three methods are safe to call multiple times — only
    the first call writes to the table.
    """

    def __init__(
        self,
        spark,
        log_table: str,
        task_key: str,
        dataset_date: str,
        experiment_uuid: str,
        env: str,
        run_id: Optional[str] = None,
    ) -> None:
        self._spark           = spark
        self._log_table       = log_table
        self._task_key        = task_key
        self._dataset_date    = dataset_date
        self._experiment_uuid = experiment_uuid
        self._env             = env
        self._started_at      = _utcnow()
        self._done            = False

        # Shared run_id ties all tasks of one job trigger together.
        # Callers can pass an explicit run_id (from job parameters) or let it
        # default to a new UUID (useful for notebook / ad-hoc runs).
        self._run_id = run_id or str(uuid.uuid4())

        # Databricks sets this env var inside job tasks.
        self._job_run_id = os.environ.get(_DATABRICKS_JOB_RUN_ID_ENV)

    # ── Public API ────────────────────────────────────────────────────────────

    def success(self, rows_written: int) -> None:
        """Record a successful task completion."""
        self._write("success", rows_written=rows_written, error_message=None)

    def skip(self) -> None:
        """Record a skipped task (check-if-exists found existing partition)."""
        self._write("skipped", rows_written=None, error_message=None)

    def fail(self, error: str) -> None:
        """Record a failed task with a truncated error message."""
        # Truncate to 1000 chars to keep the table scannable.
        self._write("failed", rows_written=None, error_message=str(error)[:1000])

    @property
    def run_id(self) -> str:
        return self._run_id

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write(
        self,
        status: str,
        rows_written: Optional[int],
        error_message: Optional[str],
    ) -> None:
        if self._done:
            log.warning(
                "[run_logger] _write called more than once for %s — ignoring",
                self._task_key,
            )
            return

        completed_at     = _utcnow()
        duration_seconds = int((completed_at - self._started_at).total_seconds())

        row = {
            "run_id":           self._run_id,
            "job_run_id":       self._job_run_id,
            "task_key":         self._task_key,
            "dataset_date":     self._dataset_date,
            "experiment_uuid":  self._experiment_uuid,
            "status":           status,
            "started_at":       self._started_at,
            "completed_at":     completed_at,
            "duration_seconds": duration_seconds,
            "rows_written":     rows_written,
            "error_message":    error_message,
            "env":              self._env,
        }

        try:
            import pandas as _pd  # noqa: PLC0415
            pdf = _pd.DataFrame([row])
            sdf = self._spark.createDataFrame(pdf)
            sdf.write.format("delta").mode("append").saveAsTable(self._log_table)
            log.info(
                "[run_logger] %s → %s (%ds, rows=%s)",
                self._task_key, status, duration_seconds, rows_written,
            )
        except Exception as exc:  # noqa: BLE001
            # Logging must never crash the pipeline.
            log.error("[run_logger] Failed to write run log: %s", exc)

        self._done = True
