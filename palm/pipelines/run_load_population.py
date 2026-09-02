"""P1 — Load experiment population.

Reads experiment_user_assignments + experiment_config, joins on experiment_uuid,
deduplicates users (keep first treatment_uuid alphabetically), drops null
account_ids, and writes to eval_experiment_population.

Mirrors production's src/offline_evaluation/run_load_population.py.
Check-if-exists: skips if population for this experiment already exists
(use --force to recompute).

Databricks job task — part of the PALM evaluation DAG.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.dont_write_bytecode = True

try:
    _script_path = os.path.abspath(__file__)
except NameError:
    _script_path = filename  # type: ignore[name-defined]  # noqa: F821
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_script_path)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from palm.common.run_logger import PipelineRunLogger  # noqa: E402
from palm.common.spark_io import write_or_create  # noqa: E402
from palm.common.tables import build_table_vars, validate_experiment_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1: Load experiment population")
    parser.add_argument("--env",             required=True, help="dev / qa / prod")
    parser.add_argument("--experiment-uuid", required=True, help="Experiment ID to process")
    parser.add_argument("--force",           action="store_true", help="Recompute even if already exists")
    return parser.parse_args()


def load_population(spark: SparkSession, tables: dict, experiment_uuid: str):
    """Join assignments + config, deduplicate, drop nulls."""
    assignments = (
        spark.table(tables["experiment_user_assignments"])
        .filter(F.col("experiment_uuid") == experiment_uuid)
        .dropna(subset=["account_id"])
    )
    config = spark.table(tables["experiment_config"]).filter(
        F.col("experiment_uuid") == experiment_uuid
    )

    # Deduplicate: keep alphabetically-first treatment_uuid per user
    deduped = (
        assignments
        .withColumn("_rank", F.row_number().over(
            __import__("pyspark.sql.window", fromlist=["Window"])
            .Window.partitionBy("account_id", "experiment_uuid")
            .orderBy("treatment_uuid")
        ))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )

    return deduped.join(
        config.select("experiment_uuid", "experiment_name",
                      "experiment_start_date", "experiment_end_date"),
        on="experiment_uuid",
        how="inner",
    ).select(
        "account_id", "experiment_uuid", "experiment_name",
        "treatment_arm", "treatment_uuid",
        "experiment_start_date", "experiment_end_date",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    validate_experiment_id(args.experiment_uuid)

    from datetime import date  # noqa: PLC0415
    spark  = SparkSession.builder.appName("PALM_P1_LoadPopulation").getOrCreate()
    tables = build_table_vars(args.env)
    output = tables["eval_experiment_population"]

    run_logger = PipelineRunLogger(
        spark=spark, log_table=tables["pipeline_run_log"],
        task_key="p1_load_population", dataset_date=str(date.today()),
        experiment_uuid=args.experiment_uuid, env=args.env,
    )

    # ── Check-if-exists (skip if data already there) ──────────────────────────
    if not args.force and spark.catalog.tableExists(output):
        existing = (
            spark.table(output)
            .filter(F.col("experiment_uuid") == args.experiment_uuid)
            .limit(1).count()
        )
        if existing > 0:
            logging.info("[*] Population already exists for %s — skipping (use --force to recompute)", args.experiment_uuid)
            run_logger.skip()
            return

    try:
        logging.info("[*] Loading population for %s", args.experiment_uuid)
        df = load_population(spark, tables, args.experiment_uuid)

        n = df.count()
        if n == 0:
            raise RuntimeError(
                f"No users found for experiment_uuid={args.experiment_uuid}. "
                "Did data_generator run first?"
            )
        logging.info("[*] Population: %d users", n)

        write_or_create(
            df, spark, output,
            partition_by=["experiment_uuid"],
            replace_where=f"experiment_uuid = '{args.experiment_uuid}'",
        )
        logging.info("[*] Written %d rows to %s", n, output)
        run_logger.success(rows_written=n)

    except Exception as exc:
        run_logger.fail(error=str(exc))
        raise


if __name__ == "__main__":
    main()
