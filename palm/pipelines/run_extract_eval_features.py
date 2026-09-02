"""P3 — Extract user features for CATE inference.

Reads eval_experiment_population, generates pre-experiment user features
anchored to experiment_start_date - 1 (dataset_date), and writes to
eval_user_features.

In production this joins against multiple feature tables
(user_features_content, user_features_ad, user_features_behavior).
Here we generate synthetic features deterministically from account_id
so the cohort structure is consistent across the whole pipeline:
    user_idx % 100 < 30  → sensitive-like:  low watch hours (~15h/30d)
    user_idx % 100 30–69 → neutral-like:    mid watch hours (~33h/30d)
    user_idx % 100 ≥ 70  → resilient-like:  high watch hours (~50h/30d)

These feature ranges feed into P4's CATE formula, which produces
predicted_effects that fall into the correct cohort buckets in P5.

Mirrors production's src/offline_evaluation/run_extract_eval_features.py.
Databricks job task — part of the PALM evaluation DAG.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

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


@F.udf(returnType=DoubleType())
def synthetic_wh7d_udf(account_id: str) -> float:
    import numpy as _np  # noqa: PLC0415
    idx  = int(account_id.replace("user_", "")) % 100
    base = 6.0 if idx < 30 else (8.0 if idx < 70 else 10.0)
    rng  = _np.random.default_rng(int(account_id.replace("user_", "")))
    return float(max(0.0, base + rng.normal(0, 0.5)))


@F.udf(returnType=DoubleType())
def synthetic_wh30d_udf(account_id: str) -> float:
    import numpy as _np  # noqa: PLC0415
    idx  = int(account_id.replace("user_", "")) % 100
    base = 15.0 if idx < 30 else (33.0 if idx < 70 else 50.0)
    rng  = _np.random.default_rng(int(account_id.replace("user_", "")) + 9999)
    return float(max(0.0, base + rng.normal(0, 2.0)))


@F.udf(returnType=StringType())
def synthetic_content_type_udf(account_id: str) -> str:
    import numpy as _np  # noqa: PLC0415
    rng = _np.random.default_rng(int(account_id.replace("user_", "")) + 7777)
    return str(rng.choice(["series", "film", "mixed"], p=[0.4, 0.3, 0.3]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P3: Extract eval user features")
    parser.add_argument("--env",             required=True)
    parser.add_argument("--experiment-uuid", required=True)
    parser.add_argument("--force",           action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    validate_experiment_id(args.experiment_uuid)

    from datetime import date  # noqa: PLC0415
    spark  = SparkSession.builder.appName("PALM_P3_EvalFeatures").getOrCreate()
    tables = build_table_vars(args.env)
    output = tables["eval_user_features"]

    run_logger = PipelineRunLogger(
        spark=spark, log_table=tables["pipeline_run_log"],
        task_key="p3_extract_eval_features", dataset_date=str(date.today()),
        experiment_uuid=args.experiment_uuid, env=args.env,
    )

    # ── Check-if-exists ───────────────────────────────────────────────────────
    if not args.force and spark.catalog.tableExists(output):
        existing = (
            spark.table(output)
            .filter(F.col("experiment_uuid") == args.experiment_uuid)
            .limit(1).count()
        )
        if existing > 0:
            logging.info("[*] Features already exist for %s — skipping", args.experiment_uuid)
            run_logger.skip()
            return

    try:
        population = (
            spark.table(tables["eval_experiment_population"])
            .filter(F.col("experiment_uuid") == args.experiment_uuid)
        )
        n = population.count()
        if n == 0:
            raise RuntimeError(f"eval_experiment_population empty for {args.experiment_uuid}")

        logging.info("[*] Extracting features for %d users", n)
        df = (
            population
            .withColumn("dataset_date",    (F.col("experiment_start_date") - F.expr("INTERVAL 1 DAY")).cast("date"))
            .withColumn("watch_hours_7d",  synthetic_wh7d_udf(F.col("account_id")))
            .withColumn("watch_hours_30d", synthetic_wh30d_udf(F.col("account_id")))
            .withColumn("content_type",    synthetic_content_type_udf(F.col("account_id")))
            .select("account_id", "experiment_uuid", "dataset_date",
                    "watch_hours_7d", "watch_hours_30d", "content_type")
        )

        write_or_create(
            df, spark, output,
            partition_by=["experiment_uuid"],
            replace_where=f"experiment_uuid = '{args.experiment_uuid}'",
        )
        logging.info("[*] Written %d rows → %s", n, output)
        run_logger.success(rows_written=n)

    except Exception as exc:
        run_logger.fail(error=str(exc))
        raise


if __name__ == "__main__":
    main()
