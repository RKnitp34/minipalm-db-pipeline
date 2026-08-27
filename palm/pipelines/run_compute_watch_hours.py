"""P2 — Compute observed watch hours.

Reads eval_experiment_population, generates synthetic per-user K-day watch hours
using a deterministic formula keyed on account_id and treatment_arm, and writes
to eval_watch_hours.

In production this reads from fact_watches (actual streaming events).
Here we generate synthetic data so the pipeline can run end-to-end without
real watch-event tables.

Synthetic logic (consistent with cohort categories derived from account_id):
    user_idx = int(account_id[-5:]) % 100
    0–29  → sensitive-like:  control=6.0h,  treatment=3.5h
    30–69 → neutral-like:    control=8.0h,  treatment=7.2h
    70–99 → resilient-like:  control=10.0h, treatment=9.5h

Mirrors production's src/offline_evaluation/run_compute_watch_hours.py.
Databricks job task — part of the PALM evaluation DAG.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

sys.dont_write_bytecode = True

try:
    _script_path = os.path.abspath(__file__)
except NameError:
    _script_path = filename  # type: ignore[name-defined]  # noqa: F821
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_script_path)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from palm.common.spark_io import write_or_create  # noqa: E402
from palm.common.tables import build_table_vars, validate_experiment_id  # noqa: E402

# Base watch hours (control, treatment) per category derived from account_id
_WATCH_PARAMS = {
    "sensitive": (6.0,  3.5),
    "neutral":   (8.0,  7.2),
    "resilient": (10.0, 9.5),
}


def _user_category(account_id: str) -> str:
    idx = int(account_id.replace("user_", "")) % 100
    if idx < 30:   return "sensitive"
    elif idx < 70: return "neutral"
    else:          return "resilient"


@F.udf(returnType=DoubleType())
def synthetic_watch_hours_udf(account_id: str, treatment_arm: str, k_value: int) -> float:
    """Deterministic synthetic watch hours based on account_id + arm."""
    import numpy as _np  # noqa: PLC0415
    idx      = int(account_id.replace("user_", "")) % 100
    category = "sensitive" if idx < 30 else ("neutral" if idx < 70 else "resilient")
    params   = {"sensitive": (6.0, 3.5), "neutral": (8.0, 7.2), "resilient": (10.0, 9.5)}
    ctrl_h, trt_h = params[category]
    base = ctrl_h if treatment_arm == "control" else trt_h
    seed = int(account_id.replace("user_", "")) + k_value * 1000
    rng  = _np.random.default_rng(seed)
    return float(max(0.0, base + rng.normal(0, 1.0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2: Compute watch hours")
    parser.add_argument("--env",             required=True)
    parser.add_argument("--experiment-uuid", required=True)
    parser.add_argument("--k-values",        default="7", help="Comma-separated, e.g. '7,14'")
    parser.add_argument("--force",           action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args     = parse_args()
    validate_experiment_id(args.experiment_uuid)
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    spark  = SparkSession.builder.appName("PALM_P2_WatchHours").getOrCreate()
    tables = build_table_vars(args.env)
    output = tables["eval_watch_hours"]

    # ── Check-if-exists ───────────────────────────────────────────────────────
    if not args.force and spark.catalog.tableExists(output):
        existing_ks = {
            r.k_value for r in
            spark.table(output)
            .filter(F.col("experiment_uuid") == args.experiment_uuid)
            .select("k_value").distinct().collect()
        }
        k_values = [k for k in k_values if k not in existing_ks]
        if not k_values:
            logging.info("[*] Watch hours already exist — skipping")
            return
        logging.info("[*] Missing k_values: %s", k_values)

    population = (
        spark.table(tables["eval_experiment_population"])
        .filter(F.col("experiment_uuid") == args.experiment_uuid)
    )
    n = population.count()
    if n == 0:
        raise RuntimeError(f"eval_experiment_population is empty for {args.experiment_uuid}")

    for k in k_values:
        logging.info("[*] Computing watch_hours for k=%d (%d users)", k, n)
        df = (
            population
            .withColumn("k_value",     F.lit(k))
            .withColumn("watch_hours", synthetic_watch_hours_udf(
                F.col("account_id"), F.col("treatment_arm"), F.lit(k)
            ))
            .select("account_id", "experiment_uuid", "experiment_name",
                    "treatment_arm", "treatment_uuid", "k_value", "watch_hours")
        )
        k_list = ",".join(str(kv) for kv in k_values)
        write_or_create(
            df, spark, output,
            partition_by=["experiment_uuid", "k_value"],
            replace_where=(
                f"experiment_uuid = '{args.experiment_uuid}' AND k_value IN ({k_list})"
            ),
        )
        logging.info("[*] Written k=%d → %s", k, output)


if __name__ == "__main__":
    main()
