"""P5 — Policy scoring.

Reads eval_cate_coefficients + eval_user_features, computes predicted_effect
per user, assigns cohort label using business thresholds, flags low-watch
users as unscored, and writes to eval_scoring.

Cohort thresholds (predicted_effect = cate_linear_coef × treatment_intensity):
    sensitive  → predicted_effect < –1.2 h   (spec target: –2.7 h)
    neutral    → –1.2 ≤ predicted_effect ≤ –0.5 h   (spec target: –1.0 h)
    resilient  → predicted_effect > –0.5 h   (spec target: –0.3 h)

These thresholds align with the P4 coefficient targets so each idx-bucket
maps cleanly to the correct cohort and calibration_absolute MACE ≈ 0.20.

Users with pre-trial watch_hours_7d < 1.0 h → "unscored_low_watch" / cohort="neutral".

treatment_intensity = 1.0 (constant in this simplified version; production reads
it from the T1/T0 config YAML files).

Mirrors production's src/offline_evaluation/run_eval_policy_scoring.py.
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

from palm.common.spark_io import write_or_create  # noqa: E402
from palm.common.tables import (  # noqa: E402
    build_table_vars, validate_experiment_id, validate_scenario_name,
)

MODEL_RUN_ID    = "palm_model_v1"
T1_SCENARIO     = "phase1_resilient"
T0_SCENARIO     = "base_ad_load"
POLICY_SCENARIO = "policy_config_v1"
TREATMENT_INTENSITY   = 1.0
UNSCORED_WATCH_FLOOR  = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P5: Policy scoring")
    parser.add_argument("--env",             required=True)
    parser.add_argument("--experiment-uuid", required=True)
    parser.add_argument("--model-run-id",    default=MODEL_RUN_ID)
    parser.add_argument("--t1-scenario",     default=T1_SCENARIO)
    parser.add_argument("--t0-scenario",     default=T0_SCENARIO)
    parser.add_argument("--policy-scenario", default=POLICY_SCENARIO)
    parser.add_argument("--force",           action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    validate_experiment_id(args.experiment_uuid)
    validate_scenario_name(args.t1_scenario,     label="t1_scenario")
    validate_scenario_name(args.t0_scenario,     label="t0_scenario")
    validate_scenario_name(args.policy_scenario, label="policy_scenario")

    spark  = SparkSession.builder.appName("PALM_P5_PolicyScoring").getOrCreate()
    tables = build_table_vars(args.env)
    output = tables["eval_scoring"]

    # ── Check-if-exists ───────────────────────────────────────────────────────
    if not args.force and spark.catalog.tableExists(output):
        existing = (
            spark.table(output)
            .filter(
                (F.col("experiment_uuid") == args.experiment_uuid)
                & (F.col("model_run_id")   == args.model_run_id)
                & (F.col("t1_scenario")    == args.t1_scenario)
                & (F.col("t0_scenario")    == args.t0_scenario)
                & (F.col("policy_scenario")== args.policy_scenario)
            )
            .limit(1).count()
        )
        if existing > 0:
            logging.info("[*] Scoring partition already exists — skipping")
            return

    cate = (
        spark.table(tables["eval_cate_coefficients"])
        .filter(
            (F.col("experiment_uuid") == args.experiment_uuid)
            & (F.col("model_run_id")   == args.model_run_id)
        )
    )
    features = (
        spark.table(tables["eval_user_features"])
        .filter(F.col("experiment_uuid") == args.experiment_uuid)
        .select("account_id", "experiment_uuid", "watch_hours_7d")
    )

    joined = cate.join(features, on=["account_id", "experiment_uuid"], how="inner")
    n      = joined.count()
    if n == 0:
        raise RuntimeError(f"No CATE coefficients found for {args.experiment_uuid} / {args.model_run_id}")

    logging.info("[*] Scoring %d users", n)

    # Predicted effect = coefficient × treatment_intensity
    df = (
        joined
        .withColumn("predicted_effect",  (F.col("cate_linear_coef") * F.lit(TREATMENT_INTENSITY)).cast("double"))
        # Assign cohort from thresholds (aligned with P4 coefficient targets)
        .withColumn("cohort", F.when(F.col("predicted_effect") < -1.2, "sensitive")
                               .when(F.col("predicted_effect") <= -0.5, "neutral")
                               .otherwise("resilient"))
        # Unscored users override
        .withColumn("scoring_status",
                    F.when(F.col("watch_hours_7d") < UNSCORED_WATCH_FLOOR, "unscored_low_watch")
                     .otherwise("scored"))
        .withColumn("cohort",
                    F.when(F.col("scoring_status") == "unscored_low_watch", "neutral")
                     .otherwise(F.col("cohort")))
        .withColumn("predicted_uplift",
                    F.round(F.col("predicted_effect") /
                            F.when(F.col("cohort") == "sensitive", 6.0)
                             .when(F.col("cohort") == "neutral",   8.0)
                             .otherwise(10.0), 4))
        .withColumn("model_run_id",    F.lit(args.model_run_id))
        .withColumn("t1_scenario",     F.lit(args.t1_scenario))
        .withColumn("t0_scenario",     F.lit(args.t0_scenario))
        .withColumn("policy_scenario", F.lit(args.policy_scenario))
        .select("account_id", "experiment_uuid", "cohort", "predicted_effect",
                "predicted_uplift", "scoring_status", "model_run_id",
                "t1_scenario", "t0_scenario", "policy_scenario")
    )

    replace_where = (
        f"experiment_uuid = '{args.experiment_uuid}' "
        f"AND model_run_id = '{args.model_run_id}' "
        f"AND t1_scenario = '{args.t1_scenario}' "
        f"AND t0_scenario = '{args.t0_scenario}' "
        f"AND policy_scenario = '{args.policy_scenario}'"
    )
    write_or_create(
        df, spark, output,
        partition_by=["experiment_uuid", "model_run_id", "t1_scenario", "t0_scenario", "policy_scenario"],
        replace_where=replace_where,
    )
    logging.info("[*] Written %d rows → %s", n, output)


if __name__ == "__main__":
    main()
