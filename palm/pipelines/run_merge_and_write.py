"""P6 — Merge watch hours + scoring → offline_evaluation_results.

Reads eval_watch_hours + eval_scoring, joins on (account_id, experiment_uuid),
derives baseline_hps (mean control-arm watch_hours per cohort), and writes the
merged result to offline_evaluation_results.

baseline_hps is the control arm mean within each cohort — used later in P7
to compute calibration metrics.

Mirrors production's src/offline_evaluation/run_merge_and_write.py.
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
CONTROL_ARM     = "control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P6: Merge and write evaluation results")
    parser.add_argument("--env",             required=True)
    parser.add_argument("--experiment-uuid", required=True)
    parser.add_argument("--k-values",        default="7")
    parser.add_argument("--model-run-id",    default=MODEL_RUN_ID)
    parser.add_argument("--t1-scenario",     default=T1_SCENARIO)
    parser.add_argument("--t0-scenario",     default=T0_SCENARIO)
    parser.add_argument("--policy-scenario", default=POLICY_SCENARIO)
    parser.add_argument("--control-arm",     default=CONTROL_ARM)
    parser.add_argument("--force",           action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args     = parse_args()
    validate_experiment_id(args.experiment_uuid)
    validate_scenario_name(args.t1_scenario,     label="t1_scenario")
    validate_scenario_name(args.t0_scenario,     label="t0_scenario")
    validate_scenario_name(args.policy_scenario, label="policy_scenario")
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    spark  = SparkSession.builder.appName("PALM_P6_MergeWrite").getOrCreate()
    tables = build_table_vars(args.env)
    output = tables["offline_evaluation_results"]

    watch_hours = (
        spark.table(tables["eval_watch_hours"])
        .filter(F.col("experiment_uuid") == args.experiment_uuid)
        .filter(F.col("k_value").isin(k_values))
    )
    scoring = (
        spark.table(tables["eval_scoring"])
        .filter(F.col("experiment_uuid") == args.experiment_uuid)
        .filter(F.col("model_run_id")    == args.model_run_id)
        .filter(F.col("t1_scenario")     == args.t1_scenario)
        .filter(F.col("t0_scenario")     == args.t0_scenario)
        .filter(F.col("policy_scenario") == args.policy_scenario)
        .select("account_id", "experiment_uuid", "cohort",
                "predicted_effect", "predicted_uplift", "scoring_status")
    )

    merged = watch_hours.join(scoring, on=["account_id", "experiment_uuid"], how="inner")
    n      = merged.count()
    if n == 0:
        raise RuntimeError(f"Join produced 0 rows for {args.experiment_uuid}")
    logging.info("[*] Merged %d rows", n)

    # Compute baseline_hps: mean control-arm watch_hours per (experiment, cohort)
    baseline = (
        merged
        .filter(F.col("treatment_arm") == args.control_arm)
        .groupBy("experiment_uuid", "cohort")
        .agg(F.round(F.mean("watch_hours"), 4).alias("baseline_hps"))
    )

    result = (
        merged
        .join(baseline, on=["experiment_uuid", "cohort"], how="left")
        .withColumn("model_run_id",    F.lit(args.model_run_id))
        .withColumn("t1_scenario",     F.lit(args.t1_scenario))
        .withColumn("t0_scenario",     F.lit(args.t0_scenario))
        .withColumn("policy_scenario", F.lit(args.policy_scenario))
        .select(
            "account_id", "experiment_uuid", "k_value", "treatment_arm",
            "treatment_uuid", "watch_hours", "cohort", "predicted_effect",
            "predicted_uplift", "scoring_status", "baseline_hps",
            "model_run_id", "t1_scenario", "t0_scenario", "policy_scenario",
        )
    )

    k_list = ",".join(str(k) for k in k_values)
    replace_where = (
        f"experiment_uuid = '{args.experiment_uuid}' "
        f"AND k_value IN ({k_list}) "
        f"AND model_run_id = '{args.model_run_id}' "
        f"AND t1_scenario = '{args.t1_scenario}' "
        f"AND t0_scenario = '{args.t0_scenario}' "
        f"AND policy_scenario = '{args.policy_scenario}'"
    )
    write_or_create(
        result, spark, output,
        partition_by=["experiment_uuid", "k_value", "model_run_id",
                      "t1_scenario", "t0_scenario", "policy_scenario"],
        replace_where=replace_where,
    )
    logging.info("[*] Written %d rows → %s", n, output)


if __name__ == "__main__":
    main()
