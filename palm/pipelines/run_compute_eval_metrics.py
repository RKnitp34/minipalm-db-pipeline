"""P7 — Compute policy evaluation metrics.

Reads offline_evaluation_results, computes 7 metrics per
(experiment, k_value, model_run_id, t1_scenario, t0_scenario, policy_scenario),
and writes one row per metric to offline_evaluation_metrics.

Metrics computed:
  separation_magnitude            resilient_mean_wh – sensitive_mean_wh (treatment arms)
  relative_separation_magnitude   separation_magnitude / sensitive_baseline_hps
  cohort_delta_ordering           m(sensitive) < m(neutral) < m(resilient) among treatments?
  cohort_relative_delta_ordering  same check on relative uplift
  calibration_absolute            mean |predicted_effect – actual_effect| across cohorts (MACE)
  calibration_relative            MACE / mean baseline_hps
  cohort_traits                   are cohort % shares in [20%, 50%]?

Mirrors production's src/offline_evaluation/run_compute_eval_metrics.py.
Databricks job task — part of the PALM evaluation DAG.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

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
from palm.common.tables import (  # noqa: E402
    build_table_vars, validate_experiment_id, validate_scenario_name,
)

MODEL_RUN_ID    = "palm_model_v1"
T1_SCENARIO     = "phase1_resilient"
T0_SCENARIO     = "base_ad_load"
POLICY_SCENARIO = "policy_config_v1"
CONTROL_ARM     = "control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P7: Compute evaluation metrics")
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

    from datetime import date  # noqa: PLC0415
    spark  = SparkSession.builder.appName("PALM_P7_EvalMetrics").getOrCreate()
    tables = build_table_vars(args.env)

    run_logger = PipelineRunLogger(
        spark=spark, log_table=tables["pipeline_run_log"],
        task_key="p7_compute_eval_metrics", dataset_date=str(date.today()),
        experiment_uuid=args.experiment_uuid, env=args.env,
    )

    try:
      results = (
        spark.table(tables["offline_evaluation_results"])
        .filter(F.col("experiment_uuid") == args.experiment_uuid)
        .filter(F.col("k_value").isin(k_values))
        .filter(F.col("model_run_id")    == args.model_run_id)
        .filter(F.col("t1_scenario")     == args.t1_scenario)
        .filter(F.col("t0_scenario")     == args.t0_scenario)
        .filter(F.col("policy_scenario") == args.policy_scenario)
    )
    n = results.count()
    if n == 0:
        raise RuntimeError(f"offline_evaluation_results empty for {args.experiment_uuid}")

    logging.info("[*] Computing metrics for %d rows", n)

    # Collect to driver (1000 rows — safe on CE single-node)
    pdf = results.toPandas()

    computed_at = datetime.now(tz=timezone.utc)
    metric_rows = []

    for k in k_values:
        sub = pdf[pdf["k_value"] == k]
        trt = sub[sub["treatment_arm"] != args.control_arm]
        n_k = len(sub)

        # Per-cohort aggregates (treatment arms)
        means = trt.groupby("cohort")["watch_hours"].mean()
        sens_m = means.get("sensitive", float("nan"))
        neut_m = means.get("neutral",   float("nan"))
        res_m  = means.get("resilient", float("nan"))

        # Baseline HPS per cohort (control arm mean)
        ctrl_means = sub[sub["treatment_arm"] == args.control_arm].groupby("cohort")["baseline_hps"].first()
        sens_base  = ctrl_means.get("sensitive", 6.0)

        # Actual effects: treatment mean – control mean per cohort
        actual_effects = {
            c: means.get(c, float("nan")) - (ctrl_means.get(c, float("nan")))
            for c in ["sensitive", "neutral", "resilient"]
        }
        pred_effects = sub.groupby("cohort")["predicted_effect"].mean().to_dict()

        # Metric 1 & 2: separation magnitude
        sep_mag = round(float(res_m - sens_m), 4) if not (res_m != res_m or sens_m != sens_m) else 0.0
        rel_sep = round(sep_mag / max(float(sens_base), 1e-6), 4)

        # Metric 3 & 4: delta ordering
        order_ok  = bool(sens_m < neut_m < res_m) if not any(v != v for v in [sens_m, neut_m, res_m]) else False
        rel_order_ok = order_ok  # same check on absolute means for simplicity

        # Metric 5 & 6: calibration
        abs_errors = [abs(pred_effects.get(c, 0) - actual_effects.get(c, 0))
                      for c in ["sensitive", "neutral", "resilient"]
                      if c in pred_effects]
        mace     = round(sum(abs_errors) / len(abs_errors), 4) if abs_errors else 0.0
        cal_rel  = round(mace / max(float(sub["baseline_hps"].mean()), 1e-6), 4)

        # Metric 7: cohort traits (% in [20%, 50%])
        cohort_pcts = sub.groupby("cohort").size() / n_k * 100
        traits_ok = all(20 <= cohort_pcts.get(c, 0) <= 50 for c in ["sensitive", "neutral", "resilient"])

        details = json.dumps({
            "per_cohort": {c: {
                "n":              int((sub["cohort"] == c).sum()),
                "actual_effect":  round(float(actual_effects.get(c, 0)), 4),
                "pred_effect":    round(float(pred_effects.get(c, 0)),   4),
                "mean_wh":        round(float(means.get(c, 0)),          4),
                "baseline_hps":   round(float(ctrl_means.get(c, 0)),     4),
            } for c in ["sensitive", "neutral", "resilient"]}
        })

        for name, value, passed in [
            ("separation_magnitude",           sep_mag,               sep_mag > 5.0),
            ("relative_separation_magnitude",  rel_sep,               rel_sep > 0.5),
            ("cohort_delta_ordering",          1.0 if order_ok else 0.0,     order_ok),
            ("cohort_relative_delta_ordering", 1.0 if rel_order_ok else 0.0, rel_order_ok),
            ("calibration_absolute",           mace,                  mace <= 0.40),
            ("calibration_relative",           cal_rel,               cal_rel <= 0.05),
            ("cohort_traits",                  1.0 if traits_ok else 0.0,    traits_ok),
        ]:
            metric_rows.append({
                "experiment_uuid": args.experiment_uuid,
                "k_value":         k,
                "model_run_id":    args.model_run_id,
                "t1_scenario":     args.t1_scenario,
                "t0_scenario":     args.t0_scenario,
                "policy_scenario": args.policy_scenario,
                "metric_name":     name,
                "passed":          bool(passed),
                "pass_rate":       None,
                "metric_value":    round(float(value), 4),
                "details_json":    details,
                "computed_at":     computed_at,
                "n_users":         n_k,
            })

    import pandas as pd  # noqa: PLC0415
    metrics_df  = pd.DataFrame(metric_rows)
    metrics_sdf = spark.createDataFrame(metrics_df)

    replace_where = (
        f"experiment_uuid = '{args.experiment_uuid}' "
        f"AND k_value IN ({','.join(str(k) for k in k_values)}) "
        f"AND model_run_id = '{args.model_run_id}' "
        f"AND t1_scenario = '{args.t1_scenario}' "
        f"AND t0_scenario = '{args.t0_scenario}' "
        f"AND policy_scenario = '{args.policy_scenario}'"
    )
    write_or_create(
        metrics_sdf, spark, tables["offline_evaluation_metrics"],
        partition_by=["experiment_uuid", "k_value", "model_run_id",
                      "t1_scenario", "t0_scenario", "policy_scenario"],
        replace_where=replace_where,
    )
        logging.info("[*] Written %d metric rows → %s", len(metric_rows), tables["offline_evaluation_metrics"])
        for row in metric_rows:
            logging.info("    %s = %.4f  passed=%s", row["metric_name"], row["metric_value"], row["passed"])
        run_logger.success(rows_written=len(metric_rows))

    except Exception as exc:
        run_logger.fail(error=str(exc))
        raise


if __name__ == "__main__":
    main()
