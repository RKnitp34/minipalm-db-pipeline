"""P4 — CATE inference.

Reads eval_user_features, computes per-user CATE (Conditional Average Treatment
Effect) linear coefficients using a simplified linear formula, and writes to
eval_cate_coefficients.

Formula (mirrors production's Causal Forest DML output structure):
    cate_linear_coef = (watch_hours_30d - 33) * 0.03 + noise

This maps the three feature clusters to distinct cohort-compatible ranges:
    sensitive  (wh30d ≈ 15h) → coef ≈ –0.54  → will be < –0.5  → sensitive cohort ✓
    neutral    (wh30d ≈ 33h) → coef ≈  0.00  → will be –0.5–0  → neutral cohort ✓
    resilient  (wh30d ≈ 50h) → coef ≈ +0.51  → will be > 0     → resilient cohort ✓

Confidence bounds are ±0.15 (fixed width for simplicity).

In production this loads an EconML / scikit-learn model from a UC Volume path
and runs distributed batch inference.

Mirrors production's src/offline_evaluation/run_eval_cate_inference.py.
Databricks job task — part of the PALM evaluation DAG.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

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

MODEL_RUN_ID = "palm_model_v1"


@F.udf(returnType=DoubleType())
def cate_coef_udf(account_id: str, watch_hours_30d: float) -> float:
    """Deterministic synthetic CATE coefficient from pre-trial features."""
    import numpy as _np  # noqa: PLC0415
    rng  = _np.random.default_rng(int(account_id.replace("user_", "")) + 5555)
    coef = (watch_hours_30d - 33.0) * 0.03 + rng.normal(0, 0.05)
    return float(round(coef, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4: CATE inference")
    parser.add_argument("--env",             required=True)
    parser.add_argument("--experiment-uuid", required=True)
    parser.add_argument("--model-run-id",    default=MODEL_RUN_ID)
    parser.add_argument("--force",           action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    validate_experiment_id(args.experiment_uuid)

    spark       = SparkSession.builder.appName("PALM_P4_CATEInference").getOrCreate()
    tables      = build_table_vars(args.env)
    output      = tables["eval_cate_coefficients"]
    model_run   = args.model_run_id

    # ── Check-if-exists ───────────────────────────────────────────────────────
    if not args.force and spark.catalog.tableExists(output):
        existing = (
            spark.table(output)
            .filter(
                (F.col("experiment_uuid") == args.experiment_uuid)
                & (F.col("model_run_id")   == model_run)
            )
            .limit(1).count()
        )
        if existing > 0:
            logging.info("[*] CATE coefficients exist for %s / %s — skipping", args.experiment_uuid, model_run)
            return

    features = (
        spark.table(tables["eval_user_features"])
        .filter(F.col("experiment_uuid") == args.experiment_uuid)
    )
    n = features.count()
    if n == 0:
        raise RuntimeError(f"eval_user_features empty for {args.experiment_uuid}")

    logging.info("[*] Running CATE inference for %d users (model=%s)", n, model_run)
    df = (
        features
        .withColumn("model_run_id",       F.lit(model_run))
        .withColumn("cate_linear_coef",   cate_coef_udf(F.col("account_id"), F.col("watch_hours_30d")))
        .withColumn("cate_linear_coef_lb", F.col("cate_linear_coef") - F.lit(0.15))
        .withColumn("cate_linear_coef_ub", F.col("cate_linear_coef") + F.lit(0.15))
        .select("account_id", "experiment_uuid", "model_run_id",
                "cate_linear_coef", "cate_linear_coef_lb", "cate_linear_coef_ub")
    )

    write_or_create(
        df, spark, output,
        partition_by=["experiment_uuid", "model_run_id"],
        replace_where=(
            f"experiment_uuid = '{args.experiment_uuid}' "
            f"AND model_run_id = '{model_run}'"
        ),
    )
    logging.info("[*] Written %d rows → %s", n, output)


if __name__ == "__main__":
    main()
