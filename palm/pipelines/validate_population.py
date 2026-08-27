"""Gate: fail if the relevant table is empty for the given experiment.

Mirrors the validate_* tasks in production's offline-evaluation-pipeline.yml,
which run pytest integration tests. Here we do a simple row-count check.
"""
from __future__ import annotations
import argparse, logging, os, sys
sys.dont_write_bytecode = True
try:
    _p = os.path.abspath(__file__)
except NameError:
    _p = filename  # type: ignore  # noqa: F821
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_p)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from palm.common.tables import build_table_vars, validate_experiment_id

TABLE_MAP = {
    "validate_population":   "eval_experiment_population",
    "validate_watch_hours":  "eval_watch_hours",
    "validate_eval_features":"eval_user_features",
    "validate_cate":         "eval_cate_coefficients",
    "validate_scoring":      "eval_scoring",
}
_SCRIPT = os.path.splitext(os.path.basename(_p))[0]

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",             required=True)
    parser.add_argument("--experiment-uuid", required=True)
    args = parser.parse_args()
    validate_experiment_id(args.experiment_uuid)

    spark  = SparkSession.builder.appName(f"PALM_{_SCRIPT}").getOrCreate()
    tables = build_table_vars(args.env)
    tbl_key = TABLE_MAP[_SCRIPT]
    table   = tables[tbl_key]

    if not spark.catalog.tableExists(table):
        raise RuntimeError(f"[GATE FAIL] {table} does not exist for {args.experiment_uuid}")

    count = spark.table(table).filter(F.col("experiment_uuid") == args.experiment_uuid).count()
    if count == 0:
        raise RuntimeError(f"[GATE FAIL] {table} is EMPTY for {args.experiment_uuid} — pipeline aborted")
    logging.info("[*] Validated %s: %d rows for %s", tbl_key, count, args.experiment_uuid)

if __name__ == "__main__":
    main()
