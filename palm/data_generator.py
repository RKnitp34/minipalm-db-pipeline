"""data_generator.py — Create PALM source tables with synthetic experiment data.

Creates ONLY the two upstream source tables that the pipeline reads:
  • experiment_user_assignments   (who is in which arm)
  • experiment_config             (experiment dates)

Every other table (eval_experiment_population, eval_watch_hours, …) is produced
by the pipeline scripts P1–P7.  This mirrors production exactly — the data
generator never pre-computes pipeline outputs.

This script runs as the FIRST task in the DAB job.  It regenerates source data
on every run so the downstream pipeline always has fresh input to process.

Usage (local, no Spark):
    python3 -m palm.data_generator --local

Usage (Databricks job task):
    python palm/data_generator.py --env dev --experiment-uuid exp-001-disney-midroll

Schema will be created automatically on first run:
    CREATE SCHEMA IF NOT EXISTS main.palm_learning_dev;
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

# ── Make palm importable whether run locally or as a Databricks task ─────────
try:
    _script_path = os.path.abspath(__file__)
except NameError:
    _script_path = filename  # type: ignore[name-defined]  # noqa: F821
_ROOT = os.path.dirname(os.path.dirname(_script_path))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from palm.common.tables import build_table_vars, validate_experiment_id  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Experiment definitions (same two experiments used throughout) ─────────────
EXPERIMENTS = {
    "exp-001-disney-midroll": {
        "experiment_name":       "disney_midroll_increase_q1",
        "experiment_start_date": date(2026, 1, 1),
        "experiment_end_date":   date(2026, 2, 28),
    },
    "exp-002-hulu-preroll": {
        "experiment_name":       "hulu_preroll_test_q1",
        "experiment_start_date": date(2026, 1, 15),
        "experiment_end_date":   date(2026, 3, 15),
    },
}

N_USERS_PER_EXP = 500
RNG = np.random.default_rng(42)

ARM_FRACS    = [("control", 0.30), ("treatment_1", 0.40), ("treatment_2", 0.30)]
TREATMENT_UUIDS = {
    "control":     "aaaaaaaa-0000-0000-0000-000000000000",
    "treatment_1": "aaaaaaaa-0000-0000-0000-000000000001",
    "treatment_2": "aaaaaaaa-0000-0000-0000-000000000002",
}


# ── Table generators ──────────────────────────────────────────────────────────

def _gen_user_assignments(experiment_uuid: str) -> pd.DataFrame:
    """500 users, randomly assigned to arms (30/40/30 split)."""
    cfg = EXPERIMENTS[experiment_uuid]
    exp_idx = list(EXPERIMENTS).index(experiment_uuid)
    start_uid = exp_idx * N_USERS_PER_EXP + 1

    arm_labels: list[str] = []
    for arm, frac in ARM_FRACS:
        arm_labels.extend([arm] * round(N_USERS_PER_EXP * frac))
    arm_labels = (arm_labels + ["treatment_1"] * N_USERS_PER_EXP)[:N_USERS_PER_EXP]
    arms = np.array(arm_labels)[RNG.permutation(N_USERS_PER_EXP)]

    span = (cfg["experiment_end_date"] - cfg["experiment_start_date"]).days
    assign_deltas = RNG.integers(0, min(span, 7), N_USERS_PER_EXP)

    rows = []
    for i in range(N_USERS_PER_EXP):
        arm = arms[i]
        rows.append({
            "account_id":      f"user_{start_uid + i:05d}",
            "experiment_uuid": experiment_uuid,
            "treatment_uuid":  TREATMENT_UUIDS[arm],
            "treatment_arm":   arm,
            "assignment_date": cfg["experiment_start_date"]
                               + __import__("datetime").timedelta(days=int(assign_deltas[i])),
        })
    return pd.DataFrame(rows)


def _gen_experiment_config() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "experiment_uuid":       k,
            "experiment_name":       v["experiment_name"],
            "experiment_start_date": v["experiment_start_date"],
            "experiment_end_date":   v["experiment_end_date"],
        }
        for k, v in EXPERIMENTS.items()
    ])


# ── Write helpers ─────────────────────────────────────────────────────────────

def _write_delta(pdf: pd.DataFrame, table_fqn: str) -> None:
    """Convert pandas → Spark and write as Delta (overwrite)."""
    sdf = spark.createDataFrame(pdf)  # noqa: F821  # spark is in scope on Databricks
    (
        sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_fqn)
    )
    log.info("[*] Written %d rows → %s", len(pdf), table_fqn)


def _write_local(pdf: pd.DataFrame, name: str) -> None:
    import tempfile, os  # noqa: E401
    out = os.path.join(tempfile.gettempdir(), f"palm_{name}.parquet")
    pdf.to_parquet(out, index=False)
    log.info("[*] Local preview saved → %s", out)
    print(f"\n── {name} ({len(pdf)} rows) ──")
    print(pdf.head(5).to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PALM source tables")
    parser.add_argument("--env",             default="dev", help="Deployment environment (default: dev)")
    parser.add_argument("--experiment-uuid", default=None,  help="Only generate data for this experiment (default: all)")
    parser.add_argument("--local",           action="store_true", help="Print preview + save parquet; no Spark write")
    args = parser.parse_args()

    if args.experiment_uuid:
        validate_experiment_id(args.experiment_uuid)
        target_exps = [args.experiment_uuid]
    else:
        target_exps = list(EXPERIMENTS)

    # ── Build DataFrames ──────────────────────────────────────────────────────
    assignments_df = pd.concat([_gen_user_assignments(exp) for exp in target_exps])
    config_df      = _gen_experiment_config()

    if args.local:
        _write_local(assignments_df, "experiment_user_assignments")
        _write_local(config_df,      "experiment_config")
        return

    # ── Spark write (Databricks) ──────────────────────────────────────────────
    tables = build_table_vars(args.env)

    # Ensure schema exists (CE: user must run `CREATE SCHEMA IF NOT EXISTS main.palm_learning_dev`)
    log.info("[*] Writing source tables to env=%s", args.env)
    _write_delta(assignments_df, tables["experiment_user_assignments"])
    _write_delta(config_df,      tables["experiment_config"])
    log.info("[*] Source tables ready.")


if __name__ == "__main__":
    main()
