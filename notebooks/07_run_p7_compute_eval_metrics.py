# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — P7: Compute Evaluation Metrics
# MAGIC
# MAGIC **What this does:**
# MAGIC - Reads `offline_evaluation_results` (P6)
# MAGIC - Computes 7 metrics per experiment:
# MAGIC   - `separation_magnitude` — resilient mean WH minus sensitive mean WH
# MAGIC   - `relative_separation_magnitude` — separation / sensitive baseline
# MAGIC   - `cohort_delta_ordering` — does sensitive < neutral < resilient?
# MAGIC   - `cohort_relative_delta_ordering` — same on relative uplift
# MAGIC   - `calibration_absolute` — MACE (mean absolute calibration error)
# MAGIC   - `calibration_relative` — MACE / baseline_hps
# MAGIC   - `cohort_traits` — are cohort % shares between 20% and 50%?
# MAGIC - Writes to `minipalm.palm_learning_dev.offline_evaluation_metrics`
# MAGIC
# MAGIC **Prerequisite:** Run notebook 06 first.

# COMMAND ----------

import sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:4])
print(f"Repo root: {repo_root}")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import sys as _sys, importlib
import palm.pipelines.run_compute_eval_metrics as p7
importlib.reload(p7)

for exp_uuid in ["exp-001-disney-midroll", "exp-002-hulu-preroll"]:
    print(f"\n{'='*60}\nRunning P7 for: {exp_uuid}\n{'='*60}")
    _sys.argv = [
        "run_compute_eval_metrics.py",
        "--env", "dev",
        "--experiment-uuid", exp_uuid,
        "--k-values", "7",
        "--force",
    ]
    p7.main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview — all metrics

# COMMAND ----------

df_metrics = spark.table("minipalm.palm_learning_dev.offline_evaluation_metrics")
print(f"Total rows: {df_metrics.count()}")
display(df_metrics.select(
    "experiment_uuid", "metric_name", "metric_value", "passed", "n_users"
).orderBy("experiment_uuid", "metric_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity check — pass/fail summary

# COMMAND ----------

from pyspark.sql.functions import count, when, col

display(
    df_metrics
    .groupBy("experiment_uuid")
    .agg(
        count(when(col("passed") == True, 1)).alias("passed"),
        count(when(col("passed") == False, 1)).alias("failed"),
        count("*").alias("total_metrics")
    )
    .orderBy("experiment_uuid")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key metric — MACE (calibration_absolute)
# MAGIC Red line threshold = 0.40. Below = good model calibration.

# COMMAND ----------

display(
    df_metrics
    .filter("metric_name = 'calibration_absolute'")
    .select("experiment_uuid", "metric_value", "passed")
    .orderBy("experiment_uuid")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ **Pipeline complete!** All 7 tables have been written end to end:
# MAGIC
# MAGIC | Table | Written by |
# MAGIC |---|---|
# MAGIC | experiment_user_assignments | data_generator |
# MAGIC | experiment_config | data_generator |
# MAGIC | eval_experiment_population | P1 |
# MAGIC | eval_watch_hours | P2 |
# MAGIC | eval_user_features | P3 |
# MAGIC | eval_cate_coefficients | P4 |
# MAGIC | eval_scoring | P5 |
# MAGIC | offline_evaluation_results | P6 |
# MAGIC | offline_evaluation_metrics | P7 ← you are here |
# MAGIC
# MAGIC **Next phase:** Build the monitoring dashboard reading from these tables.
