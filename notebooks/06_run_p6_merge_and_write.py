# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 06 — P6: Merge and Write
# MAGIC
# MAGIC **What this does:**
# MAGIC - Reads `eval_watch_hours` (P2) + `eval_scoring` (P5)
# MAGIC - Joins on `account_id + experiment_uuid`
# MAGIC - Computes `baseline_hps` = mean watch hours of control arm per cohort
# MAGIC - Writes to `minipalm.palm_learning_dev.offline_evaluation_results`
# MAGIC
# MAGIC **Prerequisite:** Run notebooks 02 and 05 first.

# COMMAND ----------

import sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:4])
print(f"Repo root: {repo_root}")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import sys as _sys, importlib
import palm.pipelines.run_merge_and_write as p6
importlib.reload(p6)

for exp_uuid in ["exp-001-disney-midroll", "exp-002-hulu-preroll"]:
    print(f"\n{'='*60}\nRunning P6 for: {exp_uuid}\n{'='*60}")
    _sys.argv = [
        "run_merge_and_write.py",
        "--env", "dev",
        "--experiment-uuid", exp_uuid,
        "--k-values", "7",
        "--force",
    ]
    p6.main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview output

# COMMAND ----------

df_results = spark.table("minipalm.palm_learning_dev.offline_evaluation_results")
print(f"Total rows: {df_results.count()}")
display(df_results.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity check — baseline_hps per cohort (control arm mean watch hours)

# COMMAND ----------

from pyspark.sql.functions import avg, round as spark_round, count

display(
    df_results
    .filter("treatment_arm = 'control'")
    .groupBy("experiment_uuid", "cohort")
    .agg(
        spark_round(avg("baseline_hps"), 2).alias("baseline_hps"),
        count("*").alias("user_count")
    )
    .orderBy("experiment_uuid", "cohort")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Each cohort should have a different baseline_hps:
# MAGIC - sensitive → ~6h
# MAGIC - neutral → ~8h
# MAGIC - resilient → ~10h
# MAGIC
# MAGIC **Next step:** Run notebook `07_run_p7_compute_eval_metrics.py`