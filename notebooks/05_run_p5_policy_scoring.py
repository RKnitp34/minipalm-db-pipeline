# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 05 — P5: Policy Scoring
# MAGIC
# MAGIC **What this does:**
# MAGIC - Reads `eval_cate_coefficients` (P4) + `eval_user_features` (P3)
# MAGIC - Computes `predicted_effect = cate_linear_coef × treatment_intensity`
# MAGIC - Assigns cohort using business thresholds:
# MAGIC   - `predicted_effect < -0.5` → **sensitive**
# MAGIC   - `-0.5 ≤ predicted_effect ≤ 0.0` → **neutral**
# MAGIC   - `predicted_effect > 0.0` → **resilient**
# MAGIC - Users with `watch_hours_7d < 1.0` → `scoring_status = unscored_low_watch` → cohort forced to neutral
# MAGIC - Writes to `minipalm.palm_learning_dev.eval_scoring`
# MAGIC
# MAGIC **Prerequisite:** Run notebooks 03 and 04 first.

# COMMAND ----------

import sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:4])
print(f"Repo root: {repo_root}")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import sys as _sys, importlib
import palm.pipelines.run_eval_policy_scoring as p5
importlib.reload(p5)

for exp_uuid in ["exp-001-disney-midroll", "exp-002-hulu-preroll"]:
    print(f"\n{'='*60}\nRunning P5 for: {exp_uuid}\n{'='*60}")
    _sys.argv = [
        "run_eval_policy_scoring.py",
        "--env", "dev",
        "--experiment-uuid", exp_uuid,
        "--model-run-id",    "palm_model_v1",
        "--t1-scenario",     "phase1_resilient",
        "--t0-scenario",     "base_ad_load",
        "--policy-scenario", "policy_config_v1",
        "--force",
    ]
    p5.main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview output

# COMMAND ----------

df_scoring = spark.table("minipalm.palm_learning_dev.eval_scoring")
print(f"Total rows: {df_scoring.count()}")
display(df_scoring.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity check — cohort distribution (should be ~30% sensitive / 40% neutral / 30% resilient)

# COMMAND ----------

from pyspark.sql.functions import count, round as spark_round

total = df_scoring.count()
display(
    df_scoring
    .groupBy("experiment_uuid", "cohort")
    .agg(
        count("*").alias("user_count"),
        spark_round(count("*") / total * 100, 1).alias("pct")
    )
    .orderBy("experiment_uuid", "cohort")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring status breakdown

# COMMAND ----------

display(
    df_scoring
    .groupBy("experiment_uuid", "scoring_status")
    .agg(count("*").alias("user_count"))
    .orderBy("experiment_uuid", "scoring_status")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Expected: ~30% sensitive, ~40% neutral, ~30% resilient. Most users should be "scored".
# MAGIC
# MAGIC **Next step:** Run notebook `06_run_p6_merge_and_write.py`