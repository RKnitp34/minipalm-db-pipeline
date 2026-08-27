# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — P3: Extract User Features
# MAGIC
# MAGIC **What this does:**
# MAGIC - Reads `eval_experiment_population` (output of P1)
# MAGIC - Generates pre-experiment features per user:
# MAGIC   - `watch_hours_7d` — 7-day watch hours before experiment start
# MAGIC   - `watch_hours_30d` — 30-day watch hours (this drives P4 CATE)
# MAGIC   - `content_type` — series / film / mixed
# MAGIC   - `dataset_date` — experiment_start_date minus 1 day
# MAGIC - Writes to `minipalm.palm_learning_dev.eval_user_features`
# MAGIC
# MAGIC **Prerequisite:** Run notebook 01 first (P2 not needed for P3).

# COMMAND ----------

import sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:4])
print(f"Repo root: {repo_root}")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import sys as _sys, importlib
import palm.pipelines.run_extract_eval_features as p3
importlib.reload(p3)

for exp_uuid in ["exp-001-disney-midroll", "exp-002-hulu-preroll"]:
    print(f"\n{'='*60}\nRunning P3 for: {exp_uuid}\n{'='*60}")
    _sys.argv = [
        "run_extract_eval_features.py",
        "--env", "dev",
        "--experiment-uuid", exp_uuid,
        "--force",
    ]
    p3.main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview output

# COMMAND ----------

df_feat = spark.table("minipalm.palm_learning_dev.eval_user_features")
print(f"Total rows: {df_feat.count()}")
display(df_feat.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity check — avg watch_hours_30d by user range
# MAGIC (sensitive users ~15h, neutral ~33h, resilient ~50h)

# COMMAND ----------

from pyspark.sql.functions import avg, round as spark_round, expr

display(
    df_feat
    .withColumn("user_idx", expr("cast(replace(account_id, 'user_', '') as int) % 100"))
    .withColumn("category",
        expr("CASE WHEN user_idx < 30 THEN 'sensitive' WHEN user_idx < 70 THEN 'neutral' ELSE 'resilient' END"))
    .groupBy("experiment_uuid", "category")
    .agg(
        spark_round(avg("watch_hours_30d"), 1).alias("avg_wh30d"),
        spark_round(avg("watch_hours_7d"), 1).alias("avg_wh7d"),
    )
    .orderBy("experiment_uuid", "category")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Expected: sensitive ~15h, neutral ~33h, resilient ~50h for watch_hours_30d
# MAGIC
# MAGIC **Next step:** Run notebook `04_run_p4_cate_inference.py`
