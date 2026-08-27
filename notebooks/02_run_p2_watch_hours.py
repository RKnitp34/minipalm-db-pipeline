# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — P2: Compute Watch Hours
# MAGIC
# MAGIC **What this does:**
# MAGIC - Reads `eval_experiment_population` (output of P1)
# MAGIC - Generates synthetic 7-day watch hours per user based on their arm
# MAGIC   - sensitive + control → ~6h, sensitive + treatment → ~3.5h
# MAGIC   - neutral + control → ~8h, neutral + treatment → ~7.2h
# MAGIC   - resilient + control → ~10h, resilient + treatment → ~9.5h
# MAGIC - Writes to `minipalm.palm_learning_dev.eval_watch_hours`
# MAGIC
# MAGIC **Prerequisite:** Run notebook 01 first.

# COMMAND ----------

import sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:4])
print(f"Repo root: {repo_root}")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import sys as _sys, importlib
import palm.pipelines.run_compute_watch_hours as p2
importlib.reload(p2)

for exp_uuid in ["exp-001-disney-midroll", "exp-002-hulu-preroll"]:
    print(f"\n{'='*60}\nRunning P2 for: {exp_uuid}\n{'='*60}")
    _sys.argv = [
        "run_compute_watch_hours.py",
        "--env", "dev",
        "--experiment-uuid", exp_uuid,
        "--k-values", "7",
        "--force",
    ]
    p2.main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview output

# COMMAND ----------

df_wh = spark.table("minipalm.palm_learning_dev.eval_watch_hours")
print(f"Total rows: {df_wh.count()}")
display(df_wh.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity check — avg watch hours by arm (should match the spec)

# COMMAND ----------

from pyspark.sql.functions import avg, count, round as spark_round

display(
    df_wh
    .groupBy("experiment_uuid", "treatment_arm")
    .agg(
        spark_round(avg("watch_hours"), 2).alias("avg_watch_hours"),
        count("*").alias("user_count")
    )
    .orderBy("experiment_uuid", "treatment_arm")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Expected avg watch hours:
# MAGIC - control: ~6-10h depending on user category
# MAGIC - treatment_1 / treatment_2: slightly lower than control
# MAGIC
# MAGIC **Next step:** Run notebook `03_run_p3_user_features.py`
