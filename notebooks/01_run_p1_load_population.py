# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 01 — P1: Load Experiment Population
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC - Reads `experiment_user_assignments` + `experiment_config`
# MAGIC - Joins on `experiment_uuid`, deduplicates users, drops nulls
# MAGIC - Writes to `minipalm.palm_learning_dev.eval_experiment_population`
# MAGIC
# MAGIC **Prerequisite:** Run `00_setup_source_tables` first.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Set up Python path

# COMMAND ----------

import sys

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:4])
print(f"Repo root: {repo_root}")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Run P1 for both experiments

# COMMAND ----------

import sys as _sys
import importlib

import palm.pipelines.run_load_population as p1
importlib.reload(p1)

for exp_uuid in ["exp-001-disney-midroll", "exp-002-hulu-preroll"]:
    print(f"\n{'='*60}")
    print(f"Running P1 for: {exp_uuid}")
    print(f"{'='*60}")
    _sys.argv = [
        "run_load_population.py",
        "--env", "dev",
        "--experiment-uuid", exp_uuid,
        "--force",
    ]
    p1.main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Preview output table

# COMMAND ----------

df_pop = spark.table("minipalm.palm_learning_dev.eval_experiment_population")
print(f"Total rows: {df_pop.count()}")
display(df_pop.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Sanity check: row counts per experiment + arm

# COMMAND ----------

from pyspark.sql.functions import count

display(
    df_pop
    .groupBy("experiment_uuid", "treatment_arm")
    .agg(count("*").alias("user_count"))
    .orderBy("experiment_uuid", "treatment_arm")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ You should see ~500 users per experiment split across 3 arms (control / treatment_1 / treatment_2).
# MAGIC
# MAGIC **Next step:** Open `notebooks/02_run_p2_watch_hours.py`

# COMMAND ----------

