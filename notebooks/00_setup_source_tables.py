# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 00 — Setup Source Tables
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Creates the Unity Catalog schema `workspace.palm_learning_dev` (one-time setup)
# MAGIC 2. Runs `data_generator.py` to populate two source tables:
# MAGIC    - `workspace.palm_learning_dev.experiment_user_assignments`
# MAGIC    - `workspace.palm_learning_dev.experiment_config`
# MAGIC 3. Previews both tables so we can confirm data looks correct before running the pipeline
# MAGIC
# MAGIC Run this notebook **once** before running any pipeline scripts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Create the schema (one-time, safe to re-run)

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW CATALOGS;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS minipalm.palm_learning_dev;
# MAGIC SHOW SCHEMAS IN minipalm;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Add repo root to Python path so `palm` is importable

# COMMAND ----------

import sys

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:4])
print(f"Repo root detected: {repo_root}")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Run data generator

# COMMAND ----------

import sys as _sys
import importlib

_sys.argv = ["data_generator.py", "--env", "dev"]

import palm.data_generator as dg
importlib.reload(dg)
dg.main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Preview source tables

# COMMAND ----------

print("=" * 60)
print("experiment_user_assignments — first 10 rows")
print("=" * 60)
df_assignments = spark.table("workspace.palm_learning_dev.experiment_user_assignments")
print(f"Total rows: {df_assignments.count()}")
display(df_assignments.limit(10))

# COMMAND ----------

print("=" * 60)
print("experiment_config — all rows")
print("=" * 60)
df_config = spark.table("workspace.palm_learning_dev.experiment_config")
display(df_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Quick sanity check

# COMMAND ----------

from pyspark.sql.functions import count

print("Arms distribution (~30% control / 40% treatment_1 / 30% treatment_2):")
display(
    df_assignments
    .groupBy("experiment_uuid", "treatment_arm")
    .agg(count("*").alias("user_count"))
    .orderBy("experiment_uuid", "treatment_arm")
)

# COMMAND ----------

print("rakesh")

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ If you see 2 experiments × 500 users each with the right arm distribution, **source tables are ready**.
# MAGIC
# MAGIC **Next step:** Open `notebooks/01_run_p1_load_population.py`