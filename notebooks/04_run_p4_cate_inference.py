# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — P4: CATE Inference
# MAGIC
# MAGIC **What this does:**
# MAGIC - Reads `eval_user_features` (output of P3)
# MAGIC - Computes a CATE (Conditional Average Treatment Effect) coefficient per user:
# MAGIC   - Formula: `cate_linear_coef = (watch_hours_30d - 33) * 0.03 + noise`
# MAGIC   - This maps low-watch users → negative coef (sensitive) and high-watch → positive (resilient)
# MAGIC - Writes to `minipalm.palm_learning_dev.eval_cate_coefficients`
# MAGIC
# MAGIC **In production:** This runs a real EconML / Causal Forest model.
# MAGIC **Here:** We use the formula above to simulate model output deterministically.
# MAGIC
# MAGIC **Prerequisite:** Run notebook 03 first.

# COMMAND ----------

import sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:4])
print(f"Repo root: {repo_root}")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import sys as _sys, importlib
import palm.pipelines.run_eval_cate_inference as p4
importlib.reload(p4)

for exp_uuid in ["exp-001-disney-midroll", "exp-002-hulu-preroll"]:
    print(f"\n{'='*60}\nRunning P4 for: {exp_uuid}\n{'='*60}")
    _sys.argv = [
        "run_eval_cate_inference.py",
        "--env", "dev",
        "--experiment-uuid", exp_uuid,
        "--model-run-id", "palm_model_v1",
        "--force",
    ]
    p4.main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview output

# COMMAND ----------

df_cate = spark.table("minipalm.palm_learning_dev.eval_cate_coefficients")
print(f"Total rows: {df_cate.count()}")
display(df_cate.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity check — avg CATE coef by user category
# MAGIC (sensitive → coef < -0.5, neutral → coef ≈ 0, resilient → coef > 0)

# COMMAND ----------

from pyspark.sql.functions import avg, round as spark_round, expr

display(
    df_cate
    .withColumn("user_idx", expr("cast(replace(account_id, 'user_', '') as int) % 100"))
    .withColumn("category",
        expr("CASE WHEN user_idx < 30 THEN 'sensitive' WHEN user_idx < 70 THEN 'neutral' ELSE 'resilient' END"))
    .groupBy("experiment_uuid", "category")
    .agg(spark_round(avg("cate_linear_coef"), 3).alias("avg_cate_coef"))
    .orderBy("experiment_uuid", "category")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Expected CATE coefs: sensitive ≈ -0.54, neutral ≈ 0.0, resilient ≈ +0.51
# MAGIC
# MAGIC **Next step:** Run notebook `05_run_p5_policy_scoring.py`
