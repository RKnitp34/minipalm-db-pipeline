# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Inspect Pipeline Outputs
# MAGIC
# MAGIC Reads the latest data from every output table and shows row counts, schemas, and sample rows.
# MAGIC Run this after any pipeline run to verify the end-to-end results.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — catalog / schema

# COMMAND ----------

CATALOG = "minipalm"
SCHEMA  = "palm_learning_dev"

def tbl(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

print(f"Reading from: {CATALOG}.{SCHEMA}.*")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SOURCE TABLES

# COMMAND ----------

# MAGIC %md
# MAGIC ### experiment_user_assignments

# COMMAND ----------

df = spark.table(tbl("experiment_user_assignments"))
print(f"Rows: {df.count()}")
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### experiment_config

# COMMAND ----------

df = spark.table(tbl("experiment_config"))
print(f"Rows: {df.count()}")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## P1 OUTPUT — eval_experiment_population

# COMMAND ----------

df_pop = spark.table(tbl("eval_experiment_population"))
print(f"Rows: {df_pop.count()}")
display(df_pop.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### P1: rows per experiment + arm

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
# MAGIC ## P2 OUTPUT — eval_watch_hours

# COMMAND ----------

df_wh = spark.table(tbl("eval_watch_hours"))
print(f"Rows: {df_wh.count()}")
display(df_wh.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### P2: avg watch hours per experiment + arm + k_value

# COMMAND ----------

from pyspark.sql.functions import avg, round as spark_round

display(
    df_wh
    .groupBy("experiment_uuid", "treatment_arm", "k_value")
    .agg(spark_round(avg("watch_hours"), 2).alias("avg_watch_hours"))
    .orderBy("experiment_uuid", "treatment_arm")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## P3 OUTPUT — eval_user_features

# COMMAND ----------

df_feat = spark.table(tbl("eval_user_features"))
print(f"Rows: {df_feat.count()}")
display(df_feat.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### P3: avg features per content_type

# COMMAND ----------

display(
    df_feat
    .groupBy("experiment_uuid", "content_type")
    .agg(
        count("*").alias("users"),
        spark_round(avg("watch_hours_7d"), 2).alias("avg_wh_7d"),
        spark_round(avg("watch_hours_30d"), 2).alias("avg_wh_30d"),
    )
    .orderBy("experiment_uuid", "content_type")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## P4 OUTPUT — eval_cate_coefficients

# COMMAND ----------

df_cate = spark.table(tbl("eval_cate_coefficients"))
print(f"Rows: {df_cate.count()}")
display(df_cate.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### P4: avg CATE coef per experiment

# COMMAND ----------

display(
    df_cate
    .groupBy("experiment_uuid")
    .agg(
        spark_round(avg("cate_linear_coef"), 4).alias("avg_coef"),
        spark_round(avg("cate_linear_coef_lb"), 4).alias("avg_lb"),
        spark_round(avg("cate_linear_coef_ub"), 4).alias("avg_ub"),
    )
    .orderBy("experiment_uuid")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## P5 OUTPUT — eval_scoring

# COMMAND ----------

df_score = spark.table(tbl("eval_scoring"))
print(f"Rows: {df_score.count()}")
display(df_score.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### P5: cohort distribution per experiment

# COMMAND ----------

display(
    df_score
    .groupBy("experiment_uuid", "cohort", "scoring_status")
    .agg(count("*").alias("user_count"))
    .orderBy("experiment_uuid", "cohort")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### P5: avg predicted_effect per cohort  (should match spec: ~-0.3 / -1.0 / -2.7)

# COMMAND ----------

display(
    df_score
    .groupBy("experiment_uuid", "cohort")
    .agg(spark_round(avg("predicted_effect"), 3).alias("avg_predicted_effect"))
    .orderBy("experiment_uuid", "cohort")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## P6 OUTPUT — offline_evaluation_results

# COMMAND ----------

df_res = spark.table(tbl("offline_evaluation_results"))
print(f"Rows: {df_res.count()}")
display(df_res.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### P6: avg actual watch hours vs predicted_effect per cohort + arm

# COMMAND ----------

display(
    df_res
    .groupBy("experiment_uuid", "cohort", "treatment_arm", "k_value")
    .agg(
        count("*").alias("users"),
        spark_round(avg("watch_hours"), 2).alias("avg_actual_wh"),
        spark_round(avg("predicted_effect"), 3).alias("avg_pred_effect"),
        spark_round(avg("baseline_hps"), 2).alias("avg_baseline_hps"),
    )
    .orderBy("experiment_uuid", "cohort", "treatment_arm")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## P7 OUTPUT — offline_evaluation_metrics  ← FINAL TABLE

# COMMAND ----------

df_metrics = spark.table(tbl("offline_evaluation_metrics"))
print(f"Rows: {df_metrics.count()}")
display(df_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### P7: pass/fail summary

# COMMAND ----------

from pyspark.sql.functions import col

display(
    df_metrics
    .select("experiment_uuid", "metric_name", "metric_value", "passed", "n_users", "computed_at")
    .orderBy("experiment_uuid", "metric_name")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### P7: quick pass rate check

# COMMAND ----------

from pyspark.sql.functions import sum as spark_sum, lit

total   = df_metrics.count()
passing = df_metrics.filter(col("passed") == True).count()
print(f"Metrics passed: {passing}/{total}")
print(f"Pass rate: {passing/total*100:.1f}%")

display(
    df_metrics
    .groupBy("experiment_uuid")
    .agg(
        spark_sum(col("passed").cast("int")).alias("passed"),
        count("*").alias("total_metrics"),
    )
    .orderBy("experiment_uuid")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema check — all output tables

# COMMAND ----------

tables_to_check = [
    "eval_experiment_population",
    "eval_watch_hours",
    "eval_user_features",
    "eval_cate_coefficients",
    "eval_scoring",
    "offline_evaluation_results",
    "offline_evaluation_metrics",
]

for t in tables_to_check:
    fqn = tbl(t)
    try:
        df = spark.table(fqn)
        print(f"\n{'='*60}")
        print(f"{fqn}  ({df.count()} rows)")
        print(f"Columns: {df.columns}")
    except Exception as e:
        print(f"\n[MISSING] {fqn}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ If all tables have rows and all 7 metrics show `passed = True`, the end-to-end pipeline is working correctly.
