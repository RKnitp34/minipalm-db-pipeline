# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Setup Pipeline Run Log Table
# MAGIC
# MAGIC **Run this notebook once** before running any pipeline.
# MAGIC It creates `workspace.palm_learning_dev.pipeline_run_log` — an append-only
# MAGIC Delta table that records every task execution (success / failed / skipped).
# MAGIC
# MAGIC After each pipeline run, this table gets one row per task:
# MAGIC
# MAGIC | task_key | status | duration_seconds | rows_written |
# MAGIC |---|---|---|---|
# MAGIC | p1_load_population | success | 12 | 1000 |
# MAGIC | p2_watch_hours | success | 8 | 1000 |
# MAGIC | p3_extract_features | skipped | 1 | null |
# MAGIC | p4_cate_inference | failed | 3 | null |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Create the table

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS workspace.palm_learning_dev.pipeline_run_log (
# MAGIC   run_id           STRING    COMMENT 'UUID shared by all tasks in one job trigger',
# MAGIC   job_run_id       STRING    COMMENT 'Databricks job run ID (null for notebook runs)',
# MAGIC   task_key         STRING    COMMENT 'e.g. p1_load_population',
# MAGIC   dataset_date     DATE      COMMENT 'Partition date being processed',
# MAGIC   experiment_uuid  STRING    COMMENT 'Which experiment was processed',
# MAGIC   status           STRING    COMMENT 'success | failed | skipped',
# MAGIC   started_at       TIMESTAMP COMMENT 'Task start time (UTC)',
# MAGIC   completed_at     TIMESTAMP COMMENT 'Task end time (UTC)',
# MAGIC   duration_seconds INT       COMMENT 'Wall-clock seconds',
# MAGIC   rows_written     INT       COMMENT 'Rows written to output table (null if skipped/failed)',
# MAGIC   error_message    STRING    COMMENT 'Truncated exception message (null if success)',
# MAGIC   env              STRING    COMMENT 'dev | prod'
# MAGIC )
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (dataset_date, env)
# MAGIC COMMENT 'Append-only pipeline run log — one row per task per run'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Confirm table was created

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE workspace.palm_learning_dev.pipeline_run_log

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Preview (empty on first run)

# COMMAND ----------

df = spark.table("workspace.palm_learning_dev.pipeline_run_log")
print(f"Rows in pipeline_run_log: {df.count()}")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Table is ready. Pull latest code and run the pipeline — each task will write one row here automatically.
# MAGIC
# MAGIC **Next step:** Run the full DAB pipeline job or any individual notebook (P1–P7).
# MAGIC After the run, come back here and re-run Step 3 to see the log entries.
