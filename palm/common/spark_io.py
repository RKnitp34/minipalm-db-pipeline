"""Delta table write helpers.

Simplified version of production's src/common/spark_io.py.
Differences from production:
  - No Liquibase check (always auto-create on first run)
  - Schema mismatch → merge (warn only, don't raise)
  - Works on Databricks Community Edition with Unity Catalog

Key function: write_or_create()
"""

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def write_or_create(
    df: DataFrame,
    spark: SparkSession,
    table: str,
    partition_by: list[str] | None = None,
    replace_where: str | None = None,
) -> None:
    """Write df to a Delta table with partition-safe overwrite.

    Behaviour:
      - Table does not exist → create via saveAsTable (with partitioning).
      - Table exists, schema matches → overwrite exactly the partition specified
        by replace_where using DataFrameWriterV2 (no other partitions touched).
      - Table exists, schema drifted → merge schema and overwrite (warns instead
        of raising, since this is a dev/learning environment).

    Args:
        df:           DataFrame to write.
        spark:        Active SparkSession.
        table:        Backtick-quoted FQN e.g. "`main`.`palm_learning_dev`.`eval_watch_hours`".
        partition_by: Partition columns (used only on first create).
        replace_where: SQL predicate scoping the overwrite, e.g.
                       "experiment_uuid = 'exp-001-disney-midroll' AND k_value = 7"
                       Must be set whenever partition_by is set — prevents
                       accidentally overwriting the whole table on re-runs.
    """
    partition_by = list(partition_by) if partition_by else []

    if not spark.catalog.tableExists(table):
        logging.info("[*] Table %s does not exist — creating", table)
        writer = df.write.format("delta").mode("overwrite")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.saveAsTable(table)
        logging.info("[*] Created %s (%d rows)", table, df.count())
        return

    # Validate schema
    target_schema = spark.table(table).schema
    target_cols   = {f.name for f in target_schema.fields}
    df_cols       = set(df.columns)
    extra         = sorted(df_cols - target_cols)
    missing       = sorted(target_cols - df_cols)

    if extra or missing:
        logging.warning(
            "[*] Schema drift on %s — allowing merge (extra: %s, missing: %s)",
            table, extra, missing,
        )
        writer = df.write.format("delta").mode("overwrite").option("mergeSchema", "true")
        if replace_where:
            writer = writer.option("replaceWhere", replace_where)
        writer.saveAsTable(table)
        return

    # Align column order to target, then overwrite only the specified partition
    aligned   = df.select(*[f.name for f in target_schema.fields])
    condition = F.expr(replace_where) if replace_where else F.lit(True)
    aligned.writeTo(table).overwrite(condition)
    logging.info("[*] Wrote to %s (replace_where=%r)", table, replace_where)
