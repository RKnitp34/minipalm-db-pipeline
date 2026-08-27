"""Databricks / Unity Catalog helpers.

Mirrors production's src/common/databricks.py.
"""

import re

_CATALOG_SCHEMA_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_TABLE_RE          = re.compile(r"^[A-Za-z0-9_]+$")


def validate_three_part_table_name(name: str) -> tuple[str, str, str]:
    """Parse and validate a fully-qualified table name: catalog.schema.table.

    Accepts both plain and backtick-quoted forms.

    Returns:
        (catalog, schema, table) tuple.

    Raises:
        ValueError: If not exactly 3 parts or invalid characters.
    """
    parts = [p.strip("`") for p in name.split(".")]
    if len(parts) != 3:
        raise ValueError(f"Table name must be catalog.schema.table, got: {name!r}")
    catalog, schema, table = parts
    for part, pattern, label in (
        (catalog, _CATALOG_SCHEMA_RE, "catalog"),
        (schema,  _CATALOG_SCHEMA_RE, "schema"),
        (table,   _TABLE_RE,          "table"),
    ):
        if not part or not pattern.match(part):
            raise ValueError(f"Invalid {label} component {part!r} in {name!r}")
    return catalog, schema, table
