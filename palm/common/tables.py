"""Utilities for resolving table FQNs from fixtures/tables.yml.

Mirrors production's src/common/tables.py.

Usage:
    from palm.common.tables import build_table_vars

    tables = build_table_vars("dev")
    df = spark.table(tables["eval_experiment_population"])
    spark.sql("SELECT * FROM {eval_experiment_population}".format(**tables))
"""

import re
from pathlib import Path

import yaml

# Resolve fixtures/ relative to this file: palm/common/tables.py → ../../fixtures/
_FIXTURES_DIR = Path(__file__).parents[2] / "fixtures"

# For Community Edition there is only "dev".
# Add "qa" / "prod" when you move to a paid workspace.
VALID_ENVS: frozenset[str] = frozenset({"dev"})

# Input-validation regexes (prevent SQL / path injection)
_SAFE_EXP_ID_RE   = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_SAFE_SCENARIO_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def validate_experiment_id(experiment_id: str) -> None:
    """Accept UUID v4 or human-readable IDs like 'exp-001-disney-midroll'.

    Raises:
        ValueError: If the ID contains unsafe characters.
    """
    if not _SAFE_EXP_ID_RE.match(experiment_id):
        raise ValueError(
            f"Invalid experiment_id: {experiment_id!r}. "
            "Use alphanumeric characters, hyphens, or underscores."
        )


def validate_scenario_name(name: str, *, label: str = "scenario") -> None:
    """Reject scenario names that could be injected into SQL."""
    if not _SAFE_SCENARIO_RE.match(name):
        raise ValueError(f"Invalid {label}: {name!r}. Use alphanumeric + hyphens/underscores.")


def build_table_vars(
    env: str,
    fixtures_dir: Path = _FIXTURES_DIR,
) -> dict[str, str]:
    """Load fixtures/tables.yml and return logical_name → backtick-quoted FQN.

    Both catalog and schema templates are resolved with {env}. For example:
        catalog: "main"                  → "main"           (unchanged)
        schema:  "palm_learning_{env}"   → "palm_learning_dev"  (when env="dev")

    Args:
        env: Target environment. Must be in VALID_ENVS.

    Returns:
        Dict like:
            {"eval_experiment_population": "`main`.`palm_learning_dev`.`eval_experiment_population`"}

    Raises:
        ValueError: If env is not in VALID_ENVS.
    """
    if env not in VALID_ENVS:
        raise ValueError(f"Unknown environment '{env}'. Valid: {sorted(VALID_ENVS)}")

    path = fixtures_dir / "tables.yml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    table_vars: dict[str, str] = {}
    for _group, tables in (raw or {}).items():
        if not isinstance(tables, dict):
            continue
        for logical_name, spec in tables.items():
            if not isinstance(spec, dict):
                continue
            catalog = spec["catalog"].format(env=env)
            schema  = spec["schema"].format(env=env)
            table   = spec["table"]
            table_vars[logical_name] = f"`{catalog}`.`{schema}`.`{table}`"

    return table_vars
