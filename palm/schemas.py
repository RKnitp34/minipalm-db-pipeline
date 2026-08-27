"""
schemas.py — Single source of truth for every table schema in PALM-LEARNING.

Column lists are plain tuples of strings so they can be used for:
  - Spark DataFrame .select(*SCHEMA_X)
  - assertion checks after writing (assert set(df.columns) == set(SCHEMA_X))
  - documentation

Catalog / schema path used everywhere in this project:
  main.palm_learning.<table_name>
"""

CATALOG = "main"
SCHEMA = "palm_learning"


def full_table(table_name: str) -> str:
    """Return fully-qualified table path: main.palm_learning.<table_name>"""
    return f"{CATALOG}.{SCHEMA}.{table_name}"


# ── Source tables ────────────────────────────────────────────────────────────

EXPERIMENT_USER_ASSIGNMENTS = (
    "account_id",
    "experiment_uuid",
    "treatment_uuid",
    "treatment_arm",
    "assignment_date",
)

EXPERIMENT_CONFIG = (
    "experiment_uuid",
    "experiment_name",
    "experiment_start_date",
    "experiment_end_date",
)

# ── Pipeline output tables ────────────────────────────────────────────────────

# P1 output
EVAL_EXPERIMENT_POPULATION = (
    "account_id",
    "experiment_uuid",
    "experiment_name",
    "treatment_arm",
    "treatment_uuid",
    "experiment_start_date",
    "experiment_end_date",
)

# P2 output  — partitioned by [experiment_uuid, k_value]
EVAL_WATCH_HOURS = (
    "account_id",
    "experiment_uuid",
    "experiment_name",
    "treatment_arm",
    "treatment_uuid",
    "k_value",
    "watch_hours",
)

# P3 output  — partitioned by [experiment_uuid]
EVAL_USER_FEATURES = (
    "account_id",
    "experiment_uuid",
    "dataset_date",
    "watch_hours_7d",
    "watch_hours_30d",
    "content_type",
)

# P4 output
EVAL_CATE_COEFFICIENTS = (
    "account_id",
    "experiment_uuid",
    "cate_linear_coef",
    "cate_linear_coef_lb",
    "cate_linear_coef_ub",
)

# P5 output  — partitioned by [experiment_uuid, model_run_id, t1_scenario, t0_scenario, policy_scenario]
EVAL_SCORING = (
    "account_id",
    "experiment_uuid",
    "cohort",
    "predicted_effect",
    "predicted_uplift",
    "scoring_status",
    "model_run_id",
    "t1_scenario",
    "t0_scenario",
    "policy_scenario",
)

# P6 output  — partitioned by [experiment_uuid, k_value, model_run_id, t1_scenario, t0_scenario, policy_scenario]
OFFLINE_EVALUATION_RESULTS = (
    "account_id",
    "experiment_uuid",
    "k_value",
    "treatment_arm",
    "treatment_uuid",
    "watch_hours",
    "cohort",
    "predicted_effect",
    "predicted_uplift",
    "scoring_status",
    "baseline_hps",
    "model_run_id",
    "t1_scenario",
    "t0_scenario",
    "policy_scenario",
)

# P7 output
OFFLINE_EVALUATION_METRICS = (
    "experiment_uuid",
    "k_value",
    "model_run_id",
    "t1_scenario",
    "t0_scenario",
    "policy_scenario",
    "metric_name",
    "passed",
    "pass_rate",
    "metric_value",
    "details_json",
    "computed_at",
    "n_users",
)

# Monitoring
PIPELINE_RUN_LOG = (
    "run_id",
    "pipeline_name",
    "experiment_uuid",
    "run_date",
    "status",
    "started_at",
    "completed_at",
    "duration_seconds",
    "rows_written",
    "error_message",
)

# ── Business constants ────────────────────────────────────────────────────────

COHORTS = ("sensitive", "neutral", "resilient")

TREATMENT_ARMS = ("control", "treatment_1", "treatment_2")

K_VALUES = [7]

# Cohort assignment thresholds based on predicted_effect (hours)
COHORT_THRESHOLDS = {
    "sensitive": lambda e: e < -0.5,
    "neutral":   lambda e: -0.5 <= e <= 0.0,
    "resilient": lambda e: e > 0.0,
}

METRIC_NAMES = (
    "separation_magnitude",
    "relative_separation_magnitude",
    "cohort_delta_ordering",
    "cohort_relative_delta_ordering",
    "calibration_absolute",
    "calibration_relative",
    "cohort_traits",
)

PIPELINE_NAMES = (
    "p1_load_population",
    "p2_watch_hours",
    "p3_user_features",
    "p4_cate_inference",
    "p5_policy_scoring",
    "p6_merge_and_write",
    "p7_compute_metrics",
)

# Scenario names used in scoring / evaluation tables
DEFAULT_T1_SCENARIO = "phase1_resilient"
DEFAULT_T0_SCENARIO = "base_ad_load"
DEFAULT_POLICY_SCENARIO = "policy_config_v1"

# MACE threshold above which a recalibration alert fires in the dashboard
MACE_RECALIBRATION_THRESHOLD = 0.40

# Unscored watch-hour floor
UNSCORED_WATCH_HOURS_FLOOR = 1.0

# Cohort-shift alert: flag if cohort % changes more than this vs previous run
COHORT_SHIFT_ALERT_PCT = 10.0
