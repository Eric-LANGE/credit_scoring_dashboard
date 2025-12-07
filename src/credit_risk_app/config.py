# config.py
"""
Configuration module for Credit Risk Dashboard.

Combines:
- HuggingFace Hub repository settings (for external asset storage)
- Application constants (columns, thresholds, paths)
"""

import os
from pathlib import Path

# =============================================================================
# HUGGING FACE HUB CONFIGURATION
# =============================================================================

# Repository IDs (format: "username/repo-name")
# Override via environment variables in HF Space settings
HF_MODEL_REPO_ID = os.environ.get(
    "HF_MODEL_REPO_ID",
    "YOUR_USERNAME/credit-risk-dashboard-model"
)
HF_DATA_REPO_ID = os.environ.get(
    "HF_DATA_REPO_ID",
    "YOUR_USERNAME/credit-risk-dashboard-data"
)

# Subdirectory within model repo containing the MLflow model
HF_MODEL_SUBDIR = "gradient_boosting"

# Cache directory for HF Hub downloads
HF_CACHE_DIR = os.environ.get("HF_HOME", "/tmp/.huggingface")

# =============================================================================
# LOCAL PATHS (inside container, populated at runtime from HF Hub)
# =============================================================================

LOCAL_MODEL_DIR = Path("/app/models/gradient_boosting")
LOCAL_DATA_DIR = Path("/app/data")
LOCAL_SHAP_DIR = Path("/app/shap")
LOCAL_PLOTS_DIR = Path("/app/plots")

# =============================================================================
# FILE NAMES ON HF HUB
# =============================================================================

RAW_DATA_FILENAME = "application_test.csv"
SHAP_EXPLANATION_FILENAME = "shap_explanation.joblib"
SHAP_BEESWARM_FILENAME = "shap_beeswarm.png"

PLOT_FILENAMES = [
    "DAYS_EMPLOYED_hist_data.json",
    "EXT_SOURCE_2_hist_data.json",
    "EXT_SOURCE_3_hist_data.json",
    "OWN_CAR_AGE_hist_data.json",
]

# =============================================================================
# APPLICATION CONSTANTS (from original config.py)
# =============================================================================

# Columns to select when loading the test data
COLUMNS_TO_IMPORT = [
    "SK_ID_CURR",
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OCCUPATION_TYPE",
    "FLAG_DOCUMENT_3",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_8",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT_W_CITY",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "ORGANIZATION_TYPE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

# Default threshold (will be overridden by model metadata)
DEFAULT_PREDICTION_THRESHOLD = 0.5

# Allowed features for distribution endpoint
DISTRIBUTION_FEATURES = frozenset(
    {"EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_EMPLOYED", "OWN_CAR_AGE"}
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_path() -> Path:
    """Return path to the MLflow model directory."""
    return LOCAL_MODEL_DIR


def get_raw_data_path() -> Path:
    """Return path to the raw data CSV file."""
    return LOCAL_DATA_DIR / RAW_DATA_FILENAME


def get_shap_explanation_path() -> Path:
    """Return path to the SHAP explanation joblib file."""
    return LOCAL_SHAP_DIR / SHAP_EXPLANATION_FILENAME


def get_shap_beeswarm_path() -> Path:
    """Return path to the SHAP beeswarm PNG file."""
    return LOCAL_SHAP_DIR / SHAP_BEESWARM_FILENAME


def get_plot_path(filename: str) -> Path:
    """Return path to a specific plot JSON file."""
    return LOCAL_PLOTS_DIR / filename


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("HF HUB CONFIGURATION")
    print("=" * 60)
    print(f"HF_MODEL_REPO_ID: {HF_MODEL_REPO_ID}")
    print(f"HF_DATA_REPO_ID:  {HF_DATA_REPO_ID}")
    print(f"HF_CACHE_DIR:     {HF_CACHE_DIR}")
    print("-" * 60)
    print(f"LOCAL_MODEL_DIR:  {LOCAL_MODEL_DIR}")
    print(f"LOCAL_DATA_DIR:   {LOCAL_DATA_DIR}")
    print(f"LOCAL_SHAP_DIR:   {LOCAL_SHAP_DIR}")
    print(f"LOCAL_PLOTS_DIR:  {LOCAL_PLOTS_DIR}")
    print("=" * 60)
