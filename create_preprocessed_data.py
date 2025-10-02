# create_preprocessed_data.py

import logging
import pandas as pd
import mlflow
from pathlib import Path
import numpy as np
from sklearn import set_config

set_config(transform_output="pandas")

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT_DIR / "models" / "gradient_boosting"
TEST_DATA_PATH = PROJECT_ROOT_DIR / "data" / "application_test.csv"
OUTPUT_DATA_PATH = PROJECT_ROOT_DIR / "data" / "dashboard_data.csv"
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
    "FLAG_DOCUMENT_3",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_8",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]
DASHBOARD_FEATURES = ["EXT_SOURCE_3", "EXT_SOURCE_2", "DAYS_EMPLOYED", "OWN_CAR_AGE"]


def apply_transformations_for_model(df, expected_features):
    """Applies all preprocessing transformations to the dataframe for model prediction."""
    df_processed = df.copy()

    # Replace placeholders
    df_processed["DAYS_EMPLOYED"] = df_processed["DAYS_EMPLOYED"].replace(
        365243, np.nan
    )
    df_processed["ORGANIZATION_TYPE"] = df_processed["ORGANIZATION_TYPE"].replace(
        "XNA", np.nan
    )

    # Fill missing values
    for col in ("OCCUPATION_TYPE", "ORGANIZATION_TYPE"):
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna("Unknown")

    # Convert time columns to positive
    for col in (
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
    ):
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].abs()

    # Fix Region Rating anachronisms
    if "REGION_RATING_CLIENT_W_CITY" in df_processed.columns:
        df_processed["REGION_RATING_CLIENT_W_CITY"] = df_processed[
            "REGION_RATING_CLIENT_W_CITY"
        ].replace(-1, 2)

    # Standardize categoricals
    mappings = {
        "CODE_GENDER": {"M": "male", "F": "female", "XNA": "male"},
        "FLAG_OWN_CAR": {"Y": "yes", "N": "no"},
        "FLAG_OWN_REALTY": {"Y": "yes", "N": "no"},
        "FLAG_MOBIL": {1: "yes", 0: "no"},
        "FLAG_EMP_PHONE": {1: "yes", 0: "no"},
        "FLAG_WORK_PHONE": {1: "yes", 0: "no"},
        "FLAG_CONT_MOBILE": {1: "yes", 0: "no"},
        "FLAG_PHONE": {1: "yes", 0: "no"},
        "FLAG_EMAIL": {1: "yes", 0: "no"},
        "FLAG_DOCUMENT_3": {1: "yes", 0: "no"},
        "FLAG_DOCUMENT_6": {1: "yes", 0: "no"},
        "FLAG_DOCUMENT_8": {1: "yes", 0: "no"},
        "REG_REGION_NOT_LIVE_REGION": {1: "different", 0: "same"},
        "REG_REGION_NOT_WORK_REGION": {1: "different", 0: "same"},
        "LIVE_REGION_NOT_WORK_REGION": {1: "different", 0: "same"},
        "REG_CITY_NOT_LIVE_CITY": {1: "different", 0: "same"},
        "REG_CITY_NOT_WORK_CITY": {1: "different", 0: "same"},
        "LIVE_CITY_NOT_WORK_CITY": {1: "different", 0: "same"},
        "REGION_RATING_CLIENT_W_CITY": {1: "A", 2: "B", 3: "C"},
    }
    for col, mapping in mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].replace(mapping)

    # Engineer ratio features
    df_processed["PAYMENT_RATE"] = df_processed["AMT_ANNUITY"] / df_processed[
        "AMT_CREDIT"
    ].replace(0, np.nan)
    df_processed["ANNUITY_INCOME_PERC"] = df_processed["AMT_ANNUITY"] / df_processed[
        "AMT_INCOME_TOTAL"
    ].replace(0, np.nan)
    df_processed["INCOME_CREDIT_PERC"] = df_processed[
        "AMT_INCOME_TOTAL"
    ] / df_processed["AMT_CREDIT"].replace(0, np.nan)
    df_processed["DEBT_TO_INCOME"] = df_processed["AMT_CREDIT"] / df_processed[
        "AMT_INCOME_TOTAL"
    ].replace(0, np.nan)
    df_processed["CREDIT_PER_PERSON"] = df_processed["AMT_CREDIT"] / df_processed[
        "CNT_FAM_MEMBERS"
    ].replace(0, np.nan)

    # Ensure all expected features are present
    for col in expected_features:
        if col not in df_processed.columns:
            df_processed[col] = np.nan

    # Cast all numeric columns to float64 to match model signature
    for col in df_processed.select_dtypes(include=np.number).columns:
        if df_processed[col].dtype != "float64":
            df_processed[col] = df_processed[col].astype("float64")

    return df_processed[expected_features]


def main():
    """Main function to run the preprocessing and prediction."""
    logger.info("Loading MLflow model...")
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    expected_features = model.metadata.get_input_schema().input_names()
    threshold = float(model.metadata.metadata.get("optimal_threshold", 0.5))

    logger.info(f"Loading raw test data from {TEST_DATA_PATH}...")
    raw_data = pd.read_csv(TEST_DATA_PATH, usecols=COLUMNS_TO_IMPORT)
    raw_data.set_index("SK_ID_CURR", inplace=True)

    logger.info("Applying transformations for model prediction...")
    processed_data_for_model = apply_transformations_for_model(
        raw_data.copy(), expected_features
    )

    logger.info("Generating predictions...")
    predictions = model.predict(processed_data_for_model)
    predictions_df = pd.DataFrame(
        predictions,
        index=raw_data.index,
        columns=["probability_neg", "probability_pos"],
    )

    logger.info("Creating final dashboard dataset...")
    # Start with the features needed for the dashboard from the raw data
    dashboard_df = raw_data[DASHBOARD_FEATURES].copy()

    # **FIX:** Clean DAYS_EMPLOYED for the dashboard output
    dashboard_df["DAYS_EMPLOYED"] = (
        dashboard_df["DAYS_EMPLOYED"].replace(365243, np.nan).abs()
    )

    # Join the predictions
    dashboard_df = dashboard_df.join(predictions_df)

    # Add threshold and decision
    dashboard_df["threshold"] = threshold
    dashboard_df["decision"] = np.where(
        dashboard_df["probability_pos"] >= threshold, "refused", "accepted"
    )

    logger.info(f"Saving preprocessed data to {OUTPUT_DATA_PATH}...")
    dashboard_df.to_csv(OUTPUT_DATA_PATH)
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
