# src/credit_risk_app/services.py

import logging
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class DashboardService:
    def __init__(self, data_path: Path, shap_explanation_path: Path):
        try:
            self.dashboard_data = pd.read_csv(data_path, index_col="SK_ID_CURR")
            logger.info(f"Dashboard data loaded successfully from {data_path}.")
        except FileNotFoundError:
            logger.error(f"FATAL: Dashboard data file not found at {data_path}.")
            raise

        try:
            self.shap_explanation = joblib.load(shap_explanation_path)
            logger.info(f"SHAP explanation loaded from {shap_explanation_path}.")
        except FileNotFoundError:
            logger.error(
                f"FATAL: SHAP explanation file not found at {shap_explanation_path}."
            )
            raise

    def get_all_customer_ids(self) -> list[int]:
        """Returns a list of all customer IDs."""
        return self.dashboard_data.index.tolist()

    def _get_customer_data(self, customer_id: int) -> pd.Series:
        """Helper to retrieve a single customer's data."""
        if customer_id not in self.dashboard_data.index:
            raise HTTPException(
                status_code=404, detail=f"Customer ID {customer_id} not found."
            )
        return self.dashboard_data.loc[customer_id]

    def get_score_data(self, customer_id: int) -> dict:
        """Returns data for the score gauge widget."""
        customer_data = self._get_customer_data(customer_id)
        return {
            "probability_pos": customer_data["probability_pos"],
            "threshold": customer_data["threshold"],
            "decision": customer_data["decision"],
        }

    def get_main_features(self, customer_id: int) -> dict:
        """Returns the four main features for a customer, handling missing values."""
        customer_data = self._get_customer_data(customer_id)
        features = ["EXT_SOURCE_3", "EXT_SOURCE_2", "DAYS_EMPLOYED", "OWN_CAR_AGE"]
        customer_features = customer_data[features].replace({np.nan: None})
        return customer_features.to_dict()

    def get_local_shap_values(self, customer_id: int) -> dict:
        """Extracts local SHAP values for a specific customer."""
        if customer_id not in self.dashboard_data.index:
            raise HTTPException(
                status_code=404, detail=f"Customer ID {customer_id} not found."
            )

        try:
            positional_idx = self.dashboard_data.index.get_loc(customer_id)
            shap_values = self.shap_explanation[positional_idx]
            return {
                "base_value": float(shap_values.base_values),
                "values": [float(v) for v in shap_values.values],
                "feature_names": shap_values.feature_names,
            }
        except Exception as e:
            logger.error(f"Error retrieving SHAP values for {customer_id}: {e}")
            raise HTTPException(
                status_code=500, detail="Could not retrieve SHAP values."
            )

    def get_bivariate_data(self, feat_x: str, feat_y: str) -> dict:
        """Returns data for the bi-variate scatter plot."""
        # **FIX:** Explicitly handle case where features are identical
        if feat_x == feat_y:
            data_series = self.dashboard_data[feat_x].dropna()
            return {
                "x_data": [x for x in data_series],
                "y_data": [y for y in data_series],
            }

        # Original logic for different features
        features_to_get = [feat_x, feat_y]
        bivariate_df = self.dashboard_data[features_to_get].dropna()
        return {
            "x_data": [x for x in bivariate_df[feat_x]],
            "y_data": [y for y in bivariate_df[feat_y]],
        }
