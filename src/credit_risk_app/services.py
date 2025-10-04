# src/credit_risk_app/services.py (NEW VERSION)

import logging
import pandas as pd
import joblib
import mlflow
import numpy as np
from pathlib import Path
from fastapi import HTTPException
from typing import Optional, Dict, Any

from .preprocessing import apply_transformations
from .config import COLUMNS_TO_IMPORT

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Service for runtime inference with intelligent caching.

    Strategy:
    - Load model + raw data at startup (~150 MB memory)
    - Lazy preprocessing + prediction on first request (~5s warmup)
    - Cache all results in memory (instant subsequent requests)
    """

    def __init__(
        self, model_path: Path, raw_data_path: Path, shap_explanation_path: Path
    ):
        # 1. Load MLflow model
        logger.info(f"Loading MLflow model from {model_path}...")
        self.model = mlflow.pyfunc.load_model(str(model_path))
        self.expected_features = self.model.metadata.get_input_schema().input_names()
        self.threshold = float(
            self.model.metadata.metadata.get("optimal_threshold", 0.5)
        )
        logger.info(f"Model loaded. Threshold: {self.threshold}")

        # 2. Load raw data
        logger.info(f"Loading raw data from {raw_data_path}...")
        self.raw_data = pd.read_csv(
            raw_data_path, usecols=COLUMNS_TO_IMPORT, index_col="SK_ID_CURR"
        )
        logger.info(f"Loaded {len(self.raw_data)} clients")

        # 3. Load SHAP explanation (pre-computed)
        logger.info(f"Loading SHAP explanation from {shap_explanation_path}...")
        self.shap_explanation = joblib.load(shap_explanation_path)
        logger.info("SHAP explanation loaded")

        # 4. Initialize cache (lazy loading)
        self._predictions_cache: Optional[pd.DataFrame] = None
        self._dashboard_features = [
            "EXT_SOURCE_3",
            "EXT_SOURCE_2",
            "DAYS_EMPLOYED",
            "OWN_CAR_AGE",
        ]

    def _ensure_predictions_cached(self) -> None:
        """
        Lazy cache initialization: compute predictions on first request.

        Warmup time: ~5-10s for 48k clients (one-time cost).
        """
        if self._predictions_cache is not None:
            return  # Already cached

        logger.info("🔥 WARMUP: Computing predictions for all clients...")
        import time

        start = time.time()

        # Apply preprocessing
        X_processed = apply_transformations(
            self.raw_data.copy(), self.expected_features
        )

        # Generate predictions
        predictions = self.model.predict(X_processed)
        predictions_df = pd.DataFrame(
            predictions,
            index=self.raw_data.index,
            columns=["probability_neg", "probability_pos"],
        )

        # Prepare dashboard features (clean DAYS_EMPLOYED)
        dashboard_df = self.raw_data[self._dashboard_features].copy()
        dashboard_df["DAYS_EMPLOYED"] = (
            dashboard_df["DAYS_EMPLOYED"].replace(365243, np.nan).abs()
        )

        # Combine everything
        self._predictions_cache = dashboard_df.join(predictions_df)
        self._predictions_cache["threshold"] = self.threshold
        self._predictions_cache["decision"] = np.where(
            self._predictions_cache["probability_pos"] >= self.threshold,
            "refused",
            "accepted",
        )

        elapsed = time.time() - start
        logger.info(f"✅ WARMUP COMPLETE in {elapsed:.2f}s. Cache ready.")

    def get_all_customer_ids(self) -> list[int]:
        """Returns list of all customer IDs (no cache needed)."""
        return self.raw_data.index.tolist()

    def _get_customer_data(self, customer_id: int) -> pd.Series:
        """
        Retrieve cached data for a customer.
        Triggers lazy cache initialization on first call.
        """
        self._ensure_predictions_cached()

        if customer_id not in self._predictions_cache.index:
            raise HTTPException(
                status_code=404, detail=f"Customer ID {customer_id} not found."
            )

        return self._predictions_cache.loc[customer_id]

    def get_score_data(self, customer_id: int) -> Dict[str, Any]:
        """Returns score data for gauge widget."""
        customer_data = self._get_customer_data(customer_id)
        return {
            "probability_pos": float(customer_data["probability_pos"]),
            "threshold": float(customer_data["threshold"]),
            "decision": customer_data["decision"],
        }

    def get_main_features(self, customer_id: int) -> Dict[str, Any]:
        """Returns the 4 main dashboard features."""
        customer_data = self._get_customer_data(customer_id)
        features = self._dashboard_features
        customer_features = customer_data[features].replace({np.nan: None})
        return customer_features.to_dict()

    def get_local_shap_values(self, customer_id: int) -> Dict[str, Any]:
        """Extracts local SHAP values (from pre-computed explanation)."""
        if customer_id not in self.raw_data.index:
            raise HTTPException(
                status_code=404, detail=f"Customer ID {customer_id} not found."
            )

        try:
            positional_idx = self.raw_data.index.get_loc(customer_id)
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

    def get_bivariate_data(self, feat_x: str, feat_y: str) -> Dict[str, Any]:
        """Returns bivariate scatter data."""
        self._ensure_predictions_cached()

        if feat_x == feat_y:
            data_series = self._predictions_cache[feat_x].dropna()
            return {
                "x_data": data_series.tolist(),
                "y_data": data_series.tolist(),
            }

        bivariate_df = self._predictions_cache[[feat_x, feat_y]].dropna()
        return {
            "x_data": bivariate_df[feat_x].tolist(),
            "y_data": bivariate_df[feat_y].tolist(),
        }
