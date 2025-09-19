# src/credit_risk_app/main.py

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Depends
from .services import DashboardService

# --- Configuration & Setup ---
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = APP_DIR.parent.parent
DATA_PATH = PROJECT_ROOT_DIR / "data" / "dashboard_data.csv"
SHAP_EXPLANATION_PATH = PROJECT_ROOT_DIR / "shap" / "shap_explanation.joblib"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Lifespan & App Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading resources...")
    try:
        service = DashboardService(DATA_PATH, SHAP_EXPLANATION_PATH)
        app.state.dashboard_service = service
        logger.info("Application startup complete. Service is ready.")
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        app.state.dashboard_service = None
    yield
    logger.info("Application shutdown.")


app = FastAPI(lifespan=lifespan, title="Credit Risk API")


def get_dashboard_service(request: Request) -> DashboardService:
    if not request.app.state.dashboard_service:
        raise HTTPException(status_code=503, detail="Service is unavailable.")
    return request.app.state.dashboard_service


# --- API Endpoints ---


@app.get("/customers", tags=["Dashboard Data"])
async def customers(service: DashboardService = Depends(get_dashboard_service)):
    """Returns a list of all available customer IDs."""
    return {"customer_ids": service.get_all_customer_ids()}


@app.get("/customer/{customer_id}/score", tags=["Dashboard Widgets"])
async def get_score(
    customer_id: int, service: DashboardService = Depends(get_dashboard_service)
):
    """Endpoint for the Score Gauge widget."""
    return service.get_score_data(customer_id)


@app.get("/customer/{customer_id}/features", tags=["Dashboard Widgets"])
async def get_features(
    customer_id: int, service: DashboardService = Depends(get_dashboard_service)
):
    """Endpoint for the Main Features display."""
    return service.get_main_features(customer_id)


@app.get("/customer/{customer_id}/shap", tags=["Dashboard Widgets"])
async def get_shap_values(
    customer_id: int, service: DashboardService = Depends(get_dashboard_service)
):
    """Endpoint for the Local SHAP Importance (waterfall) plot."""
    return service.get_local_shap_values(customer_id)


@app.get("/features/bivariate_data", tags=["Dashboard Widgets"])
async def get_bivariate_data(
    feat_x: str, feat_y: str, service: DashboardService = Depends(get_dashboard_service)
):
    """Endpoint for the Bi-variate Analysis scatter plot."""
    return service.get_bivariate_data(feat_x, feat_y)


# Note: The distribution data is now handled directly by Streamlit reading static JSON files.
# If you wanted to serve them via API, you could add an endpoint like this:
#
# import json
# @app.get("/feature/{feature_name}/distribution", tags=["Dashboard Widgets"])
# async def get_distribution(feature_name: str):
#     """Endpoint for the Feature Distribution histogram."""
#     plot_path = PROJECT_ROOT_DIR / "plots" / f"{feature_name}_hist_data.json"
#     if not plot_path.exists():
#         raise HTTPException(status_code=404, detail=f"Distribution data for {feature_name} not found.")
#     with open(plot_path, "r") as f:
#         return json.load(f)
