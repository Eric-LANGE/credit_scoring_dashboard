---
title: Credit Risk Dashboard
emoji: 📊📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
port: 7860
---

# Credit risk analysis dashboard

**Live demo**: [view on Hugging Face Spaces](https://shaolins-p8-dashboard.hf.space)

## Overview

This project provides a **web dashboard** for analyzing credit risk predictions. 
Users can explore credit scores, feature importance (SHAP), and customer positioning through an interactive interface. 
The application uses a **FastAPI backend** for runtime inference and a **Streamlit frontend** for visualization, optimized for low-latency performance through caching.

**Architecture**: the system loads an MLflow model and raw customer data at startup. 
On the first dashboard request, it performs preprocessing and prediction for all customers (~5-10s warmup), then caches results in memory for instant subsequent requests (<50ms).

## Features

### Dashboard capabilities

- **Interactive credit scoring**: dynamic gauge visualization showing approval/rejection decisions
- **Feature importance analysis**: 
  - global view via SHAP beeswarm plots
  - local explanations via SHAP waterfall charts
- **Customer positioning**: 
  - univariate distribution analysis with histogram overlays
  - bivariate scatter plot analysis for feature relationships

### Technical features

- **Runtime inference**: MLflow model with lazy cache initialization for optimal memory usage
- **REST API**: FastAPI backend with automatic OpenAPI documentation
- **Containerized deployment**: Docker + micromamba for reproducible environments
- **CI/CD pipeline**: automated testing, linting, and deployment to Hugging Face Spaces
- **Responsive UI**: Streamlit frontend with custom CSS for stable image rendering
- **Preprocessing integrity**: runtime preprocessing matches training pipeline exactly to prevent drift

## Architecture

The application consists of three main components:

1. **InferenceService (Backend Core):**
   - loads MLflow model and raw customer data at startup
   - applies preprocessing pipeline matching training exactly
   - implements lazy caching: predictions computed on first request, cached for subsequent requests
   - serves SHAP explanations from pre-computed explanations

2. **FastAPI API layer:**
   - exposes RESTful endpoints for customer data, scores, and SHAP values
   - composite endpoint (`/customer/{id}/dashboard`) fetches all data in one request
   - serves static assets (SHAP plots, distribution data) with HTTP caching
   - automatic OpenAPI documentation at `/docs`

3. **Streamlit frontend:**
   - interactive web dashboard querying FastAPI backend
   - generates plots and visualizations client-side
   - custom CSS for stable image rendering and responsive layout

The services are designed to run together, as orchestrated by the `entrypoint.sh` script.

### Project structure

```
.
├── .dockerignore
├── .gitattributes
├── .github
│   └── workflows
│       └── deploy.yml
├── .gitignore
├── Dockerfile
├── README.md
├── credit_risk_env.yml
├── data
│   └── application_test.csv        # Raw customer data (26.5 MB, Git LFS)
├── entrypoint.sh                   # Startup orchestration script
├── models
│   └── gradient_boosting           # MLflow model directory
│       ├── MLmodel
│       ├── model.pkl               # Trained model (479 KB, Git LFS)
│       ├── conda.yaml
│       ├── python_env.yaml
│       ├── requirements.txt
│       └── code                    # Model code dependencies
│           └── p7_utils/
├── plots
│   ├── DAYS_EMPLOYED_hist_data.json    # Pre-computed histograms (4 features)
│   ├── EXT_SOURCE_2_hist_data.json
│   ├── EXT_SOURCE_3_hist_data.json
│   └── OWN_CAR_AGE_hist_data.json
├── shap
│   ├── shap_beeswarm.png           # Global SHAP plot (200 KB, Git LFS)
│   └── shap_explanation.joblib     # Pre-computed SHAP values (93 MB, Git LFS)
├── src
│   └── credit_risk_app
│       ├── __init__.py
│       ├── config.py               # Configuration constants
│       ├── dashboard.py            # Streamlit frontend
│       ├── main.py                 # FastAPI backend
│       ├── preprocessing.py        # Feature engineering pipeline
│       └── services.py             # InferenceService with lazy caching
└── tests
    └── test_main.py
```

### API endpoints

The FastAPI backend (`http://localhost:8000`) exposes:

#### Primary endpoints

- **`GET /customers`**: returns all available customer IDs.
- **`GET /customer/{customer_id}/dashboard`**: composite endpoint fetching all dashboard data (score, features, SHAP) in a single request. First call triggers warmup (~5-10s).

#### Widget endpoints

- **`GET /customer/{customer_id}/score`**: returns score data for the gauge widget.
  - Response: `{ "probability_pos": float, "threshold": float, "decision": "accepted"|"refused" }`
- **`GET /customer/{customer_id}/features`**: returns the 4 main dashboard features.
  - Response: `{ "EXT_SOURCE_3": float|null, "EXT_SOURCE_2": float|null, "DAYS_EMPLOYED": int|null, "OWN_CAR_AGE": int|null }`
- **`GET /customer/{customer_id}/shap`**: returns local SHAP values for the waterfall plot.
  - Response: `{ "base_value": float, "values": float[], "feature_names": string[] }`
- **`GET /features/bivariate_data?feat_x={feature}&feat_y={feature}`**: returns scatter plot data for bivariate analysis.
  - Response: `{ "x_data": float[], "y_data": float[] }`

#### Static assets endpoints

- **`GET /shap/global`**: returns the pre-computed global SHAP beeswarm plot (PNG image).
  - Response: `image/png` with `Cache-Control: public, max-age=86400`
- **`GET /features/{feature_name}/distribution`**: returns pre-computed histogram data for a feature.
  - Available features: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_EMPLOYED`, `OWN_CAR_AGE`
  - Response: `{ "feature": string, "counts": int[], "bin_edges": float[], "median": float }`

**API documentation**: visit `http://localhost:8000/docs` for interactive Swagger UI

### Data source

The application loads raw customer data from `data/application_test.csv` at startup. 
This file contains unprocessed customer applications from the "Home Credit Default Risk" Kaggle competition.

**Runtime processing:**
- preprocessing is applied at runtime using `preprocessing.py`
- the preprocessing pipeline **must match training exactly** to prevent prediction drift
- critical preprocessing steps include:
- placeholder replacement (e.g., DAYS_EMPLOYED = 365243 → NaN)
- missing value imputation
- time column conversion (negative days → absolute values)
- categorical standardization (e.g., CODE_GENDER XNA → "male")
- feature engineering (5 financial ratios)
- type casting (all numeric columns → float64 for MLflow compatibility)

**Pre-computed artifacts:**
- **SHAP explanations** (`shap/shap_explanation.joblib`, 93 MB): generated once during model training
- **Distribution histograms** (`plots/*.json`): pre-computed for 4 key features to accelerate dashboard rendering

**Note**: large files (`.csv`, `.joblib`, `.png`, `.pkl`) are managed via Git LFS as specified in `.gitattributes`.

### Testing

The project includes automated tests:
```bash
# Run tests locally
pytest tests/

# Run tests in Docker (matches CI/CD)
docker run --rm --entrypoint pytest -e PYTHONPATH=/app:/app/src credit-risk-app:latest tests/
```

Tests cover:
- API endpoint responses
- service layer caching behavior
- mock-based isolation of InferenceService

### Critical development notes

1. **MLflow model loading**: the model is loaded via `mlflow.pyfunc.load_model()`, which:
   - validates input schema against expected features
   - loads metadata
   - ensures reproducible predictions

2. **Git LFS**: clone with `git lfs pull` to fetch large files. 

3. **Environment variables**:
   - `MLFLOW_TRACKING_URI="file:///tmp/mlruns-disabled"` (disable tracking)
   - `MPLCONFIGDIR="/tmp/matplotlib"` (writable config directory)
   - `PYTHONPATH="/app/src"` (module resolution)
