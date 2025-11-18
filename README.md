---
title: Credit Risk Dashboard
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
port: 7860
---

# Credit Risk Analysis Dashboard

**Live Demo**: [View on Hugging Face Spaces](https://shaolins-p8-dashboard.hf.space)

## Overview

This project provides a **web dashboard** for analyzing credit risk predictions. 
Users can explore credit scores, feature importance (SHAP), and customer positioning through an interactive interface. 
The application uses a **FastAPI backend** for runtime inference and a **Streamlit frontend** for visualization, optimized for low-latency performance through caching.

**Architecture**: the system loads an MLflow model and raw customer data at startup. 
On the first dashboard request, it performs preprocessing and prediction for all customers (~5-10s warmup), then caches results in memory for instant subsequent requests (<50ms).

## Features

### Dashboard Capabilities

- **Interactive Credit Scoring**: dynamic gauge visualization showing approval/rejection decisions
- **Feature Importance Analysis**: 
  - Global view via SHAP beeswarm plots
  - Local explanations via SHAP waterfall charts
- **Customer Positioning**: 
  - Univariate distribution analysis with histogram overlays
  - Bivariate scatter plot analysis for feature relationships

### Technical Features

- **Runtime Inference**: MLflow model with lazy cache initialization for optimal memory usage
- **REST API**: FastAPI backend with automatic OpenAPI documentation
- **Containerized Deployment**: Docker + micromamba for reproducible environments
- **CI/CD Pipeline**: automated testing, linting, and deployment to Hugging Face Spaces
- **Responsive UI**: Streamlit frontend with custom CSS for stable image rendering
- **Preprocessing Integrity**: runtime preprocessing matches training pipeline exactly to prevent drift

## Architecture

The application consists of three main components:

1. **InferenceService (Backend Core):**
   - Loads MLflow model and raw customer data at startup
   - Applies preprocessing pipeline matching training exactly
   - Implements lazy caching: predictions computed on first request, cached for subsequent requests
   - Serves SHAP explanations from pre-computed explanations

2. **FastAPI API Layer:**
   - Exposes RESTful endpoints for customer data, scores, and SHAP values
   - Composite endpoint (`/customer/{id}/dashboard`) fetches all data in one request
   - Automatic OpenAPI documentation at `/docs`

3. **Streamlit Frontend:**
   - Interactive web dashboard querying FastAPI backend
   - Generates plots and visualizations client-side
   - Custom CSS for stable image rendering and responsive layout

The services are designed to run together, as orchestrated by the `entrypoint.sh` script.

### Project Structure

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

### API Endpoints

The FastAPI backend (`http://localhost:8000`) exposes:

#### Primary Endpoints

- **`GET /customers`**: returns all available customer IDs
- **`GET /customer/{customer_id}/dashboard`**: fetches all dashboard data (score, features, SHAP) in a single request

**API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger UI

### Data Source

The application loads raw customer data from `data/application_test.csv` at startup. 
This file contains unprocessed customer applications from the "Home Credit Default Risk" Kaggle competition.

**Runtime Processing:**
- Preprocessing is applied at runtime using `preprocessing.py`
- The preprocessing pipeline **must match training exactly** to prevent prediction drift
- Critical preprocessing steps include:
  - Placeholder replacement (e.g., DAYS_EMPLOYED = 365243 → NaN)
  - Missing value imputation
  - Time column conversion (negative days → absolute values)
  - Categorical standardization (e.g., CODE_GENDER XNA → "male")
  - Feature engineering (5 financial ratios)
  - Type casting (all numeric columns → float64 for MLflow compatibility)

**Pre-computed Artifacts:**
- **SHAP explanations** (`shap/shap_explanation.joblib`, 93 MB): generated once during model training
- **Distribution histograms** (`plots/*.json`): pre-computed for 4 key features to accelerate dashboard rendering

**Note**: Large files (`.csv`, `.joblib`, `.png`, `.pkl`) are managed via Git LFS as specified in `.gitattributes`.



### Critical Development Notes

1. **MLflow Model Loading**: The model is loaded via `mlflow.pyfunc.load_model()`, which:
   - Validates input schema against expected features
   - Loads metadata (optimal threshold, model version)
   - Ensures reproducible predictions

2. **Git LFS**: clone with `git lfs pull` to fetch large files. 

3. **Environment Variables**: the entrypoint script sets:
   - `MLFLOW_TRACKING_URI="file:///tmp/mlruns-disabled"` (disable tracking)
   - `MPLCONFIGDIR="/tmp/matplotlib"` (writable config directory)
   - `PYTHONPATH="/app/src"` (module resolution)
