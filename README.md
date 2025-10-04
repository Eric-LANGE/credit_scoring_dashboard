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

**Live Demo**: [View on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/p8_dashboard)

## Overview

This project provides a **production-ready web dashboard** for analyzing credit risk predictions in real-time. Users can explore credit scores, feature importance (SHAP), and customer positioning through an interactive interface. The application uses a **FastAPI backend** for runtime inference and a **Streamlit frontend** for visualization, optimized for low-latency performance through intelligent caching.

**Architecture**: The system loads an MLflow model and raw customer data at startup (~150 MB memory footprint). On the first dashboard request, it performs preprocessing and prediction for all customers (~5-10s warmup), then caches results in memory for instant subsequent requests (<50ms).

## Features

### Dashboard Capabilities

- **Interactive Credit Scoring**: Dynamic gauge visualization showing approval/rejection decisions
- **Feature Importance Analysis**: 
  - Global view via SHAP beeswarm plots
  - Local explanations via SHAP waterfall charts (per customer)
- **Customer Positioning**: 
  - Univariate distribution analysis with histogram overlays
  - Bivariate scatter plot analysis for feature relationships
- **Performance Optimized**: Composite API endpoint reduces dashboard load time by ~60%

### Technical Features

- **Runtime Inference**: MLflow model with lazy cache initialization for optimal memory usage
- **REST API**: FastAPI backend with automatic OpenAPI documentation
- **Containerized Deployment**: Docker + micromamba for reproducible environments
- **CI/CD Pipeline**: Automated testing, linting, and deployment to Hugging Face Spaces
- **Responsive UI**: Streamlit frontend with custom CSS for stable image rendering
- **Preprocessing Integrity**: Runtime preprocessing matches training pipeline exactly to prevent drift

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

## Performance Characteristics

### Memory Usage
- **Startup**: ~150 MB (model + raw data + SHAP explanations)
- **After warmup**: ~250-300 MB (cached predictions for all customers)

### Response Times
- **First dashboard request**: 5-10s (warmup: preprocessing + prediction for ~48k customers)
- **Subsequent requests**: <50ms (cached results)
- **API endpoint**: `/customer/{id}/dashboard` (composite, optimized)

### Warmup Behavior
The warmup happens automatically on the first customer dashboard request. The API logs clearly indicate when warmup is in progress and when it completes.

## Quick Start

### Prerequisites

- Docker 20.10+ installed
- 2GB+ available disk space
- Port 7860 available (or modify `docker run` command)

### Using Docker (Recommended)

The easiest way to run the application:

```bash
# Build the Docker image
docker build -t credit-risk-dashboard .

# Run the container
docker run -p 7860:7860 credit-risk-dashboard
```

Access the dashboard at `http://localhost:7860`

### Alternative: Manual Setup

For development without Docker:

```bash
micromamba create -f credit_risk_env.yml
micromamba activate base
bash entrypoint.sh
```

## For Developers & Data Scientists

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

- **`GET /customers`**: Returns all available customer IDs
- **`GET /customer/{customer_id}/dashboard`**: ⚡ **Optimized composite endpoint** — fetches all dashboard data (score, features, SHAP) in a single request (~60% latency reduction vs. 3 separate calls)

#### Legacy Widget Endpoints

These are maintained for backward compatibility but the composite endpoint is preferred:

- `GET /customer/{customer_id}/score`: Credit score and decision
- `GET /customer/{customer_id}/features`: Main customer features
- `GET /customer/{customer_id}/shap`: Local SHAP explanation
- `GET /features/bivariate_data?feat_x=X&feat_y=Y`: Bivariate scatter data

**API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger UI

### Data Source

The application loads raw customer data from `data/application_test.csv` at startup. This file contains unprocessed customer applications from the "Home Credit Default Risk" Kaggle competition.

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
- **SHAP explanations** (`shap/shap_explanation.joblib`, 93 MB): Generated once during model training
- **Distribution histograms** (`plots/*.json`): Pre-computed for 4 key features to accelerate dashboard rendering

**Note**: Large files (`.csv`, `.joblib`, `.png`, `.pkl`) are managed via Git LFS as specified in `.gitattributes`.

### Testing

The project uses `pytest` for testing the API endpoints.

To run the tests:

```bash
docker build -t credit-risk-dashboard .
docker run --rm --entrypoint pytest credit-risk-dashboard tests/
```

Tests cover:
- All API endpoints with mocked service layer
- The composite endpoint structure
- Error handling for missing customers
- Lazy cache initialization behavior

### Development Workflow

1. **Code Quality Checks** (pre-commit):
   ```bash
   black --check .
   flake8 .
   ```

2. **Local Testing**:
   ```bash
   docker build -t credit-risk-dashboard .
   docker run --rm --entrypoint pytest credit-risk-dashboard tests/
   ```

3. **Deployment**: Push to `main` branch triggers:
   - Automated linting (flake8, black)
   - Docker build and test
   - Deployment to Hugging Face Spaces (requires `HF_TOKEN` secret)

### Technology Stack

- **Backend**: FastAPI 0.115+, uvicorn, Pydantic
- **Frontend**: Streamlit 1.41+, Plotly, Matplotlib
- **ML/Data**: pandas 2.2+, numpy 2.0+, scikit-learn 1.6+, SHAP 0.46+
- **Model Management**: MLflow 2.22+ (model loading, metadata, signature validation)
- **Container**: Docker with mambaorg/micromamba base image
- **CI/CD**: GitHub Actions

### Critical Development Notes

1. **Preprocessing Consistency**: Any changes to `preprocessing.py` MUST be validated against the training pipeline to prevent prediction drift.

2. **Memory Management**: The lazy cache strategy balances memory usage and performance. First request triggers warmup (~5-10s), but subsequent requests are instant.

3. **MLflow Model Loading**: The model is loaded via `mlflow.pyfunc.load_model()`, which:
   - Validates input schema against expected features
   - Loads metadata (optimal threshold, model version)
   - Ensures reproducible predictions

4. **Git LFS**: Clone with `git lfs pull` to fetch large files. Without LFS, the application will fail to load models and data.

5. **Environment Variables**: The entrypoint script sets:
   - `MLFLOW_TRACKING_URI="file:///tmp/mlruns-disabled"` (disable tracking)
   - `MPLCONFIGDIR="/tmp/matplotlib"` (writable config directory)
   - `PYTHONPATH="/app/src"` (module resolution)
