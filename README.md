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

This project provides a **production-ready web dashboard** for analyzing pre-calculated credit risk predictions. Users can explore credit scores, feature importance (SHAP), and customer positioning through an interactive interface. The application uses a **FastAPI backend** for data serving and a **Streamlit frontend** for visualization, optimized for low-latency performance.

**Important**: This is an inference-only dashboard. All ML predictions and SHAP explanations are pre-calculated and served from static files—no model training or live inference occurs at runtime.

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

- **REST API**: FastAPI backend with automatic OpenAPI documentation
- **Containerized Deployment**: Docker + micromamba for reproducible environments
- **CI/CD Pipeline**: Automated testing, linting, and deployment to Hugging Face Spaces
- **Responsive UI**: Streamlit frontend with custom CSS for stable image rendering

## Architecture

The application consists of two main components:

1.  **FastAPI Backend:** A Python backend that loads pre-calculated prediction data from a CSV file (`data/dashboard_data.csv`) and SHAP values from a joblib file (`shap/shap_explanation.joblib`). It exposes several API endpoints to make this data available to the frontend.
2.  **Streamlit Frontend:** An interactive web dashboard that queries the FastAPI backend to retrieve data for a selected client. It then uses this data to generate various plots and visualizations.

The two services are designed to run together, as defined in the `entrypoint.sh` script.

## Quick Start

### Prerequisites

- Docker 20.10+ installed
- 2GB+ available disk space
- Port 7860 available (or modify `docker run` command)

### Using Docker (Recommended)

The easiest way to run the application:

### Running the Application

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Build and run the Docker container:**
    ```bash
    docker build -t credit-risk-dashboard .
    docker run -p 7860:7860 credit-risk-dashboard
    ```

3.  **Access the dashboard:**
    Open your web browser and navigate to `http://localhost:7860`.

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
│   └── dashboard_data.csv
├── entrypoint.sh               # Startup orchestration script
├── plots
│   ├── DAYS_EMPLOYED_hist_data.json  # Pre-computed histograms (4 features)
│   ├── EXT_SOURCE_2_hist_data.json
│   ├── EXT_SOURCE_3_hist_data.json
│   └── OWN_CAR_AGE_hist_data.json
├── shap
│   ├── shap_beeswarm.png
│   └── shap_explanation.joblib
├── src
│   └── credit_risk_app
│       ├── __init__.py
│       ├── config.py
│       ├── dashboard.py        # Streamlit frontend
│       ├── main.py
│       ├── preprocessing.py
│       └── services.py
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

The data used in this application is stored in `data/dashboard_data.csv`. This file contains pre-calculated predictions, SHAP values, and other client information. The original data for this project comes from the "Home Credit Default Risk" competition on Kaggle.

**Note**: Large files (`.csv`, `.joblib`, `.png`) are managed via Git LFS as specified in `.gitattributes`.

### Testing

The project uses `pytest` for testing the API endpoints.

To run the tests, execute the following command from the root of the project:
```bash
docker build -t credit-risk-dashboard .
docker run --rm --entrypoint pytest credit-risk-dashboard tests/
```

Tests cover:
- All API endpoints with mocked service layer
- The new composite endpoint structure
- Error handling for missing customers

### Development Workflow

1. **Code Quality Checks** (pre-commit):
   ```bash
   black --check .
   flake8 .
   ```

2. **Local Testing**:
   ```bash
   docker build -t credit-risk-dashboard .
   docker run --rm --entrypoint pytest credit-risk-dashboard
   ```

3. **Deployment**: Push to `main` branch triggers:
   - Automated linting (flake8, black)
   - Docker build and test
   - Deployment to Hugging Face Spaces (requires `HF_TOKEN` secret)

### Technology Stack

- **Backend**: FastAPI 0.115+, uvicorn, Pydantic
- **Frontend**: Streamlit 1.41+, Plotly, Matplotlib
- **ML/Data**: pandas 2.2+, numpy 2.0+, scikit-learn 1.6+, SHAP 0.46+
- **Container**: Docker with mambaorg/micromamba base image
- **CI/CD**: GitHub Actions
