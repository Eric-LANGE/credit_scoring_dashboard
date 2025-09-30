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

## Overview

This project provides a web-based dashboard for analyzing pre-calculated credit risk predictions. It allows users to explore the credit scores, feature importance, and other details for a set of clients. The application is built with a FastAPI backend to serve the data and a Streamlit frontend for the interactive dashboard.

**Note:** This application does not contain a live machine learning model. All predictions and SHAP values are pre-calculated and loaded from static files.

## Features

*   **Interactive Dashboard:** A web-based interface to visualize credit risk predictions.
*   **Credit Score Visualization:** View the credit score of each client on a dynamic gauge.
*   **Feature Importance:** Explore both global (SHAP beeswarm) and local (SHAP waterfall) feature importance to understand the factors influencing the predictions.
*   **Customer Data Exploration:** View the main features of each client and analyze the distribution of individual features.
*   **Bivariate Analysis:** Compare pairs of features in a scatter plot to identify relationships.
*   **REST API:** A FastAPI backend provides endpoints for all the data displayed on the dashboard.
*   **Dockerized Application:** The entire application is containerized for easy setup and deployment.

## How it Works

The application consists of two main components:

1.  **FastAPI Backend:** A Python backend that loads pre-calculated prediction data from a CSV file (`data/dashboard_data.csv`) and SHAP values from a joblib file (`shap/shap_explanation.joblib`). It exposes several API endpoints to make this data available to the frontend.
2.  **Streamlit Frontend:** An interactive web dashboard that queries the FastAPI backend to retrieve data for a selected client. It then uses this data to generate various plots and visualizations.

The two services are designed to run together, as defined in the `entrypoint.sh` script.

## Getting Started

The easiest way to run the application is by using Docker.

### Prerequisites

*   Docker installed on your machine.

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
├── entrypoint.sh
├── plots
│   ├── DAYS_EMPLOYED_hist_data.json
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
│       ├── dashboard.py
│       ├── main.py
│       ├── preprocessing.py
│       └── services.py
└── tests
    └── test_main.py
```

### API Endpoints

The FastAPI application exposes the following endpoints:

*   `GET /customers`: Returns a list of all available customer IDs.
*   `GET /customer/{customer_id}/score`: Returns the credit score data for a specific customer.
*   `GET /customer/{customer_id}/features`: Returns the main features for a specific customer.
*   `GET /customer/{customer_id}/shap`: Returns the local SHAP values for a specific customer.
*   `GET /features/bivariate_data?feat_x=<feature1>&feat_y=<feature2>`: Returns data for a bivariate analysis plot.

### Data Source

The data used in this application is stored in `data/dashboard_data.csv`. This file contains pre-calculated predictions, SHAP values, and other client information. The original data for this project comes from the "Home Credit Default Risk" competition on Kaggle.

### Testing

The project uses `pytest` for testing the API endpoints.

To run the tests, execute the following command from the root of the project:
```bash
docker build -t credit-risk-dashboard .
docker run credit-risk-dashboard pytest
```
