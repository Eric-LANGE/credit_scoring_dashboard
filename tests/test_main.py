# tests/test_main.py (UPDATED)

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.credit_risk_app.main import app, get_inference_service

# Create mock service
mock_service_instance = MagicMock()


def override_get_inference_service():
    return mock_service_instance


app.dependency_overrides[get_inference_service] = override_get_inference_service


@pytest.fixture
def client():
    return TestClient(app)


def test_get_customers(client):
    """Test /customers endpoint."""
    mock_service_instance.get_all_customer_ids.return_value = [100001, 100002]
    response = client.get("/customers")
    assert response.status_code == 200
    assert response.json() == {"customer_ids": [100001, 100002]}


def test_get_dashboard_composite(client):
    """Test composite endpoint with lazy cache behavior."""
    customer_id = 100001

    mock_service_instance.get_score_data.return_value = {
        "probability_pos": 0.2,
        "threshold": 0.5064,  # From model metadata
        "decision": "accepted",
    }
    mock_service_instance.get_main_features.return_value = {
        "EXT_SOURCE_3": 0.7,
        "EXT_SOURCE_2": 0.6,
        "DAYS_EMPLOYED": 1234,
        "OWN_CAR_AGE": 5,
    }
    mock_service_instance.get_local_shap_values.return_value = {
        "base_value": 0.15,
        "values": [0.1, -0.05, 0.08, -0.02],
        "feature_names": [
            "EXT_SOURCE_3",
            "EXT_SOURCE_2",
            "DAYS_EMPLOYED",
            "OWN_CAR_AGE",
        ],
    }

    response = client.get(f"/customer/{customer_id}/dashboard")
    assert response.status_code == 200
    data = response.json()

    assert "score" in data
    assert "features" in data
    assert "shap" in data
    assert "metadata" in data

    # Verify warmup was triggered (via service method calls)
    mock_service_instance.get_score_data.assert_called_with(customer_id)
    mock_service_instance.get_main_features.assert_called_with(customer_id)
    mock_service_instance.get_local_shap_values.assert_called_with(customer_id)
