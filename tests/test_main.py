# tests/test_main.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Mock the service before importing the app
# This ensures the app uses our mock during initialization
mock_service_instance = MagicMock()
with patch(
    "src.credit_risk_app.main.DashboardService", return_value=mock_service_instance
):
    from src.credit_risk_app.main import app


@pytest.fixture
def client():
    """Fixture to create a TestClient for the FastAPI app."""
    return TestClient(app)


def test_get_customers(client):
    """Test the /customers endpoint."""
    mock_service_instance.get_all_customer_ids.return_value = [100001, 100002]
    response = client.get("/customers")
    assert response.status_code == 200
    assert response.json() == {"customer_ids": [100001, 100002]}
    mock_service_instance.get_all_customer_ids.assert_called_once()
    print("\nTest Passed: /customers endpoint returned correct mock data.")


def test_get_score(client):
    """Test the /customer/{customer_id}/score endpoint."""
    customer_id = 100001
    mock_data = {"probabilityClass1": 0.2, "threshold": 0.5, "decision": "accepted"}
    mock_service_instance.get_score_data.return_value = mock_data

    response = client.get(f"/customer/{customer_id}/score")
    assert response.status_code == 200
    assert response.json() == mock_data
    mock_service_instance.get_score_data.assert_called_with(customer_id)
    print("\nTest Passed: /score endpoint returned correct mock data.")


def test_get_features(client):
    """Test the /customer/{customer_id}/features endpoint."""
    customer_id = 100001
    mock_data = {"EXT_SOURCE_3": 0.7, "EXT_SOURCE_2": 0.6}
    mock_service_instance.get_main_features.return_value = mock_data

    response = client.get(f"/customer/{customer_id}/features")
    assert response.status_code == 200
    assert response.json() == mock_data
    mock_service_instance.get_main_features.assert_called_with(customer_id)
    print("\nTest Passed: /features endpoint returned correct mock data.")


def test_get_bivariate_data(client):
    """Test the /features/bivariate_data endpoint."""
    mock_data = {"x_data": [1, 2], "y_data": [3, 4]}
    mock_service_instance.get_bivariate_data.return_value = mock_data

    response = client.get(
        "/features/bivariate_data?feat_x=EXT_SOURCE_2&feat_y=EXT_SOURCE_3"
    )
    assert response.status_code == 200
    assert response.json() == mock_data
    mock_service_instance.get_bivariate_data.assert_called_with(
        "EXT_SOURCE_2", "EXT_SOURCE_3"
    )
    print("\nTest Passed: /bivariate_data endpoint returned correct mock data.")
