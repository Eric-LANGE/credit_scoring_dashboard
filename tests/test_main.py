import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Import the app and the dependency we want to override
from src.credit_risk_app.main import app, get_dashboard_service

# Create a mock instance of the service
mock_service_instance = MagicMock()


# Create an override function that returns our mock instance
def override_get_dashboard_service():
    return mock_service_instance


# Apply the override to the FastAPI app
app.dependency_overrides[get_dashboard_service] = override_get_dashboard_service


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


def test_get_dashboard_composite(client):
    """Test the /customer/{customer_id}/dashboard composite endpoint."""
    customer_id = 100001

    # Configure mock responses for all service methods
    mock_service_instance.get_score_data.return_value = {
        "probability_pos": 0.2,
        "threshold": 0.5,
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

    # Verify structure - all expected keys are present
    assert "score" in data
    assert "features" in data
    assert "shap" in data
    assert "metadata" in data
    assert "timestamp" in data["metadata"]

    # Verify content - data matches what service methods returned
    assert data["score"]["probability_pos"] == 0.2
    assert data["score"]["threshold"] == 0.5
    assert data["score"]["decision"] == "accepted"

    assert data["features"]["EXT_SOURCE_3"] == 0.7
    assert data["features"]["EXT_SOURCE_2"] == 0.6
    assert data["features"]["DAYS_EMPLOYED"] == 1234
    assert data["features"]["OWN_CAR_AGE"] == 5

    assert data["shap"]["base_value"] == 0.15
    assert data["shap"]["values"] == [0.1, -0.05, 0.08, -0.02]
    assert len(data["shap"]["feature_names"]) == 4

    # Verify all service methods were called with correct arguments
    mock_service_instance.get_score_data.assert_called_with(customer_id)
    mock_service_instance.get_main_features.assert_called_with(customer_id)
    mock_service_instance.get_local_shap_values.assert_called_with(customer_id)

    # Verify timestamp is ISO 8601 format (basic check)
    assert "T" in data["metadata"]["timestamp"]
    assert len(data["metadata"]["timestamp"]) > 20  # ISO format is at least 20 chars

    print(
        "\nTest Passed: /dashboard composite endpoint returned complete data structure."
    )
