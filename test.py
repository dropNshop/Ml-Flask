#!/usr/bin/env python3
import requests
import json

# Update this after deploying to Vercel
BASE_URL = "https://your-vercel-app-name.vercel.app"


def test_train_endpoint():
    """Test the /api/train endpoint to train the model."""
    print("\nTesting /api/train endpoint...")
    url = f"{BASE_URL}/api/train"
    try:
        response = requests.get(url)
        print("Status Code:", response.status_code)
        data = response.json()
        print("Response:", json.dumps(data, indent=4))
    except Exception as e:
        print("Exception during /api/train test:", e)


def test_predict_endpoint():
    """Test the /api/predict endpoint to get price predictions. Ensure the model is trained first."""
    print("\nTesting /api/predict endpoint...")
    url = f"{BASE_URL}/api/predict"
    headers = {"Content-Type": "application/json"}

    # Sample payload; adjust values to ones present in your CSV file
    payload = {
        "category": "Meat & Poultry",
        "product": "Chicken Breast",
        "month": 5,
        "year": 2024
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        print("Status Code:", response.status_code)
        data = response.json()
        print("Response:", json.dumps(data, indent=4))
    except Exception as e:
        print("Exception during /api/predict test:", e)


def test_dashboard_endpoint():
    """Test the /api/dashboard endpoint for analytics data."""
    print("\nTesting /api/dashboard endpoint...")
    url = f"{BASE_URL}/api/dashboard"
    try:
        response = requests.get(url)
        print("Status Code:", response.status_code)
        data = response.json()
        print("Response:", json.dumps(data, indent=4))
    except Exception as e:
        print("Exception during /api/dashboard test:", e)


def test_stats_endpoint():
    """Test the /api/stats endpoint for summary statistics."""
    print("\nTesting /api/stats endpoint...")
    url = f"{BASE_URL}/api/stats"
    try:
        response = requests.get(url)
        print("Status Code:", response.status_code)
        data = response.json()
        print("Response:", json.dumps(data, indent=4))
    except Exception as e:
        print("Exception during /api/stats test:", e)


if __name__ == "__main__":
    print("Ensure that your Flask application is running at", BASE_URL)
    input("Press Enter to run tests...")

    # Test endpoints sequentially
    test_train_endpoint()
    test_predict_endpoint()
    test_dashboard_endpoint()
    test_stats_endpoint() 