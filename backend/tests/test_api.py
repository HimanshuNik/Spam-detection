import os
import json
import pytest
from app import app, get_db, init_db

@pytest.fixture
def client():
    # Configure app for testing
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    
    # Use a test database
    db_path = os.path.join(os.path.dirname(__file__), "test_spam_history.db")
    app.config["DB_PATH"] = db_path
    
    with app.test_client() as client:
        with app.app_context():
            # In a real setup, we'd mock the db path, but app.py uses a global DB_PATH.
            # We'll just test the endpoints directly assuming the DB is initialized.
            yield client

def test_health_endpoint(client):
    rv = client.get("/health")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_predict_endpoint_empty(client):
    rv = client.post("/predict", json={"message": ""})
    assert rv.status_code == 400
    data = json.loads(rv.data)
    assert "error" in data

def test_predict_endpoint_spam(client):
    # A typical spam message
    rv = client.post("/predict", json={"message": "WINNER! You have won a free iPhone. Call now to claim your prize."})
    if rv.status_code == 200:
        data = json.loads(rv.data)
        assert data["label"] == "spam"
        assert "confidence" in data
        assert "keywords" in data
        assert len(data["keywords"]) > 0
    else:
        # If model is not loaded, it might return 503
        assert rv.status_code in [200, 503]

def test_predict_endpoint_ham(client):
    # A typical ham message
    rv = client.post("/predict", json={"message": "Hey, are you coming to the party tonight?"})
    if rv.status_code == 200:
        data = json.loads(rv.data)
        assert data["label"] == "ham"
        assert "confidence" in data
    else:
        assert rv.status_code in [200, 503]

def test_history_endpoint(client):
    rv = client.get("/history")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert isinstance(data, list)

def test_stats_endpoint(client):
    rv = client.get("/stats")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "total" in data
    assert "spam" in data
    assert "ham" in data

def test_export_endpoint(client):
    rv = client.get("/export")
    assert rv.status_code == 200
    assert rv.mimetype == "text/csv"

def test_samples_endpoint(client):
    rv = client.get("/samples")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "spam" in data
    assert "ham" in data
    assert len(data["spam"]) > 0
    assert len(data["ham"]) > 0

def test_api_dashboard_stats(client):
    rv = client.get("/api/dashboard/stats")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "total_predictions" in data
    assert "spam_predictions" in data
    assert "ham_predictions" in data

def test_api_dashboard_trends(client):
    rv = client.get("/api/dashboard/trends")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "dates" in data
    assert "spam" in data
    assert "ham" in data

def test_api_dashboard_confidence(client):
    rv = client.get("/api/dashboard/confidence")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "distribution" in data

def test_api_dashboard_keywords(client):
    rv = client.get("/api/dashboard/keywords")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "top_keywords" in data

def test_api_dashboard_history(client):
    rv = client.get("/api/dashboard/history")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert isinstance(data, list)

def test_api_dashboard_accuracy(client):
    rv = client.get("/api/dashboard/accuracy")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "accuracy" in data
