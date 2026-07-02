import pytest
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test the home page loads correctly."""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"Disease" in rv.data

def test_heart_page(client):
    """Test the heart disease page loads correctly."""
    rv = client.get('/heart')
    assert rv.status_code == 200

def test_diabetes_page(client):
    """Test the diabetes page loads correctly."""
    rv = client.get('/diabetes')
    assert rv.status_code == 200

def test_parkinsons_page(client):
    """Test the parkinsons page loads correctly."""
    rv = client.get('/parkinsons')
    assert rv.status_code == 200
