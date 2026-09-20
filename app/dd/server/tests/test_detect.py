import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../app')))
from fastapi.testclient import TestClient
from server.api.main import app

client = TestClient(app)

def test_detect_endpoint_no_file():
    response = client.post("/api/detect")
    assert response.status_code == 422

def test_detect_endpoint_with_image():
    # Use a sample image from the repo
    sample_path = os.path.join(os.path.dirname(__file__), '../../app/samples/mt_normal.jpg')
    if not os.path.exists(sample_path):
        sample_path = os.path.join(os.path.dirname(__file__), '../../app/samples/mt_defect.jpg')
    
    with open(sample_path, 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        response = client.post("/api/detect", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert 'result' in data
    assert 'class_name' in data
