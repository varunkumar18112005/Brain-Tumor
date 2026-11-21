import pytest
from pathlib import Path
import sys
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))
sys.path.insert(0, str(BASE_DIR))

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"

def test_model_status_endpoint():
    response = client.get("/api/model/status")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 2

def test_knowledge_base_endpoint():
    response = client.get("/api/knowledge-base")
    assert response.status_code == 200
    data = response.json()
    assert "tumor_categories" in data
    assert len(data["tumor_categories"]) >= 3

def test_full_pipeline_via_api():
    # 1. Create in-memory test PNG scan
    img = Image.new("L", (128, 128), color=100)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    # 2. Upload
    upload_res = client.post(
        "/api/upload",
        files={"file": ("test_scan.png", buf, "image/png")},
        data={"modality": "MRI", "patient_id": "PAT-TEST-001"}
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    file_id = upload_data["file_id"]
    
    # 3. Analyze complete pipeline
    analysis_res = client.post(
        "/api/analyze",
        json={
            "file_id": file_id,
            "patient_id": "PAT-TEST-001",
            "modality": "MRI",
            "selected_slice": 0
        }
    )
    assert analysis_res.status_code == 200
    analysis_data = analysis_res.json()
    
    assert "analysis_id" in analysis_data
    assert "tumor_detected" in analysis_data
    assert "confidence_score" in analysis_data
    assert "fuzzy_uncertainty_level" in analysis_data
    assert "report_pdf_url" in analysis_data
    
    # 4. Check history includes new analysis
    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    hist_list = hist_res.json()
    assert len(hist_list) > 0
    assert any(h["id"] == analysis_data["analysis_id"] for h in hist_list)
