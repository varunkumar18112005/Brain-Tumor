import os
import sys
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

BACKEND_DIR = BASE_DIR / "backend"
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
CHECKPOINTS_DIR = BASE_DIR / "ml" / "checkpoints"
DATASETS_DIR = BASE_DIR / "ml" / "datasets"

for d in [UPLOADS_DIR, REPORTS_DIR, KNOWLEDGE_BASE_DIR, CHECKPOINTS_DIR, DATASETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    PROJECT_NAME: str = "Intelligent Medical Image-Based Tumor Detection and Uncertainty Assessment System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    BASE_DIR: Path = BASE_DIR
    UPLOADS_DIR: Path = UPLOADS_DIR
    REPORTS_DIR: Path = REPORTS_DIR
    KNOWLEDGE_BASE_DIR: Path = KNOWLEDGE_BASE_DIR
    CHECKPOINTS_DIR: Path = CHECKPOINTS_DIR
    DATASETS_DIR: Path = DATASETS_DIR
    
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/medical_tumor_system.db"
    
    # Model & Uncertainty Settings
    DEFAULT_IMG_SIZE: int = 224
    MC_DROPOUT_SAMPLES: int = 15
    ABSTENTION_THRESHOLD: float = 0.55
    EXPERT_REVIEW_THRESHOLD: float = 0.40
    
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".dcm", ".nii", ".nii.gz"}

settings = Settings()
