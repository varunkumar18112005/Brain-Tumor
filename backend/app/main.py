from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.logger import logger
from app.database.connection import engine, Base
from app.api.endpoints import router as api_router

# Initialize Database Tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Intelligent Medical Tumor AI System Backend...")
    logger.info(f"Database connected: {settings.DATABASE_URL}")
    logger.info(f"Uploads dir: {settings.UPLOADS_DIR}")
    logger.info(f"Reports dir: {settings.REPORTS_DIR}")
    yield
    logger.info("Shutting down Medical Tumor AI System Backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Intelligent Medical Image-Based Tumor Detection and Uncertainty Assessment System API. "
                "Integrates MRI/CT preprocessing, U-Net segmentation, radiomic feature extraction, "
                "Monte Carlo Dropout CNN classification, and Mamdani fuzzy reasoning.",
    lifespan=lifespan
)

# Enable CORS for local Vite development & general access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads and reports directories for direct image & PDF preview
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")
app.mount("/reports", StaticFiles(directory=str(settings.REPORTS_DIR)), name="reports")

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "title": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR,
        "health": f"{settings.API_V1_STR}/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
