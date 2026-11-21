# Intelligent Medical Image-Based Tumor Detection and Uncertainty Assessment System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-cyan.svg)](https://react.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-blueviolet.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic clinical decision-support and artificial intelligence software system that analyzes multi-modal MRI and CT scans, executes automatic tumor segmentation using deep U-Net architectures, performs radiomic feature extraction, classifies neoplasms using deep CNNs, estimates dual epistemic and aleatoric uncertainty via Monte Carlo Dropout, and evaluates risk using a Mamdani fuzzy reasoning engine.

---

## 1. Project Objectives

1. **Medical Image Acquisition & Multimodal Support**: Ingest DICOM (`.dcm`), NIfTI (`.nii`, `.nii.gz`), PNG, and JPEG scans with patient ID anonymization and slice selection.
2. **Medical Image Preprocessing**: Real-time noise suppression (Gaussian & Median filtering), contrast enhancement (CLAHE), and intensity standardization.
3. **Deep Tumor Segmentation**: PyTorch 2D Convolutional U-Net with skip connections, calculating tumor area, perimeter, circularity, boundary irregularity, and Dice/IoU coefficients.
4. **Radiomic Feature Extraction**: Intensity statistics, GLCM texture descriptors (Contrast, Homogeneity, Energy, ASM, Correlation, Entropy), and morphological shape descriptors.
5. **Medical Knowledge Base & Rule Engine**: Structured profiles for Glioma, Meningioma, Pituitary, Normal Brain, and CT lesions with editable inference rules.
6. **Deep CNN Neoplasm Classification**: Multi-class detection and classification with softmax probabilities.
7. **Dual Uncertainty Assessment**:
   - **Probabilistic Reasoning**: Monte Carlo Dropout stochastic passes ($T=15$), predictive variance, Shannon predictive entropy ($H(p)$), and epistemic vs. aleatoric decomposition.
   - **Fuzzy Reasoning**: Mamdani fuzzy logic engine with configurable triangular/trapezoidal membership functions linking model confidence, segmentation quality, boundary clarity, and radiomic consistency.
   - **Abstention & Expert Review Triggering**: Formally flags low-confidence or high-uncertainty scans to avoid false diagnostic claims.
8. **Automated PDF Report Generation**: Full ReportLab clinical report generation with side-by-side transformation previews, radiomic profiles, and disclaimers.
9. **Model Training & Evaluation Dashboard**: Accuracy, sensitivity, specificity, precision, F1-score, confusion matrix, Dice, IoU, loss curves, Expected Calibration Error (ECE), and reliability diagrams.

---

## 2. Technology Stack

### Frontend
- **Framework**: React 19 + TypeScript + Vite
- **Styling**: Tailwind CSS v4 (Clean medical theme: Deep blue `#0f3d64`, Teal `#0d9488`, Slate background)
- **Icons**: Lucide React
- **Data Visualization**: Recharts (Calibration diagrams, loss curves, confusion matrices, probability bars)
- **API Client**: Axios

### Backend
- **Framework**: FastAPI + Starlette
- **Server**: Uvicorn ASGI
- **Data Validation**: Pydantic v2
- **Database & ORM**: SQLite + SQLAlchemy 2.0
- **PDF Generation**: ReportLab 5.0

### Machine Learning & Medical Image Processing
- **Deep Learning**: PyTorch 2.13 (CPU / CUDA compatible)
- **Medical Formats**: PyDicom (DICOM `.dcm`), NiBabel (NIfTI `.nii`, `.nii.gz`)
- **Computer Vision & Filtering**: Pillow, SciPy (`scipy.ndimage`), NumPy
- **Metrics & Calibration**: Scikit-learn, Scipy Stats

---

## 3. System Architecture & Pipeline Workflow

```
[1. INPUT] MRI / CT Scans (DICOM, NIfTI, PNG, JPG)
    │
    ▼
[2. ACQUISITION & ANONYMIZATION]
    │   • Header parsing, voxel normalization, slice selection
    ▼
[3. IMAGE PREPROCESSING]
    │   • Gaussian Filtering (k=3)
    │   • Median Filtering (k=3)
    │   • CLAHE Contrast Enhancement (clip=2.0)
    │   • Intensity Standardization [0.0, 1.0]
    ▼
[4. TUMOR SEGMENTATION & EXTRACTION]
    │   • PyTorch U-Net (Skip Connections)
    │   • Mask, Color Overlay, and Isolated ROI generation
    │   • Area, Perimeter, Circularity, Irregularity
    ▼
[5. RADIOMIC FEATURE EXTRACTION]
    │   • First-Order Intensity (Mean, Std, Skewness)
    │   • GLCM Texture (Contrast, Homogeneity, ASM, Correlation, Entropy)
    │   • Morphometry (Solidity, Aspect Ratio, Circularity)
    ▼
[6. KNOWLEDGE BASE CORRELATION]
    │   • Pathological profiles & rule-based reasoning engine
    ▼
[7. CNN CLASSIFICATION & MC DROPOUT]
    │   • Stochastic forward passes with active dropout
    │   • Mean probabilities, predictive variance
    ▼
[8. DUAL UNCERTAINTY ASSESSMENT]
    │   • Shannon Predictive Entropy & Epistemic/Aleatoric Decomposition
    │   • Mamdani Fuzzy Inference Engine (Confidence, Quality, Boundary, Consistency)
    │   • Safety Abstention & Expert Review Flagging
    ▼
[9. DECISION SUPPORT & PDF REPORT GENERATION]
    │   • Clinical finding synthesis
    │   • Automated downloadable ReportLab PDF
    │   • SQLite Database Archiving
```

---

## 4. Project Folder Structure

```
c:\study\Project 2\
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── endpoints.py          # FastAPI REST endpoints
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py             # System paths and thresholds
│   │   │   └── logger.py             # Structured logger
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── connection.py         # SQLAlchemy engine & session
│   │   │   └── models.py             # AnalysisHistory & ModelMetadata
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── analysis.py           # Pydantic request & response models
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── acquisition.py        # DICOM/NIfTI/Standard image ingestion
│   │   │   ├── preprocessing.py      # Noise removal & CLAHE
│   │   │   ├── segmentation.py       # U-Net lesion segmentation
│   │   │   ├── feature_extraction.py # GLCM and radiomic descriptors
│   │   │   ├── knowledge_base.py     # Knowledge base rule engine
│   │   │   ├── classification.py     # CNN model & MC Dropout passes
│   │   │   ├── fuzzy_engine.py       # Mamdani fuzzy inference engine
│   │   │   ├── uncertainty.py        # Entropy & dual uncertainty synthesis
│   │   │   ├── decision_support.py   # Clinical narrative generator
│   │   │   ├── report_generator.py   # ReportLab PDF generator
│   │   │   └── model_trainer.py      # PyTorch training & evaluation pipeline
│   │   ├── __init__.py
│   │   └── main.py                   # FastAPI main application
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.tsx            # Header & system status
│   │   │   └── Sidebar.tsx           # Clinical navigation
│   │   ├── pages/
│   │   │   ├── DashboardPage.tsx     # Metrics, trends & recent activity
│   │   │   ├── NewScanPage.tsx       # Scan upload, preview, slice selector
│   │   │   ├── ImageProcessingPage.tsx # 7-way side-by-side pipeline views
│   │   │   ├── ClassificationPage.tsx  # Findings, probability bars, uncertainty
│   │   │   ├── KnowledgeBasePage.tsx   # Categories & rule manager
│   │   │   ├── ModelPerformancePage.tsx # Confusion matrix, loss curves, ECE
│   │   │   ├── HistoryPage.tsx       # SQLite history archive & PDF downloads
│   │   │   └── SettingsPage.tsx      # Thresholds & preprocessing controls
│   │   ├── services/
│   │   │   └── api.ts                # Axios client
│   │   ├── types/
│   │   │   └── index.ts              # TypeScript interfaces
│   │   ├── App.tsx                   # Master router & state
│   │   ├── main.tsx
│   │   └── index.css                 # Tailwind CSS styling
│   ├── package.json
│   ├── vite.config.ts                # Proxy setup for /api, /uploads, /reports
│   └── tsconfig.json
├── ml/
│   ├── checkpoints/                  # Trained PyTorch weights (.pt) & metrics (.json)
│   ├── datasets/                     # Dataset storage & import staging
│   ├── evaluation/
│   │   └── metrics.py                # Sensitivity, specificity, ECE, entropy
│   ├── models/
│   │   ├── cnn_classifier.py         # PyTorch CNN with MC Dropout
│   │   └── unet.py                   # PyTorch U-Net architecture
│   └── training/
│       ├── train_classification.py   # CLI training script
│       └── train_segmentation.py     # CLI segmentation script
├── knowledge_base/
│   ├── tumor_knowledge.json          # Medical profiles & clinical rules
│   └── fuzzy_rules.json              # Mamdani membership functions & rules
├── uploads/                          # Patient scan storage & preprocessed slices
├── reports/                          # Generated PDF reports
├── tests/
│   ├── fixtures/                     # Test scan fixtures (labeled synthetic)
│   ├── generate_fixtures.py
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   ├── test_fuzzy_uncertainty.py
│   ├── test_segmentation_classification.py
│   └── test_api.py
├── .env.example
└── README.md
```

---

## 5. Execution Instructions (Windows)

### Prerequisites
- Python 3.10+ (Installed and on PATH)
- Node.js 18+ and npm 10+ (Installed and on PATH)

### Step 1: Install Backend Dependencies
Open PowerShell in the project root:
```powershell
cd "c:\study\Project 2\backend"
python -m pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies
In PowerShell:
```powershell
cd "c:\study\Project 2\frontend"
npm install
```

### Step 3: Train or Verify PyTorch Checkpoints (Optional CLI)
Models come initialized with trained weights in `ml/checkpoints/`. To train further epochs:
```powershell
cd "c:\study\Project 2"
python ml/training/train_classification.py --epochs 5 --modality MRI
```

### Step 4: Run Automated Verification Tests
```powershell
cd "c:\study\Project 2"
python -m pytest tests/
```
*(All 11 tests will execute and pass)*

### Step 5: Start the Backend Server
```powershell
cd "c:\study\Project 2\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/api/health`

### Step 6: Start the Frontend Application
In a second PowerShell window:
```powershell
cd "c:\study\Project 2\frontend"
npm run dev
```
Open your browser at:
`http://localhost:5173`

---

## 6. End-to-End Application Workflow

1. **Dashboard**: Inspect overall scan metrics, diagnostic distribution, and model readiness status.
2. **New Scan Analysis**:
   - Provide an anonymized Patient Reference ID (e.g., `PAT-9021`).
   - Select Modality (`Brain MRI` or `Head CT`).
   - Drag & drop a medical image (`.dcm`, `.nii`, `.nii.gz`, `.png`, or `.jpg`). Sample test fixtures are available in `tests/fixtures/`.
   - For 3D scans, adjust the axial slice slider.
   - Click **Execute Complete AI Pipeline**.
3. **Image Processing**: Visually compare the 7 transformation outputs:
   - Original Scan, Denoised (Gaussian/Median), Enhanced (CLAHE), Normalized (224x224), U-Net Mask, Tumor Overlay, and Cropped ROI.
4. **Classification & Uncertainty Results**:
   - Review the predicted neoplasm category, confidence percentage, and candidate probability bar chart.
   - Inspect Monte Carlo predictive entropy and epistemic model variance.
   - Inspect the Mamdani fuzzy uncertainty rating and view each individual rule that fired.
   - Review the extracted radiomic table and clinical narrative.
   - Click **Download Official PDF Report** to view or print the generated clinical document.
5. **Knowledge Base**: Search tumor categories, inspect radiomic bounds, and edit decision rules.
6. **Model Performance**: Review real empirical validation metrics (Accuracy, Sensitivity, Specificity, Dice, IoU, ECE), view the confusion matrix, and trigger on-demand model retraining.
7. **Analysis History**: Filter previous patient scans by modality or diagnostic outcome, inspect past analyses, and download reports.
8. **Settings**: Adjust Monte Carlo dropout passes ($T$), abstention thresholds, and filter kernels.

---

## 7. Medical & Regulatory Disclaimer

> [!WARNING]
> This application is an **academic research prototype and intelligent decision support system**. It is designed solely for scientific research, algorithmic verification, and academic demonstration. It is **NOT** cleared by the FDA, CE-marked, or certified as a primary medical diagnostic device. It does not replace the specialized judgment of licensed radiologists or medical practitioners. All automated predictions, confidence values, and uncertainty assessments must be reviewed by qualified healthcare professionals.
