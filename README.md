# ✈️ Flight Delay Detective
### AI-Powered Flight Analytics & Prediction System
*Internship Project — Flight Delay Dataset 2024 (Kaggle)*

---

## Project Overview

Flight Delay Detective is an end-to-end data science and AI system that ingests 7+ million US domestic flight records, performs exploratory data analysis, trains a machine-learning classifier to predict departure delays, and exposes the results through a FastAPI backend and a browser-based dashboard. A **Grounded Analyst** component answers natural-language questions about the data using only pre-computed JSON statistics — no hallucinated numbers.

---

## Repository Structure

```
Flight-Delay-Detective/
│
├── data/
│   ├── processed/
│   │   ├── dev_sample.csv          # 5% stratified sample (348,266 rows)
│   │   └── metadata.json           # Processing stats
│   ├── analytics/
│   │   └── eda_stats.json          # All EDA computed statistics
│   └── metrics/
│       ├── val_metrics.json        # Final model metrics
│       └── model_comparison.json   # All 5-model comparison results
│
├── models/
│   ├── best_model.joblib           # GradientBoostingClassifier (2.2 MB)
│   └── best_encoders.joblib        # OOF target encoders (10 KB)
│
├── reports/
│   └── figures/                    # 15 PNG plots (EDA, CM, feature importance)
│
├── src/
│   ├── data/
│   │   └── processing.py           # Chunked CSV pipeline, dev sample creation
│   ├── eda/
│   │   └── eda.py                  # EDA plots and analytics JSON
│   └── ml/
│       ├── train.py                # RandomForest baseline training
│       ├── experiment.py           # 5-model comparison experiment
│       └── verify_artifacts.py     # End-to-end artifact verification
│
├── backend/
│   ├── main.py                     # FastAPI app (3 endpoints, CORS)
│   └── analyst.py                  # Grounded analyst engine
│
├── frontend/
│   └── index.html                  # Single-file dashboard (HTML/CSS/JS)
│
├── flight_data_2024.csv            # Raw dataset (DO NOT MODIFY)
├── FlightDelay_Project.ipynb       # Full project notebook
├── FlightDelay_Project.docx        # Internship report
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Flight Delay Dataset 2024 |
| Raw rows | 7,079,081 |
| Valid rows (after cleaning) | 6,965,267 |
| Dev sample | 348,266 rows (5% stratified) |
| Overall delay rate | 20.82% |
| Key columns | `fl_date`, `airline`, `origin`, `dest`, `dep_time`, `arr_time`, `dep_delay`, `arr_delay`, `cancelled`, `diverted`, `distance`, delay cause columns |

---

## Machine Learning

### Task
Binary classification: **Will this flight be delayed by ≥ 15 minutes at departure?**

### Models Evaluated
| Model | Val ROC-AUC |
|---|---|
| Logistic Regression | ~58% |
| Decision Tree | ~56% |
| Random Forest (baseline) | ~62% |
| GradientBoosting (default) | ~61% |
| **GradientBoosting (tuned) ✅** | **62.13%** |

### Final Model — GradientBoostingClassifier
| Metric | Value |
|---|---|
| ROC-AUC | **62.13%** |
| Recall | **27.80%** |
| Precision | **22.21%** |
| F1 Score | **24.69%** |
| Accuracy | **76.41%** |

**Confusion Matrix (validation set)**

|  | Predicted Not Delayed | Predicted Delayed |
|---|---|---|
| Actual Not Delayed | 42,814 (TN) | 7,997 (FP) |
| Actual Delayed | 5,929 (FN) | 2,283 (TP) |

### Top Features by Importance
| Feature | Importance |
|---|---|
| arr_hour | 17.76% |
| dep_hour | 17.51% |
| month | 15.64% |

### Leakage Warning
`dep_delay`, `arr_delay`, `arr_time`, `taxi_out`, and all delay-cause columns (`carrier_delay`, `weather_delay`, etc.) are **excluded** from features — they are only known after the flight departs and would constitute data leakage.

---

## API Endpoints

Start the backend:
```bash
cd backend
uvicorn main:app --reload --port 8000
```

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Predict delay probability for a flight |
| GET | `/analyse?question=...` | Grounded analyst — answers from JSON stats |

### Example `/predict` request
```json
{
  "airline": "AA",
  "origin": "DFW",
  "dest": "ORD",
  "month": 7,
  "day_of_week": 5,
  "dep_hour": 17,
  "distance": 802
}
```

### Example `/predict` response
```json
{
  "prob_delayed": 0.821962,
  "prediction": "delayed",
  "confidence": "high"
}
```

---

## Frontend Dashboard

Open `frontend/index.html` directly in a browser (or serve it with any static server).

Features:
- Flight input form (airline, route, date/time, distance)
- Real-time delay prediction with probability bar
- Grounded Analyst chat panel — ask questions about delay statistics
- All answers sourced from pre-computed `eda_stats.json` and `val_metrics.json`

---

## Grounded Analyst

The analyst component (`backend/analyst.py`) answers questions about flight delays using **only** pre-computed statistics stored in `data/analytics/eda_stats.json` and `data/metrics/val_metrics.json`. It never calls an LLM and never invents numbers — every figure in every answer is traceable to a computed JSON value.

Example questions:
- *"What is the overall delay rate?"*
- *"Which airline has the most delays?"*
- *"What time of day has the highest delay rate?"*
- *"How accurate is the model?"*

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the data pipeline (optional — artifacts already present)
```bash
python src/data/processing.py
python src/eda/eda.py
python src/ml/experiment.py
```

### Start the backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### Open the frontend
Open `frontend/index.html` in your browser, or:
```bash
python -m http.server 3000 --directory frontend
```
Then visit `http://localhost:3000`.

---

## Key Design Decisions

1. **Chunked processing** — the 7M-row CSV is read in 50,000-row chunks to avoid OOM errors.
2. **Stratified sampling** — the 5% dev sample preserves the original 20.82% delay rate.
3. **OOF target encoding** — categorical columns (`airline`, `origin`, `dest`) are encoded using out-of-fold target means to prevent leakage during cross-validation.
4. **Leakage-free features** — all post-departure columns are excluded from model training.
5. **Grounded AI** — the analyst is 100% grounded in computed JSON; no generative hallucination risk.

---

## Requirements

```
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
matplotlib>=3.7
seaborn>=0.12
numpy>=1.24
fastapi>=0.110
uvicorn[standard]>=0.29
httpx>=0.27
python-multipart>=0.0.9
pydantic>=2.0
```

---

## Project Timeline

| Phase | Description | Status |
|---|---|---|
| 1 | Dataset inspection & architecture planning | ✅ Complete |
| 2 | Data pipeline & dev sample | ✅ Complete |
| 3 | Exploratory Data Analysis | ✅ Complete |
| 4 | ML baseline & experiment | ✅ Complete |
| 5 | FastAPI backend | ✅ Complete |
| 6 | Frontend dashboard | ✅ Complete |
| 7 | Grounded Analyst | ✅ Complete |
| 8 | Notebook & report packaging | ✅ Complete |

---

## Limitations

- Model ROC-AUC of 62.13% reflects the inherent difficulty of predicting delays without real-time weather and ATC data.
- The dev sample covers only 5% of the full dataset; retraining on the full 7M rows would likely improve performance.
- The grounded analyst uses keyword matching, not semantic understanding; ambiguous questions may not match any template.
- The frontend requires the backend to be running locally on port 8000.

---

## Author

Internship Project — Flight Delay Detective  
Dataset: [Kaggle Flight Delay Dataset 2024](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)
