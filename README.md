# Multiple Disease Prediction System

A production-ready Machine Learning project that predicts multiple diseases (Heart Disease, Diabetes, and Parkinson's) using Scikit-Learn with a full MLOps pipeline (MLflow + DVC) and a Flask web application for real-time predictions.

---

## 🚀 Features

- **Flask Web App** – Interactive web interface to get disease predictions in real time.
- **3 Disease Predictions** – Heart Disease, Diabetes, and Parkinson's Disease.
- **Modular ML Pipeline** – Separate modules for data ingestion, validation, training, evaluation, and model registration.
- **Experiment Tracking** – MLflow + DAGsHub for tracking parameters, metrics, and models.
- **Pipeline Orchestration** – DVC manages and reproduces every pipeline stage.
- **Model Registry** – Automated model registration and staging via MLflow Model Registry.
- **Multi-Dataset Support** – Handles Heart, Diabetes, and Parkinson's datasets with specific preprocessing.

---

## 🖼️ App Pages

| Route | Description |
|---|---|
| `/` | Home page with links to all disease predictors |
| `/heart` | Heart Disease prediction form (13 features) |
| `/diabetes` | Diabetes prediction form (8 features) |
| `/parkinsons` | Parkinson's Disease prediction form (22 voice features) |

---

## 📁 Project Structure

```
disease-prediction/
├── app/
│   ├── app.py                    # Flask web application
│   ├── static/
│   │   └── style.css             # App stylesheet
│   └── templates/
│       ├── index.html            # Home page
│       ├── heart.html            # Heart disease prediction page
│       ├── diabetes.html         # Diabetes prediction page
│       └── parkinsons.html       # Parkinson's prediction page
├── artifacts/
│   ├── models/                   # Trained .pkl model files
│   └── metrics/                  # Evaluation metrics (metrics.json)
├── data/
│   ├── raw/                      # Raw datasets downloaded from Kaggle
│   └── processed/                # Train/test split CSVs
├── notebooks/
│   ├── Multiple disease prediction system - heart.ipynb
│   ├── Multiple disease prediction system - diabetes.ipynb
│   └── Multiple disease prediction system - Parkinsons.ipynb
├── src/
│   ├── data_pre/
│   │   ├── data_ingestion.py     # Downloads data from Kaggle, splits train/test
│   │   └── data_validation.py    # Validates schema and data integrity
│   ├── model/
│   │   ├── model_training.py     # Trains models and logs to MLflow
│   │   ├── model_evaluation.py   # Evaluates models, logs metrics
│   │   └── model_register.py     # Registers best models to MLflow Registry
│   ├── pipeline.py               # End-to-end pipeline entry point
│   ├── utils.py                  # Shared utility functions
│   ├── logger.py                 # Logging configuration
│   └── exception.py              # Custom exception handling
├── dvc.yaml                      # DVC pipeline stage definitions
├── dvc.lock                      # DVC lock file
├── params.yaml                   # Model & data configuration
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Kaggle Account (for dataset download)
- DAGsHub Account (for MLflow tracking)

### 2. Clone & Install

```bash
git clone <repository_url>
cd disease-prediction
pip install -r requirements.txt
pip install -e .
```

### 3. Configuration

**Kaggle API:**
Place your `kaggle.json` in `~/.kaggle/` or set environment variables:
```bash
export KAGGLE_USERNAME="<your_username>"
export KAGGLE_KEY="<your_api_key>"
```

**MLflow / DAGsHub:**
```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<your_username>/<repo_name>.mlflow"
export MLFLOW_TRACKING_USERNAME="<your_username>"
export MLFLOW_TRACKING_PASSWORD="<your_dagshub_token>"
```
> 💡 Find your DAGsHub token at: **DAGsHub → Settings → Tokens**

You can also place these in a `.env` file at the project root.

---

## 🔄 DVC Pipeline

The ML pipeline has 5 stages managed by DVC:

| Stage | Command | Description |
|---|---|---|
| `data_ingestion` | `python -m src.data_pre.data_ingestion` | Downloads datasets from Kaggle, splits into train/test |
| `data_validation` | `python -m src.data_pre.data_validation` | Validates schema and data integrity |
| `model_training` | `python -m src.model.model_training` | Trains models, logs to MLflow |
| `model_evaluation` | `python -m src.model.model_evaluation` | Evaluates models, saves metrics |
| `model_register` | `python -m src.model.model_register` | Registers best models to MLflow Registry |

**Run the full pipeline:**
```bash
# Using DVC (recommended)
dvc repro

# Using Python
python -m src.pipeline
```

---

## 🤖 Models

| Disease | Model | Target Column | Key Params |
|---|---|---|---|
| Heart Disease | Logistic Regression | `target` | `solver: liblinear` |
| Diabetes | SVM | `Outcome` | `kernel: linear` |
| Parkinson's | SVM | `status` | `kernel: linear` |

Datasets are sourced from Kaggle:
- `johnsmith88/heart-disease-dataset`
- `aarikethjariwala/diabities`
- `imkrkannan/parkinsons`

---

## 🌐 Running the Flask Web App

Make sure the pipeline has been run first so model `.pkl` files exist in `artifacts/models/`.

```bash
cd app
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

### App Routes & Input Features

**Heart Disease** (`/heart`) — 13 features:
`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

**Diabetes** (`/diabetes`) — 8 features:
`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

**Parkinson's** (`/parkinsons`) — 22 voice features:
`fo`, `fhi`, `flo`, `jitter_percent`, `jitter_abs`, `rap`, `ppq`, `ddp`, `shimmer`, `shimmer_db`, `apq3`, `apq5`, `apq`, `dda`, `nhr`, `hnr`, `rpde`, `dfa`, `spread1`, `spread2`, `d2`, `ppe`

---

## 🔧 Troubleshooting

### DAGsHub Authentication Errors
If you see 404 or authentication errors:
1. Verify `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are set correctly.
2. Ensure you have write access to the repository in `MLFLOW_TRACKING_URI`.
3. For local-only usage, set:
   ```bash
   export MLFLOW_TRACKING_URI="file:./mlruns"
   ```

### Model Not Loading in App
If you see **"Error: Model not loaded"** in the web app:
- Ensure the DVC pipeline has been run (`dvc repro`) so `.pkl` files exist in `artifacts/models/`.
- Check that filenames match: `heart-disease-dataset_model.pkl`, `diabities_model.pkl`, `parkinsons_model.pkl`.

### Registry Errors
If you see `"Unable to find a logged_model"`:
- Ensure the training step completed successfully.
- Use a consistent `MLFLOW_TRACKING_URI` across all pipeline steps.

---

## 📦 Dependencies

```
pandas, numpy, scikit-learn, mlflow, dagshub, pyyaml, kaggle, dvc, python-dotenv, flask
```

---

## 📄 License

MIT