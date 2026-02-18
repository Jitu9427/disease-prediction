# Multiple Disease Prediction System

A production-ready Machine Learning pipeline for predicting multiple diseases (Heart Disease, Diabetes, Parkinson's) using Scikit-Learn, MLflow, and DVC.

## Features
- **Modular Architecture**: Separate modules for ingestion, validation, training, evaluation, and registration.
- **Multi-Dataset Support**: Handles Heart, Diabetes, and Parkinson's datasets with specific preprocessing.
- **Experiment Tracking**: Uses MLflow (DAGsHub) for tracking parameters, metrics, and models.
- **Pipeline Orchestration**: Uses DVC (Data Version Control) to manage pipeline stages.
- **Model Registry**: Automated model registration and staging.

## Project Structure
```
├── .dvc/
├── data/
├── notebooks/
├── src/
│   ├── data_pre/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   ├── model/
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   ├── model_register.py
│   ├── logger.py
│   ├── exception.py
│   ├── utils.py
│   └── pipeline.py
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- Kaggle Account (for dataset download)
- DAGsHub Account (for MLflow tracking)

### 2. Installation
```bash
git clone <repository_url>
cd disease-prediction
pip install -r requirements.txt
pip install -e .
```

### 3. Configuration
1. **Kaggle API**: Place your `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.
2. **MLflow / DAGsHub**:
   Set environment variables for authentication:
   ```bash
   export MLFLOW_TRACKING_URI="https://dagshub.com/<your_username>/<repo_name>.mlflow"
   export MLFLOW_TRACKING_USERNAME="<your_username>"
   export MLFLOW_TRACKING_PASSWORD="<your_token>"
   ```
   *Note: Using DAGsHub token is recommended. Find it in DAGsHub Settings -> Tokens.*
   
   **Verification:**
   Run `python debug_dagshub.py` to check your connection.

### 4. Running the Pipeline
You can run the entire pipeline using DVC or the Python entry point.

**Using Python:**
```bash
python -m src.pipeline
```

**Using DVC:**
```bash
dvc repro
```

## Troubleshooting

### DAGsHub Authentication Errors
If you see 404 or Authentication errors:
1. Ensure `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are set correctly.
2. Ensure you have write access to the repository specified in `MLFLOW_TRACKING_URI`.
3. If running locally without DAGsHub, set `MLFLOW_TRACKING_URI` to a local path:
   ```bash
   export MLFLOW_TRACKING_URI="file:./mlruns"
   ```

### Registry Errors
If you see "Unable to find a logged_model", ensure that the training step completed successfully and that you are using a consistent Tracking URI across all steps.

## Components

- **Data Ingestion**: Downloads data from Kaggle and splits it into train/test sets.
- **Data Validation**: Validates the schema and data integrity.
- **Model Training**: Trains specific models (Logistic Regression, SVM) for each disease dataset and logs to MLflow.
- **Model Evaluation**: Evaluates models on test data and logs metrics (Accuracy, Precision, Recall, F1).
- **Model Registration**: Registers the best models in the MLflow Model Registry and transitions to Staging.

## Configuration
Adjust model parameters and dataset settings in `params.yaml`.

## License
MIT