import os
import sys
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request

# ── Load .env (MLFLOW_TRACKING_URI, credentials) ──────────────────────────────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

import mlflow
import mlflow.pyfunc

# ── DAGsHub / MLflow auth setup ───────────────────────────────────────────────
def _setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if "dagshub.com" in tracking_uri:
        try:
            import dagshub
            parts = tracking_uri.split("dagshub.com/")[-1].split(".mlflow")[0].split("/")
            if len(parts) == 2:
                owner, repo = parts
                token = os.getenv("MLFLOW_TRACKING_PASSWORD")
                if token:
                    os.environ["DAGSHUB_USER_TOKEN"] = token
                dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
                return
        except Exception as e:
            print(f"[WARN] dagshub init failed: {e}")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

_setup_mlflow()

# ── Load models from MLflow Registry (alias = "Staging") ─────────────────────
#    URI format:  models:/<RegisteredModelName>@<alias>
#    mlflow.pyfunc.load_model returns a PythonModel wrapper that exposes .predict()

def _load_from_registry(model_name: str):
    """Fetch the version aliased as 'Staging' from the MLflow Model Registry."""
    uri = f"models:/{model_name}@Staging"
    try:
        print(f"[MLflow] Loading '{model_name}' from {uri} ...")
        model = mlflow.pyfunc.load_model(uri)
        print(f"[MLflow] ✅ Loaded '{model_name}' successfully.")
        return model
    except Exception as e:
        print(f"[MLflow] ❌ Failed to load '{model_name}': {e}", file=sys.stderr)
        return None


# Load all three models at startup
heart_model      = _load_from_registry("Model_heart-disease-dataset")
diabetes_model   = _load_from_registry("Model_diabities")
parkinsons_model = _load_from_registry("Model_parkinsons")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/heart", methods=["GET", "POST"])
def heart():
    prediction_text = ""
    if request.method == "POST":
        try:
            if heart_model is None:
                return render_template("heart.html", prediction_text="Error: Heart model not loaded from MLflow.")

            features = [
                float(request.form["age"]),
                float(request.form["sex"]),
                float(request.form["cp"]),
                float(request.form["trestbps"]),
                float(request.form["chol"]),
                float(request.form["fbs"]),
                float(request.form["restecg"]),
                float(request.form["thalach"]),
                float(request.form["exang"]),
                float(request.form["oldpeak"]),
                float(request.form["slope"]),
                float(request.form["ca"]),
                float(request.form["thal"]),
            ]

            import pandas as pd
            input_df = pd.DataFrame([features])
            prediction = heart_model.predict(input_df)

            if int(prediction[0]) == 1:
                prediction_text = "The person has Heart Disease."
            else:
                prediction_text = "The person does not have Heart Disease."

        except Exception as e:
            prediction_text = f"Error in prediction: {e}"

    return render_template("heart.html", prediction_text=prediction_text)


@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    prediction_text = ""
    if request.method == "POST":
        try:
            if diabetes_model is None:
                return render_template("diabetes.html", prediction_text="Error: Diabetes model not loaded from MLflow.")

            features = [
                float(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                float(request.form["Age"]),
            ]

            import pandas as pd
            input_df = pd.DataFrame([features])
            prediction = diabetes_model.predict(input_df)

            if int(prediction[0]) == 1:
                prediction_text = "The person is Diabetic."
            else:
                prediction_text = "The person is not Diabetic."

        except Exception as e:
            prediction_text = f"Error in prediction: {e}"

    return render_template("diabetes.html", prediction_text=prediction_text)


@app.route("/parkinsons", methods=["GET", "POST"])
def parkinsons():
    prediction_text = ""
    if request.method == "POST":
        try:
            if parkinsons_model is None:
                return render_template("parkinsons.html", prediction_text="Error: Parkinson's model not loaded from MLflow.")

            feature_names = [
                "fo", "fhi", "flo", "jitter_percent", "jitter_abs", "rap", "ppq", "ddp",
                "shimmer", "shimmer_db", "apq3", "apq5", "apq", "dda", "nhr", "hnr",
                "rpde", "dfa", "spread1", "spread2", "d2", "ppe",
            ]

            features = [float(request.form[name]) for name in feature_names]

            import pandas as pd
            input_df = pd.DataFrame([features])
            prediction = parkinsons_model.predict(input_df)

            if int(prediction[0]) == 1:
                prediction_text = "The person has Parkinson's Disease."
            else:
                prediction_text = "The person does not have Parkinson's Disease."

        except Exception as e:
            prediction_text = f"Error in prediction: {e}"

    return render_template("parkinsons.html", prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
