import os
import sys
import numpy as np
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pyrebase

# ── Load .env (MLFLOW_TRACKING_URI, credentials, Firebase config) ──────────────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

import mlflow
import mlflow.pyfunc

# ── Firebase Configuration ────────────────────────────────────────────────────
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL")
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

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

# ── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "medi-predict-fallback-key-123")

# ── Auth Decorator ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to access this feature.")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ── Auth Routes ───────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session["user"] = user["localId"]
            session["email"] = email
            
            # Fetch user name from Realtime Database
            profile = db.child("users").child(user["localId"]).get().val()
            if profile:
                session["name"] = profile.get("name", "User")
            else:
                session["name"] = "User"
                
            return redirect(url_for("index"))
        except Exception as e:
            print(f"Login error: {e}")
            flash("Invalid email or password.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        sex = request.form.get("sex")
        email = request.form.get("email")
        password = request.form.get("password")
        try:
            user = auth.create_user_with_email_and_password(email, password)
            
            # Store profile data in Realtime Database
            data = {"name": name, "age": age, "sex": sex, "email": email}
            db.child("users").child(user["localId"]).set(data)
            
            flash("Account created successfully! Please log in.")
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Registration failed: {str(e)}")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("email", None)
    session.pop("name", None)
    return redirect(url_for("index"))

# ── App Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/heart", methods=["GET", "POST"])
@login_required
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
@login_required
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
@login_required
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
