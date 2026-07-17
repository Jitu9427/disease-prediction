import os
import json
import pickle
import pandas as pd

from functools import wraps
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
)

app = Flask(__name__)
app.secret_key = "disease_prediction_secret_key"

# ----------------------------
# Paths
# ----------------------------
# app.py is inside: disease-prediction-main/app/app.py
# Models are inside: disease-prediction-main/notebooks/

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "notebooks")
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

# ----------------------------
# Load Models (.sav files)
# ----------------------------

with open(os.path.join(MODEL_DIR, "heart_disease_model.sav"), "rb") as f:
    heart_model = pickle.load(f)  # nosec

with open(os.path.join(MODEL_DIR, "diabetes_model.sav"), "rb") as f:
    diabetes_model = pickle.load(f)  # nosec

with open(os.path.join(MODEL_DIR, "parkinsons_model.sav"), "rb") as f:
    parkinsons_model = pickle.load(f)  # nosec

# ----------------------------
# User Functions
# ----------------------------

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# ----------------------------
# Login Required Decorator
# ----------------------------

def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper

# ---------------------------------
# Root
# ---------------------------------

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("index"))

# ---------------------------------
# Register
# ---------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        name     = request.form["name"]
        age      = request.form["age"]
        sex      = request.form["sex"]
        email    = request.form["email"]
        password = request.form["password"]

        users = load_users()

        if email in users:
            flash("Email already registered!")
            return redirect(url_for("register"))

        users[email] = {
            "name": name,
            "age": age,
            "sex": sex,
            "email": email,
            "password": password
        }
        save_users(users)

        flash("Registration Successful! Please log in.")
        return redirect(url_for("login"))

    return render_template("register.html")

# ---------------------------------
# Login
# ---------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        email    = request.form["email"]
        password = request.form["password"]

        users = load_users()

        if email in users and users[email]["password"] == password:
            session["user"] = email
            session["name"] = users[email]["name"]
            return redirect(url_for("index"))

        flash("Invalid Email or Password")

    return render_template("login (1).html")

# ---------------------------------
# Logout
# ---------------------------------

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for("login"))

# ---------------------------------
# Home Page
# ---------------------------------

@app.route("/home")
@login_required
def index():
    return render_template("index.html", username=session["name"])

# ---------------------------------
# Heart Disease
# ---------------------------------

@app.route("/heart", methods=["GET", "POST"])
@login_required
def heart():
    prediction_text = None
    if request.method == "POST":
        try:
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
            prediction = heart_model.predict([features])[0]
            if prediction == 1:
                prediction_text = "The person has Heart Disease."
            else:
                prediction_text = "The person does not have Heart Disease."
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
    return render_template("heart (1).html", prediction_text=prediction_text)

# ---------------------------------
# Diabetes
# ---------------------------------

@app.route("/diabetes", methods=["GET", "POST"])
@login_required
def diabetes():
    prediction_text = None
    if request.method == "POST":
        try:
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
            prediction = diabetes_model.predict([features])[0]
            if prediction == 1:
                prediction_text = "The person is Diabetic."
            else:
                prediction_text = "The person is not Diabetic."
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
    return render_template("diabetes.html", prediction_text=prediction_text)

# ---------------------------------
# Parkinson's
# ---------------------------------

@app.route("/parkinsons", methods=["GET", "POST"])
@login_required
def parkinsons():
    prediction_text = None
    if request.method == "POST":
        try:
            features = [
                float(request.form["fo"]),
                float(request.form["fhi"]),
                float(request.form["flo"]),
                float(request.form["jitter_percent"]),
                float(request.form["jitter_abs"]),
                float(request.form["rap"]),
                float(request.form["ppq"]),
                float(request.form["ddp"]),
                float(request.form["shimmer"]),
                float(request.form["shimmer_db"]),
                float(request.form["apq3"]),
                float(request.form["apq5"]),
                float(request.form["apq"]),
                float(request.form["dda"]),
                float(request.form["nhr"]),
                float(request.form["hnr"]),
                float(request.form["rpde"]),
                float(request.form["dfa"]),
                float(request.form["spread1"]),
                float(request.form["spread2"]),
                float(request.form["d2"]),
                float(request.form["ppe"]),
            ]
            prediction = parkinsons_model.predict([features])[0]
            if prediction == 1:
                prediction_text = "The person has Parkinson's Disease."
            else:
                prediction_text = "The person does not have Parkinson's Disease."
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
    return render_template("parkinsons.html", prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)  # nosec