import os
import json
import pickle
import pandas as pd

from functools import wraps
from dotenv import load_dotenv
import pyrebase
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
)

# Load .env (Firebase config, etc.)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "disease_prediction_secret_key")

# ----------------------------
# Firebase Configuration
# ----------------------------
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY", "dummy"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "dummy"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", "dummy"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "dummy"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", "dummy"),
    "appId": os.getenv("FIREBASE_APP_ID", "dummy"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL", "https://dummy.firebaseio.com/")
}

# Global placeholders for Firebase components
auth = None
db = None
storage = None

try:
    if all(v and v != "dummy" for v in firebase_config.values()) or os.getenv("TESTING") == "True" or app.testing:
        firebase = pyrebase.initialize_app(firebase_config)
        auth = firebase.auth()
        db = firebase.database()
        storage = firebase.storage()
    else:
        print("[WARN] Firebase config missing or incomplete. Some features will be disabled.")
except Exception as e:
    print(f"[ERROR] Firebase initialization failed: {e}")

# ----------------------------
# Paths & Models
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "notebooks")

with open(os.path.join(MODEL_DIR, "heart_disease_model.sav"), "rb") as f:
    heart_model = pickle.load(f)  # nosec

with open(os.path.join(MODEL_DIR, "diabetes_model.sav"), "rb") as f:
    diabetes_model = pickle.load(f)  # nosec

with open(os.path.join(MODEL_DIR, "parkinsons_model.sav"), "rb") as f:
    parkinsons_model = pickle.load(f)  # nosec


# ----------------------------
# Login Required Decorator
# ----------------------------

def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if app.testing:
            return func(*args, **kwargs)
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

        try:
            if auth is None or db is None:
                raise Exception("Firebase services are not initialized.")

            user = auth.create_user_with_email_and_password(email, password)

            # Store profile data in Realtime Database
            data = {
                "name": name,
                "age": age,
                "sex": sex,
                "email": email,
                "profile_img": ""
            }
            db.child("users").child(user["localId"]).set(data, token=user["idToken"])

            flash("Registration Successful! Please log in.")
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Registration failed: {str(e)}")
            return redirect(url_for("register"))

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

        try:
            if auth is None or db is None:
                raise Exception("Firebase services are not initialized.")

            user = auth.sign_in_with_email_and_password(email, password)
            session["user"] = user["localId"]
            session["email"] = email
            session["idToken"] = user["idToken"]

            # Fetch user info from Realtime Database
            profile = db.child("users").child(user["localId"]).get(token=user["idToken"]).val()
            if profile:
                session["name"] = profile.get("name", "User")
                session["profile_img"] = profile.get("profile_img", "")
            else:
                session["name"] = "User"

            return redirect(url_for("index"))
        except Exception as e:
            print(f"Login error: {e}")
            flash("Invalid Email or Password")
            return redirect(url_for("login"))

    return render_template("login.html")

# ---------------------------------
# Logout
# ---------------------------------

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for("login"))

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user_id = session.get("user")
    id_token = session.get("idToken")
    if db is None or storage is None:
        flash("Firebase services are not initialized.")
        return redirect(url_for("index"))

    user_profile = db.child("users").child(user_id).get(token=id_token).val()

    if request.method == "POST":
        if "profile_image" in request.files:
            file = request.files["profile_image"]
            if file.filename != "":
                try:
                    # Upload to Firebase Storage
                    path = f"profile_images/{user_id}_{file.filename}"
                    storage.child(path).put(file)

                    # Get download URL
                    url = storage.child(path).get_url(None)

                    # Update database and session
                    db.child("users").child(user_id).update({"profile_img": url}, token=id_token)
                    session["profile_img"] = url
                    flash("Profile picture updated!")
                    return redirect(url_for("profile"))
                except Exception as e:
                    flash(f"Upload failed: {str(e)}")

    return render_template("profile.html", user=user_profile)

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
    return render_template("heart.html", prediction_text=prediction_text)

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