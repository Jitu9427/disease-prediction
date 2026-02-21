import os
import pickle
import numpy as np
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load the trained models
# Assuming models are in 'artifacts/models' relative to project root
# If running from 'd:\disease-prediction', the path is 'artifacts/models'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")

print(f"Loading models from: {MODELS_DIR}")

def load_model(filename):
    try:
        path = os.path.join(MODELS_DIR, filename)
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file {filename} not found.")
        return None

try:
    heart_model = load_model("heart-disease-dataset_model.pkl")
    diabetes_model = load_model("diabities_model.pkl")
    parkinsons_model = load_model("parkinsons_model.pkl")
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    heart_model = None
    diabetes_model = None
    parkinsons_model = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/heart", methods=["GET", "POST"])
def heart():
    prediction_text = ""
    if request.method == "POST":
        try:
            # Get input values from form
            if not heart_model:
                return render_template("heart.html", prediction_text="Error: Model not loaded.")

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
            
            final_features = [np.array(features)]
            prediction = heart_model.predict(final_features)
            
            if prediction[0] == 1:
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
            if not diabetes_model:
                return render_template("diabetes.html", prediction_text="Error: Model not loaded.")

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
            
            final_features = [np.array(features)]
            prediction = diabetes_model.predict(final_features)
            
            if prediction[0] == 1:
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
            if not parkinsons_model:
                return render_template("parkinsons.html", prediction_text="Error: Model not loaded.")

            # Note: The input features must match the order during training
            # Based on the CSV header:
            # MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),
            # MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,MDVP:Shimmer(dB),
            # Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,NHR,HNR,
            # RPDE,DFA,spread1,spread2,D2,PPE
            
            feature_names = [
                "fo", "fhi", "flo", "jitter_percent", "jitter_abs", "rap", "ppq", "ddp",
                "shimmer", "shimmer_db", "apq3", "apq5", "apq", "dda", "nhr", "hnr",
                "rpde", "dfa", "spread1", "spread2", "d2", "ppe"
            ]
            
            features = [float(request.form[name]) for name in feature_names]
            
            final_features = [np.array(features)]
            prediction = parkinsons_model.predict(final_features)
            
            if prediction[0] == 1:
                prediction_text = "The person has Parkinson's Disease."
            else:
                prediction_text = "The person does not have Parkinson's Disease."
                
        except Exception as e:
            prediction_text = f"Error in prediction: {e}"

    return render_template("parkinsons.html", prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
