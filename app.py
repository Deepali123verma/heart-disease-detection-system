
"""
Flask backend for the Heart Disease Detection System.

How to run:
    1. Train and save the model: python train_model.py
    2. Start the web server: python app.py
    3. Open: http://127.0.0.1:5000
"""

from pathlib import Path

import joblib
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for

# ---------------------------------------------------------------------------
# Paths and Flask app setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

app = Flask(__name__)

app.secret_key = "change-this-secret-key-in-production"

# Feature names must match train_model.py and HTML form fields
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# ---------------------------------------------------------------------------
# Load trained model and scaler
# ---------------------------------------------------------------------------

def load_artifacts():
    """
    Load the trained model and scaler from the models folder.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Please run train_model.py first."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}. "
            "Please run train_model.py again."
        )

    trained_model = joblib.load(MODEL_PATH)
    fitted_scaler = joblib.load(SCALER_PATH)

    if not hasattr(fitted_scaler, "transform"):
        raise TypeError("Loaded scaler is invalid.")

    if not hasattr(trained_model, "predict"):
        raise TypeError("Loaded ML model is invalid.")

    return trained_model, fitted_scaler


# Load artifacts when the Flask application is imported.
# This is important for Vercel and Gunicorn deployments.

try:
    model, scaler = load_artifacts()
    print("Model and scaler loaded successfully.")

except (FileNotFoundError, TypeError, ValueError, OSError) as error:
    print(f"Model loading error: {error}")
    model = None
    scaler = None


# ---------------------------------------------------------------------------
# Parse HTML form inputs
# ---------------------------------------------------------------------------

def parse_form_inputs():

    values = {}

    for field_name in FEATURE_COLUMNS:

        raw_value = request.form.get(field_name, "").strip()

        if raw_value == "":
            return None, f"Please fill in all fields. Missing: {field_name}"

        try:
            values[field_name] = float(raw_value)

        except ValueError:
            return None, (
                f"Invalid number for '{field_name}'. "
                "Please enter numeric values only."
            )

    return values, None


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def make_prediction(values):

    if model is None or scaler is None:
        raise RuntimeError(
            "ML model or scaler is not loaded. "
            "Please check the models folder and deployment logs."
        )

    # Prepare input in the exact training feature order
    feature_row = np.array(
        [[values[col] for col in FEATURE_COLUMNS]],
        dtype=float
    )

    # Apply the trained scaler
    scaled_features = scaler.transform(feature_row)

    # Generate prediction
    prediction = int(model.predict(scaled_features)[0])

    probabilities = model.predict_proba(scaled_features)[0]

    confidence = float(probabilities[prediction]) * 100

    if prediction == 1:
        label = "Heart Disease Detected"
        css_class = "danger"

    else:
        label = "No Heart Disease Detected"
        css_class = "success"

    return label, round(confidence, 1), css_class


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Step 1: Parse patient inputs
        values, parse_error = parse_form_inputs()

        if parse_error:
            flash(parse_error, "danger")
            return redirect(url_for("home"))

        # Step 2: Generate ML prediction
        result_label, probability, result_class = make_prediction(values)

        # Step 3: Display result
        return render_template(
            "result.html",
            result_label=result_label,
            result_class=result_class,
            probability=probability,
            values=values,
        )

    except Exception as error:

        app.logger.exception("Prediction failed")

        flash(f"Prediction failed: {error}", "danger")

        return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# Start Flask server (Local development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    app.run(debug=False)

