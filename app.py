"""
Flask backend for the Heart Disease Detection System.

How to run:
    1. Train and save the model:  python train_model.py
    2. Start the web server:       python app.py
    3. Open in browser:            http://127.0.0.1:5000
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

# Create the Flask application
app = Flask(__name__)
# Secret key is required for flash messages (temporary user notifications)
app.secret_key = "change-this-secret-key-in-production"

# Feature names must match the HTML form fields and train_model.py
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

# Loaded once at startup (set by load_artifacts)
model = None
scaler = None


def load_artifacts():
    """
    Load the trained model and scaler from the models/ folder using joblib.

    Returns:
        tuple: (model, scaler)

    Raises:
        FileNotFoundError: if training files are missing
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run 'python train_model.py' first."
        )
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found at {SCALER_PATH}. "
            "Please run 'python train_model.py' again to save the scaler."
        )

    trained_model = joblib.load(MODEL_PATH)
    fitted_scaler = joblib.load(SCALER_PATH)
    return trained_model, fitted_scaler


def parse_form_inputs():
    """
    Read the HTML form and convert every field to a float (numerical format).

    Returns:
        tuple: (values_dict, error_message)
               error_message is None when parsing succeeds
    """
    values = {}

    for field_name in FEATURE_COLUMNS:
        raw_value = request.form.get(field_name, "").strip()

        # Check for empty fields
        if raw_value == "":
            return None, f"Please fill in all fields. Missing: {field_name}"

        # Convert text from the form into a number the model can use
        try:
            values[field_name] = float(raw_value)
        except ValueError:
            return None, f"Invalid number for '{field_name}'. Please enter numeric values only."

    return values, None


def make_prediction(values):
    """
    Scale patient data and run the machine learning model.

    Args:
        values: dict of feature name -> float

    Returns:
        tuple: (prediction_label, probability_percent, prediction_class)
    """
    # Build one row of data in the same column order used during training
    feature_row = np.array([[values[col] for col in FEATURE_COLUMNS]])

    # Apply the same scaling used in train_model.py
    scaled_features = scaler.transform(feature_row)

    # predict() returns 0 or 1; predict_proba() returns probabilities for each class
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
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Home page — show the patient input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction route — read form data, run the model, show the result.
    Only accepts POST requests (form submission).
    """
    try:
        # Step 1: Convert form inputs to numbers
        values, parse_error = parse_form_inputs()
        if parse_error:
            flash(parse_error, "danger")
            return redirect(url_for("home"))

        # Step 2: Run the trained model
        result_label, probability, result_class = make_prediction(values)

        # Step 3: Show the result page
        return render_template(
            "result.html",
            result_label=result_label,
            result_class=result_class,
            probability=probability,
            values=values,
        )

    except FileNotFoundError as error:
        flash(str(error), "danger")
        return redirect(url_for("home"))

    except Exception as error:
        # Catch unexpected errors so the user sees a friendly message
        flash(f"Prediction failed: {error}", "danger")
        return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# Start the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        model, scaler = load_artifacts()
        print("Model and scaler loaded successfully.")
        print(f"Model path: {MODEL_PATH}")
        app.run(debug=True)
    except FileNotFoundError as error:
        print(f"Error: {error}")
