# Heart Disease Detection System

A beginner-friendly web application that predicts heart disease risk using **Python**, **Flask**, **Scikit-learn**, **HTML**, **CSS**, and **Bootstrap**.

> **Note:** This project is for learning only. It is not a substitute for professional medical diagnosis.

---

## Project Structure

```
Heart_disease_detection_system/
│
├── app.py                 # Flask web server and prediction routes
├── train_model.py         # Trains the ML model and saves it to models/
├── requirements.txt     # Python dependencies
├── README.md              # This file
│
├── dataset/               # Training data (CSV files)
│   └── heart_disease.csv
│
├── models/                # Saved trained model (created after training)
│   └── heart_model.pkl
│
├── templates/             # HTML pages (Jinja2 templates for Flask)
│   ├── index.html         # Input form
│   └── result.html        # Prediction result
│
└── static/                # CSS, images, JavaScript
    └── css/
        └── style.css
```

---

## How It Works

1. **`train_model.py`** reads `dataset/heart_disease.csv`, trains a Random Forest classifier, and saves `models/heart_model.pkl`.
2. **`app.py`** loads the saved model and serves a web form.
3. When you submit the form, Flask runs a prediction and shows the result in `templates/result.html`.

---

## Setup Instructions

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

You should see accuracy metrics and a message that the model was saved to `models/heart_model.pkl`.

### 4. Run the web app

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## Using a Larger Dataset

The included `heart_disease.csv` is a small sample so you can run the project immediately. For better accuracy, replace it with the full [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) (Cleveland format) and keep the same column names.

---

## Technologies Used

| Layer        | Technology        |
|-------------|-------------------|
| Backend     | Python, Flask     |
| Machine Learning | Scikit-learn |
| Frontend    | HTML, CSS, Bootstrap 5 |
| Data        | Pandas, CSV       |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Run `python train_model.py` first |
| `Dataset not found` | Ensure `dataset/heart_disease.csv` exists |
| Port already in use | Stop other apps on port 5000 or change the port in `app.py` |

---

## License

Educational use. Dataset attributes belong to their original sources (e.g., UCI Machine Learning Repository).
