"""
Explore, preprocess, and train heart disease classification models.

Run this script from the project root:
    python train_model.py

Dataset location: dataset/heart.csv
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# File paths and column names
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "heart.csv"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Human-readable names for the two classes (used in reports and plots)
CLASS_LABELS = ["No Heart Disease", "Heart Disease"]

# The column we want to predict (1 = heart disease, 0 = no heart disease)
TARGET_COLUMN = "target"


def load_dataset():
    """
    Load heart.csv into a pandas DataFrame.

    Returns:
        pd.DataFrame: The loaded dataset
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Please place heart.csv inside the dataset/ folder."
        )

    df = pd.read_csv(DATASET_PATH)
    return df


def explore_dataset(df):
    """
    Print useful information about the dataset for beginners.
    Each section helps you understand data quality and content.
    """

    print("=" * 60)
    print("HEART DISEASE DATASET — EXPLORATORY ANALYSIS")
    print("=" * 60)

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Dataset shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Full shape (rows, columns): {df.shape}")

    print("\n--- Column names ---")
    print(df.columns.tolist())

    print("\n--- Missing values (per column) ---")
    missing = df.isnull().sum()
    print(missing)
    print(f"\nTotal missing values in dataset: {missing.sum()}")

    print("\n--- Duplicate rows ---")
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")

    if duplicate_count > 0:
        print("\nSample of duplicate rows:")
        print(df[df.duplicated(keep=False)].head(10))

    print("\n--- Statistical summary ---")
    print(df.describe())


def preprocess_dataset(df):
    """
    Clean the data and prepare it for machine learning.

    Steps:
        1. Handle missing values
        2. Remove duplicate rows
        3. Split into features (X) and target (y)
        4. Split into training and testing sets
        5. Scale features with StandardScaler

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Work on a copy so the original DataFrame passed in is not changed
    df = df.copy()

    # ------------------------------------------------------------------
    # Step 1: Handle missing values
    # ------------------------------------------------------------------
    # Missing cells are empty/NaN values. We replace them with the median
    # of each column (the middle value), which is safe for numeric data.
    print("\n--- Step 1: Handling missing values ---")
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values before: {missing_before}")

    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
            print(f"  Filled missing values in '{column}' with median: {median_value}")

    missing_after = df.isnull().sum().sum()
    print(f"Total missing values after: {missing_after}")

    # ------------------------------------------------------------------
    # Step 2: Remove duplicate rows
    # ------------------------------------------------------------------
    # Duplicate rows are exact copies; they can bias the model, so we drop them.
    print("\n--- Step 2: Removing duplicate rows ---")
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_removed = rows_before - len(df)
    print(f"Rows before: {rows_before}")
    print(f"Duplicate rows removed: {rows_removed}")
    print(f"Rows after: {len(df)}")

    # ------------------------------------------------------------------
    # Step 3: Separate features (X) and target (y)
    # ------------------------------------------------------------------
    # Features (X) = all input columns used to make a prediction
    # Target (y)   = the column we want to predict (heart disease or not)
    print("\n--- Step 3: Separating features and target ---")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    print(f"Feature columns (X): {X.columns.tolist()}")
    print(f"Target column (y): '{TARGET_COLUMN}'")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # ------------------------------------------------------------------
    # Step 4: Train-test split
    # ------------------------------------------------------------------
    # We hold back 20% of the data for testing so we can check model accuracy
    # on data it has never seen during training.
    # random_state=42 keeps the split the same every time you run the script.
    print("\n--- Step 4: Train-test split (80% train, 20% test) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # Step 5: StandardScaler
    # ------------------------------------------------------------------
    # Scaling puts all features on a similar scale (mean=0, std=1).
    # Important: fit the scaler ONLY on training data, then apply to both sets.
    # Fitting on the full dataset would leak test information into training.
    print("\n--- Step 5: Applying StandardScaler ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaler fitted on training data and applied to train and test sets.")

    # ------------------------------------------------------------------
    # Step 6: Print final shapes
    # ------------------------------------------------------------------
    print("\n--- Step 6: Train and test dataset shapes ---")
    print(f"X_train shape: {X_train_scaled.shape}")
    print(f"X_test shape:  {X_test_scaled.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classifiers, measure accuracy on the test set,
    and report which model performed best.

    Args:
        X_train, X_test: Scaled feature arrays (from preprocessing)
        y_train, y_test: Target labels for training and testing

    Returns:
        trained_models: dict of model name -> fitted model
        accuracies: dict of model name -> test accuracy (0.0 to 1.0)
        best_name: name of the highest-accuracy model
    """

    print("\n" + "=" * 60)
    print("MODEL TRAINING & COMPARISON")
    print("=" * 60)

    # Dictionary of models to train.
    # Each key is a display name; each value is an untrained model instance.
    # random_state=42 makes results repeatable (same split, same randomness).
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        # probability=True is required for ROC curves (predict_proba)
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
    }

    # Store accuracy for each model so we can compare them later
    accuracies = {}
    trained_models = {}

    print("\n--- Training models and calculating accuracy ---\n")

    for name, model in models.items():
        # Step 1: Train (fit) the model on training data
        # The model learns patterns from X_train and y_train
        model.fit(X_train, y_train)

        # Step 2: Predict labels for the test set (data the model has not seen)
        y_pred = model.predict(X_test)

        # Step 3: Accuracy = fraction of correct predictions (0.0 to 1.0)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        trained_models[name] = model

        # Print accuracy as a percentage for easy reading
        print(f"{name}: {accuracy:.2%}")

    # ------------------------------------------------------------------
    # Compare all models side by side
    # ------------------------------------------------------------------
    print("\n--- Model accuracy comparison ---")
    # Sort models from highest to lowest accuracy
    sorted_results = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)

    for rank, (name, accuracy) in enumerate(sorted_results, start=1):
        print(f"  {rank}. {name}: {accuracy:.2%}")

    # ------------------------------------------------------------------
    # Best-performing model
    # ------------------------------------------------------------------
    best_name, best_accuracy = sorted_results[0]
    print("\n--- Best-performing model ---")
    print(f"Winner: {best_name} with test accuracy {best_accuracy:.2%}")

    return trained_models, accuracies, best_name


def save_best_model(trained_models, best_name, scaler):
    """
    Save the best-performing trained model to disk using joblib.

    joblib is efficient for large numpy arrays inside scikit-learn models.
    The Flask app (app.py) loads this file to make predictions.

    Args:
        trained_models: dict of model name -> fitted model
        best_name: name of the model with highest test accuracy
        scaler: fitted StandardScaler (same preprocessing as training)
    """

    print("\n" + "=" * 60)
    print("SAVE BEST MODEL")
    print("=" * 60)

    # Create the models/ folder if it does not exist yet
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Pick the winning model from the dictionary of trained models
    best_model = trained_models[best_name]

    # joblib.dump() writes the Python object to a .pkl file on disk
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\nBest model saved successfully as models/model.pkl")


def evaluate_and_visualize(trained_models, accuracies, best_name, X_test, y_test):
    """
    Print detailed metrics and save evaluation plots.

    Creates:
        - Classification report for every model (printed)
        - Confusion matrix heatmap for the best model
        - ROC curve for the best model
        - Bar chart comparing test accuracy of all models

    Plots are saved in the plots/ folder and shown on screen (close each
    window to continue to the next graph).
    """

    print("\n" + "=" * 60)
    print("MODEL EVALUATION & VISUALIZATION")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Seaborn makes statistical plots look clean with minimal code
    sns.set_theme(style="whitegrid")

    # ------------------------------------------------------------------
    # Classification report for each model
    # ------------------------------------------------------------------
    # Shows precision, recall, and F1-score per class (more detail than accuracy)
    print("\n--- Classification reports (all models) ---\n")

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        print(f"{'=' * 40}")
        print(f"Model: {name}")
        print(f"{'=' * 40}")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=CLASS_LABELS,
            )
        )

    # ------------------------------------------------------------------
    # Best model: predictions and probability scores for positive class
    # ------------------------------------------------------------------
    best_model = trained_models[best_name]
    y_pred_best = best_model.predict(X_test)
    # Probability that each sample belongs to class 1 (heart disease)
    y_proba_best = best_model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 1. Confusion matrix (best model)
    # ------------------------------------------------------------------
    # Rows = actual class, columns = predicted class
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.title(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.tight_layout()

    confusion_path = PLOTS_DIR / "confusion_matrix_best_model.png"
    plt.savefig(confusion_path, dpi=150)
    # Display the plot in a window; close it to continue to the next graph
    plt.show()
    plt.close()
    print(f"\nConfusion matrix saved to: {confusion_path}")

    # ------------------------------------------------------------------
    # 2. ROC curve (best model)
    # ------------------------------------------------------------------
    # ROC shows trade-off between true positive rate and false positive rate.
    # AUC (area under curve) near 1.0 means strong discrimination ability.
    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.2f})",
    )
    # Diagonal line = random guessing (no skill)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve — {best_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = PLOTS_DIR / "roc_curve_best_model.png"
    plt.savefig(roc_path, dpi=150)
    plt.show()
    plt.close()
    print(f"ROC curve saved to: {roc_path}")

    # ------------------------------------------------------------------
    # 3. Accuracy comparison bar chart (all models)
    # ------------------------------------------------------------------
    model_names = list(accuracies.keys())
    accuracy_values = [accuracies[name] * 100 for name in model_names]

    plt.figure(figsize=(10, 6))
    bar_colors = sns.color_palette("viridis", len(model_names))
    bars = plt.bar(model_names, accuracy_values, color=bar_colors, edgecolor="black")

    # Highlight the best model with a different edge color
    best_index = model_names.index(best_name)
    bars[best_index].set_edgecolor("red")
    bars[best_index].set_linewidth(2.5)

    plt.title("Model Accuracy Comparison (Test Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Machine Learning Model", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=15, ha="right")

    # Show percentage on top of each bar
    for bar, value in zip(bars, accuracy_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    accuracy_path = PLOTS_DIR / "accuracy_comparison.png"
    plt.savefig(accuracy_path, dpi=150)
    plt.show()
    plt.close()
    print(f"Accuracy comparison chart saved to: {accuracy_path}")

    print("\nAll evaluation plots saved in the 'plots/' folder and displayed on screen.")


if __name__ == "__main__":
    # Step 1: Load the data
    print("Loading dataset...")
    data = load_dataset()

    # Step 2: Explore the raw data
    explore_dataset(data)

    # Step 3: Preprocess for machine learning
    X_train, X_test, y_train, y_test, scaler = preprocess_dataset(data)

    # Step 4: Train models and compare accuracy
    trained_models, accuracies, best_name = train_and_compare_models(
        X_train, X_test, y_train, y_test
    )

    # Step 5: Save the best model for use in the Flask web app
    save_best_model(trained_models, best_name, scaler)

    # Step 6: Detailed evaluation and visualization
    evaluate_and_visualize(
        trained_models, accuracies, best_name, X_test, y_test
    )
