
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"


def train_model():
    # Load the sample Iris flower dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Build a machine-learning pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=500))
    ])

    # Train the model
    model.fit(X, y)

    # Save the trained model and its class names
    joblib.dump({
        "model": model,
        "target_names": iris.target_names.tolist(),
        "feature_names": iris.feature_names
    }, MODEL_PATH)

    print(f"Model trained and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
