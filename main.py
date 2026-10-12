
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = FastAPI(
    title="Real-Time ML Inference API",
    description="An Iris flower classification API",
    version="1.0.0"
)

# Load the trained model
if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model file not found. Run python train_model.py first."
    )

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
target_names = saved_data["target_names"]


class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


@app.get("/")
def home():
    return {
        "message": "ML Inference API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(data: FlowerInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    try:
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        return {
            "prediction": target_names[prediction],
            "class_id": prediction,
            "confidence": round(
                float(max(probabilities)), 4
            )
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from error
