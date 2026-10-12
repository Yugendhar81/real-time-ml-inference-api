
# Real-Time ML Inference REST API

A machine-learning REST API built with Python, FastAPI, and Scikit-learn to classify Iris flowers in real time.

## Features

- Train an ML classification model
- REST API prediction endpoint
- JSON request and response
- Input validation using Pydantic
- Prediction confidence score
- Health-check endpoint
- Interactive API documentation

## Project Structure

```text
real-time-ml-inference-api/
├── main.py
├── train_model.py
├── requirements.txt
├── README.md
└── model.joblib   # Generated during model training
```

## Installation

Install Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

This creates `model.joblib`.

## Start the API

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to test the API.

## Prediction Endpoint

POST `/predict`

Example JSON request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:

```json
{
  "prediction": "setosa",
  "class_id": 0,
  "confidence": 0.98
}
```

The confidence value is illustrative; actual output depends on the trained model.

## Health Check

GET `/health`

Returns the API health status and whether the model is loaded.

## Technologies

- Python
- FastAPI
- Scikit-learn
- Pydantic
- Joblib

## Limitations

This is an educational demonstration using the Iris dataset. It is not intended for critical real-world decisions.
