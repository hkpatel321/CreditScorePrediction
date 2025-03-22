import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load the trained model
model = joblib.load("random_forest_credit_score.pkl")

# Load the trained columns used for one-hot encoding
trained_columns = joblib.load("trained_columns.pkl")

# Initialize FastAPI app
app = FastAPI(title="Credit Score Prediction API",
              description="API for predicting farmer's credit score")

# Define the request schema
class CreditInput(BaseModel):
    year: str
    country: str
    region: str
    landSize: float
    soilType: str
    pastYield: float
    cropTypes: str
    annualIncome: int
    soilPH: float
    nitrogenLevel: int
    organicMatterLevel: int
    landQualityScore: int
    pastRainfall: float
    avgTemperature: float
    creditScore: float

# Function to preprocess input data
def preprocess_input(data: CreditInput):
    """Convert input data to DataFrame and apply one-hot encoding."""
    input_data = pd.DataFrame([data.dict()])  # Convert input to DataFrame

    # One-hot encoding for categorical variables
    input_data = pd.get_dummies(input_data)

    # Align columns with trained model (add missing columns with default value 0)
    for col in trained_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Ensure column order is the same as in training
    input_data = input_data[trained_columns]

    return input_data

@app.post("/predict")
def predict_credit_score(data: CreditInput):
    """API Endpoint to predict credit score."""
    processed_data = preprocess_input(data)  # Preprocess input
    prediction = model.predict(processed_data)[0]  # Make prediction

    return {"predicted_credit_score": round(prediction, 2)}

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
