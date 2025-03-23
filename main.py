import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = joblib.load("random_forest_credit_score.pkl")
trained_columns = joblib.load("trained_columns.pkl")

app = FastAPI(title="Credit Score Prediction API",
              description="API for predicting farmer's credit score")

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

def preprocess_input(data: CreditInput):
    input_data = pd.DataFrame([data.dict()])
    logger.info("\n📌 Raw Input Data:\n%s", input_data.to_string())

    numerical_cols = ['landSize', 'pastYield', 'annualIncome', 'soilPH',
                      'nitrogenLevel', 'organicMatterLevel', 'landQualityScore',
                      'pastRainfall', 'avgTemperature']
    categorical_cols = ['year', 'country', 'region', 'soilType', 'cropTypes']

    numerical_data = input_data[numerical_cols]

    numerical_data.columns = ['LandSize', 'PastYield', 'Annual Income (₹)', 'SoilPH',
                             'Nitrogen Level', 'Organic Matter Level', 'Land Quality Score',
                             'PastRainfall', 'AvgTemperature']

    categorical_data = pd.get_dummies(input_data[categorical_cols],
                                     columns=categorical_cols,
                                     prefix=['Year', 'Country', 'Region', 'Soil Type', 'Crop Type'])

    input_data_processed = pd.concat([numerical_data, categorical_data], axis=1)

    for col in trained_columns:
        if col not in input_data_processed.columns:
            input_data_processed[col] = 0

    input_data_processed = input_data_processed[trained_columns]

    logger.info("\n✅ Processed Input Data (Ready for Prediction):\n%s", input_data_processed.to_string())

    return input_data_processed

@app.post("/predict")
def predict_credit_score(data: CreditInput):
    """API Endpoint to predict credit score."""
    try:
        processed_data = preprocess_input(data)  # Preprocess input
        prediction = model.predict(processed_data)[0]  # Make prediction

        logger.info("\n🎯 Predicted Credit Score: %.2f", prediction)

        return {"predicted_credit_score": round(prediction, 2)}
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)