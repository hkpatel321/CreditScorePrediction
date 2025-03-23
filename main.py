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

@app.post("/predict_formula")
def predict_credit_score_formula(data: CreditInput):
    """API Endpoint to calculate credit score using the custom formula."""
    try:
        past_yield = data.pastYield
        annual_income = data.annualIncome
        land_quality = data.landQualityScore
        soil_ph = data.soilPH
        nitrogen_level = data.nitrogenLevel
        organic_matter = data.organicMatterLevel
        past_rainfall = data.pastRainfall
        avg_temp = data.avgTemperature
        crop_type = data.cropTypes

        regional_avg_yield = 2.5
        regional_avg_income = 300000
        regional_avg_rainfall = 800
        crop_ideal_temp = 22

        yield_score = min((past_yield / regional_avg_yield) * 100, 100)
        income_score = min((annual_income / regional_avg_income) * 100, 100)
        land_quality_score = land_quality
        soil_ph_score = 100 - 20 * abs(soil_ph - 6.5)
        nitrogen_score = 100 - 30 * abs(nitrogen_level - 2)
        organic_matter_score = organic_matter * 25
        rainfall_score = 100 - 2 * abs(past_rainfall - regional_avg_rainfall)
        temp_score = 100 - 5 * abs(avg_temp - crop_ideal_temp)

        crop_risk_factors = {
            "Wheat": 1.0,
            "Rice": 1.0,
            "Corn": 0.95,
            "Soybeans": 0.95,
            "Coffee": 0.85,
            "Fruits": 0.85,
            "Barley": 0.9,
            "Cotton": 0.9,
            "Vegetables": 0.9,
            "Sugarcane": 0.9,
        }
        crop_risk_factor = crop_risk_factors.get(crop_type, 1.0)
        weighted_score = (
            (yield_score * 0.25) +
            (income_score * 0.20) +
            (land_quality_score * 0.15) +
            (soil_ph_score * 0.10) +
            (nitrogen_score * 0.10) +
            (organic_matter_score * 0.10) +
            (rainfall_score * 0.05) +
            (temp_score * 0.05)
        )

        credit_score = weighted_score * crop_risk_factor

        logger.info("\n🎯 Formula-Based Credit Score: %.2f", credit_score)

        return {"predicted_credit_score": round(credit_score, 2)}
    except Exception as e:
        logger.error("Error during formula-based calculation: %s", str(e))
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)