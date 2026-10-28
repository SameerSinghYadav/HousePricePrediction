
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction API")

# Load the trained model
model = joblib.load("model.joblib")


class HouseFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    street: str
    city: str
    statezip: str


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict_price(house: HouseFeatures):
    data = pd.DataFrame([house.model_dump()])
    prediction = model.predict(data)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }
