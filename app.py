from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,ValidationError
from fastapi.exceptions import RequestValidationError
import pandas as pd
import joblib
import numpy as np
from typing import List, Dict
import nest_asyncio
import logging
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"An error occurred: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "details": str(exc)}
    )

# Load model and encoders
model = joblib.load('LogReg.pkl')
le = joblib.load('label_encoders.pkl')
column_transformer = joblib.load('column_transformer.pkl')
y_encoder = joblib.load('y_encoder.pkl')

class CropPrediction(BaseModel):
    crop: str
    probability: float

# Request model (existing)
class CropRequest(BaseModel):
    state: str
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    soil_type: str
    land_size: float

def preprocess_input(data: pd.DataFrame):
    # Apply label encoding
    for col, encoder in le.items():
        if col in data.columns:
            # Handle unseen labels
            if not data[col].isin(encoder.classes_).all():
                invalid = data[col][~data[col].isin(encoder.classes_)].unique()
                raise ValueError(f"Invalid {col} value(s): {invalid}. Allowed values: {encoder.classes_.tolist()}")
            data[col] = encoder.transform(data[col])
    
    # Apply column transformations
    if column_transformer:
        data = column_transformer.transform(data)
    
    return data

@app.post("/predict", response_model=List[CropPrediction])
async def predict(request: CropRequest):
    try:
        # Convert request to DataFrame and capitalize state and soil type
        input_data = request.dict()
        input_data['state'] = input_data['state'].capitalize()  # Capitalize state
        input_data['Soil Type'] = input_data.pop('soil_type').capitalize()  # Capitalize soil type
        
        df = pd.DataFrame([input_data])

        # Preprocess data
        processed_data = preprocess_input(df)

        # Make prediction
        probabilities = model.predict_proba(processed_data)[0]
        
        # Get top 5 predictions
        top5_idx = np.argsort(probabilities)[-5:][::-1]
        crops = y_encoder.inverse_transform(top5_idx)
        scores = probabilities[top5_idx]

        return [{"crop": crop, "probability": float(score)} 
                for crop, score in zip(crops, scores)]
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Crop Recommendation API - Send POST requests to /predict"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

