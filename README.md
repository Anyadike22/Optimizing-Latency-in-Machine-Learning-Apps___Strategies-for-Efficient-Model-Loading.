# Optimizing-Latency-in-Machine-Learning-Apps___Strategies-for-Efficient-Model-Loading.

## Model Loading on Every Prediction Call (Loading Per Request )
The way the machine learning model is being loaded in this script can negatively impact latency due to the following reasons:

## Issues Affecting Latency with this strategy

## Model Loading on Every Prediction Call

In the Predict method, the model is checked (if self.model is None), and if it is None, the load_model() method is called.
This means that if self.model is not already loaded, the system will read the model from disk (pickle.load(f)) before making predictions.
Loading a model from disk introduces I/O overhead, which increases latency.

## Lack of Model Persistence

The model is not preloaded into memory when the FastAPI server starts.
Instead, it's only loaded when a prediction request is made (if self.model is None).
This causes unnecessary disk reads, slowing down response times.

(strategy 1 script)

pip install fastapi uvicorn scikit-learn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np
import pickle

# Step 1: Data and Model class
class ModelHandler:
    def __init__(self):
        self.model = None

    def train_model(self):
        # Generate Synthetic data
        X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.save_model()


    def save_model(self):
          # Save model to the disc
              with open('model.pkl', 'wb') as f:
                        pickle.dump(self.model, f)

    def load_model(self):
          try:
            with open('model.pkl', 'rb') as f:
                 self.model = pickle.load(f)
          except FileNotFoundError:
                raise HTTPException(status_code=500, detail="Model not trained yet")


    def Predict(self, input_data: np.ndarray):
      # Predict using the loaded model
      if self.model is None:
        self.load_model()
        return self.model.predict(input_data)

# Step 2: API Schema for Request and Response

class PredictionRequest(BaseModel):
    feature: float  # single feature input for simplicity

class PredictionResponse(BaseModel):
    Prediction: float

# Step 3: API Definition
api = FastAPI(title="ML Model Serving API with OOP", version="1.0")

#Initialize the model handler
model_handler = ModelHandler()




# Strategies for Optimizations to Reduce Latency

## Strategy 1: Preload the Model on API Startup
Modify the script to load the model when the API starts instead of on every request.


@api.on_event("startup")
def load_model_on_startup():
    model_handler.load_model()
This ensures that the model is in memory before any request comes in.
Remove Unnecessary File I/O
Modify the Predict method to assume the model is always loaded:


def Predict(self, input_data: np.ndarray):
    if self.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return self.model.predict(input_data)
Use a More Efficient Model Storage Method

## Note: Instead of using pickle, consider joblib, which is optimized for storing and loading large NumPy arrays.


import joblib
def save_model(self):
    joblib.dump(self.model, 'model.pkl')

def load_model(self):
    self.model = joblib.load('model.pkl')
    
Expected Performance Improvement
Faster Predictions: Model is already in memory when requests come in.
Lower Latency: No disk reads for every prediction request.
Better Scalability: Reduces I/O bottlenecks, improving API response times under load.


# Strategy 2: Preloading Models at Startup (Global Instance)

Load the model during application startup and reuse it globally.

from fastapi import FastAPI

app = FastAPI()
predictor = None  # Global instance

# --- Preload Model at Startup ---
@app.on_event("startup")
def load_model():
    global predictor
    predictor = ModelPredictor("model_v1.pkl")  # Initialize once

# --- Endpoint ---
@app.post("/predict")
async def predict_endpoint(data: dict):
    processed_data = predictor.preprocess(data)
    return {"prediction": predictor.predict(processed_data)}


## Key Features:

* Simple and explicit initialization at startup.

* Avoids repeated model loading.

