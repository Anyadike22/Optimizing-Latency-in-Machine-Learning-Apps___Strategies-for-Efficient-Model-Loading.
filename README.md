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

```python
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
```

This ensures that the model is in memory before any request comes in.
Remove Unnecessary File I/O



Modify the Predict method to assume the model is always loaded:

```
def Predict(self, input_data: np.ndarray):
    if self.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return self.model.predict(input_data)
Use a More Efficient Model Storage Method
```

## Note: Instead of using pickle, consider joblib, which is optimized for storing and loading large NumPy arrays.

```python 
import joblib
def save_model(self):
    joblib.dump(self.model, 'model.pkl')

def load_model(self):
    self.model = joblib.load('model.pkl')
```
    
Expected Performance Improvement
Faster Predictions: Model is already in memory when requests come in.
Lower Latency: No disk reads for every prediction request.
Better Scalability: Reduces I/O bottlenecks, improving API response times under load.


## Strategy 2: Preloading Models at Startup (Global Instance)

Load the model during application startup and reuse it globally.

```python 
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
```

## Key Features:

* Simple and explicit initialization at startup.

* Avoids repeated model loading.

## Strategy 3 : Dependency Injection (Recommended for FastAPI)
Load the model once and reuse it across requests, avoiding redundant instantiation.

```python
from fastapi import FastAPI, Depends
from functools import lru_cache

app = FastAPI()

# --- ModelPredictor Class (OOP) ---
class ModelPredictor:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)  # Preload at startup
    
    def _load_model(self, model_path: str):
        # Simulate model loading (e.g., from pickle, ONNX, TF SavedModel)
        print(f"Loading model from {model_path}...")
        return "Mock Model"  # Replace with actual model
    
    def predict(self, data: dict):
        return self.model  # Replace with actual prediction logic

# --- Dependency Injection with Caching ---
@lru_cache  # Ensures the model is loaded ONCE, not per request
def get_predictor():
    return ModelPredictor("model_v1.pkl")

# --- FastAPI Endpoint ---
@app.post("/predict")
async def predict_endpoint(
    data: dict, 
    predictor: ModelPredictor = Depends(get_predictor)  # Reuse the same instance
):
    return {"prediction": predictor.predict(data)}
```


## Strategy 4: Singleton Pattern
Enforce a single instance of ModelPredictor across the application.

```python 
class SingletonModelPredictor:
    _instance = None

    def __new__(cls, model_path: str):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.model = cls._load_model(model_path)
        return cls._instance

    @staticmethod
    def _load_model(model_path: str):
        print(f"Loading model from {model_path}...")
        return "Mock Model"

# --- FastAPI Usage ---
@app.on_event("startup")  # Preload at startup
def initialize_model():
    global predictor
    predictor = SingletonModelPredictor("model_v1.pkl")

@app.post("/predict")
async def predict_endpoint(data: dict):
    return {"prediction": predictor.predict(data)}
```

# Conclusion 

The latency optimization strategies discussed—preloading models at startup, singleton patterns, and dependency injection—address the inefficiencies of reloading ML models per request, ensuring scalable, low-latency real-time predictions.

## Preloading Models at Startup:

Loads models into memory during application initialization, eliminating redundant I/O operations during inference.

Simplifies resource management but requires careful handling of global state.

## Singleton Pattern:

Enforces a single instance of the model across requests, reducing memory overhead and ensuring consistency.

Ideal for scenarios requiring strict control over model access but less flexible for dynamic model switching.

## Dependency Injection (FastAPI):

Recommended Approach: Uses FastAPI’s Depends with @lru_cache to load models once and reuse them across requests.

Balances thread safety, testability, and scalability while aligning with modern API design practices.

## Final Recommendation:
For production-grade ML APIs, dependency injection is optimal, combining FastAPI’s async capabilities with OOP principles to minimize latency while maintaining clean, maintainable code. Preloading models at startup (via singletons or global instances) ensures consistent performance, making these strategies essential for high-traffic, real-time systems.
