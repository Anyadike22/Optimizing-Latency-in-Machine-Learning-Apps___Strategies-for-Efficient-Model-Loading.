# Optimizing-Latency-in-Machine-Learning-Apps___Strategies-for-Efficient-Model-Loading.

## Strategy 1: Model Loading on Every Prediction Call (Loading Per Request )
The way the machine learning model is being loaded in this script can negatively impact latency due to the following reasons:

Issues Affecting Latency with this strategy 
Model Loading on Every Prediction Call

In the Predict method, the model is checked (if self.model is None), and if it is None, the load_model() method is called.
This means that if self.model is not already loaded, the system will read the model from disk (pickle.load(f)) before making predictions.
Loading a model from disk introduces I/O overhead, which increases latency.
Lack of Model Persistence

The model is not preloaded into memory when the FastAPI server starts.
Instead, it's only loaded when a prediction request is made (if self.model is None).
This causes unnecessary disk reads, slowing down response times.
