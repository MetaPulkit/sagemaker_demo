# inference.py

import joblib
import os
import numpy as np
import json

def model_fn(model_dir):
    """Load the trained model from the model directory."""
    model = joblib.load(os.path.join(model_dir, 'iris_model.tar.gz'))
    return model

def predict_fn(input_data, model):
    """Make prediction using the trained model."""
    input_data = np.array(input_data['instances'])
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, accept):
    """Format the predictions as a response."""
    result = json.dumps({"predictions": prediction.tolist()})
    return result, 'application/json'
