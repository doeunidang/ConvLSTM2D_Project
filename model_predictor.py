#model_predictor.py
import numpy as np
from tensorflow.keras.models import load_model
from custom_function import custom_loss, custom_accuracy
from config import MODEL_PATH

def preprocess_input_data(rainfall_data):
    terrain_data = np.load("F:\\Flooding_Predict_System\\data\\terrain_data.npy")
    if terrain_data.shape != (4, 64, 64, 4):
        raise ValueError(f"Invalid terrain shape: {terrain_data.shape}")
    input_data = np.concatenate((rainfall_data, terrain_data), axis=-1)
    return input_data

def predict_discharge(input_data):
    model = load_model(
        MODEL_PATH,
        custom_objects={"custom_loss": custom_loss, "custom_accuracy": custom_accuracy},
    )
    input_data = np.expand_dims(input_data, axis=0)
    y_pred = model.predict(input_data)
    return np.squeeze(y_pred, axis=0)

