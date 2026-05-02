import os
import json
import numpy as np
import tensorflow as tf
from app.utils.image_processing import load_and_preprocess_image

# Determine static paths relative to this file
# This file is at backend/app/services/prediction.py
# BACKEND_DIR needs to be backend/
# BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BACKEND_DIR,"..","plant_final.keras")
# CLASS_INDICES_PATH = os.path.join(BACKEND_DIR,"..","class_indices.json")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # services/

APP_DIR = os.path.abspath(os.path.join(BASE_DIR, "../.."))

MODEL_PATH = os.path.join(APP_DIR, "plant_final.keras")
CLASS_INDICES_PATH = os.path.join(APP_DIR, "class_indices.json")

print("Resolved MODEL PATH:", MODEL_PATH)
print("Resolved CLASS PATH:", CLASS_INDICES_PATH)

# Load model and class mapping ONCE at server startup to prevent blocking per-request
print(f"Loading ML model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        class_names = json.load(f)
    print("ML model loaded successfully!")
except Exception as e:
    print(f"Error loading model or class indices: {e}")
    model = None
    class_names = {}

def predict_image(image_bytes: bytes) -> tuple[str, float]:
    """
    Takes raw image bytes, preprocesses it, runs inference,
    and returns a tuple of (predicted_class_name, confidence).
    """
    if model is None:
        raise RuntimeError("ML model is not loaded. Check server logs.")
        
    preprocessed_img = load_and_preprocess_image(image_bytes)
    
    # Run prediction silently without tqdm verbose spam
    predictions = model.predict(preprocessed_img, verbose=0)
    
    # Extract highest probability index and its underlying probability score
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    
    predicted_class_name = class_names[str(predicted_class_index)]
    
    return predicted_class_name, float(confidence)
