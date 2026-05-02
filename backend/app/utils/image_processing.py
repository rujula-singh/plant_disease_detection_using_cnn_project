from PIL import Image
import numpy as np
import io

def load_and_preprocess_image(image_bytes: bytes, target_size=(224, 224)):
    # Read image from raw bytes uploaded via HTTP
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize to exact dimensions expected by model
    img = img.resize(target_size)
    # Convert image to numpy array
    img_array = np.array(img)
    # Expand dimensions (add batch axis to shape [1, 224, 224, 3])
    img_array = np.expand_dims(img_array, axis=0)
    # Ensure precision and normalize pixel values to [0, 1] range
    img_array = img_array.astype('float32') / 255.0
    
    return img_array
