
import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir=os.path.dirname(os.path.abspath(__file__))
model_path=f"{working_dir}/plant_final.keras"

model=tf.keras.models.load_model(model_path)

class_names=json.load(open(f"{working_dir}/class_indices.json"))

def load_and_preprocess_image(image_path,target_size=(224,224)):
    img=Image.open(image_path)
    img=img.resize(target_size)
    img_array=np.array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model,image_path,class_indices):
    preprocessed_img=load_and_preprocess_image(image_path)
    predictions=model.predict(preprocessed_img)
    predicted_class_index=np.argmax(predictions,axis=1)[0]
    predicted_class_name=class_indices[str(predicted_class_index)]
    return predicted_class_name

st.set_page_config(page_title="🌱 Plant Disease Classifier", layout="centered")
st.markdown("<h1 style='text-align: center; color: green;'>🌿 Plant Disease Classifier</h1>", unsafe_allow_html=True)
uploaded_image=st.file_uploader("Upload an image...",type=['jpg','jpeg','png'])
st.markdown("<p style='text-align: center;'>Upload a leaf image to detect plant disease.</p>", unsafe_allow_html=True)

with st.sidebar:
     st.header("🌱 About")
     st.markdown("""
      This Plant Disease Classifier uses a Convolutional Neural Network (CNN) trained on a dataset of plant leaf images.

       🧠 **Model:** Custom CNN trained with TensorFlow  
       🌿 **Goal:** Detect plant diseases from images  
       📁 **Supported files:** JPG, JPEG, PNG  
    

    """)

if uploaded_image is not None:
    image=Image.open(uploaded_image)
    col1,col2=st.columns(2)

    with col1:
        resized_img=image.resize((150,150))
        st.image(resized_img)

    with col2:
        if st.button('🔍Classify'):
            prediction=predict_image_class(model,uploaded_image,class_names) 
            st.success(f'Prediction: {str(prediction)}')


   
   
