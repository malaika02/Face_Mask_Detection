import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
# Load the pre-trained model
model = tf.keras.models.load_model('D:\mask_model.h5')

def preprocess(img):
    # Assuming 'img' is already a PIL image object
    img = img.resize((128, 128))  # Resize the image to match model input size
    img = np.array(img)  # Convert to NumPy array
    img = img / 255.0  # Normalize the image
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)  # Make prediction
    return prediction[0][0] > 0.5  # Return True if prediction > 0.5, else False
st.title('Mask Detection Model')

uploaded_file = st.file_uploader('Upload an image ',type='jpeg')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Prediction
    is_wearing_mask = preprocess(image)

    if is_wearing_mask:
        st.write("The person is wearing a mask.")
    else:
        st.write("The person is not wearing a mask.")
