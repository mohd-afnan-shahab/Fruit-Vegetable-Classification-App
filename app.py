import streamlit as st
import pandas as pd
from collections import Counter
import joblib
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import streamlit as st
import torch
import torchvision
from PIL import Image
import joblib
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import json

image_file = 'background/bg2.jpg'

with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


# Load the saved model
loaded_model = keras.models.load_model('cnn_model.h5')

# Load class names
class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Define sample images
sample_images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg', 'sample4.jpg', 'sample5.jpg']

# Load data from JSON file
with open('data_analysis.json', 'r') as f:
    data_analysis = json.load(f)


def preprocess_image(image):
    img_array = np.array(image.resize((160, 160))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    
# Main app
def main():
    # Sidebar
    st.sidebar.title("Fruit Vegetable Classification App")
    pages = ["Data & Analytics", "Classification"]
    selection = st.sidebar.radio("Go To", pages)

    if selection == "Data & Analytics":
        st.title("Data & Analytics")

        # Data details
        st.header("Data Details")
        da1 = data_analysis['num_classes']
        da2= data_analysis['num_train_images']
        da3= data_analysis['num_val_images']
        da4 = data_analysis['num_test_images']
        st.markdown(f'<span style="background-color:#009900; padding: 5px">Number of classes: {da1}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="background-color:#009900; padding: 5px">Number of images in train set: {da1}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="background-color:#009900; padding: 5px">Number of images in validation set: {da1}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="background-color:#009900; padding: 5px">Number of images in test set: {da1}</span>', unsafe_allow_html=True)

        # Data distribution pie charts
        st.header("Data Distribution")
        total_images = data_analysis['num_train_images'] + data_analysis['num_val_images'] + data_analysis['num_test_images']
        train_percentage = (data_analysis['num_train_images'] / total_images) * 100
        val_percentage = (data_analysis['num_val_images'] / total_images) * 100
        test_percentage = (data_analysis['num_test_images'] / total_images) * 100

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].pie([train_percentage, val_percentage, test_percentage], labels=['Train', 'Validation', 'Test'], autopct='%1.1f%%', startangle=90)
        axes[0].axis('equal')
        axes[0].set_title('Data Distribution (Percentage)')

        axes[1].pie([data_analysis['num_train_images'], data_analysis['num_val_images'], data_analysis['num_test_images']], labels=['Train', 'Validation', 'Test'], autopct='%1.0f', startangle=90)
        axes[1].axis('equal')
        axes[1].set_title('Data Distribution (Counts)')

        st.pyplot(fig)

        # Class distribution
        st.header("Class Distribution")
        class_counts = data_analysis['class_counts']
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(class_counts)), list(class_counts.values()), align='center')
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels(list(class_counts.keys()), rotation=90)
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Images')
        ax.set_title('Class Distribution (Train Set)')
        st.pyplot(fig)

        # Percentage distribution across classes
        st.header("Percentage Distribution across Classes (Train Set)")
        class_percentages = data_analysis['class_percentages']
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(class_percentages.values(), labels=class_percentages.keys(), autopct='%1.1f%%')
        ax.axis('equal')
        ax.set_title('Percentage Distribution across Classes (Train Set)')
        st.pyplot(fig)

        # Model summary
        st.header("Model Summary")
        st.code(data_analysis['model_summary'], language='text')

        # Model history
        st.header("Model Training History")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].plot(data_analysis['model_history']['accuracy'])
        axes[0].plot(data_analysis['model_history']['val_accuracy'])
        axes[0].set_title('Model Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(['Train', 'Validation'], loc='upper left')

        axes[1].plot(data_analysis['model_history']['loss'])
        axes[1].plot(data_analysis['model_history']['val_loss'])
        axes[1].set_title('Model Loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(['Train', 'Validation'], loc='upper left')

        st.pyplot(fig)

    elif selection == "Classification":
        st.title("Find Your Flavor")
        image_source = st.sidebar.radio("Select image source", ("Sample Images", "Upload Your Own"))

        if image_source == "Sample Images":
            sample_image_name = st.sidebar.selectbox("Select a sample image", sample_images)
            sample_image_path = f"sample_images/{sample_image_name}"
            image = Image.open(sample_image_path)
        elif image_source == "Upload Your Own":
            uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
            else:
                st.warning("Please upload an image.")

        if 'image' in locals():
            st.image(image, caption="Input Image", use_column_width=True)

            if st.button("Make Prediction"):
                img_array = preprocess_image(image)
                predictions = loaded_model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = class_names[predicted_class_index]
                predicted_probability = predictions[0][predicted_class_index]

                st.subheader("Prediction")
                st.markdown(f'<span style="background-color:#009900; padding: 5px">The image is classified as: {predicted_class}</span>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
        