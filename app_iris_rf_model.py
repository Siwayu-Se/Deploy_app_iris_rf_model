
# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

# Set background image URL
background_image_url = "https://images.pexels.com/photos/713054/pexels-photo-713054.jpeg"

# Set desired colors
text_color = "#007ACC"  # Text color
result_bg_color = "#E0F7FA"  # Result background color

# Apply CSS for background and text colors
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        height: 100vh;
    }}
    h1, h2, h3, p, div {{
        color: {text_color} !important;
    }}
    .result-container {{
        background-color: {result_bg_color};
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        opacity: 0.9;
        border: 2px solid {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained Random Forest model and Label Encoder
with open('iris_best_rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('iris_label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Streamlit App
st.title("🌸 Iris Species Prediction ")
st.write("This app predicts the **species** of an Iris flower based on its physical characteristics. "
         "Please enter the details on the left and click 'Predict Species' to see the result.")

# Sidebar inputs for user to input iris features
st.sidebar.header("Iris Flower Features")
st.sidebar.write("Provide the following features to predict the iris species:")

# Input options in sidebar with descriptions
sepal_length_cm = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width_cm = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length_cm = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.5)
petal_width_cm = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Move the Predict button to the sidebar
if st.sidebar.button("Predict Species"):
    # Convert inputs to model-compatible format
    input_data = np.array([[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]])

    # Predict species
    prediction = rf_model.predict(input_data)
    species = label_encoder.inverse_transform(prediction)[0]

    # Display results with style
    st.subheader("Prediction Result")

    # Using markdown for better styling
    st.markdown(
        f"<div class='result-container'>"
        f"<h3 style='color: {text_color};'>The predicted species of the iris is: <strong>{species}</strong></h3>"
        "</div>",
        unsafe_allow_html=True
    )
    # Show species image in the center
    st.image(f"{species.lower()}.jpg", width=300, caption=f"{species} Penguin", use_column_width='auto')

    # Display characteristics with a result container
    st.markdown(
        f"<div class='result-container'>"
        f"<h4 style='color: {text_color};'>Iris Flower Characteristics</h4>"
        f"<p style='color: {text_color};'>- Sepal Length: {sepal_length_cm} cm</p>"
        f"<p style='color: {text_color};'>- Sepal Width: {sepal_width_cm} cm</p>"
        f"<p style='color: {text_color};'>- Petal Length: {petal_length_cm} cm</p>"
        f"<p style='color: {text_color};'>- Petal Width: {petal_width_cm} cm</p>"
        "</div>",
        unsafe_allow_html=True
    )
else:
    st.info("Please enter the iris flower features on the left and click 'Predict Species'.")
