import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Crime Prediction ANN Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

scaler = load_scaler()
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("crime_ann_model.h5", compile=False)
    return model

model = load_model()

# -----------------------------
# TITLE
# -----------------------------

st.title("🚔 Crime Prediction using Artificial Neural Network")
st.write("Predict total crime levels using crime indicators.")

st.sidebar.header("Adjust Crime Indicators")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------

murder = st.sidebar.slider("Murder Cases",0,500,50)
rape = st.sidebar.slider("Rape Cases",0,500,50)
kidnapping = st.sidebar.slider("Kidnapping",0,500,50)
theft = st.sidebar.slider("Theft",0,5000,500)
robbery = st.sidebar.slider("Robbery",0,500,50)

# -----------------------------
# CREATE INPUT ARRAY
# -----------------------------

input_data = np.array([[murder, rape, kidnapping, theft, robbery]])

# -----------------------------
# DISPLAY INPUT DATA
# -----------------------------

st.subheader("Input Data")

input_df = pd.DataFrame(
    input_data,
    columns=[
        "Murder",
        "Rape",
        "Kidnapping",
        "Theft",
        "Robbery"
    ]
)

st.dataframe(input_df)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------

if st.button("Predict Crime Level"):

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    st.subheader("Prediction Result")

    st.success(f"Predicted Total Crimes: {int(prediction[0][0])}")

# -----------------------------
# VISUALIZATION
# -----------------------------

st.subheader("Crime Input Distribution")

chart_data = pd.DataFrame({
    "Crime Type":[
        "Murder",
        "Rape",
        "Kidnapping",
        "Theft",
        "Robbery"
    ],
    "Cases":[
        murder,
        rape,
        kidnapping,
        theft,
        robbery
    ]
})

st.bar_chart(chart_data.set_index("Crime Type"))

# -----------------------------
# FOOTER
# -----------------------------

st.write("---")

st.write("AI Crime Forecasting System using ANN")
st.write("Built with TensorFlow + Streamlit")
