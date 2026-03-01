import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Crime Prediction ANN", layout="wide")

st.title("🚔 Crime Prediction using ANN (Hyperparameter Tuned)")
st.write("Upload a crime dataset row to predict TOTAL IPC CRIMES.")

# -------------------------
# Load Model & Scaler
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crime_ann_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# -------------------------
# Upload CSV
# -------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())
    
    # Drop columns used during training
    if 'STATE/UT' in df.columns:
        df = df.drop(['STATE/UT'], axis=1)
    if 'DISTRICT' in df.columns:
        df = df.drop(['DISTRICT'], axis=1)
    
    # If TOTAL column exists, remove it (since we are predicting it)
    if 'TOTAL IPC CRIMES' in df.columns:
        df = df.drop(['TOTAL IPC CRIMES'], axis=1)
    
    # Scale input
    scaled_input = scaler.transform(df)
    
    # Predict
    predictions = model.predict(scaled_input)
    
    st.subheader("Predicted TOTAL IPC CRIMES")
    
    df["Predicted TOTAL IPC CRIMES"] = predictions.astype(int)
    
    st.dataframe(df)