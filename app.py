import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(page_title="District Crime Forecast", layout="wide")

st.title("🚔 District Crime Forecasting (ANN + Hyperparameter Tuned)")

# -------------------------
# Load Model & Scaler
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crime_growth_model.keras", compile=False)

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# -------------------------
# Load Dataset (Include CSV in GitHub repo)
# -------------------------
df = pd.read_csv("Crimes_in_india_2001-2013.csv")

df = df.sort_values(["STATE/UT", "DISTRICT", "YEAR"])

# -------------------------
# Yearly Total Display
# -------------------------
st.subheader("📊 Year-wise Total Crimes (All India)")

yearly_total = df.groupby("YEAR")["TOTAL IPC CRIMES"].sum()
st.line_chart(yearly_total)

# -------------------------
# User Selection
# -------------------------
st.subheader("🔎 Select Crime Forecast Parameters")

state = st.selectbox("Select State", df["STATE/UT"].unique())

district_df = df[df["STATE/UT"] == state]
district = st.selectbox("Select District", district_df["DISTRICT"].unique())

crime_columns = df.columns[3:-1]
crime = st.selectbox("Select Crime Type", crime_columns)

# -------------------------
# Forecast Logic
# -------------------------
if st.button("Predict Next Year Crime"):

   state_data = df[
    df["STATE/UT"] == state
].groupby("YEAR")[crime].sum().reset_index().sort_values("YEAR")

    if len(district_data) < 4:
        st.error("Not enough historical data for this district.")
    else:
        # Take last 3 years
        last_3 = district_data[crime].values[-3:]

        input_data = np.array(last_3).reshape(1, -1)

        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)

        next_year = district_data["YEAR"].max() + 1

        st.success(f"Predicted {crime} cases in {district}, {state} for {next_year}: {int(prediction[0][0])}")
