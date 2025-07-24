import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load('Admission_predictor_model.pkl')
scaler = joblib.load('feature_scaler_file.pkl')

# Set up the Streamlit app
st.set_page_config(page_title="Admission Predictor", layout="wide")
st.title("🎓 Admission Predictor")
st.markdown("Predict your chances of getting admitted based on your profile.")

# User input
gre_score = st.number_input("GRE Score", min_value=260, max_value=340)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120)
university_rating = st.slider("University Rating", 1, 5)
sop = st.slider("SOP Strength", 1.0, 5.0, step=0.5)
lor = st.slider("LOR Strength", 1.0, 5.0, step=0.5)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
research = st.selectbox("Research Experience", ["No", "Yes"])

# Convert input to model format
research = 1 if research == "Yes" else 0
input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Admission Chance"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"🎯 Your predicted admission chance is: **{prediction * 100:.2f}%**")
