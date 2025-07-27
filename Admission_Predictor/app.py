import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load the model and scaler
model = joblib.load('Admission_predictor_model.pkl')
scaler = joblib.load('feature_scaler_file.pkl')

# Set up the Streamlit app
st.set_page_config(page_title="Admission Predictor", layout="wide")
st.title("🎓 Admission Predictor")
st.markdown("Predict your chances of getting admitted based on your profile.")

# User input
gre_score = st.number_input("GRE Score", min_value=260, max_value=340, step=1)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)
university_rating = st.slider("University Rating", 1, 5)
sop = st.slider("SOP Strength", 1.0, 5.0, step=0.1)
lor = st.slider("LOR Strength", 1.0, 5.0, step=0.1)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
research = st.selectbox("Research Experience", ["No", "Yes"])

# Convert input to model format
research = 1 if research == "Yes" else 0
input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Admission Chance"):
    prediction_original_scale = np.exp(model.predict(scaled_input)[0])
    st.success(f'predicted chance of admit: {round(prediction_original_scale[0] * 100)}')


    
feature_names = ['gre_score', 'toefl_score', 'university_rating', 'sop', 'lor', 'cgpa', 'research']
importances = [0.026671, 0.018226, 0.002940, 0.001788, 0.015866, 0.067581, 0.011940]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


st.write("### 🔍 Feature Importance in Model")
st.dataframe(importance_df)


import altair as alt

chart = alt.Chart(importance_df).mark_bar().encode(
    x=alt.X('Importance', scale=alt.Scale(domain=[0, max(importances)+0.01])),
    y=alt.Y('Feature', sort='-x')
).properties(title="Feature Importance")

st.altair_chart(chart, use_container_width=True)