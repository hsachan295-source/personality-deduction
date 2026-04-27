import streamlit as st
import pandas as pd
import pickle

# ------------------------------
# Load Model + Scaler
# ------------------------------
with open('personality_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Personality Predictor", page_icon="🧠", layout="wide")

st.title("🧠 Personality Type Predictor")
st.write("Enter your personality trait scores (0-10) and predict whether you are **Extrovert / Introvert / Ambivert**.")

# Features used in training
features = [
    'social_energy',
    'alone_time_preference',
    'talkativeness',
    'deep_reflection',
    'group_comfort',
    'party_liking',
    'listening_skill',
    'empathy',
    'organization',
    'leadership',
    'risk_taking',
    'public_speaking_comfort',
    'curiosity',
    'routine_preference',
    'excitement_seeking',
    'friendliness',
    'planning',
    'spontaneity',
    'adventurousness',
    'reading_habit',
    'sports_interest',
    'online_social_usage',
    'travel_desire',
    'gadget_usage',
    'work_style_collaborative',
    'decision_speed'
]

# Note:
# personality_type = target column
# emotional_stability, stress_handling, creativity were dropped during training
# so they should NOT be included in prediction input.

user_input = {}

# Create sliders in 2 columns
col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    if i % 2 == 0:
        user_input[feature] = col1.slider(
            feature.replace('_', ' ').title(),
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )
    else:
        user_input[feature] = col2.slider(
            feature.replace('_', ' ').title(),
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

# Predict button
if st.button("Predict Personality"):
    input_df = pd.DataFrame([user_input])

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input).max() * 100

    # Convert numeric prediction to actual label
    label_map = {
        0: "Ambivert",
        1: "Extrovert",
        2: "Introvert"
    }

    final_prediction = label_map.get(prediction, "Unknown")

    st.success(f"Predicted Personality Type: **{final_prediction}**")
    st.info(f"Confidence Score: **{probability:.2f}%**")

    st.subheader("Your Input Summary")
    st.dataframe(input_df)

