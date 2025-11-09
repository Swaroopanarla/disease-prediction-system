# --------------------------------------------
# Streamlit App: AI-Based Career Recommendation System
# --------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import os

# Get current folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model and encoder safely
MODEL_PATH = os.path.join(BASE_DIR, "career_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

st.set_page_config(page_title="AI Career Recommender", page_icon="🎯", layout="centered")

st.title("🎯 AI-Based Career Recommendation System")
st.write("### Predict your ideal career using personality and aptitude scores!")

# Try to load model and encoder
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    st.success("✅ Model and encoder loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load model or encoder.\n\nError: {e}")
    st.stop()

st.markdown("---")

# Input sliders for the 10 numeric scores
col1, col2 = st.columns(2)

with col1:
    O_score = st.slider("Openness (O_score)", 1.0, 10.0, 5.0)
    C_score = st.slider("Conscientiousness (C_score)", 1.0, 10.0, 5.0)
    E_score = st.slider("Extraversion (E_score)", 1.0, 10.0, 5.0)
    A_score = st.slider("Agreeableness (A_score)", 1.0, 10.0, 5.0)
    N_score = st.slider("Neuroticism (N_score)", 1.0, 10.0, 5.0)

with col2:
    Numerical = st.slider("Numerical Aptitude", 1.0, 10.0, 5.0)
    Spatial = st.slider("Spatial Aptitude", 1.0, 10.0, 5.0)
    Perceptual = st.slider("Perceptual Aptitude", 1.0, 10.0, 5.0)
    Abstract = st.slider("Abstract Reasoning", 1.0, 10.0, 5.0)
    Verbal = st.slider("Verbal Reasoning", 1.0, 10.0, 5.0)

st.markdown("---")

# Button to predict
if st.button("🔍 Recommend My Career"):
    features = pd.DataFrame([[
        O_score, C_score, E_score, A_score, N_score,
        Numerical, Spatial, Perceptual, Abstract, Verbal
    ]], columns=[
        'O_score', 'C_score', 'E_score', 'A_score', 'N_score',
        'Numerical Aptitude', 'Spatial Aptitude',
        'Perceptual Aptitude', 'Abstract Reasoning', 'Verbal Reasoning'
    ])

    try:
        pred_encoded = model.predict(features)
        pred_label = encoder.inverse_transform(pred_encoded)[0]
        st.success(f"### 💡 Recommended Career: **{pred_label}**")
        st.balloons()
    except Exception as e:
        st.error(f"⚠️ Prediction failed:\n\n{e}")

st.markdown("---")
st.caption("Developed by Narla Jyothi Swaroopa | NASSCOM FutureSkills AI Internship")
