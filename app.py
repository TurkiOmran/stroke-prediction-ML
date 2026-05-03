"""
Stroke Risk Prediction — Streamlit app.

A simple clinical-screening UI around the trained AutoGluon model.
Loads the model from ./model/, collects patient inputs, and returns
a stroke probability + flag/routine decision based on the chosen threshold.

Run locally:
    streamlit run app.py

Deployed via Streamlit Community Cloud (or Hugging Face Spaces) — see README.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from autogluon.tabular import TabularPredictor


# --- Configuration ---------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "model"

# Decision threshold (see README, Point 7).
# 0.08 → ~60% recall, ~17% of patients flagged.
# 0.05 → high-sensitivity alternative for "missing a stroke is unacceptable".
DEFAULT_THRESHOLD = 0.08


# --- Model loading (cached so it only runs once per session) ---------------

@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    # require_py_version_match=False — model was trained on Colab's Python 3.12;
    # this lets it load on 3.11/3.12 deploy environments without re-training.
    return TabularPredictor.load(str(MODEL_PATH), require_py_version_match=False)


# --- Feature engineering ---------------------------------------------------

def bmi_band(b: float) -> str:
    """Discretize BMI into WHO clinical bands (matches the notebook)."""
    if b is None or pd.isna(b):
        return "Unknown"
    if b < 18.5:
        return "Underweight"
    if b < 25:
        return "Normal"
    if b < 30:
        return "Overweight"
    return "Obese"


# --- Streamlit UI ----------------------------------------------------------

st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon=None,
    layout="centered",
)

st.title("Stroke Risk Prediction")
st.write(
    "Clinical-screening tool that estimates a patient's stroke risk from "
    "routinely-collected health data. Built on an AutoGluon model trained "
    "on the Kaggle Stroke Prediction Dataset."
)
st.caption(
    "For educational use only. Not a diagnostic tool. "
    "See the README for model framing, metrics, and limitations."
)

st.divider()

predictor = load_predictor()

# Sidebar — adjustable threshold
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.01,
        max_value=0.50,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help=(
            "Probability cutoff above which the patient is flagged for "
            "follow-up. Default 0.08 (catches ~60% of stroke patients). "
            "Lower to 0.05 for higher sensitivity."
        ),
    )

# Two-column input form
st.subheader("Patient details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", min_value=0, max_value=100, value=50)
    hypertension = st.selectbox(
        "Hypertension",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    heart_disease = st.selectbox(
        "Heart disease",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    ever_married = st.selectbox("Ever married", ["Yes", "No"])

with col2:
    work_type = st.selectbox(
        "Work type",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    )
    residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
    avg_glucose_level = st.slider(
        "Average glucose level (mg/dL)",
        min_value=40.0,
        max_value=300.0,
        value=100.0,
        step=1.0,
    )
    bmi = st.slider(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
    )
    smoking_status = st.selectbox(
        "Smoking status",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
    )

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    patient = {
        "gender": gender,
        "age": float(age),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": float(avg_glucose_level),
        "smoking_status": smoking_status,
        "bmi_category": bmi_band(bmi),
    }

    df = pd.DataFrame([patient])
    proba = float(predictor.predict_proba(df)[1].iloc[0])

    st.subheader("Result")
    st.metric("Stroke probability", f"{proba:.2%}")

    if proba >= threshold:
        st.error(
            f"FLAG for follow-up — probability {proba:.2%} is at or above "
            f"the threshold of {threshold:.0%}."
        )
        st.write(
            "**Suggested action:** schedule an additional consultation, "
            "run further screening, or recommend lifestyle counseling."
        )
    else:
        st.success(
            f"Routine care — probability {proba:.2%} is below the "
            f"threshold of {threshold:.0%}."
        )

st.divider()
st.caption(
    "**Data source:** [Stroke Prediction Dataset (Kaggle, fedesoriano)]"
    "(https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).  \n"
    "**Model:** AutoGluon TabularPredictor — `WeightedEnsemble_L2` "
    "(95% NeuralNetFastAI + 5% ExtraTreesGini)."
)
