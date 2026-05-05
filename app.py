"""
Stroke Risk Prediction — Streamlit app.

A simple clinical-screening UI around the trained AutoGluon model.
Loads the model from ./model/, collects patient inputs, and returns
a stroke probability + flag/routine decision based on the chosen threshold.

Run locally:
    streamlit run app.py

Deployed via Streamlit Community Cloud (or Hugging Face Spaces) — see README.
"""

import json
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

# Form-field defaults — also the canonical key set for JSON import.
FIELD_DEFAULTS = {
    "gender": "Male",
    "age": 50,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 100.0,
    "bmi": 25.0,
    "smoking_status": "never smoked",
}

# Built-in demo patients — one-click presets for the presentation.
PRESETS = {
    "Low risk": {
        "gender": "Female",
        "age": 28,
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "No",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 85.0,
        "bmi": 22.0,
        "smoking_status": "never smoked",
    },
    "High risk": {
        "gender": "Male",
        "age": 78,
        "hypertension": 1,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 240.0,
        "bmi": 36.0,
        "smoking_status": "smokes",
    },
}


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

# Seed session-state defaults BEFORE widgets render so JSON import can
# overwrite them on a subsequent run.
for _k, _v in FIELD_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _apply_patient(p: dict) -> None:
    """Copy fields from a patient dict into session_state, ignoring unknown keys."""
    for k, v in p.items():
        if k in FIELD_DEFAULTS:
            st.session_state[k] = v


# Sidebar — quick-pick presets, JSON import, adjustable threshold
with st.sidebar:
    st.header("Examples")
    st.caption("One-click patients for the presentation demo.")
    preset_cols = st.columns(len(PRESETS))
    for col, (label, patient) in zip(preset_cols, PRESETS.items()):
        with col:
            if st.button(label, use_container_width=True, key=f"preset_{label}"):
                _apply_patient(patient)
                st.rerun()

    st.header("Import patient")
    uploaded = st.file_uploader(
        "Patient JSON",
        type=["json"],
        help=(
            "Upload either a single patient dict, or a dict of named "
            "patients (e.g. `{\"Mr. A\": {...}, \"Ms. B\": {...}}`). "
            "Recognized keys: " + ", ".join(FIELD_DEFAULTS) + "."
        ),
    )
    if uploaded is not None:
        try:
            data = json.load(uploaded)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            data = None

        if isinstance(data, dict) and data:
            if all(isinstance(v, dict) for v in data.values()):
                # Dict of named patients
                choice = st.selectbox("Choose patient", list(data.keys()))
                if st.button("Load patient", use_container_width=True):
                    _apply_patient(data[choice])
                    st.rerun()
            else:
                # Single patient dict
                if st.button("Load patient", use_container_width=True):
                    _apply_patient(data)
                    st.rerun()
        elif data is not None:
            st.error("JSON must be a non-empty object.")

    with st.expander("JSON format example"):
        st.code(
            json.dumps(
                {
                    "Example patient": {
                        "gender": "Male",
                        "age": 67,
                        "hypertension": 1,
                        "heart_disease": 0,
                        "ever_married": "Yes",
                        "work_type": "Private",
                        "Residence_type": "Urban",
                        "avg_glucose_level": 180.5,
                        "bmi": 32.0,
                        "smoking_status": "formerly smoked",
                    },
                },
                indent=2,
            ),
            language="json",
        )

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
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
    age = st.slider("Age", min_value=0, max_value=100, key="age")
    hypertension = st.selectbox(
        "Hypertension",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="hypertension",
    )
    heart_disease = st.selectbox(
        "Heart disease",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="heart_disease",
    )
    ever_married = st.selectbox("Ever married", ["Yes", "No"], key="ever_married")

with col2:
    work_type = st.selectbox(
        "Work type",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        key="work_type",
    )
    residence_type = st.selectbox(
        "Residence type", ["Urban", "Rural"], key="Residence_type"
    )
    avg_glucose_level = st.slider(
        "Average glucose level (mg/dL)",
        min_value=40.0,
        max_value=300.0,
        step=1.0,
        key="avg_glucose_level",
    )
    bmi = st.slider(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        step=0.1,
        key="bmi",
    )
    smoking_status = st.selectbox(
        "Smoking status",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
        key="smoking_status",
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
