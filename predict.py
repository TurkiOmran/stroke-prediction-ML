"""
Stroke Risk Prediction — inference script.

Loads the trained AutoGluon model from ./model/ and runs inference on a
single patient. Returns the probability of stroke and a clinical decision
based on the chosen threshold (0.08).

Usage
-----
    # As a script (uses the example patient defined below):
    python predict.py

    # As a module:
    from predict import predict_stroke
    result = predict_stroke({
        "gender": "Male",
        "age": 67,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 180.5,
        "smoking_status": "formerly smoked",
        "bmi": 32.0,
    })
    print(result)
"""

from pathlib import Path
import pandas as pd
from autogluon.tabular import TabularPredictor


# --- Configuration -----------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "model"

# Decision threshold (see README, Point 7).
# 0.08 → ~60% recall, ~17% of patients flagged.
# 0.05 → high-sensitivity alternative (~74% recall) for "missing a stroke
#        is unacceptable" use cases.
THRESHOLD = 0.08


# --- Feature engineering -----------------------------------------------------

def bmi_band(b):
    """Discretize BMI into WHO clinical bands (matches notebook Cell 25)."""
    if b is None or pd.isna(b):
        return "Unknown"
    if b < 18.5:
        return "Underweight"
    if b < 25:
        return "Normal"
    if b < 30:
        return "Overweight"
    return "Obese"


# --- Load the predictor once at import-time ---------------------------------

_predictor = TabularPredictor.load(str(MODEL_PATH), require_py_version_match=False)


# --- Public API --------------------------------------------------------------

def predict_stroke(patient: dict, threshold: float = THRESHOLD) -> dict:
    """
    Run inference on a single patient.

    Parameters
    ----------
    patient : dict
        Keys expected: gender, age, hypertension, heart_disease, ever_married,
        work_type, Residence_type, avg_glucose_level, smoking_status, bmi.
        (`bmi` is converted to `bmi_category` automatically — same engineering
        as in the notebook.)
    threshold : float, optional
        Probability threshold for the FLAG decision. Default = THRESHOLD (0.08).

    Returns
    -------
    dict with keys:
        probability : float — model's predicted probability of stroke (0..1)
        decision    : str   — "FLAG for follow-up" or "routine care"
        threshold   : float — the threshold actually applied
    """
    # Match the feature engineering used at training time
    row = dict(patient)
    if "bmi_category" not in row:
        row["bmi_category"] = bmi_band(row.pop("bmi", None))

    df = pd.DataFrame([row])

    proba = _predictor.predict_proba(df)[1].iloc[0]
    decision = "FLAG for follow-up" if proba >= threshold else "routine care"

    return {
        "probability": float(proba),
        "decision": decision,
        "threshold": threshold,
    }


# --- Script entry point ------------------------------------------------------

if __name__ == "__main__":
    example_patient = {
        "gender": "Male",
        "age": 67,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 180.5,
        "smoking_status": "formerly smoked",
        "bmi": 32.0,
    }

    result = predict_stroke(example_patient)

    print("Patient profile:")
    for k, v in example_patient.items():
        print(f"  {k}: {v}")
    print()
    print(f"Stroke probability: {result['probability']:.4f}")
    print(f"Threshold:          {result['threshold']}")
    print(f"Decision:           {result['decision']}")
