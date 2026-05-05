# Stroke Risk Prediction — Applied AI Bootcamp (Week 3)

A binary classification project that predicts whether a patient is at risk of stroke, using **AutoGluon TabularPredictor**.

**Live demo:** https://stroke-prediction-ml-ewi5zpm22bn6z9g38rbwky.streamlit.app/

---

## Dataset

**Source:** [Stroke Prediction Dataset (Kaggle, fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

**Description:** ~5,110 anonymized patient records with 11 features and a binary label.

**Features:**

| Feature | Type | Description |
|---|---|---|
| `gender` | categorical | Male / Female / Other |
| `age` | numerical | Patient age in years |
| `hypertension` | binary | 1 if patient has hypertension |
| `heart_disease` | binary | 1 if patient has heart disease |
| `ever_married` | categorical | Yes / No |
| `work_type` | categorical | Private / Self-employed / Govt_job / children / Never_worked |
| `Residence_type` | categorical | Urban / Rural |
| `avg_glucose_level` | numerical | Average glucose level in blood |
| `bmi` | numerical | Body Mass Index (has missing values) |
| `smoking_status` | categorical | formerly smoked / never smoked / smokes / Unknown |
| `stroke` | **target** | 1 if patient had a stroke, 0 otherwise |

### Why this dataset

1. **Real-world impact.** Stroke is a leading cause of death and disability worldwide. Early identification of high-risk patients lets clinicians intervene with lifestyle counseling, medication, or follow-up screening.
2. **Tabular and clean.** Mix of numerical and categorical columns — a perfect fit for AutoGluon's TabularPredictor.
3. **Manageable scale.** ~5,110 rows trains in minutes on a laptop, but is large enough to produce meaningful metrics.
4. **Interesting modeling challenges to discuss:**
   - **Severe class imbalance** — only ~5% of patients had a stroke. Plain accuracy would be misleading; we use **ROC-AUC** as the primary metric.
   - **Missing values in BMI** — about 4% of rows are missing BMI. AutoGluon handles this automatically.

---

## ML Problem Framing

### 1. Goal (in non-ML terms)

**Application:** Healthcare — preventive stroke screening in a primary-care setting.

**Goal:** Help clinicians and nurses flag patients who are at elevated risk of stroke, so those patients can be prioritized for follow-up evaluation, lifestyle counseling, or preventive treatment.

**Description:** During a routine primary-care visit, a clinician enters the patient's basic demographics and routinely-collected clinical measurements (age, blood pressure status, heart disease history, BMI, glucose level, smoking status, etc.). The tool returns a stroke risk score, helping the clinician decide whether the patient needs further attention.

**Current (non-ML) solution:** Today, this kind of screening relies almost entirely on the clinician's intuition and experience. There is no consistent formal scoring step — risk assessment varies from doctor to doctor, depends heavily on years of experience, and can miss patients who don't fit a clinician's mental "high-risk" profile.

**ML Task type:** **Binary Classification** — *Healthcare → Stroke risk → Classification* (per the bootcamp's ML Task table). Output is `1` (at-risk) or `0` (not at-risk), with a probability score that supports thresholding.

### 2. Why ML over the non-ML baseline

| | ML approach |
|---|---|
| **Difference** | Captures non-linear interactions between risk factors (e.g., age × glucose × BMI) that a doctor's intuition cannot. Flags at-risk patients whose individual measurements look unremarkable in isolation but are dangerous in combination. |
| **Cost** | Low. Small dataset, minutes to train, milliseconds per prediction. No specialized hardware. |
| **Maintenance** | Low. Quarterly retraining as new records arrive. AutoGluon handles preprocessing, model selection, and ensembling. |
| **Expertise** | Low to moderate. AutoGluon abstracts away most ML complexity — one data scientist can maintain the pipeline. |

### 3. Does ART apply to the data?

| | Verdict | Detail |
|---|---|---|
| **Available** | ✅ Yes | All 11 features are routinely captured at primary-care visits (age, BMI, glucose, smoking status, etc.). Heart disease and hypertension flags are typically already in the patient record. |
| **Representative** | ⚠️ Unknown | The dataset does not disclose country, hospital system, or time period of collection. Cannot be assumed representative of any specific population. |
| **Trusted** | ⚠️ Limited | The author lists the source as "Confidential" and recommends the dataset for **educational use only**. Sufficient for this bootcamp project; not sufficient for real clinical deployment. |

### 4. Data quantity and quality

**Quantity:**
5,110 rows, 11 features. Small in absolute terms but typical for healthcare studies. The real bottleneck is the **rare positive class — only 249 stroke cases (~5%)**, which limits how confidently the model can learn the "stroke=1" pattern.

**Quality:**

| Aspect | Finding |
|---|---|
| **Duplicates** | 0 duplicate rows, 0 duplicate IDs |
| **Missing values** | `bmi` missing in 201 rows (3.93%) |
| **Class balance** | 4,861 negative vs 249 positive (~5%) — severe imbalance |
| **`smoking_status = "Unknown"`** | 30.2% of rows — information gap, treated as its own category |
| **Edge cases** | 43 rows with age < 1 (infants) |
| **Outliers** | Max BMI 97.6 (13 rows > 60); glucose up to 271 |

### 5. Feature engineering

One engineered feature was added on top of the raw dataset:

| New feature | How it's built | Why |
|---|---|---|
| `bmi_category` | Discretize `bmi` into WHO clinical bands: **Underweight** (<18.5), **Normal** (18.5–24.9), **Overweight** (25.0–29.9), **Obese** (≥30.0). Missing values → **Unknown**. | Mirrors how clinicians actually interpret BMI. Gives the model a categorical signal that matches medical guidelines, and keeps missing-BMI rows usable without dropping them. |

No other manual feature engineering was applied. AutoGluon handles categorical encoding, missing-value imputation, and standard preprocessing internally.

### 6. Predictive power

**Method:** AutoGluon's permutation feature importance on the test set.

| Tier | Features | Importance |
|---|---|---|
| **Dominant** | `age` | **0.265** |
| **Meaningful** | `avg_glucose_level`, `ever_married`, `bmi_category` | 0.005 – 0.019 |
| **Negligible** | `work_type`, `smoking_status`, `gender`, `heart_disease`, `hypertension`, `Residence_type` | ≤ 0.003 |

**Matched expectation:** `age` dominates by ~14× the next feature.

**Surprise:** `hypertension` and `heart_disease` — textbook stroke risk factors — show near-zero importance. Both are strongly age-correlated, so once `age` is in the model they add no extra signal. `ever_married` ranks above them as an age proxy.

### 7. From prediction to decision

The model outputs a **probability of stroke** for each patient. To turn that into a clinical action, we apply a probability threshold.

**Chosen decision rule:**

| Predicted probability | Action |
|---|---|
| `prob ≥ 0.08` | **Flag for follow-up** — schedule an additional consultation, run further screening, or recommend lifestyle counseling. |
| `prob < 0.08` | **Routine care** — no additional action. |

At this threshold the model catches ~60% of stroke patients, while flagging ~17% of all patients for follow-up (precision ~17%, i.e., roughly 1 in 6 flagged patients actually had a stroke).

**Note:** if the clinical priority is *"missing a stroke is unacceptable"*, the threshold can be lowered to **0.05**, which raises recall to ~74% (catches 37 of every 50 stroke patients) at the cost of a larger flagged group (~26% of all patients) and slightly lower precision (~14%). The threshold is a deployment-time decision and can be tuned per clinical workflow.

### 8. Model metrics

**Primary metric: ROC-AUC.** Chosen because the dataset is severely imbalanced (~5% positive class), making accuracy misleading.

**Headline result on the held-out test set (1,022 patients, 50 positives):**

| Metric | Value | Notes |
|---|---|---|
| **ROC-AUC** | **0.806** | Solid; comparable to published results on this dataset. |
| Accuracy | 0.951 | Misleading — a "predict no stroke for everyone" baseline scores the same. |

**Threshold-dependent metrics at the chosen decision threshold (0.08):**

| Metric | Value |
|---|---|
| Precision | 0.17 |
| Recall | 0.60 |
| F1 | 0.27 |
| Patients flagged | 176 / 1,022 (~17%) |

AutoGluon also reports these metrics automatically via `predictor.evaluate()` and `predictor.leaderboard()`.

### 9. Success / failure criteria

Success and failure are measured on **clinical outcomes**, not on the model's metrics directly.

**Success:** in a 6-month pilot, patients flagged by the tool receive follow-up screening, lifestyle counseling, or preventive treatment at a measurably higher rate than the pre-tool baseline — and the tool catches at-risk patients earlier than clinician intuition alone would have.

**Failure:** patients flagged by the tool show no different follow-up, intervention, or outcome rates than non-flagged patients, OR clinicians override the tool's recommendations on the majority of visits (indicating low trust / poor fit with workflow).

---

## Setup and Usage

### Try the live demo (no install required)

Open the deployed Streamlit app: https://stroke-prediction-ml-ewi5zpm22bn6z9g38rbwky.streamlit.app/

Enter patient details, adjust the decision threshold if you want, and click **Predict**.

### Run the app locally

Requires **Python 3.11** (recommended) and the trained model artifacts under `model/`.

```bash
# Clone the repo
git clone https://github.com/TurkiOmran/stroke-prediction-ML.git
cd stroke-prediction-ML

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Re-train the model

Open `Copy_of_ML_Project.ipynb` in Google Colab (or a local Jupyter), upload the dataset CSV, and run all cells. The notebook covers EDA, feature engineering, training with AutoGluon, threshold selection, feature importance, and saving the model. The saved model folder can be downloaded as `autogluon_model.zip` (last cell) and unzipped into `model/`.

### Programmatic inference (without the Streamlit UI)

```python
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
# → {'probability': 0.1433, 'decision': 'FLAG for follow-up', 'threshold': 0.08}
```

