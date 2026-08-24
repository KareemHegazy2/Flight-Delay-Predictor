# ✈️ Flight Delay Predictor: Multi-Task Aviation Risk Analytics

Flight-Delay-Predictor is an end-to-end machine learning system and interactive analytics dashboard designed to assess flight delay risks before departure. The pipeline combines three specialized XGBoost models to deliver delay probability, duration estimates, and root cause diagnostics.

---

## 📊 Dataset Access
Due to GitHub file size limits (500MB+), the full dataset is hosted on Kaggle:
👉 **[Download Dataset from Kaggle](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)**

---

## 🌟 Multi-Task Architecture
* **Task A: Delay Classification (Binary)**
  * Predicts whether a flight will be delayed (>15 minutes threshold).
  * Uses a calibrated **custom threshold ($0.22$)** to optimize sensitivity and address class imbalance.
* **Task B: Delay Duration Estimation (Regression)**
  * Estimates delay length in minutes exclusively for flights flagged as delayed.
* **Task C: Root Cause Diagnostics (Multi-Class)**
  * Identifies the primary delay driver (*Carrier, Weather, NAS, Late Aircraft, Security*) using balanced sample weighting.

---

## 🛠️ Tech Stack & Dependencies
* **Core ML & Processing:** `Python`, `XGBoost`, `Scikit-Learn`, `Pandas`, `NumPy`, `Joblib`
* **Inference App & Dashboard:** `Streamlit`

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/KareemHegazy2/Flight-Delay-Predictor.git](https://github.com/KareemHegazy2/Flight-Delay-Predictor.git)
   cd Flight-Delay-Predictor
