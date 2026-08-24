# Flight-Delay-Predictor
# ✈️ SkyPredict AI: Multi-Task Flight Delay Prediction & Risk Diagnostics

SkyPredict AI is an end-to-end machine learning pipeline and interactive web application designed to assess flight delay risks before departure.

### 🌟 Key Capabilities
* **Task A (Classification):** Binary classification to predict flight delay probability using an optimized custom threshold (0.22) to handle class imbalance.
* **Task B (Regression):** Predicts the exact delay duration in minutes for flagged flights.
* **Task C (Multi-Class):** Diagnoses the primary root cause (Carrier, Weather, NAS, Late Aircraft, Security) using balanced sample weights.
* **Interactive UI:** A production-ready Streamlit dashboard for real-time inference and risk analytics.

### 🛠️ Tech Stack
* **Machine Learning:** XGBoost (Classifier & Regressor), Scikit-Learn
* **Data Processing:** Pandas, NumPy, Joblib
* **Deployment & UI:** Streamlit
