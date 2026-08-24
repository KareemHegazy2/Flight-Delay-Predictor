# =================================================================
# FLIGHT DELAY PREDICTION SYSTEM - Production Ready Pipeline
# =================================================================

import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_squared_error, 
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42

# =================================================================
# STEP 1: LOAD DATA + SAMPLING
# =================================================================
print("=" * 60)
print("STEP 1: Loading data and taking a 50,000-row sample...")
print("=" * 60)

df = pd.read_csv('../data/Flight Delay and Cancellation Dataset.csv')
df = df.sample(50000, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"✓ Sampled dataset shape: {df.shape}")

# =================================================================
# STEP 2: TARGET VARIABLE ENGINEERING & NEW FEATURES
# =================================================================
print("\n" + "=" * 60)
print("STEP 2: Engineering targets and extracting features...")
print("=" * 60)

# Targets
df["Is_Delayed"] = (df["ARR_DELAY"] > 15).astype(int)
df["Delay_Minutes"] = df["ARR_DELAY"]

delay_cause_cols = [
    "DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY", "DELAY_DUE_LATE_AIRCRAFT"
]

def get_delay_reason(row):
    values = row[delay_cause_cols]
    if values.isna().all() or values.fillna(0).sum() == 0:
        return "No Delay"
    reason = values.fillna(0).idxmax()
    return reason.replace("DELAY_DUE_", "").replace("_", " ").title()

df["Delay_Reason"] = df.apply(get_delay_reason, axis=1)

df["Dep_Hour"] = (df["CRS_DEP_TIME"] // 100).astype(int)

print("✓ Targets and new features created.")

# =================================================================
# STEP 3: STRICT FEATURE SELECTION (PREVENT NOISE)
# =================================================================
print("\n" + "=" * 60)
print("STEP 3: Selecting pre-flight operational features...")
print("=" * 60)

df = df.drop_duplicates().reset_index(drop=True)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in numerical_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode()[0])

important_features = ["AIRLINE_CODE", "DISTANCE", "Dep_Hour", "CRS_ARR_TIME", "CRS_ELAPSED_TIME"]
X = df[important_features].copy()

# =================================================================
# STEP 4 & 5: ENCODING & TARGET PREPARATION
# =================================================================
print("\n" + "=" * 60)
print("STEP 4 & 5: Encoding features and targets...")
print("=" * 60)

X_encoded = pd.get_dummies(X, drop_first=True)

y1 = df["Is_Delayed"]
y2 = df["Delay_Minutes"]

label_encoder = LabelEncoder()
y3 = label_encoder.fit_transform(df["Delay_Reason"])

# =================================================================
# STEP 6 & 7: SPLIT & SCALE
# =================================================================
print("\n" + "=" * 60)
print("STEP 6 & 7: Splitting and Scaling data...")
print("=" * 60)

X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = (
    train_test_split(X_encoded, y1, y2, y3, test_size=0.2, random_state=RANDOM_STATE, stratify=y1)
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


train_delayed_idx = (y1_train == 1)
test_delayed_idx = (y1_test == 1)

X_train_scaled_b = X_train_scaled[train_delayed_idx]
y2_train_b = y2_train[train_delayed_idx]
X_test_scaled_b = X_test_scaled[test_delayed_idx]
y2_test_b = y2_test[test_delayed_idx]

# =================================================================
# STEP 8 & 9: TRAINING & BALANCED EVALUATION
# =================================================================
print("\n" + "=" * 60)
print("STEP 8 & 9: Training & Evaluating Optimized Models...")
print("=" * 60)

# ---- TASK A: Will the flight be delayed? (مع تعديل العتبة ذكياً) ----
print("\n" + "-"*40 + "\n[Task A] Binary Classification (Is_Delayed)\n" + "-"*40)
model_a = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
model_a.fit(X_train_scaled, y1_train, eval_set=[(X_test_scaled, y1_test)], verbose=False)

probs_a = model_a.predict_proba(X_test_scaled)[:, 1]
CUSTOM_THRESHOLD = 0.22
preds_a_custom = (probs_a > CUSTOM_THRESHOLD).astype(int)

print(f"\n[Optimized with Custom Threshold = {CUSTOM_THRESHOLD}]")
print("\nConfusion Matrix for Model A:")
print(confusion_matrix(y1_test, preds_a_custom))
print("\nClassification Report (Balanced Precision & Recall):")
print(classification_report(y1_test, preds_a_custom, target_names=["On-Time", "Delayed"]))

# ---- TASK B: How many minutes delayed? ----
print("\n" + "-"*40 + "\n[Task B] Regression (Delay_Minutes - Only Delayed Flights)\n" + "-"*40)
model_b = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1)
model_b.fit(X_train_scaled_b, y2_train_b, eval_set=[(X_test_scaled_b, y2_test_b)], verbose=False)

preds_b = model_b.predict(X_test_scaled_b)
r2_b = r2_score(y2_test_b, preds_b)
rmse_b = np.sqrt(mean_squared_error(y2_test_b, preds_b))
print(f"✓ Adjusted R2-Score: {r2_b:.4f}")
print(f"✓ RMSE: {rmse_b:.2f} Minutes")

# ---- TASK C: What caused the delay? ----
print("\n" + "-"*40 + "\n[Task C] Multi-class Classification (Delay_Reason)\n" + "-"*40)
sample_weights_c = compute_sample_weight(class_weight='balanced', y=y3_train)

model_c = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1)
model_c.fit(X_train_scaled, y3_train, sample_weight=sample_weights_c, eval_set=[(X_test_scaled, y3_test)], verbose=False)

preds_c = model_c.predict(X_test_scaled)
print("\nClassification Report for Model C (Balanced Exploration):")
print(classification_report(y3_test, preds_c, target_names=label_encoder.classes_, zero_division=0))

# =================================================================
# STEP 10: SAVE ARTIFACTS
# =================================================================
print("\n" + "=" * 60)
print("STEP 10: Saving Models & Metadata...")
print("=" * 60)

joblib.dump(model_a, "best_model_taskA_is_delayed.pkl")
joblib.dump(model_b, "best_model_taskB_delay_minutes.pkl")
joblib.dump(model_c, "best_model_taskC_delay_reason.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(label_encoder, "delay_reason_label_encoder.pkl")
joblib.dump(X_encoded.columns.tolist(), "feature_columns.pkl")
joblib.dump(important_features, "original_feature_cols.pkl")

print("✓ All models saved.")

# =================================================================
# STEP 11: PREDICTION DEMO WITH CUSTOM THRESHOLD INTEGRATION
# =================================================================
print("\n" + "=" * 60)
print("STEP 11: Production Prediction Demo...")
print("=" * 60)

def predict_flight_pipeline(flight_features_df):
    scaler_loaded = joblib.load("feature_scaler.pkl")
    label_encoder_loaded = joblib.load("delay_reason_label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    
    m_a = joblib.load("best_model_taskA_is_delayed.pkl")
    m_b = joblib.load("best_model_taskB_delay_minutes.pkl")
    m_c = joblib.load("best_model_taskC_delay_reason.pkl")
    
    # Preprocessing
    flight_encoded = pd.get_dummies(flight_features_df, drop_first=True)
    for col in feature_columns:
        if col not in flight_encoded.columns:
            flight_encoded[col] = 0
    flight_encoded = flight_encoded[feature_columns]
    flight_scaled = scaler_loaded.transform(flight_encoded)
    
    prob_delayed = m_a.predict_proba(flight_scaled)[0][1]
    is_delayed = bool(prob_delayed > 0.22) # Threshold
    
    delay_minutes = m_b.predict(flight_scaled)[0] if is_delayed else 0.0
    
    reason_encoded = m_c.predict(flight_scaled)[0]
    delay_reason = label_encoder_loaded.inverse_transform([reason_encoded])[0]
    if not is_delayed:
        delay_reason = "No Delay"
        
    return {
        "is_delayed": is_delayed,
        "probability_of_delay": f"{prob_delayed*100:.1f}%",
        "predicted_delay_minutes": round(float(delay_minutes), 2) if is_delayed else 0,
        "primary_cause": delay_reason
    }


sample_original = df[important_features].iloc[X_test.index[0]:X_test.index[0]+1].copy()
demo_prediction = predict_flight_pipeline(sample_original)
print("\n🎯 Production Simulation Example:")
print(demo_prediction)
print("\n" + "=" * 60)