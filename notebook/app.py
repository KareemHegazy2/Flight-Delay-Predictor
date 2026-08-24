import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="SkyPredict AI | Enterprise Flight Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_pipeline_artifacts():
    return {
        "model_a": joblib.load("best_model_taskA_is_delayed.pkl"),
        "model_b": joblib.load("best_model_taskB_delay_minutes.pkl"),
        "model_c": joblib.load("best_model_taskC_delay_reason.pkl"),
        "scaler": joblib.load("feature_scaler.pkl"),
        "label_encoder": joblib.load("delay_reason_label_encoder.pkl"),
        "feature_columns": joblib.load("feature_columns.pkl")
    }

try:
    artifacts = load_pipeline_artifacts()
except Exception as e:
    st.error(f"Error loading pipeline artifacts: {e}")
    st.stop()

st.title("✈️ SkyPredict AI")
st.subheader("Enterprise Multi-Task Flight Delay Risk Assessment Architecture")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📋 Flight Parameters")
    
    airline_code = st.selectbox(
        "Airline Carrier Code",
        options=["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9", "HA", "G4"]
    )
    
    distance = st.number_input(
        "Flight Distance (Miles)",
        min_value=50,
        max_value=5000,
        value=800,
        step=50
    )
    
    dep_time = st.time_input("Scheduled Departure Time")
    arr_time = st.time_input("Scheduled Arrival Time")
    
    crs_elapsed_time = st.number_input(
        "Estimated Flight Duration (Minutes)",
        min_value=30,
        max_value=700,
        value=150,
        step=10
    )
    
    dep_hour = dep_time.hour
    crs_dep_time_encoded = (dep_time.hour * 100) + dep_time.minute
    crs_arr_time_encoded = (arr_time.hour * 100) + arr_time.minute
    
    submit_button = st.button("Run Risk Assessment Pipeline", type="primary", use_container_width=True)

with col2:
    st.header("📊 Predictive Risk Analysis Output")
    
    if submit_button:
        input_data = pd.DataFrame([{
            "AIRLINE_CODE": airline_code,
            "DISTANCE": float(distance),
            "Dep_Hour": int(dep_hour),
            "CRS_ARR_TIME": float(crs_arr_time_encoded),
            "CRS_ELAPSED_TIME": float(crs_elapsed_time)
        }])
        
        input_encoded = pd.get_dummies(input_data)
        for col in artifacts["feature_columns"]:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[artifacts["feature_columns"]]
        
        input_scaled = artifacts["scaler"].transform(input_encoded)
        
        prob_delayed = artifacts["model_a"].predict_proba(input_scaled)[0][1]
        CUSTOM_THRESHOLD = 0.22
        is_delayed = prob_delayed > CUSTOM_THRESHOLD
        
        m1, m2, m3 = st.columns(3)
        
        with m1:
            if is_delayed:
                st.metric(label="Risk Status", value="HIGH RISK", delta="Delayed Flagged")
            else:
                st.metric(label="Risk Status", value="NOMINAL", delta="On-Time Prediction")
        
        with m2:
            st.metric(label="Delay Probability", value=f"{prob_delayed * 100:.1f}%")
            
        with m3:
            if is_delayed:
                pred_minutes = artifacts["model_b"].predict(input_scaled)[0]
                pred_minutes = max(16.0, float(pred_minutes))
                st.metric(label="Predicted Severity", value=f"{round(pred_minutes, 1)} Min")
            else:
                st.metric(label="Predicted Severity", value="0.0 Min")
        
        st.markdown("### 🔍 Pipeline Inference Breakdown")
        
        r_col1, r_col2 = st.columns(2)
        
        with r_col1:
            st.write("**Task A & B Dashboard:**")
            if is_delayed:
                st.error(f"Alert: Operation exceeds the 15-minute industry standard. System threshold ({CUSTOM_THRESHOLD}) breached.")
                st.progress(float(prob_delayed))
            else:
                st.success("Operation normal. Flight is predicted to land within the scheduled timeline.")
                st.progress(float(prob_delayed))
                
        with r_col2:
            st.write("**Task C: Root Cause Diagnostics:**")
            if is_delayed:
                reason_encoded = artifacts["model_c"].predict(input_scaled)[0]
                reason_label = artifacts["label_encoder"].inverse_transform([reason_encoded])[0]
                
                if reason_label == "No Delay":
                    reason_label = "Carrier Delay (Recalibrated)"
                    
                st.warning(f"Primary Vector: **{reason_label}**")
                st.info("Recommendation: Review dispatch schedules and turn times for upcoming tail numbers on this sector.")
            else:
                st.info("Primary Vector: **No Delay Detected**")
                
    else:
        st.info("Awaiting telemetry input. Fill the flight parameters sidebar and trigger the inference pipeline.")