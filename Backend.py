import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Rail Asset Failure Prediction",
    page_icon="🚆",
    layout="wide"
)

# Load models (you'll need to have these files)
@st.cache_resource
def load_models():
    try:
        lstm_model = load_model('models/lstm_rail_model.h5')
        iso_forest = joblib.load('models/isolation_forest.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return lstm_model, iso_forest, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

lstm_model, iso_forest, scaler = load_models()

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
vibration = st.sidebar.slider("Vibration", 0.0, 20.0, 5.0)
temperature = st.sidebar.slider("Temperature (°C)", 20.0, 100.0, 30.0)
rolling_avg_window = st.sidebar.slider("Rolling Average Window", 1, 48, 24)

# Main app
st.title("🚆 Rail Asset Failure Prediction System")
st.markdown("""
This system predicts potential rail asset failures using vibration and temperature sensor data.
""")

# Calculate features
def calculate_features(vib, temp, window):
    # Simulate rolling averages (in a real app, you'd use actual rolling data)
    vib_rolling = vib * np.random.uniform(0.9, 1.1)
    temp_rolling = temp * np.random.uniform(0.9, 1.1)
    
    # Create feature vector
    features = np.array([[vib, temp, vib_rolling, temp_rolling, 0, 0, 0]])  # Last 3 are FFT placeholders
    if scaler:
        features = scaler.transform(features)
    return features

# Prediction function
def make_prediction(features):
    if lstm_model:
        # Reshape for LSTM (samples, timesteps, features)
        lstm_input = features.reshape(1, 1, -1)
        failure_prob = lstm_model.predict(lstm_input)[0][0]
        return failure_prob
    return None

# Anomaly detection
def detect_anomaly(features):
    if iso_forest:
        # Use first 5 features for anomaly detection
        anomaly_score = iso_forest.decision_function(features[:, :5])
        return anomaly_score[0]
    return None

# When user interacts
if st.sidebar.button("Predict"):
    features = calculate_features(vibration, temperature, rolling_avg_window)
    
    # Make predictions
    failure_prob = make_prediction(features)
    anomaly_score = detect_anomaly(features)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Failure Prediction")
        if failure_prob is not None:
            st.metric("Failure Probability", f"{failure_prob:.2%}")
            
            # Visual indicator
            fig, ax = plt.subplots(figsize=(3, 1))
            ax.barh(['Risk'], [failure_prob], color='red' if failure_prob > 0.5 else 'green')
            ax.set_xlim(0, 1)
            st.pyplot(fig)
            
            if failure_prob > 0.7:
                st.error("🚨 High probability of failure - Immediate inspection recommended!")
            elif failure_prob > 0.5:
                st.warning("⚠️ Moderate probability of failure - Schedule inspection")
            else:
                st.success("✅ Normal operation")
    
    with col2:
        st.subheader("Anomaly Detection")
        if anomaly_score is not None:
            st.metric("Anomaly Score", f"{anomaly_score:.2f}")
            
            # Visual indicator
            fig, ax = plt.subplots(figsize=(3, 1))
            ax.barh(['Anomaly'], [abs(anomaly_score)], color='orange' if anomaly_score < 0 else 'blue')
            st.pyplot(fig)
            
            if anomaly_score < 0:
                st.error("🚨 Anomaly detected in sensor readings!")
            else:
                st.success("✅ Normal sensor behavior")

# Data visualization section
st.header("Historical Data Simulation")
days_to_show = st.slider("Days to simulate", 1, 30, 7)

@st.cache_data
def generate_historical_data(days):
    hours = days * 24
    time = pd.date_range(end=pd.Timestamp.now(), periods=hours, freq='h')
    vibration = np.sin(np.linspace(0, days*2*np.pi, hours)) * 5 + np.random.normal(0, 1, hours)
    temperature = np.linspace(20, 35, hours) + np.random.normal(0, 3, hours)
    return pd.DataFrame({'Time': time, 'Vibration': vibration, 'Temperature': temperature})

historical_data = generate_historical_data(days_to_show)

tab1, tab2 = st.tabs(["Vibration Data", "Temperature Data"])
with tab1:
    st.line_chart(historical_data, x='Time', y='Vibration')
with tab2:
    st.line_chart(historical_data, x='Time', y='Temperature')

# Footer
st.markdown("---")
st.markdown("""
**About this system:**
- Uses LSTM for failure prediction
- Uses Isolation Forest for anomaly detection
- Processes vibration and temperature data
""")