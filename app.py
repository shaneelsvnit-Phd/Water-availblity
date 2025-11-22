import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
# NOTE: Ensure 'reservoir_model.pkl' is uploaded to GitHub in the same folder!
try:
    model = joblib.load('reservoir_model.pkl')
except FileNotFoundError:
    st.error("Model file 'reservoir_model.pkl' not found. Please upload it to your GitHub repository.")
    st.stop()

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Khadakwasla AI Predictor", page_icon="💧")

st.title("💧 Khadakwasla Dam Evaporation Predictor")
st.markdown("""
This AI tool predicts daily water loss based on weather conditions. 
It also visualizes how **Temperature** impacts evaporation trends.
""")
st.write("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("🌤️ Input Weather Conditions")

temp = st.sidebar.slider("Temperature (°C)", min_value=10.0, max_value=45.0, value=30.0)
hum = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
wind = st.sidebar.slider("Wind Speed (km/h)", min_value=0.0, max_value=30.0, value=10.0)
solar = st.sidebar.slider("Solar Radiation (MJ/m²)", min_value=0.0, max_value=30.0, value=20.0)

# --- MAIN PREDICTION SECTION ---
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Predict Evaporation Loss"):
        # Prepare input for the model
        input_data = np.array([[temp, hum, wind, solar]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display Result
        st.success(f"**{prediction:.2f} mm/day**")
        
        # Interpretation
        if prediction > 6:
            st.warning("⚠️ High Loss Alert!")
        else:
            st.info("✅ Normal Levels")

with col2:
    st.write("### Current Inputs")
    st.write(f"🌡️ **Temp:** {temp} °C")
    st.write(f"💧 **Humidity:** {hum}%")
    st.write(f"🌬️ **Wind:** {wind} km/h")

st.write("---")

# --- NEW: VISUALIZATION FEATURE (Sensitivity Analysis) ---
st.header("📈 Temperature Impact Analysis")
st.write("The graph below shows how evaporation would change if the **Temperature** rises, keeping Humidity, Wind, and Solar constant at your selected levels.")

# 1. Generate a range of temperatures (from 10C to 50C)
temp_range = list(range(10, 51))
predictions = []

# 2. Predict evaporation for every temperature in that range
for t in temp_range:
    pred = model.predict([[t, hum, wind, solar]])[0]
    predictions.append(pred)

# 3. Create Dataframe for the chart
chart_data = pd.DataFrame({
    'Temperature (°C)': temp_range,
    'Predicted Evaporation (mm)': predictions
})

# 4. Plot the Interactive Chart
st.line_chart(chart_data.set_index('Temperature (°C)'))

st.caption("Developed by shaneel AI Scientist (Simulation)")
