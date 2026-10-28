import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.title("Rainfall Category Prediction from ENSO")
st.write("Please enter all the informations")

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "ENSO_India_Climate_Dataset_1950_2026.xlsx"

#Load Pipeline and Model

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "fe_pipeline.pkl", "rb") as f:
        fe_pipeline = pickle.load(f)

    with open(BASE_DIR / "best_rf.pkl", "rb") as f:
        model = pickle.load(f)

    return fe_pipeline, model

@st.cache_data
def load_data():
    if DATA_FILE.exists():
        return pd.read_excel(DATA_FILE)
    return None

fe_pipeline, model = load_artifacts()
df = load_data()

enso_options = list(fe_pipeline.named_transformers_["nom"]["onehot"].categories_[0])
intensity_options = list(fe_pipeline.named_transformers_["ord"]["ordinal"].categories_[0])

#Buttons
col1, col2 = st.columns(2)

with col1:
    year = st.slider("Year", 2024, 2030, 2026)
    enso = st.selectbox("ENSO Type", enso_options)
    intensity = st.selectbox("ENSO Intensity", intensity_options)

with col2:
    oni = st.number_input("ONI Index", min_value = -2.0, 
                          max_value = 2.0, value = 0.0, step = 0.1, 
                          help = "Positive value = El Nino, Negative value = La Nina")
    duration = st.number_input("Duration (Months)", min_value = 0, max_value = 24,
                               value = 6, step = 1)
    temp = st.number_input("Avg Temp Anomaly", min_value = 2.0, max_value = 3.0,
                           value = 2.0, step = 0.5)

oni_intensity = oni * duration

#Prediction Button

if st.button("Predict Rainfall Category", type = "primary"):

    input_data = pd.DataFrame({
        "Year": [year],
        "ENSO_Type": [enso],
        "Intensity": [intensity],
        "ONI_Index": [oni],
        "Duration_Months": [duration],
        "Avg_Temp_Anomaly_C": [temp],
        "ONI_Intensity": [oni_intensity]
    })

    X_fe = fe_pipeline.transform(input_data)
    X_fe = pd.DataFrame(X_fe, columns= fe_pipeline.get_feature_names_out())

    prediction = model.predict(X_fe)[0]
    probabilities = model.predict_proba(X_fe)[0]

    st.success(f"Predicted Rainfall Category: **{prediction}**")

    prob_df = pd.DataFrame({
        "category": model.classes_,
        "Probability": probabilities
    }).sort_values("Probability", ascending = False)

    