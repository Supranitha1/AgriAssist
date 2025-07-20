import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Agri Assistant 🌿", layout="wide")
st.title("🌾 Crop and Fertilizer Recommendation System")

@st.cache_data
def load_crop_data():
    return pd.read_csv("dataset/Crop_recommendation.csv")

@st.cache_data
def load_fertilizer_data():
    return pd.read_csv("dataset/Fertilizer_Prediction.csv")

crop_df = load_crop_data()
fert_df = load_fertilizer_data()

# Encode target fertilizer name
fert_label = LabelEncoder()
fert_df['Fertilizer Name'] = fert_label.fit_transform(fert_df['Fertilizer Name'])

# Tabs
tab1, tab2 = st.tabs(["🌱 Crop Recommendation", "🧪 Fertilizer Recommendation"])

# ------------------------
# 🌱 Crop Recommendation Tab
# ------------------------
with tab1:
    st.subheader("🌿 Crop Recommendation Based on Soil & Weather")

    X_crop = crop_df.drop('label', axis=1)
    y_crop = crop_df['label']

    crop_model = RandomForestClassifier()
    crop_model.fit(X_crop, y_crop)

    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (Crop)", 0, 200)
        temperature = st.number_input("Temperature (Crop) (°C)", 0.0, 60.0)
        ph = st.number_input("pH (Crop)", 0.0, 14.0)
    with col2:
        P = st.number_input("Phosphorous (Crop)", 0, 200)
        humidity = st.number_input("Humidity (Crop) (%)", 0.0, 100.0)
    with col3:
        K = st.number_input("Potassium (Crop)", 0, 200)
        rainfall = st.number_input("Rainfall (Crop) (mm)", 0.0, 400.0)

    if st.button("🔍 Recommend Crop"):
        data = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = crop_model.predict(data)[0]
        st.success(f"✅ Recommended Crop: **{prediction.capitalize()}**")

# ------------------------
# 🧪 Fertilizer Recommendation Tab
# ------------------------
with tab2:
    st.subheader("🧪 Fertilizer Recommendation Based on Crop & Conditions")

    fert_X_raw = fert_df.drop(['Fertilizer Name'], axis=1)
    fert_y = fert_df['Fertilizer Name']
    fert_X = pd.get_dummies(fert_X_raw)

    fert_model = RandomForestClassifier()
    fert_model.fit(fert_X, fert_y)

    crops = fert_df['Crop'].unique().tolist()
    soil_types = fert_df['Soil Type'].unique().tolist()
    nutrients = fert_df['Nutrient'].unique().tolist()

    crop_input = st.selectbox("Select Crop (Fertilizer)", sorted(crops))
    temp = st.number_input("Temperature (Fertilizer) (°C)", 0.0, 60.0)
    humidity_f = st.number_input("Humidity (Fertilizer) (%)", 0.0, 100.0)
    moisture = st.number_input("Moisture (Fertilizer) (%)", 0.0, 100.0)
    soil = st.selectbox("Soil Type (Fertilizer)", sorted(soil_types))
    nutrient = st.selectbox("Deficient Nutrient (Fertilizer)", sorted(nutrients))

    if st.button("🧪 Recommend Fertilizer"):
        input_dict = {
            'Temperature': temp,
            'Humidity': humidity_f,
            'Moisture': moisture,
            'Soil Type': soil,
            'Crop': crop_input,
            'Nutrient': nutrient
        }

        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=fert_X.columns, fill_value=0)

        prediction = fert_model.predict(input_df)[0]
        fert_name = fert_label.inverse_transform([prediction])[0]
        st.success(f"🧾 Recommended Fertilizer: **{fert_name}**")
