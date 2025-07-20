# 🌾 Agri Assistant

Agri Assistant is a simple and smart web application that recommends the **best crop to grow** and the **right fertilizer to use** based on soil and environmental conditions. It helps farmers make informed decisions using machine learning.

---

## 🚀 Live Application

👉 [Click here to try the app](https://agriassistt.streamlit.app/)

---

## 🧠 Features

### 🌱 Crop Recommendation
- Input: N, P, K, Temperature, Humidity, pH, Rainfall
- Output: Most suitable crop to grow in the given conditions

### 🧪 Fertilizer Recommendation
- Input: Current crop, Soil type, Deficient nutrient, Temperature, Humidity, Moisture
- Output: Ideal fertilizer to use for the selected crop and condition

---

## 📊 Dataset Overview

All data used in this project is synthetically generated for demonstration purposes.

### 📁 `Crop_recommendation.csv`
- 2200 rows × 8 columns
- Features: N, P, K, temperature, humidity, pH, rainfall
- Target: `label` (crop name)

### 📁 `Fertilizer_Prediction.csv`
- 2200 rows × 7 columns
- Features: Temperature, Humidity, Moisture, Soil Type, Crop, Nutrient
- Target: `Fertilizer Name`

---

## 🛠️ Technologies Used

| Technology     | Purpose                         |
|----------------|----------------------------------|
| Streamlit      | Web app framework                |
| Pandas         | Data manipulation                |
| scikit-learn   | Machine learning models          |
| LabelEncoder   | Encoding fertilizer names        |
| RandomForest   | Classification model for both tasks |

---

## 🧾 How It Works

- The app uses `RandomForestClassifier` for both crop and fertilizer prediction.
- Inputs from the user are collected through an interactive Streamlit interface.
- The prediction result is shown instantly.

