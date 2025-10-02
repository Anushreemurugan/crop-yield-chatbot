import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import json
from datetime import datetime

# OpenWeatherMap API Key (Replace with your actual API key)
API_KEY = 'a5c4d7596f1d44f689f39ccec6f68de4'  # REPLACE WITH YOUR VALID KEY

# Function to Get District Coordinates
def get_district_coords(district):
    """Fetch latitude and longitude using OpenWeatherMap Geocoding API"""
    if API_KEY == 'YOUR_OPENWEATHERMAP_API_KEY':
        st.error("Invalid API key. Please replace 'YOUR_OPENWEATHERMAP_API_KEY' with a valid key.")
        return None
    url = f'http://api.openweathermap.org/geo/1.0/direct?q={district},India&limit=1&appid={API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data:
            return {'lat': data[0]['lat'], 'lon': data[0]['lon']}
        else:
            st.warning(f"No coordinates found for district '{district}'. Using historical averages.")
            return None
    except requests.RequestException as e:
        st.warning(f"Error fetching coordinates for {district}: {e}. Using historical averages.")
        return None

# Function to Fetch Real-Time Climate Data
def get_realtime_climate(district):
    """Fetch current weather data using coordinates from Geocoding API"""
    coords = get_district_coords(district)
    if coords is None:
        return None
    lat = coords['lat']
    lon = coords['lon']
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        climate = {
            'T2M': data['main']['temp'] - 273.15,
            'RH2M': data['main']['humidity'],
            'PRECTOTCORR': data.get('rain', {}).get('1h', 0) * 24,  # Convert hourly to daily
            'WS2M': data['wind']['speed'],
            'T2M_MAX': data['main']['temp_max'] - 273.15 if 'temp_max' in data['main'] else data['main']['temp'] - 273.15,
            'T2M_MIN': data['main']['temp_min'] - 273.15 if 'temp_min' in data['main'] else data['main']['temp'] - 273.15,
            'ALLSKY_SFC_SW_DWN': 0,  # Placeholder
            'EVPTRNS': 0  # Placeholder
        }
        return climate
    except requests.RequestException as e:
        st.warning(f"Error fetching weather data for {district}: {e}. Using historical averages.")
        return None

# Load Artifacts
try:
    with open('project/lgb_model.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    with open('project/le_district.pkl', 'rb') as f:
        le_district = pickle.load(f)
    with open('project/le_crop.pkl', 'rb') as f:
        le_crop = pickle.load(f)
    with open('project/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('project/means.pkl', 'rb') as f:
        means = pickle.load(f)
    with open('project/season_map.pkl', 'rb') as f:
        season_map = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error: Missing artifact file {e.filename}. Please ensure all .pkl files are in the same directory.")
    st.stop()

# Load dataset for historical averages
try:
    df = pd.read_excel('merged_monthly_dataset.xlsx')
    df['Year'] = pd.to_datetime(df['Year'], errors='coerce').dt.year
    df_clean = df[df['Crop'].notna() & (df['Area (Hectare)'] > 0) & (df['Yield (Tonne/Hectare)'] > 0)].copy()
    df_clean['Season'] = df_clean['Season'].map(season_map).fillna(0)
    df_clean['District_Enc'] = le_district.transform(df_clean['District'])
    df_clean['Crop_Enc'] = le_crop.transform(df_clean['Crop'])
    df_clean = df_clean.rename(columns={'Area (Hectare)': 'Area_Hectare'})
except FileNotFoundError:
    st.error("Error: 'merged_monthly_dataset.xlsx' not found. Please ensure the dataset is in the same directory.")
    st.stop()

# Features
features = ['Year', 'District_Enc', 'Crop_Enc', 'Season', 'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PRECTOTCORR',
            'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'WS2M', 'Precip_Temp_Interact', 'Area_Hectare']
num_features = ['ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PRECTOTCORR', 'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'WS2M',
                'Precip_Temp_Interact', 'Area_Hectare']

# Get Historical Climate
def get_historical_climate(district):
    district_data = df_clean[df_clean['District'] == district]
    if district_data.empty:
        st.warning(f"No data for {district}. Using global averages.")
        return df_clean[num_features[:-2]].mean().to_dict()
    historical = district_data[num_features[:-2]].mean().to_dict()
    return historical

# Prediction Function
def predict_suitability(district, crop, season='Kharif', year=2025, area=5000, climate_data=None):
    try:
        dist_enc = le_district.transform([district])[0]
        crop_enc = le_crop.transform([crop])[0]
        season_num = season_map[season]
    except ValueError:
        st.error(f"Invalid district '{district}' or crop '{crop}'. Use exact spelling (e.g., Ariyalur, Rice).")
        return None, None, None
    except KeyError:
        st.error(f"Invalid season '{season}'. Choose from {list(season_map.keys())}.")
        return None, None, None
    # Use provided climate data
    climate = climate_data
    climate['ALLSKY_SFC_SW_DWN'] = df_clean['ALLSKY_SFC_SW_DWN'].mean()
    climate['EVPTRNS'] = df_clean['EVPTRNS'].mean()
    input_df = pd.DataFrame({
        'Year': [year],
        'District_Enc': [dist_enc],
        'Crop_Enc': [crop_enc],
        'Season': [season_num],
        'Area_Hectare': [area]
    })
    for key in num_features[:-2]:
        input_df[key] = climate[key]
    input_df['Precip_Temp_Interact'] = input_df['PRECTOTCORR'] * input_df['T2M']
    input_scaled = input_df[features].copy()
    input_scaled[num_features] = scaler.transform(input_scaled[num_features])
    try:
        yield_pred = lgb_model.predict(input_scaled)[0]
        thresh = means.get((district, crop), np.median(df_clean['Yield (Tonne/Hectare)']))
        suitable = yield_pred > thresh
        return yield_pred, suitable, thresh
    except ValueError:
        st.error(f"Error during prediction for district '{district}' or crop '{crop}'.")
        return None, None, None

# Suggest Crops
def suggest_crops(district, season, current_crop=None, year=2025, top_k=2, climate_data=None):
    try:
        dist_enc = le_district.transform([district])[0]
    except ValueError:
        st.error(f"No crops found for district '{district}'.")
        return []
    possible_crops = df_clean[df_clean['District'] == district]['Crop'].unique()
    if len(possible_crops) == 0:
        st.error(f"No crops found for district '{district}'.")
        return []
    if current_crop:
        possible_crops = [c for c in possible_crops if c != current_crop]
    preds = []
    global_median = np.median(df_clean['Yield (Tonne/Hectare)'])
    for crop in possible_crops:
        yield_p, _, thresh = predict_suitability(district, crop, season, year, climate_data=climate_data)
        if yield_p is not None:
            hist = thresh
            relative_score = yield_p / hist if hist > 0 else yield_p / global_median
            preds.append((crop, yield_p, relative_score))
    # Sort by relative_score descending
    return sorted(preds, key=lambda x: x[2], reverse=True)[:top_k]

# Streamlit App
st.title("Crop Yield Prediction Chatbot")
st.write("Enter details to predict crop yield and get crop recommendations.")

# User Inputs
districts = list(le_district.classes_)
crops = list(le_crop.classes_)
seasons = list(season_map.keys())

with st.form("prediction_form"):
    user_district = st.selectbox("Select District", districts, help="Choose a district (e.g., Ariyalur)")
    user_crop = st.selectbox("Select Crop", crops, help="Choose a crop (e.g., Rice)")
    user_season = st.selectbox("Select Season", seasons, index=seasons.index('Kharif'), help="Choose a season")
    user_area = st.number_input("Enter Area (Hectare)", min_value=1.0, value=5000.0, step=100.0)
    submitted = st.form_submit_button("Predict Yield and Suggest Crops")

if submitted:
    # Fetch climate data once
    climate = get_realtime_climate(user_district)
    if climate is None:
        climate = get_historical_climate(user_district)
        st.info(f"Using historical climate for {user_district}: Temp={climate['T2M']:.1f}°C, Humidity={climate['RH2M']}%, Precip={climate['PRECTOTCORR']}mm")
    else:
        st.success(f"Fetched real-time climate data for {user_district}: Temp={climate['T2M']:.1f}°C, Humidity={climate['RH2M']}%, Precip={climate['PRECTOTCORR']}mm (daily)")

    # Predict Yield
    st.subheader(f"Prediction for {user_crop} in {user_district} ({user_season})")
    yield_p, suitable, thresh = predict_suitability(user_district, user_crop, user_season, area=user_area, climate_data=climate)
    if yield_p is not None:
        st.write(f"**Predicted Yield**: {yield_p:.2f} T/Ha")
        st.write(f"**Suitable**: {'Yes' if suitable else 'No'} (Historical Mean: {thresh:.2f} T/Ha)")
    else:
        st.error("Prediction failed. Check inputs.")

    # Suggest Crops
    st.subheader(f"Top Alternative Crop Suggestions for {user_district} ({user_season})")

    suggestions = suggest_crops(user_district, user_season, current_crop=user_crop, year=2025, top_k=2, climate_data=climate)
    if suggestions:
        data = {
            'Crop': [s[0] for s in suggestions],
            'Predicted Yield (T/Ha)': [round(s[1], 2) for s in suggestions],
            'Relative Score': [f"{s[2]:.2f}" for s in suggestions]
        }
        suggestions_df = pd.DataFrame(data)
        st.table(suggestions_df)
    else:
        st.error("No crop suggestions available.")
# Add after imports
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #f0f8f0;  /* Light green background */
    }
    .stApp > header { background-color: #2e7d32; }  /* Dark green header */
    .stMarkdown { font-family: 'Arial', sans-serif; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# Set light theme (or dark: st.set_page_config(initial_sidebar_state="collapsed", theme="dark"))
st.set_page_config(page_title="Crop Yield Predictor", layout="wide", theme="light")