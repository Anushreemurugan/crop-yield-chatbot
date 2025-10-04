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
from scipy.stats.mstats import winsorize
# OpenWeatherMap API Key (Replace with your actual API key)
API_KEY = 'a5c4d7596f1d44f689f39ccec6f68de4' # REPLACE WITH YOUR VALID KEY
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
            'PRECTOTCORR': data.get('rain', {}).get('1h', 0) * 24, # Convert hourly to daily
            'WS2M': data['wind']['speed'],
            'T2M_MAX': data['main']['temp_max'] - 273.15 if 'temp_max' in data['main'] else data['main']['temp'] - 273.15,
            'T2M_MIN': data['main']['temp_min'] - 273.15 if 'temp_min' in data['main'] else data['main']['temp'] - 273.15,
            'ALLSKY_SFC_SW_DWN': 0, # Placeholder
            'EVPTRNS': 0 # Placeholder
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
    df['Year'] = df['Year'].astype(str).str.split('-').str[0].astype(int)
    # Aggregate monthly data to seasonal level
    weather_cols = ['ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PRECTOTCORR', 'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'WS2M']
    df_agg = df.groupby(['District', 'Year', 'Crop', 'Season', 'Month']).agg({
        'Area (Hectare)': 'first',
        'Production (Tonnes)': 'first',
        'Yield (Tonne/Hectare)': 'first',
        **{col: 'mean' for col in weather_cols}
    }).reset_index()
    season_months = {
        'Kharif': [6, 7, 8, 9],
        'Rabi': [10, 11, 12, 1, 2, 3],
        'Autumn': [9, 10],
        'Summer': [4, 5],
        'Winter': [12, 1, 2],
        'Whole Year': list(range(1, 13))
    }
    # Fixed apply to avoid deprecation
    df_filtered = df_agg.groupby(['District', 'Year', 'Crop', 'Season'], group_keys=False).apply(
        lambda g: g[g['Month'].isin(season_months.get(g.name[3], list(range(1,13))))]
    ).reset_index(drop=True)
    # Final aggregation
    df_clean = df_filtered.groupby(['District', 'Year', 'Crop', 'Season']).agg({
        'Area (Hectare)': 'first',
        'Yield (Tonne/Hectare)': 'first',
        **{col: 'mean' for col in weather_cols}
    }).reset_index()
    df_clean = df_clean[df_clean['Crop'].notna() & (df_clean['Area (Hectare)'] > 0) & (df_clean['Yield (Tonne/Hectare)'] > 0)].copy()
    # Outlier removal
    def remove_outliers(df, columns):
        df_out = df.copy()
        initial_shape = df.shape[0]
        for col in columns:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
            print(f"Removed outliers in {col}: {initial_shape - df_out.shape[0]} rows")
            initial_shape = df_out.shape[0]
        return df_out
    outlier_columns = ['Yield (Tonne/Hectare)', 'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PRECTOTCORR',
                       'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'WS2M']
    df_clean = remove_outliers(df_clean, outlier_columns)
    # Feature Engineering
    df_clean['Precip_Temp_Interact'] = df_clean['PRECTOTCORR'] * df_clean['T2M']
    df_clean['Season'] = df_clean['Season'].map(season_map).fillna(0)
    df_clean['District_Enc'] = le_district.transform(df_clean['District'])
    df_clean['Crop_Enc'] = le_crop.transform(df_clean['Crop'])
    df_clean = df_clean.rename(columns={'Area (Hectare)': 'Area_Hectare'})
    # Compute diversity factor
    crop_freq = df_clean['Crop'].value_counts()
    diversity_factor = 1 / crop_freq
except FileNotFoundError:
    st.error("Error: 'merged_monthly_dataset.xlsx' not found. Please ensure the dataset is in the same directory.")
    st.stop()
# Features
features = ['Year', 'District_Enc', 'Crop_Enc', 'Season', 'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PRECTOTCORR',
            'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'WS2M', 'Precip_Temp_Interact', 'Area_Hectare']
num_features = ['ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PRECTOTCORR', 'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'WS2M',
                'Precip_Temp_Interact', 'Area_Hectare']
# Get Historical Climate (season-specific)
def get_historical_climate(district, season):
    district_data = df_clean[df_clean['District'] == district]
    if district_data.empty:
        st.warning(f"No data for {district}. Using global averages.")
        return df_clean[num_features[:-2]].mean().to_dict()
    season_num = season_map.get(season, 0)
    season_data = district_data[district_data['Season'] == season_num]
    if season_data.empty:
        season_data = district_data  # Fallback to all seasons
    historical = season_data[num_features[:-2]].mean().to_dict()
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
# Suggest Crops (updated with balanced)
def suggest_crops(district, season, exclude_crop=None, year=2025, top_k=2, climate_data=None, sort_by='balanced'):
    """
    Suggest top crops for a district/season.
   
    Args:
        ... (existing args)
        sort_by: 'absolute' (raw yield), 'relative' (normalized), or 'balanced' (default: yield * diversity_factor).
    """
    try:
        dist_enc = le_district.transform([district])[0]
    except ValueError:
        st.error(f"No crops found for district '{district}'.")
        return []
    possible_crops = df_clean[df_clean['District_Enc'] == dist_enc]['Crop'].unique()
    if len(possible_crops) == 0:
        st.error(f"No crops found for district '{district}'.")
        return []
    if exclude_crop:
        possible_crops = [c for c in possible_crops if c != exclude_crop]
    preds = []
    for crop in possible_crops:
        yield_p, _, thresh = predict_suitability(district, crop, season, year, climate_data=climate_data)
        if yield_p is not None:
            if sort_by == 'relative':
                normalized_yield = yield_p / (thresh if thresh > 0 else 1)
                sort_key = normalized_yield
            elif sort_by == 'balanced':
                div_factor = diversity_factor.get(crop, 1.0)
                sort_key = yield_p * div_factor
            else:  # 'absolute'
                sort_key = yield_p
            preds.append((crop, yield_p, sort_key))
    # Sort by the chosen key
    return sorted(preds, key=lambda x: x[2], reverse=True)[:top_k]
# Set page config
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")
# Custom CSS for theme
st.markdown("""
<style>
    /* Background and header */
    .stApp {
        background: linear-gradient(to bottom, #f0f8f0, #e8f5e8);
    }
    .stApp > header {
        background-color: #2e7d32; /* Dark green header */
        color: white;
    }
    /* Fonts and spacing */
    .stMarkdown {
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
    }
    /* Button styling */
    .stButton > button {
        background-color: #4caf50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        white-space: nowrap;
    }
    /* Form submit button full width */
    .stForm > .stButton > button {
        width: 100%;
        justify-content: center;
    }
    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
# Title with emoji
st.title("🌾 Crop Yield Prediction Chatbot")
st.markdown("**Predict yields and get smart crop suggestions based on real-time weather.**")
# Initialize session state for prediction persistence
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'yield_p' not in st.session_state:
    st.session_state.yield_p = None
if 'suitable' not in st.session_state:
    st.session_state.suitable = None
if 'thresh' not in st.session_state:
    st.session_state.thresh = None
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'climate_msg' not in st.session_state:
    st.session_state.climate_msg = None
if 'user_district' not in st.session_state:
    st.session_state.user_district = None
if 'user_crop' not in st.session_state:
    st.session_state.user_crop = None
if 'user_season' not in st.session_state:
    st.session_state.user_season = None
if 'user_area' not in st.session_state:
    st.session_state.user_area = 5000.0
if 'user_sort_by' not in st.session_state:
    st.session_state.user_sort_by = 'balanced'
# User Inputs
districts = list(le_district.classes_)
crops = list(le_crop.classes_)
seasons = list(season_map.keys())
with st.sidebar:
    st.header("🛠️ Settings")
    st.write("**App Version**: 1.3 (Updated)")
    if st.button("Reset Form"):
        for key in list(st.session_state.keys()):
            if key not in ['predicted', 'yield_p', 'suitable', 'thresh', 'suggestions', 'climate_msg', 'user_district', 'user_crop', 'user_season', 'user_area', 'user_sort_by']:
                del st.session_state[key]
        st.session_state.predicted = False # Re-init flag
        st.rerun()
    # Layout with columns
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📝 Input Details")
    with st.form("inputs_form"):
        with st.expander("Select Parameters", expanded=True):
            user_district = st.selectbox("🌍 District", districts, key="district", help="Choose a district (e.g., Ariyalur)")
            user_crop = st.selectbox("🌾 Crop", crops, key="crop", help="Choose a crop (e.g., Rice)")
            user_season = st.selectbox("☀️ Season", seasons, index=seasons.index('Kharif'), key="season", help="Choose a season")
            user_area = st.number_input("📏 Area (Hectare)", min_value=1.0, value=5000.0, step=100.0, key="area")
            user_sort_by = st.selectbox("Sort Suggestions By", ['absolute', 'balanced'], index=1, key="sort_by", help="'balanced' favors diverse crops; 'absolute' favors highest yield")
        # Full-width horizontal button
        submitted = st.form_submit_button("🚀 Predict & Suggest", type="primary", use_container_width=True)
with col2:
    if st.session_state.predicted:
        with st.spinner("Fetching weather and predicting..."):
            # Fetch climate data once (but since persisted, show stored)
            if st.session_state.climate_msg:
                st.info(st.session_state.climate_msg)
            # Predict Yield (use stored, but recompute if needed; here display stored)
            st.subheader(f"Prediction for {st.session_state.user_crop} in {st.session_state.user_district} ({st.session_state.user_season})")
            if st.session_state.yield_p is not None:
                st.balloons() # Confetti animation (runs once per view)
                st.success(f"Prediction complete! 🌟 Yield: {st.session_state.yield_p:.2f} T/Ha")
                st.markdown('<div class="metric-card"><h3>Predicted Yield</h3><p>{:.2f} T/Ha</p></div>'.format(st.session_state.yield_p), unsafe_allow_html=True)
                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric("Suitability", "Yes" if st.session_state.suitable else "No", delta=f"vs Mean {st.session_state.thresh:.2f}")
                with col_metrics2:
                    st.metric("Area Input", f"{st.session_state.user_area} Ha")
                # Suggest Crops
                sort_label = "balanced for diversity" if st.session_state.user_sort_by == 'balanced' else "absolute yield"
                st.subheader(f"💡 Top Alternative Crop Suggestions ({sort_label}) for {st.session_state.user_district} ({st.session_state.user_season})")
                if st.session_state.suggestions:
                    suggestions_df = pd.DataFrame([[s[0], round(s[1], 2)] for s in st.session_state.suggestions], columns=['Crop', 'Predicted Yield (T/Ha)'])
                    st.table(suggestions_df.style.background_gradient(cmap='Greens'))
                else:
                    st.error("No crop suggestions available.")
            else:
                st.error("Prediction failed. Check inputs.")
        # Button to hide results or new prediction
        if st.button("New Prediction"):
            st.session_state.predicted = False
            st.rerun()
if submitted:
    # Update session state with current inputs
    st.session_state.user_district = user_district
    st.session_state.user_crop = user_crop
    st.session_state.user_season = user_season
    st.session_state.user_area = user_area
    st.session_state.user_sort_by = user_sort_by
    # Fetch climate data once
    climate = get_realtime_climate(user_district)
    if climate is None:
        climate = get_historical_climate(user_district, user_season)
        st.session_state.climate_msg = f"Using historical climate for {user_district} ({user_season}): Temp={climate['T2M']:.1f}°C, Humidity={climate['RH2M']}%, Precip={climate['PRECTOTCORR']}mm"
    else:
        st.session_state.climate_msg = f"Fetched real-time climate data for {user_district}: Temp={climate['T2M']:.1f}°C, Humidity={climate['RH2M']}%, Precip={climate['PRECTOTCORR']}mm (daily)"
    # Predict Yield
    yield_p, suitable, thresh = predict_suitability(user_district, user_crop, user_season, area=user_area, climate_data=climate)
    st.session_state.yield_p = yield_p
    st.session_state.suitable = suitable
    st.session_state.thresh = thresh
    # Suggest Crops
    suggestions = suggest_crops(user_district, user_season, exclude_crop=user_crop, year=2025, top_k=2, climate_data=climate, sort_by=user_sort_by)
    st.session_state.suggestions = suggestions
    # Set flag
    st.session_state.predicted = True
    st.rerun() # Rerun to show results immediately