import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
from datetime import datetime
import io
import base64

# Page config matching old code
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS from old code
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

# Global variables
API_KEY = 'a5c4d7596f1d44f689f39ccec6f68de4'  # Default, will be updated from sidebar

def get_district_coords(district):
    """Fetch latitude and longitude using OpenWeatherMap Geocoding API"""
    if API_KEY == 'YOUR_OPENWEATHERMAP_API_KEY':
        st.warning("Error: Invalid API key. Please replace with a valid key.")
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
        st.error(f"Error fetching coordinates for {district}: {e}. Using historical averages.")
        return None

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
        st.error(f"Error fetching weather data for {district}: {e}. Using historical averages.")
        return None

@st.cache_data
def load_and_train_model(_api_key):
    global API_KEY
    API_KEY = _api_key
    # Load and Preprocess Data (matching notebook exactly)
    df = pd.read_excel('merged_monthly_dataset.xlsx')
    # Fix: Extract the starting year from "2013-2014" format
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
    def filter_season_months(group):
        season = group.name[3]
        months = season_months.get(season, list(range(1, 13)))
        return group[group['Month'].isin(months)]
    df_filtered = df_agg.groupby(['District', 'Year', 'Crop', 'Season']).apply(filter_season_months).reset_index(drop=True)
    df_clean = df_filtered.groupby(['District', 'Year', 'Crop', 'Season']).agg({
        'Area (Hectare)': 'first',
        'Yield (Tonne/Hectare)': 'first',
        **{col: 'mean' for col in weather_cols}
    }).reset_index()
    # Clean further (drop if no yield)
    df_clean = df_clean[df_clean['Crop'].notna() & (df_clean['Area (Hectare)'] > 0) & (df_clean['Yield (Tonne/Hectare)'] > 0)].copy()
    # Feature Engineering
    df_clean['Precip_Temp_Interact'] = df_clean['PRECTOTCORR'] * df_clean['T2M']
    season_map = {'Kharif': 1, 'Rabi': 2, 'Autumn': 3, 'Summer': 4, 'Winter': 5, 'Whole Year': 6}
    df_clean['Season'] = df_clean['Season'].map(season_map).fillna(0)
    # Encode categorical features
    le_district = LabelEncoder()
    le_crop = LabelEncoder()
    df_clean['District_Enc'] = le_district.fit_transform(df_clean['District'])
    df_clean['Crop_Enc'] = le_crop.fit_transform(df_clean['Crop'])
    # Features and Target
    features = ['Year', 'District_Enc', 'Crop_Enc', 'Season', 'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PRECTOTCORR',
                'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'WS2M', 'Precip_Temp_Interact', 'Area_Hectare']
    df_clean = df_clean.rename(columns={'Area (Hectare)': 'Area_Hectare'})
    X = df_clean[features]
    y = df_clean['Yield (Tonne/Hectare)']
    # Scale numerical features
    num_features = ['ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PRECTOTCORR', 'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'WS2M',
                    'Precip_Temp_Interact', 'Area_Hectare']
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[num_features] = scaler.fit_transform(X_scaled[num_features])
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # Train LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42,
                                  num_leaves=50, min_child_samples=100, feature_fraction=0.8, verbose=-1)
    lgb_model.fit(X_train, y_train, categorical_feature=['District_Enc', 'Crop_Enc', 'Season'])
    # Evaluate
    y_pred_test = lgb_model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    st.info(f"Model loaded: Test RMSE: {rmse_test:.3f} T/Ha, R²: {r2_test:.3f}")
    # Compute Mean for Suitability Threshold (before outlier removal)
    means = df_clean.groupby(['District', 'Crop'])['Yield (Tonne/Hectare)'].mean().to_dict()
    # Compute crop diversity factor (before outlier removal)
    crop_freq = df_clean['Crop'].value_counts()
    diversity_factor = 1 / crop_freq
    # Outlier Detection (after means and diversity)
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
            initial_shape = df_out.shape[0]
        return df_out
    outlier_columns = ['Yield (Tonne/Hectare)', 'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PRECTOTCORR',
                       'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'WS2M']
    df_clean = remove_outliers(df_clean, outlier_columns)
    # Return only serializable objects
    return lgb_model, le_district, le_crop, scaler, means, season_map, df_clean, diversity_factor, features, num_features

# Load serializable model parts
api_key_input = st.sidebar.text_input("OpenWeatherMap API Key", type="password", value='a5c4d7596f1d44f689f39ccec6f68de4')
lgb_model, le_district, le_crop, scaler, means, season_map, df_clean, diversity_factor, features, num_features = load_and_train_model(api_key_input)

# Define functions after loading (not cached)
def get_historical_climate(district, season):
    district_data = df_clean[df_clean['District'] == district]
    if district_data.empty:
        return df_clean[num_features[:-2]].mean().to_dict()
    season_num = season_map.get(season, 0)
    season_data = district_data[district_data['Season'] == season_num]
    if season_data.empty:
        season_data = district_data  # Fallback to all seasons
    historical = season_data[num_features[:-2]].mean().to_dict()
    return historical

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
    # Use provided climate data or fetch real-time
    if climate_data is None:
        climate = get_realtime_climate(district)
        if climate is None:
            climate = get_historical_climate(district, season)
    else:
        climate = climate_data
    # Set unavailable features to historical means
    climate['ALLSKY_SFC_SW_DWN'] = df_clean['ALLSKY_SFC_SW_DWN'].mean()
    climate['EVPTRNS'] = df_clean['EVPTRNS'].mean()
    input_df = pd.DataFrame({
        'Year': [year], 'District_Enc': [dist_enc], 'Crop_Enc': [crop_enc], 'Season': [season_num],
        'Area_Hectare': [area]
    })
    for key in num_features[:-2]:
        input_df[key] = climate.get(key, df_clean[key].mean())
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

def suggest_crops(district, season, year=2025, top_k=2, climate_data=None, exclude_crop=None, sort_by='balanced'):
    try:
        dist_enc = le_district.transform([district])[0]
    except ValueError:
        st.error(f"No crops found for district '{district}'.")
        return []
    possible_crops = df_clean[df_clean['District_Enc'] == dist_enc]['Crop'].unique()
    if len(possible_crops) == 0:
        st.error(f"No crops found for district '{district}'.")
        return []
    # Fetch climate data once if not provided
    if climate_data is None:
        climate = get_realtime_climate(district)
        if climate is None:
            climate = get_historical_climate(district, season)
    else:
        climate = climate_data
    preds = []
    for crop in possible_crops:
        if exclude_crop and crop == exclude_crop:
            continue
        yield_p, _, thresh = predict_suitability(district, crop, season, year, climate_data=climate)
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

# Title from old code
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

# Districts, Crops, Seasons with icons
districts_list = list(le_district.classes_)
crops_list = list(le_crop.classes_)
seasons_list = list(season_map.keys())
district_options = [f"🏛️ {d}" for d in districts_list]
crop_options = [f"🌾 {c}" for c in crops_list]
season_options = ['🌾 Kharif', '❄️ Rabi', '🍂 Autumn', '☀️ Summer', '🌨️ Winter', '📅 Whole Year']

# User Inputs
with st.sidebar:
    st.header("🛠️ Settings")
    st.write("**App Version**: 1.3")
    if st.button("Reset Form"):
        for key in ['predicted', 'yield_p', 'suitable', 'thresh', 'suggestions', 'climate_msg', 'user_district', 'user_crop', 'user_season', 'user_area']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.predicted = False
        st.rerun()

# Layout with columns
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📝 Input Details")
    with st.form("inputs_form"):
        with st.expander("Select Parameters", expanded=True):
            selected_district_display = st.selectbox("🌍 District", district_options, index=districts_list.index("Thanjavur") if "Thanjavur" in districts_list else 0)
            user_district = selected_district_display.split(' ', 1)[1]
            selected_crop_display = st.selectbox("🌾 Crop", crop_options, index=crops_list.index("Rice") if "Rice" in crops_list else 0)
            user_crop = selected_crop_display.split(' ', 1)[1]
            selected_season_display = st.selectbox("☀️ Season", season_options, index=0)
            user_season = selected_season_display.split(' ', 1)[1]
            user_area = st.number_input("📏 Area (Hectare)", min_value=1.0, value=5000.0, step=100.0)
        # Full-width horizontal button
        submitted = st.form_submit_button("🚀 Predict & Suggest", type="primary", use_container_width=True)

with col2:
    if st.session_state.predicted:
        with st.spinner("Fetching weather and predicting..."):
            # Fetch climate data once (but since persisted, show stored)
            if st.session_state.climate_msg:
                st.info(st.session_state.climate_msg)
            # Predict Yield
            st.subheader(f"Prediction for {st.session_state.user_crop} in {st.session_state.user_district} ({st.session_state.user_season})")
            if st.session_state.yield_p is not None:
                st.balloons()  # Confetti animation
                st.success(f"Prediction complete! 🌟 Yield: {st.session_state.yield_p:.2f} T/Ha")
                st.markdown(f'<div class="metric-card"><h3>Predicted Yield</h3><p>{st.session_state.yield_p:.2f} T/Ha</p></div>', unsafe_allow_html=True)
                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric("Suitability", "Yes" if st.session_state.suitable else "No", delta=f"vs Mean {st.session_state.thresh:.2f}")
                with col_metrics2:
                    st.metric("Area Input", f"{st.session_state.user_area} Ha")
                # Suggest Crops
                st.subheader(f"💡 Top Alternative Crop Suggestions (balanced for diversity) for {st.session_state.user_district} ({st.session_state.user_season})")
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
    # Fetch climate data once
    climate = get_realtime_climate(user_district)
    if climate is None:
        climate = get_historical_climate(user_district, user_season)
        st.session_state.climate_msg = f"Using historical climate for {user_district}: Temp={climate['T2M']:.1f}°C, Humidity={climate['RH2M']}%, Precip={climate['PRECTOTCORR']}mm"
    else:
        st.session_state.climate_msg = f"Fetched real-time climate data for {user_district}: Temp={climate['T2M']:.1f}°C, Humidity={climate['RH2M']}%, Precip={climate['PRECTOTCORR']}mm (daily)"
    # Predict Yield
    yield_p, suitable, thresh = predict_suitability(user_district, user_crop, user_season, area=user_area, climate_data=climate)
    st.session_state.yield_p = yield_p
    st.session_state.suitable = suitable
    st.session_state.thresh = thresh
    # Suggest Crops
    suggestions = suggest_crops(user_district, user_season, exclude_crop=user_crop, year=2025, top_k=2, climate_data=climate, sort_by='balanced')
    st.session_state.suggestions = suggestions
    # Set flag
    st.session_state.predicted = True
    st.rerun()  # Rerun to show results immediately

