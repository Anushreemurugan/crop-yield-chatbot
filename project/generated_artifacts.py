import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle

# Load and Preprocess Data
try:
    df = pd.read_excel('merged_monthly_dataset.xlsx')
except FileNotFoundError:
    print("Error: 'merged_monthly_dataset.xlsx' not found in C:\\project")
    exit(1)

print(f"Original dataset shape: {df.shape}")
df['Year'] = pd.to_datetime(df['Year'], errors='coerce').dt.year
df_clean = df[df['Crop'].notna() & (df['Area (Hectare)'] > 0) & (df['Yield (Tonne/Hectare)'] > 0)].copy()
print(f"Cleaned dataset shape: {df_clean.shape}")

# Feature Engineering
df_clean['Precip_Temp_Interact'] = df_clean['PRECTOTCORR'] * df_clean['T2M']
season_map = {'Kharif': 1, 'Rabi': 2, 'Autumn': 3, 'Summer': 4, 'Winter': 5, 'Whole Year': 6}
df_clean['Season'] = df_clean['Season'].map(season_map).fillna(0)

# Encode Categorical Features
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

# Scale Numerical Features
scaler = StandardScaler()
num_features = ['ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PRECTOTCORR', 'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'WS2M',
                'Precip_Temp_Interact', 'Area_Hectare']
X_scaled = X.copy()
X_scaled[num_features] = scaler.fit_transform(X_scaled[num_features])

# Split Data
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42,
                              num_leaves=31, min_child_samples=20, feature_fraction=0.9, verbose=-1)
lgb_model.fit(X_train, y_train, categorical_feature=['District_Enc', 'Crop_Enc', 'Season'])

# Compute Mean Yields
means = df_clean.groupby(['District', 'Crop'])['Yield (Tonne/Hectare)'].mean().to_dict()

# Save Artifacts
with open('lgb_model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)
with open('le_district.pkl', 'wb') as f:
    pickle.dump(le_district, f)
with open('le_crop.pkl', 'wb') as f:
    pickle.dump(le_crop, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('means.pkl', 'wb') as f:
    pickle.dump(means, f)
with open('season_map.pkl', 'wb') as f:
    pickle.dump(season_map, f)

print("Artifacts generated successfully: lgb_model.pkl, le_district.pkl, le_crop.pkl, scaler.pkl, means.pkl, season_map.pkl")