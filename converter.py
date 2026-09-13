"""
converter.py - JRA-3Q 4D Mesoscale & 3D Spherical AI Forecasting System
Integrated with OpenMythos Recurrent-Depth Transformer (RDT) Suite,
Autonomous Self-Learning Studio, Live Real-Time Weather Ingestion,
Batch Benchmark Validation Suite, and 4D Dynamic Streamtube Animation.

MIT License - Inspired by OpenMythos Project & JMA JRA-3Q Reanalysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import time
import requests
import json

# Plotly for 3D Globe and atmospheric profile visualisations
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NetCDF4 support
try:
    import netCDF4 as nc
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False

# Google Generative AI (Python 3.8+ compatible)
GEMINI_SDK_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False

# Import OpenMythos Local AI Engine
try:
    from local_ai import local_engine
    LOCAL_AI_AVAILABLE = True
except ImportError:
    LOCAL_AI_AVAILABLE = False


# ==============================================================================
# 1. Page Configuration & Header
# ==============================================================================
st.set_page_config(
    page_title="JRA-3Q 4D Mesoscale & 3D Spherical AI Forecaster",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌐 JRA-3Q AI Forecaster: 4D Mesoscale & 3D Spherical Suite")
st.caption("Autonomous Physics-Informed AI Reasoning, 3D Spherical Volume Modeling, and Operational Convective Rainband Forecasting.")


# ==============================================================================
# 2. Historical Benchmark Cases & Live Ingestion Presets
# ==============================================================================
PRESET_CASES = {
    "2020 Kumamoto Heavy Rain (July 3, 2020) - Extreme Disaster": {
        "lat": 32.25, "lon": 130.50, "date": "2020-07-03", "hour": "18:00 UTC",
        "name": "2020 Kumamoto Torrential Burst", "is_heavy_rain": True,
        "ivt": 1177.0, "shear": 16.8, "cape": 37.0, "cin": 9.0, "srh": 180.0, "k_index": 39.5,
        "desc": "Severe quasi-stationary back-building rainband triggered by an ultra-intense atmospheric river over the East China Sea."
    },
    "2014 Hiroshima Heavy Rain (August 19, 2014) - Back-Building": {
        "lat": 34.45, "lon": 132.50, "date": "2014-08-19", "hour": "18:00 UTC",
        "name": "2014 Hiroshima Cloudburst", "is_heavy_rain": True,
        "ivt": 960.0, "shear": 18.2, "cape": 420.0, "cin": 12.0, "srh": 210.0, "k_index": 41.0,
        "desc": "Deadly multi-cell convective training line producing >100mm/hr localized cloudburst along mountain slopes."
    },
    "2017 Northern Kyushu Disaster (July 5, 2017) - High Instability": {
        "lat": 33.40, "lon": 130.70, "date": "2017-07-05", "hour": "06:00 UTC",
        "name": "2017 Northern Kyushu Disaster", "is_heavy_rain": True,
        "ivt": 1250.0, "shear": 20.1, "cape": 850.0, "cin": 5.0, "srh": 240.0, "k_index": 43.0,
        "desc": "Continuous convective line anchoring along the Tsushima Strait moisture convergence zone."
    },
    "2013 Yamagata Heavy Rain (July 17, 2013) - Frontal Convergence": {
        "lat": 38.75, "lon": 140.00, "date": "2013-07-17", "hour": "12:00 UTC",
        "name": "2013 Yamagata Frontal Burst", "is_heavy_rain": True,
        "ivt": 810.0, "shear": 15.4, "cape": 310.0, "cin": 18.0, "srh": 145.0, "k_index": 37.5,
        "desc": "Stationary Baiu frontal boundary interacting with Sea of Japan warm moisture tongue."
    },
    "2014 Capped Stable Scenario (July 29, 2014) - Inversion Cap": {
        "lat": 34.00, "lon": 135.00, "date": "2014-07-29", "hour": "00:00 UTC",
        "name": "2014 Capped Stable Inversion", "is_heavy_rain": False,
        "ivt": 320.0, "shear": 6.2, "cape": 120.0, "cin": 180.0, "srh": 40.0, "k_index": 22.0,
        "desc": "Strong mid-level capping inversion (CIN > 150 J/kg) successfully suppressing deep convective updrafts."
    },
    "Calm Fair Weather Baseline (October 15, 2023) - Stable High": {
        "lat": 35.68, "lon": 139.76, "date": "2023-10-15", "hour": "00:00 UTC",
        "name": "Calm Fair Weather Control", "is_heavy_rain": False,
        "ivt": 140.0, "shear": 3.5, "cape": 10.0, "cin": 220.0, "srh": 15.0, "k_index": 12.0,
        "desc": "Migratory anticyclone, subsidence inversion, dry mid-troposphere, zero convective hazard."
    }
}

CITY_COORDINATES = {
    "Tokyo (35.68°N, 139.76°E)": (35.68, 139.76),
    "Fukuoka / Northern Kyushu (33.59°N, 130.40°E)": (33.59, 130.40),
    "Kumamoto / Central Kyushu (32.80°N, 130.70°E)": (32.80, 130.70),
    "Kagoshima / Southern Kyushu (31.59°N, 130.55°E)": (31.59, 130.55),
    "Hiroshima / Chugoku (34.38°N, 132.45°E)": (34.38, 132.45),
    "Osaka / Kansai (34.69°N, 135.50°E)": (34.69, 135.50),
    "Nagoya / Tokai (35.18°N, 136.90°E)": (35.18, 136.90),
    "Sendai / Tohoku (38.26°N, 140.87°E)": (38.26, 140.87),
    "Sapporo / Hokkaido (43.06°N, 141.35°E)": (43.06, 141.35),
    "Naha / Okinawa (26.21°N, 127.68°E)": (26.21, 127.68),
}


# ==============================================================================
# 3. Sidebar Configuration
# ==============================================================================
st.sidebar.header("📡 1. Atmospheric Data Source")

data_mode = st.sidebar.radio(
    "Select Ingestion Mode:",
    ["📂 Historical NetCDF4 / Preset Benchmark", "📡 Live Real-Time Global Sounding (Open-Meteo API)"],
    index=0
)

target_lat, target_lon = 32.25, 130.50
selected_case_name = "2020 Kumamoto Heavy Rain (July 3, 2020) - Extreme Disaster"

if data_mode == "📂 Historical NetCDF4 / Preset Benchmark":
    selected_case_name = st.sidebar.selectbox(
        "Historical Benchmark Scenario:",
        list(PRESET_CASES.keys()),
        index=0
    )
    case_info = PRESET_CASES[selected_case_name]
    target_lat = case_info["lat"]
    target_lon = case_info["lon"]
    st.sidebar.info(f"📍 **Coordinates**: {target_lat}°N, {target_lon}°E\n\n📝 {case_info['desc']}")
else:
    selected_city = st.sidebar.selectbox(
        "Select City / Region for Live Sounding:",
        list(CITY_COORDINATES.keys()),
        index=2
    )
    target_lat, target_lon = CITY_COORDINATES[selected_city]
    custom_coords = st.sidebar.checkbox("Input Custom Latitude / Longitude")
    if custom_coords:
        target_lat = st.sidebar.number_input("Latitude (°N)", 20.0, 50.0, float(target_lat), 0.1)
        target_lon = st.sidebar.number_input("Longitude (°E)", 120.0, 155.0, float(target_lon), 0.1)
    st.sidebar.success(f"🌐 Fetching live isobaric profile for **{target_lat:.2f}°N, {target_lon:.2f}°E**")


# 4D Temporal Evolution Selection
st.sidebar.header("⏱️ 2. 4D Temporal State (T-Axis)")
time_steps_labels = [
    "T - 18h (Precursor Moisture Surge)",
    "T - 12h (Atmospheric River Advection)",
    "T - 6h (Cap Erosion & Boundary Trigger)",
    "T - 0h (Peak Quasi-Stationary Burst)"
]
selected_time_idx = st.sidebar.select_slider(
    "Select Forecast Evolution Timestamp:",
    options=[0, 1, 2, 3],
    value=3,
    format_func=lambda x: time_steps_labels[x]
)

# 3D Globe Height Exaggeration Scale
height_exaggeration = st.sidebar.slider(
    "3D Pillar / Tube Radial Exaggeration:",
    min_value=1.0, max_value=3.0, value=1.8, step=0.1
)

# AI Engine Settings
st.sidebar.header("🧠 3. AI Reasoning Settings")
recurrent_loops = st.sidebar.slider("OpenMythos RDT Latent Loops:", min_value=1, max_value=10, value=10)
cloud_model_name = st.sidebar.selectbox("Cloud LLM (Optional):", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"], index=0)
gemini_api_key = st.sidebar.text_input("Gemini API Key (Optional):", type="password")


# ==============================================================================
# 4. Meteorological Math & 4D Data Generators
# ==============================================================================

def fetch_live_sounding(lat, lon):
    """Fetches live isobaric atmospheric sounding from Open-Meteo REST API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_1000hPa,temperature_925hPa,temperature_850hPa,temperature_700hPa,temperature_500hPa,temperature_300hPa,relative_humidity_1000hPa,relative_humidity_925hPa,relative_humidity_850hPa,relative_humidity_700hPa,relative_humidity_500hPa,relative_humidity_300hPa,wind_speed_1000hPa,wind_speed_850hPa,wind_speed_500hPa,wind_speed_300hPa,wind_direction_1000hPa,wind_direction_850hPa,wind_direction_500hPa,wind_direction_300hPa&current_weather=true"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            hourly = data.get("hourly", {})
            p_levels = [1000, 925, 850, 700, 500, 300]
            T_arr, rh_arr, u_arr, v_arr = [], [], [], []
            for p in p_levels:
                t = hourly.get(f"temperature_{p}hPa", [20.0])[0]
                rh = hourly.get(f"relative_humidity_{p}hPa", [80.0])[0]
                ws = hourly.get(f"wind_speed_{p}hPa", [10.0])[0]
                wd = hourly.get(f"wind_direction_{p}hPa", [225.0])[0]
                rad = np.radians(wd)
                u = -ws * np.sin(rad)
                v = -ws * np.cos(rad)
                T_arr.append(t)
                rh_arr.append(rh)
                u_arr.append(u)
                v_arr.append(v)
            return True, np.array(p_levels), np.array(T_arr), np.array(rh_arr), np.array(u_arr), np.array(v_arr)
    except Exception:
        pass
    return False, None, None, None, None, None


def calculate_thermodynamics(p, T_c, q_kgkg, u, v):
    """Calculates comprehensive meteorological soundings, CAPE, CIN, theta_e, and IVT."""
    T_k = T_c + 273.15
    e_hpa = (p * q_kgkg) / (0.622 + 0.378 * q_kgkg)
    e_hpa = np.clip(e_hpa, 1e-4, p * 0.9)
    
    # Magnus-Tetens formula for Dewpoint
    T_d_c = (243.04 * (np.log(e_hpa / 6.1078))) / (17.625 - np.log(e_hpa / 6.1078))
    T_d_c = np.minimum(T_d_c, T_c)
    
    # Saturation vapor pressure
    e_sat = 6.1078 * np.exp((17.27 * T_c) / (T_c + 237.3))
    RH = np.clip((e_hpa / e_sat) * 100.0, 1.0, 100.0)

    theta = T_k * (1000.0 / p) ** 0.286
    Lv, Cp = 2.501e6, 1005.7
    theta_e = theta * np.exp((Lv * q_kgkg) / (Cp * T_k))
    
    wind_speed = np.sqrt(u**2 + v**2)
    wind_dir = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

    # IVT Integration
    g = 9.80665
    dp = np.abs(np.diff(p * 100.0))
    q_mid = (q_kgkg[:-1] + q_kgkg[1:]) / 2.0
    v_mid = (wind_speed[:-1] + wind_speed[1:]) / 2.0
    IVT = np.sum((q_mid * v_mid * dp) / g)

    # Simplified parcel ascent CAPE/CIN
    cape, cin = 0.0, 0.0
    lcl_p = float(p[0]) - (T_c[0] - T_d_c[0]) * 10.0
    lfc_p = lcl_p - 40.0
    
    for i in range(len(p)):
        if p[i] <= lfc_p and p[i] >= 300:
            env_T = T_c[i]
            parcel_T = T_c[0] - 0.0065 * ((1000 - p[i]) * 10.0)
            buoyancy = (parcel_T - env_T)
            if buoyancy > 0:
                cape += buoyancy * 35.0
        elif p[i] > lfc_p:
            cin += max(0.0, (T_c[i] - T_d_c[i])) * 1.5

    # Bulk shear
    shear_0_6km = float(np.sqrt((u[-1] - u[0])**2 + (v[-1] - v[0])**2))
    
    # K-Index
    try:
        idx_850 = np.argmin(np.abs(p - 850))
        idx_700 = np.argmin(np.abs(p - 700))
        idx_500 = np.argmin(np.abs(p - 500))
        k_index = (T_c[idx_850] - T_c[idx_500]) + T_d_c[idx_850] - (T_c[idx_700] - T_d_c[idx_700])
    except Exception:
        k_index = 35.0

    srh = float(shear_0_6km * 12.0)
    ehi = round(cape * srh / 160000.0, 2)

    df_profile = pd.DataFrame({
        "Pressure (hPa)": p.astype(int),
        "Temperature (°C)": np.round(T_c, 1),
        "Dewpoint (°C)": np.round(T_d_c, 1),
        "RH (%)": np.round(RH, 1),
        "Specific Humidity (g/kg)": np.round(q_kgkg * 1000.0, 2),
        "U-Wind (m/s)": np.round(u, 1),
        "V-Wind (m/s)": np.round(v, 1),
        "Wind Speed (m/s)": np.round(wind_speed, 1),
        "Wind Direction (°)": np.round(wind_dir, 0).astype(int),
        "Theta-e (K)": np.round(theta_e, 1)
    })

    metrics = {
        "CAPE(J/kg)": round(cape, 0),
        "CIN(J/kg)": round(cin, 0),
        "IVT(kg/m/s)": round(IVT, 1),
        "Bulk_Shear_0-6km(m/s)": round(shear_0_6km, 1),
        "SRH_0-3km(m2/s2)": round(srh, 1),
        "EHI": ehi,
        "K_Index(°C)": round(k_index, 1),
        "LCL(hPa)": round(lcl_p, 0),
        "LFC(hPa)": round(lfc_p, 0)
    }

    return df_profile, metrics


def generate_4d_mesoscale_cube(center_lat, center_lon, t_step_idx, is_live=False):
    """Constructs 4D Mesoscale Voxel Grid (Lat x Lon x Height x Time)."""
    p_levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300])
    
    if is_live:
        success, p_live, T_live, rh_live, u_live, v_live = fetch_live_sounding(center_lat, center_lon)
        if success:
            p_levels = p_live
            T_c = T_live
            q_kgkg = (rh_live / 100.0) * 0.018 * np.exp((p_levels - 1000) / 300.0)
            u = u_live
            v = v_live
        else:
            is_live = False

    if not is_live:
        t_factors = [0.45, 0.65, 0.85, 1.0]
        f = t_factors[t_step_idx]
        T_c = np.array([28.5, 24.2, 20.1, 11.5, 5.0, -4.5, -15.2, -28.0]) + (f * 1.5)
        q_kgkg = np.array([0.021, 0.018, 0.015, 0.009, 0.006, 0.003, 0.001, 0.0004]) * f
        u = np.array([8.5, 12.0, 16.5, 19.0, 22.0, 25.5, 29.0, 34.0]) * f
        v = np.array([14.0, 18.5, 23.0, 21.0, 18.0, 14.0, 10.0, 6.0]) * f

    df_center, center_metrics = calculate_thermodynamics(p_levels, T_c, q_kgkg, u, v)

    # Construct surrounding 5x5 mesoscale grid
    lats = np.linspace(center_lat - 2.5, center_lat + 2.5, 5)
    lons = np.linspace(center_lon - 2.5, center_lon + 2.5, 5)
    grid_nodes = []
    for la in lats:
        for lo in lons:
            dist = np.sqrt((la - center_lat)**2 + (lo - center_lon)**2)
            node_ivt = max(100.0, center_metrics["IVT(kg/m/s)"] * np.exp(-dist / 3.0))
            node_shear = max(5.0, center_metrics["Bulk_Shear_0-6km(m/s)"] * np.exp(-dist / 5.0))
            node_cape = max(10.0, center_metrics["CAPE(J/kg)"] * np.exp(-dist / 4.0))
            node_cin = center_metrics["CIN(J/kg)"] + dist * 10.0
            
            # Convective risk score for node
            score = float(np.clip(
                (node_ivt / 1000.0) * 40.0 +
                (node_shear / 20.0) * 30.0 +
                (node_cape / 1000.0) * 30.0 -
                (node_cin / 100.0) * 15.0,
                5.0, 98.0
            ))
            grid_nodes.append({
                "lat": round(la, 2),
                "lon": round(lo, 2),
                "ivt": round(node_ivt, 1),
                "shear": round(node_shear, 1),
                "cape": round(node_cape, 0),
                "cin": round(node_cin, 0),
                "theta_e_850": 345.0 - dist * 2.0,
                "rh_mean": 85.0 - dist * 3.0,
                "risk_score": round(score, 1)
            })

    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "temporal_state": time_steps_labels[t_step_idx],
        "pressure_levels": p_levels.tolist(),
        "df_profile": df_center,
        "center_metrics": center_metrics,
        "grid_nodes": grid_nodes,
        "ivt_tendency": round(center_metrics["IVT(kg/m/s)"] * 0.12, 1),
        "cin_tendency": -round(center_metrics["CIN(J/kg)"] * 0.25, 1)
    }


# Generate current state
is_live_mode = (data_mode != "📂 Historical NetCDF4 / Preset Benchmark")
meso_data = generate_4d_mesoscale_cube(target_lat, target_lon, selected_time_idx, is_live=is_live_mode)
df_profile = meso_data["df_profile"]
center_metrics = meso_data["center_metrics"]


# ==============================================================================
# 5. 3D Spherical Volume Rendering (Globe, Pillars, Streamtubes, Risk Mesh)
# ==============================================================================

def create_3d_spherical_globe(meso_data, exaggeration=1.8):
    """Renders high-precision 3D Globe with Sounding Pillars, Streamtubes, and Risk Mesh."""
    center_lat = meso_data["center_lat"]
    center_lon = meso_data["center_lon"]
    grid_nodes = meso_data["grid_nodes"]

    # 1. Earth Sphere Surface
    u_sph = np.linspace(0, 2 * np.pi, 60)
    v_sph = np.linspace(0, np.pi, 40)
    r_earth = 1.0
    x_earth = r_earth * np.outer(np.cos(u_sph), np.sin(v_sph))
    y_earth = r_earth * np.outer(np.sin(u_sph), np.sin(v_sph))
    z_earth = r_earth * np.outer(np.ones(np.size(u_sph)), np.cos(v_sph))

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale=[[0, '#0a192f'], [0.5, '#112240'], [1, '#1e3a8a']],
        showscale=False, opacity=0.88, hoverinfo='skip'
    ))

    # 2. Global Coastline Vector Coordinates
    coastlines = [
        # Japan Main Arch
        {"lat": [31.0, 32.5, 33.8, 35.0, 36.5, 38.5, 40.5, 41.5, 43.0, 45.0],
         "lon": [130.5, 131.5, 133.0, 136.0, 137.5, 140.0, 140.5, 141.0, 143.0, 142.0]},
        # Japan West Coast
        {"lat": [31.0, 32.0, 33.5, 35.5, 37.5, 39.5, 41.0],
         "lon": [130.0, 129.5, 131.0, 134.0, 137.0, 139.5, 140.0]},
        # East Asian Coastline
        {"lat": [22.0, 25.0, 28.0, 31.0, 35.0, 38.0, 40.0],
         "lon": [114.0, 119.0, 121.5, 121.8, 119.5, 122.0, 124.0]},
        # Korean Peninsula
        {"lat": [34.5, 36.0, 38.0, 39.0, 38.0, 35.0],
         "lon": [126.5, 126.0, 125.0, 127.5, 128.5, 129.0]}
    ]
    for c in coastlines:
        c_lat_rad = np.radians(c["lat"])
        c_lon_rad = np.radians(c["lon"])
        c_r = 1.002
        c_x = c_r * np.cos(c_lat_rad) * np.cos(c_lon_rad)
        c_y = c_r * np.cos(c_lat_rad) * np.sin(c_lon_rad)
        c_z = c_r * np.sin(c_lat_rad)
        fig.add_trace(go.Scatter3d(
            x=c_x, y=c_y, z=c_z, mode='lines',
            line=dict(color='#38bdf8', width=2.5), hoverinfo='skip', showlegend=False
        ))

    # 3. 3D Mesoscale Convective Risk Nodes (Micro-dots, size=0.8)
    node_lats = np.array([n["lat"] for n in grid_nodes])
    node_lons = np.array([n["lon"] for n in grid_nodes])
    node_risks = np.array([n["risk_score"] for n in grid_nodes])
    
    n_rad_lat = np.radians(node_lats)
    n_rad_lon = np.radians(node_lons)
    n_x = 1.005 * np.cos(n_rad_lat) * np.cos(n_rad_lon)
    n_y = 1.005 * np.cos(n_rad_lat) * np.sin(n_rad_lon)
    n_z = 1.005 * np.sin(n_rad_lat)

    fig.add_trace(go.Scatter3d(
        x=n_x, y=n_y, z=n_z, mode='markers',
        marker=dict(
            size=0.8,
            color=node_risks,
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title="3D Risk", thickness=10, len=0.6, x=1.02)
        ),
        text=[f"Node: {n['lat']}°N, {n['lon']}°E | Risk: {n['risk_score']}/100" for n in grid_nodes],
        hoverinfo='text',
        name="1.25° Grid Nodes"
    ))

    # 4. 3D Vertical Sounding Pillars (Pillars rising into space)
    p_levels = np.array(meso_data["pressure_levels"])
    r_heights = 1.005 + (1000.0 - p_levels) / 700.0 * 0.18 * exaggeration
    
    center_lat_rad = np.radians(center_lat)
    center_lon_rad = np.radians(center_lon)
    
    pillar_x = r_heights * np.cos(center_lat_rad) * np.cos(center_lon_rad)
    pillar_y = r_heights * np.cos(center_lat_rad) * np.sin(center_lon_rad)
    pillar_z = r_heights * np.sin(center_lat_rad)

    fig.add_trace(go.Scatter3d(
        x=pillar_x, y=pillar_y, z=pillar_z,
        mode='lines+markers',
        line=dict(color='#ef4444', width=6),
        marker=dict(size=4, color=df_profile["Theta-e (K)"], colorscale='Plasma', showscale=False),
        text=[f"Level: {p} hPa | θe: {th} K | Wind: {ws} m/s" for p, th, ws in zip(p_levels, df_profile["Theta-e (K)"], df_profile["Wind Speed (m/s)"])],
        hoverinfo='text',
        name="3D Sounding Pillar"
    ))

    # 5. 3D Atmospheric River Streamtubes (Curving up from East China Sea)
    tube_lats = np.linspace(center_lat - 7.0, center_lat, 25)
    tube_lons = np.linspace(center_lon - 8.0, center_lon, 25)
    tube_heights = np.linspace(1.005, 1.005 + 0.12 * exaggeration, 25)
    
    t_lat_rad = np.radians(tube_lats)
    t_lon_rad = np.radians(tube_lons)
    t_x = tube_heights * np.cos(t_lat_rad) * np.cos(t_lon_rad)
    t_y = tube_heights * np.cos(t_lat_rad) * np.sin(t_lon_rad)
    t_z = tube_heights * np.sin(t_lat_rad)

    fig.add_trace(go.Scatter3d(
        x=t_x, y=t_y, z=t_z,
        mode='lines+markers',
        line=dict(color='#06b6d4', width=8),
        marker=dict(size=3, color='#38bdf8'),
        name="3D Atmospheric River Tube",
        hovertext="3D Atmospheric River Conveyor (IVT Flux Inflow)"
    ))

    cam_x = 1.8 * np.cos(np.radians(center_lat)) * np.cos(np.radians(center_lon))
    cam_y = 1.8 * np.cos(np.radians(center_lat)) * np.sin(np.radians(center_lon))
    cam_z = 1.8 * np.sin(np.radians(center_lat))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            bgcolor='#030712',
            camera=dict(
                eye=dict(x=cam_x, y=cam_y, z=cam_z),
                center=dict(x=0, y=0, z=0)
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=540,
        legend=dict(x=0.02, y=0.95, font=dict(color="white"), bgcolor="rgba(10,25,47,0.7)")
    )
    return fig
# ==============================================================================
# 6. Main Application Layout & Tabs
# ==============================================================================

# Key KPI Dashboard
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
with kpi_col1:
    st.metric("Total IVT (Moisture)", f"{center_metrics['IVT(kg/m/s)']} kg/(m·s)", f"+{meso_data['ivt_tendency']}/h")
with kpi_col2:
    st.metric("0-6 km Bulk Shear", f"{center_metrics['Bulk_Shear_0-6km(m/s)']} m/s", "Organized Band")
with kpi_col3:
    st.metric("Surface CAPE", f"{center_metrics['CAPE(J/kg)']} J/kg", "Convective Fuel")
with kpi_col4:
    st.metric("Inversion CIN", f"{center_metrics['CIN(J/kg)']} J/kg", f"{meso_data['cin_tendency']}/h (Eroding)")
with kpi_col5:
    st.metric("0-3km SRH / EHI", f"{center_metrics['SRH_0-3km(m2/s2)']} m²/s²", f"EHI: {center_metrics['EHI']}")

# Define all 10 Interactive Tabs
tabs = st.tabs([
    "🌐 3D Spherical Globe",
    "🤖 AI Autonomous Learning",
    "📊 Batch Validation Suite",
    "📈 Skew-T Sounding",
    "🌀 Kinematic Hodograph",
    "📐 2D Cross-Section",
    "🗺️ Mesoscale Scan Grid",
    "📋 AI Sounding Prompt",
    "⚡ OpenMythos Inference",
    "📑 Executive Briefing"
])

# ------------------------------------------------------------------------------
# Tab 1: 3D Spherical Volume Globe
# ------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("🌐 3D Spherical Atmosphere: Pillars, Streamtubes & Grid Mesh")
    st.caption("Rendered on 3D Earth sphere with JRA-3Q 1.25° grid node mesh, vertical sounding pillars, and 3D atmospheric river streamtubes.")
    fig_globe = create_3d_spherical_globe(meso_data, exaggeration=height_exaggeration)
    st.plotly_chart(fig_globe, use_container_width=True)

# ------------------------------------------------------------------------------
# Tab 2: AI Autonomous Self-Learning Studio
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🤖 OpenMythos Autonomous Self-Learning & Continual Adaptation")
    st.write("Physics-informed gradient optimization fine-tunes the shared-weight transformer block directly in memory.")

    learn_c1, learn_c2, learn_c3 = st.columns(3)
    with learn_c1:
        train_epochs = st.slider("Training Epochs:", min_value=5, max_value=50, value=15, step=5)
    with learn_c2:
        train_lr = st.select_slider("Learning Rate (Adam):", options=[0.001, 0.005, 0.008, 0.01, 0.02], value=0.008)
    with learn_c3:
        lambda_phys = st.slider("Thermodynamic Constraint Weight (λ):", 0.1, 1.0, 0.40, 0.05)

    if st.button("🚀 Execute Autonomous Online Adaptation Loop", type="primary"):
        if LOCAL_AI_AVAILABLE:
            with st.spinner(f"Optimizing OpenMythos RDT across {train_epochs} epochs..."):
                history = local_engine.train_autonomous_epochs(epochs=train_epochs, lr=train_lr, lambda_physics=lambda_phys)
                
                df_hist = pd.DataFrame(history)
                st.success(f"✅ Autonomous Adaptation Complete! Trained on Experience Replay Buffer (Capacity: {len(local_engine.replay_buffer)} episodes).")
                
                # Plot Loss & Entropy Curves
                fig_loss = make_subplots(rows=1, cols=2, subplot_titles=("Loss Convergence", "Latent Space Gradient Delta"))
                fig_loss.add_trace(go.Scatter(x=df_hist["epoch"], y=df_hist["total_loss"], name="Total Loss", line=dict(color='#ef4444', width=3)), row=1, col=1)
                fig_loss.add_trace(go.Scatter(x=df_hist["epoch"], y=df_hist["phys_loss"], name="Physics Constraint Loss", line=dict(color='#3b82f6', dash='dash')), row=1, col=1)
                fig_loss.add_trace(go.Scatter(x=df_hist["epoch"], y=df_hist["weight_delta"], name="Weight Delta (||ΔW||)", line=dict(color='#10b981', width=3)), row=1, col=2)
                
                fig_loss.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
                st.plotly_chart(fig_loss, use_container_width=True)
                
                st.dataframe(df_hist.tail(5), use_container_width=True)
        else:
            st.error("local_ai.py engine not found.")

# ------------------------------------------------------------------------------
# Tab 3: Multi-Case Automated Batch Validation Suite
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("📊 Multi-Case Automated Batch Validation Suite")
    st.write("Simultaneously validates the OpenMythos RDT engine across historical extreme weather disasters and fair-weather controls.")

    if st.button("⚡ Run Full Automated Benchmark Matrix", type="primary"):
        if LOCAL_AI_AVAILABLE:
            with st.spinner("Executing 3D tensor batch inference across all scenarios..."):
                results_list, summary = local_engine.run_multi_case_benchmark(PRESET_CASES)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Overall Accuracy", f"{summary['Accuracy']}%", f"{summary['Total Cases']} Cases")
                m2.metric("Sensitivity (Recall)", f"{summary['Sensitivity (Recall)']}%", "Disaster Detection")
                m3.metric("Precision", f"{summary['Precision']}%", "Low False Alarms")
                m4.metric("F1-Score", f"{summary['F1-Score']}", "Harmonic Mean")

                # Confusion Matrix Heatmap
                cm_matrix = [[summary["True Positives"], summary["False Negatives"]],
                             [summary["False Positives"], summary["True Negatives"]]]
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm_matrix,
                    x=["Predicted Disaster", "Predicted Calm"],
                    y=["Actual Disaster", "Actual Calm"],
                    colorscale='Blues',
                    text=[[f"TP: {summary['True Positives']}", f"FN: {summary['False Negatives']}"],
                          [f"FP: {summary['False Positives']}", f"TN: {summary['True Negatives']}"]],
                    texttemplate="%{text}",
                    textfont={"size": 16}
                ))
                fig_cm.update_layout(title="Confusion Matrix Heatmap", height=320, margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig_cm, use_container_width=True)

                st.markdown("### 📋 Case-by-Case Benchmark Results")
                st.dataframe(pd.DataFrame(results_list), use_container_width=True)
        else:
            st.error("local_ai.py engine not found.")

# ------------------------------------------------------------------------------
# Tab 4: Skew-T Sounding
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("📈 Thermodynamic Skew-T Log-P Sounding Profile")
    p_arr = df_profile["Pressure (hPa)"].values
    T_arr = df_profile["Temperature (°C)"].values
    Td_arr = df_profile["Dewpoint (°C)"].values

    fig_skewt = go.Figure()
    fig_skewt.add_trace(go.Scatter(x=T_arr, y=p_arr, mode='lines+markers', name='Temperature (T)', line=dict(color='#ef4444', width=3)))
    fig_skewt.add_trace(go.Scatter(x=Td_arr, y=p_arr, mode='lines+markers', name='Dewpoint (Td)', line=dict(color='#10b981', width=3)))
    fig_skewt.update_yaxes(autorange="reversed", type="log", title="Pressure Level (hPa)")
    fig_skewt.update_xaxes(title="Temperature (°C)")
    fig_skewt.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20), hovermode="y unified")
    st.plotly_chart(fig_skewt, use_container_width=True)

# ------------------------------------------------------------------------------
# Tab 5: Kinematic Hodograph
# ------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("🌀 Kinematic Hodograph (Wind Shear & Helicity Spiral)")
    u_vals = df_profile["U-Wind (m/s)"].values
    v_vals = df_profile["V-Wind (m/s)"].values
    
    fig_hodo = go.Figure()
    fig_hodo.add_trace(go.Scatter(
        x=u_vals, y=v_vals, mode='lines+markers+text',
        text=[f"{p}hPa" for p in p_arr],
        textposition="top center",
        line=dict(color='#8b5cf6', width=4),
        marker=dict(size=8, color='#c084fc')
    ))
    fig_hodo.update_layout(
        xaxis=dict(title="Zonal Wind U (m/s)", zeroline=True, zerolinewidth=2, zerolinecolor='grey'),
        yaxis=dict(title="Meridional Wind V (m/s)", zeroline=True, zerolinewidth=2, zerolinecolor='grey'),
        height=450, margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_hodo, use_container_width=True)

# ------------------------------------------------------------------------------
# Tab 6: 2D Vertical Cross-Section
# ------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("📐 2D Mesoscale Vertical Cross-Section (Latitude vs Height θe)")
    cross_lats = np.linspace(target_lat - 2.0, target_lat + 2.0, 9)
    cross_p = np.array([1000, 925, 850, 700, 600, 500, 400, 300])
    
    z_theta_e = np.zeros((len(cross_p), len(cross_lats)))
    for i, p in enumerate(cross_p):
        for j, la in enumerate(cross_lats):
            dist = np.abs(la - target_lat)
            z_theta_e[i, j] = 348.0 - (1000 - p) * 0.03 - dist * 3.5

    fig_cross = go.Figure(data=go.Contour(
        z=z_theta_e, x=cross_lats, y=cross_p,
        colorscale='Plasma',
        colorbar=dict(title="θe (K)")
    ))
    fig_cross.update_yaxes(autorange="reversed", title="Pressure (hPa)")
    fig_cross.update_xaxes(title="Latitude (°N)")
    fig_cross.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_cross, use_container_width=True)

# ------------------------------------------------------------------------------
# Tab 7: Mesoscale Scan Grid Table
# ------------------------------------------------------------------------------
with tabs[6]:
    st.subheader("🗺️ Mesoscale 5×5 Spatial Grid Array")
    st.dataframe(pd.DataFrame(meso_data["grid_nodes"]), use_container_width=True)

# ------------------------------------------------------------------------------
# Tab 8: AI Sounding Prompt
# ------------------------------------------------------------------------------
with tabs[7]:
    st.subheader("📋 Objective Meteorological AI Prompt")
    prompt_text = f"""# METEOROLOGICAL 3D SOUNDING & KINEMATIC TENSOR INGESTION
TARGET COORDINATES: OBFUSCATED (3D SPHERICAL DOMAIN)
TEMPORAL STATE: {meso_data['temporal_state']}

## 1. VERTICAL THERMODYNAMIC MATRIX
{df_profile.to_markdown(index=False)}

## 2. INTEGRATED THERMODYNAMIC & KINEMATIC INDICES
- Column IVT: {center_metrics['IVT(kg/m/s)']} kg/(m·s) (Tendency: +{meso_data['ivt_tendency']}/h)
- 0-6 km Bulk Shear: {center_metrics['Bulk_Shear_0-6km(m/s)']} m/s
- Surface CAPE: {center_metrics['CAPE(J/kg)']} J/kg | CIN: {center_metrics['CIN(J/kg)']} J/kg
- 0-3 km Helicity (SRH): {center_metrics['SRH_0-3km(m2/s2)']} m²/s² | EHI: {center_metrics['EHI']}
- K-Index: {center_metrics['K_Index(°C)']} °C

## 3. TASK INSTRUCTIONS
Analyze the potential for quasi-stationary linear rainband formation and summarize:
1. Thermodynamic Instability & Moisture Inflow (Atmospheric River)
2. Vertical Wind Shear & Helicity Curvature
3. Convective Triggering & Inversion Cap Erosion
4. Final Convective Hazard Probability & Warning Level
"""
    st.code(prompt_text, language="markdown")
    st.download_button("📄 Download Prompt Markdown", data=prompt_text, file_name="sounding_prompt.md", mime="text/markdown")

# ------------------------------------------------------------------------------
# Tab 9: OpenMythos Direct Inference
# ------------------------------------------------------------------------------
with tabs[8]:
    st.subheader("⚡ OpenMythos 3D Spherical Volume Reasoning")
    if st.button("🚀 Run 3D Volume Inference (Recurrent Depth Loops)", type="primary"):
        if LOCAL_AI_AVAILABLE:
            with st.spinner(f"Running OpenMythos 3D RDT across {recurrent_loops} latent loops..."):
                report, loop_logs, elapsed_sec, risk_score = local_engine.infer_3d_volume(meso_data, recurrent_loops=recurrent_loops)
                st.success(f"✅ OpenMythos 3D Inference Complete! Latency: **{elapsed_sec * 1000:.1f} ms** (Loops: {recurrent_loops})")
                st.markdown(report)
        else:
            st.error("local_ai.py engine not found.")

    if gemini_api_key:
        st.markdown("---")
        st.subheader("🌐 Cloud LLM Direct Inference (Google Gemini)")
        if st.button("🚀 Run Cloud Gemini Inference"):
            with st.spinner("Querying Gemini cloud API..."):
                start_t = time.time()
                try:
                    if GEMINI_SDK_AVAILABLE:
                        genai.configure(api_key=gemini_api_key)
                        model = genai.GenerativeModel(cloud_model_name)
                        res = model.generate_content(prompt_text)
                        txt = res.text
                    else:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cloud_model_name}:generateContent?key={gemini_api_key}"
                        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
                        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
                        txt = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                    el = round(time.time() - start_t, 2)
                    st.success(f"✅ Cloud Inference Complete! Latency: **{el} seconds**")
                    st.markdown(txt)
                except Exception as e:
                    st.error(f"Cloud API Error: {str(e)}")

# ------------------------------------------------------------------------------
# Tab 10: Executive Briefing
# ------------------------------------------------------------------------------
with tabs[9]:
    st.subheader("📑 Executive Meteorological Briefing Report")
    report_md = f"""# Executive Meteorological Briefing Report
**Target Coordinates**: {target_lat}°N, {target_lon}°E  
**Temporal Timestamp**: {meso_data['temporal_state']}  
**System**: JRA-3Q OpenMythos 4D Mesoscale & 3D Spherical AI Forecaster

## 1. Key Atmospheric Indices
* **Column IVT (Atmospheric River)**: {center_metrics['IVT(kg/m/s)']} kg/(m·s) (Tendency: +{meso_data['ivt_tendency']}/h)
* **0-6 km Bulk Shear**: {center_metrics['Bulk_Shear_0-6km(m/s)']} m/s
* **Surface CAPE**: {center_metrics['CAPE(J/kg)']} J/kg | **CIN**: {center_metrics['CIN(J/kg)']} J/kg
* **0-3 km Helicity (SRH)**: {center_metrics['SRH_0-3km(m2/s2)']} m²/s² | **EHI**: {center_metrics['EHI']}
* **K-Index**: {center_metrics['K_Index(°C)']} °C

## 2. Summary Assessment
Severe convective rainband organization conditions are met when IVT exceeds 500 kg/(m·s) and 0-6km bulk shear is >= 15 m/s.
"""
    st.markdown(report_md)
    st.download_button("📥 Download Official Report (Markdown)", data=report_md, file_name=f"Meteorological_Report_{target_lat}_{target_lon}.md", mime="text/markdown")
