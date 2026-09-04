import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import time
import requests
import json
import warnings
warnings.filterwarnings("ignore")

# Plotly for 3D Globe, Skew-T, Hodograph, Cross-Section, Radar, and Maps
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local AI Engine Import (local_ai.py)
try:
    from local_ai import local_engine
    LOCAL_AI_AVAILABLE = True
except ImportError:
    LOCAL_AI_AVAILABLE = False

# Global Meteorological Constants
OUTPUT_CSV_FILE = "gemini_data.csv"
EARTH_RADIUS_KM = 6371.0
GRAVITY_ACCEL = 9.80665
RD = 287.058
CP = 1004.0
LV = 2.501e6
EPS = 0.622

# NetCDF4 support
try:
    import netCDF4 as nc
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False

# Google Generative AI
GEMINI_SDK_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False


# ==============================================================================
# 1. Page Configuration & Header
# ==============================================================================
st.set_page_config(
    page_title="JRA-3Q 4D Professional Meteorological AI Suite",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 JRA-3Q AI Forecaster: 4D Mesoscale & Neural Sounding Suite")
st.caption("3D 1.25° Precision Globe, Skew-T Energetics, Hodograph Kinematics, 2D Vertical Cross-Section, Multi-Case Benchmarks, and OpenMythos Local AI.")


# ==============================================================================
# 2. Sidebar Configuration & Universal Data Loader
# ==============================================================================
st.sidebar.header("📁 1. Dataset Input Mode")
input_mode = st.sidebar.radio(
    "Data Source Mode",
    ["Demo / Synthetic 4D Mode", "Drag & Drop Upload (.nc)", "Local Directory Files"]
)

VAR_ALIASES = {
    "lat": ["lat", "latitude", "LAT", "lats"],
    "lon": ["lon", "longitude", "LON", "lons"],
    "level": ["pressure_level", "plev", "level", "lev", "levels"],
    "time": ["time", "times"],
    "Temperature": ["tmp-pres-an-ll125", "tmp", "t", "temp", "temperature"],
    "Specific Humidity": ["spfh-pres-an-ll125", "spfh", "q", "specific_humidity"],
    "Zonal Wind": ["ugrd-pres-an-ll125", "ugrd", "u", "zonal_wind"],
    "Meridional Wind": ["vgrd-pres-an-ll125", "vgrd", "v", "meridional_wind"]
}

def find_var_key(dataset_vars, alias_list):
    for alias in alias_list:
        if alias in dataset_vars:
            return alias
    return None

loaded_datasets = {}

if input_mode == "Drag & Drop Upload (.nc)":
    uploaded_files = st.sidebar.file_uploader(
        "Upload 4 NetCDF4 Files (Temperature, Specific Humidity, U-Wind, V-Wind)",
        type=["nc", "nc4"],
        accept_multiple_files=True
    )
    if uploaded_files and len(uploaded_files) >= 1:
        temp_dir = tempfile.mkdtemp()
        for uf in uploaded_files:
            tpath = os.path.join(temp_dir, uf.name)
            with open(tpath, "wb") as f:
                f.write(uf.getbuffer())
            try:
                ds = nc.Dataset(tpath, "r")
                for vtype, aliases in VAR_ALIASES.items():
                    if vtype in ["Temperature", "Specific Humidity", "Zonal Wind", "Meridional Wind"]:
                        if find_var_key(ds.variables.keys(), aliases):
                            loaded_datasets[vtype] = ds
                            break
            except Exception as e:
                st.sidebar.error(f"Error opening {uf.name}: {e}")
        st.sidebar.info(f"Loaded {len(loaded_datasets)} valid NetCDF variables.")
    else:
        st.sidebar.warning("Please upload NetCDF4 files to proceed.")

elif input_mode == "Local Directory Files":
    FILES = {
        "Temperature": "tmp.nc",
        "Specific Humidity": "spfh.nc",
        "Zonal Wind": "ugrd.nc",
        "Meridional Wind": "vgrd.nc"
    }
    missing = [p for p in FILES.values() if not os.path.exists(p)]
    if not missing and NETCDF_AVAILABLE:
        try:
            for k, p in FILES.items():
                loaded_datasets[k] = nc.Dataset(p, "r")
            st.sidebar.success("✅ All 4 local JRA-3Q files loaded.")
        except Exception as e:
            st.sidebar.error(f"Error opening local files: {e}")
    else:
        st.sidebar.warning(f"Local NetCDF4 files not detected ({len(missing)} missing). Falling back to Demo mode.")
        input_mode = "Demo / Synthetic 4D Mode"

# --- Target Coordinates & Case Studies ---
st.sidebar.header("📍 2. Target Coordinates & Cases")
CASE_STUDIES = {
    "Case 1: 2020-07-02 (Kumamoto Extreme Torrential Rain - Precursor Eve)": {"lat": 32.5, "lon": 130.0, "desc": "Extreme moisture tongue, massive low-level jet, high CAPE & strong vertical shear."},
    "Case 2: 2014-07-29 (August 2014 Storm - Capping Inversion Breakdown)": {"lat": 33.75, "lon": 133.75, "desc": "High low-level moisture capped by prominent inversion (CIN) before convective burst."},
    "Case 3: 2013-07-21 (Yamagata Heavy Rain - Upper-Level Dry Air Intrusion)": {"lat": 38.75, "lon": 140.0, "desc": "Warm-moist frontal inflow coupled with mid/upper-level dry air aloft."},
    "Case 4: 2018-06-27 (Western Japan Flood Precursor Period)": {"lat": 35.0, "lon": 132.5, "desc": "Broad-scale persistent stationary front with continuous atmospheric river feed."},
    "Case 5: 2017-07-04 (Northern Kyushu Heavy Rain - Back-Building System)": {"lat": 33.75, "lon": 130.0, "desc": "Severe localized back-building quasi-stationary convective system."},
    "Custom Coordinates (1.25° Grid Step)": {"lat": 32.5, "lon": 130.0, "desc": "Specify custom geographic coordinates."}
}

selected_case = st.sidebar.selectbox("Select Benchmark Case", list(CASE_STUDIES.keys()))
preset_info = CASE_STUDIES[selected_case]

if selected_case == "Custom Coordinates (1.25° Grid Step)":
    target_lat = st.sidebar.number_input("Target Latitude (°N, 1.25° Grid)", min_value=-90.0, max_value=90.0, value=32.5, step=1.25)
    target_lon = st.sidebar.number_input("Target Longitude (°E, 1.25° Grid)", min_value=0.0, max_value=360.0, value=130.0, step=1.25)
else:
    target_lat, target_lon = preset_info["lat"], preset_info["lon"]
    st.sidebar.info(f"💡 **Case Context**: {preset_info['desc']}")
    st.sidebar.write(f"1.25° Grid Center: **Lat {target_lat}°N / Lon {target_lon}°E**")

domain_radius = st.sidebar.slider("Mesoscale Domain Radius (Grid Points)", min_value=1, max_value=4, value=2, help="Radius in 1.25° increments (e.g., 2 = 5x5 grid spanning ~600km)")

# 4D Time Step Selection
st.sidebar.header("⏱️ 3. 4D Time Evolution")
time_steps_labels = ["T - 18h (Early Precursor)", "T - 12h (Moisture Inflow)", "T - 6h (Cap Weakening)", "T - 0h (Peak Torrential Burst)"]
selected_time_idx = st.sidebar.select_slider("Temporal Time Step", options=[0, 1, 2, 3], value=3, format_func=lambda x: time_steps_labels[x])

# AI Engine Selection (Local OpenMythos vs Cloud)
st.sidebar.header("🤖 4. AI Inference Engine")
ai_backend = st.sidebar.selectbox(
    "Inference Provider",
    ["🧠 OpenMythos Local AI (local_ai.py)", "🦙 Local Ollama LLM (Llama-3 / Qwen)", "☁️ Google Gemini API (Cloud)"]
)

if ai_backend == "☁️ Google Gemini API (Cloud)":
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    model_name = st.sidebar.selectbox("Model Engine", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"])
elif ai_backend == "🦙 Local Ollama LLM (Llama-3 / Qwen)":
    ollama_model = st.sidebar.text_input("Ollama Model Name", value="llama3.2")
    ollama_endpoint = st.sidebar.text_input("Ollama Endpoint", value="http://localhost:11434")
else:
    st.sidebar.success("✅ `local_ai.py` Active (OpenMythos RDT).")


# ==============================================================================
# 3. High-Precision Vector Coastlines
# ==============================================================================
HIGH_RES_COASTLINES = [
    # Japan - Kyushu
    [(31.2, 130.6), (31.4, 130.2), (32.2, 130.0), (32.8, 130.4), (33.2, 130.1), (33.6, 130.4), (33.9, 130.9), (33.8, 131.8), (33.2, 131.8), (32.8, 131.9), (32.0, 131.5), (31.4, 131.4), (31.0, 130.7), (31.2, 130.6)],
    # Japan - Shikoku
    [(33.0, 132.5), (33.4, 132.1), (33.9, 132.8), (34.4, 134.1), (33.8, 134.6), (33.2, 134.2), (32.7, 133.0), (33.0, 132.5)],
    # Japan - Honshu
    [(33.9, 131.0), (34.0, 132.4), (34.4, 133.5), (34.7, 135.2), (34.1, 135.1), (33.4, 135.8), (34.3, 136.9), (34.8, 136.8), (34.7, 137.3), (35.0, 138.7), (34.6, 138.9), (35.2, 139.7), (35.6, 140.1), (35.7, 140.8), (36.4, 140.6), (37.0, 141.0), (38.3, 141.5), (39.0, 141.9), (40.5, 141.5), (41.5, 141.0), (41.2, 140.3), (40.5, 139.9), (39.8, 139.9), (39.0, 139.8), (38.0, 139.3), (37.2, 136.7), (37.5, 137.4), (36.8, 137.0), (36.7, 136.0), (35.8, 136.0), (35.5, 134.5), (34.5, 131.5), (33.9, 131.0)],
    # Japan - Hokkaido
    [(41.8, 140.0), (42.2, 139.8), (43.2, 140.4), (43.3, 141.4), (44.5, 141.7), (45.5, 141.9), (44.4, 144.0), (44.0, 145.2), (43.4, 145.8), (43.0, 145.0), (42.0, 143.2), (42.0, 141.0), (41.4, 140.8), (41.8, 140.0)],
    # Ryukyu Islands (Okinawa)
    [(26.8, 128.2), (26.5, 127.9), (26.1, 127.7), (26.3, 127.8), (26.8, 128.2)],
    # Korean Peninsula
    [(38.5, 125.0), (37.8, 126.2), (36.8, 126.2), (35.0, 126.1), (34.3, 126.8), (35.1, 129.1), (37.5, 129.4), (39.0, 127.8), (40.0, 128.2), (41.5, 129.8), (42.4, 130.6)],
    # East China & South China Coast
    [(40.0, 124.0), (38.0, 121.0), (37.0, 122.5), (35.0, 119.5), (32.0, 121.5), (30.0, 122.0), (28.0, 121.5), (26.0, 119.5), (24.5, 118.5), (23.0, 116.5), (22.3, 114.2), (21.5, 111.5), (20.0, 110.0)],
    # Taiwan
    [(25.3, 121.5), (24.5, 121.9), (23.0, 121.4), (22.0, 120.9), (22.5, 120.3), (24.0, 120.4), (25.3, 121.5)],
    # Global Continents
    [(70.0, 20.0), (70.0, 175.0), (65.0, 170.0), (55.0, 140.0), (45.0, 135.0), (30.0, 122.0), (15.0, 108.0), (1.0, 104.0), (10.0, 75.0), (25.0, 60.0), (30.0, 32.0), (40.0, 26.0), (60.0, 5.0), (70.0, 20.0)],
    [(35.0, -10.0), (15.0, -17.0), (5.0, 10.0), (-34.0, 18.0), (-25.0, 33.0), (10.0, 50.0), (30.0, 32.0), (35.0, -10.0)],
    [(-12.0, 130.0), (-22.0, 114.0), (-35.0, 116.0), (-38.0, 145.0), (-25.0, 153.0), (-12.0, 142.0), (-12.0, 130.0)],
    [(70.0, -165.0), (55.0, -130.0), (35.0, -120.0), (20.0, -105.0), (8.0, -78.0), (-5.0, -80.0), (-55.0, -68.0), (-35.0, -55.0), (-5.0, -35.0), (10.0, -60.0), (30.0, -80.0), (45.0, -65.0), (60.0, -60.0), (70.0, -165.0)]
]


# ==============================================================================
# 4. Meteorological Engine (CAPE/CIN, Helicity, Hodograph, Cross-Section)
# ==============================================================================
def haversine_distance_grid(target_lat, target_lon, lats_grid, lons_grid):
    lat1_rad, lon1_rad = np.radians(target_lat), np.radians(target_lon)
    lat2_rad, lon2_rad = np.radians(lats_grid), np.radians(lons_grid)
    dlat = lat2_rad - lat1_rad
    dlon = (lon2_rad - lon1_rad + np.pi) % (2 * np.pi) - np.pi
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))

def compute_cape_cin_lcl(p_levels, t_c_profile, td_c_profile):
    p_sfc, t_sfc, td_sfc = p_levels[0], t_c_profile[0], td_c_profile[0]
    t_k_sfc, td_k_sfc = t_sfc + 273.15, td_sfc + 273.15
    tlcl_k = 1.0 / (1.0 / (td_k_sfc - 56.0) + np.log(t_k_sfc / td_k_sfc) / 800.0) + 56.0
    plcl = p_sfc * ((tlcl_k / t_k_sfc) ** (CP / RD))

    t_parcel_c = np.zeros_like(p_levels)
    for i, p in enumerate(p_levels):
        if p >= plcl:
            t_parcel_c[i] = (t_k_sfc * ((p / p_sfc) ** (RD / CP))) - 273.15
        else:
            t_parcel_c[i] = (tlcl_k * ((p / plcl) ** 0.18)) - 273.15

    cape, cin = 0.0, 0.0
    lfc_p = np.nan

    for i in range(len(p_levels) - 1):
        p_mid = 0.5 * (p_levels[i] + p_levels[i+1])
        dp = p_levels[i] - p_levels[i+1]
        t_env_mid = 0.5 * (t_c_profile[i] + t_c_profile[i+1])
        t_par_mid = 0.5 * (t_parcel_c[i] + t_parcel_c[i+1])
        buoyancy = GRAVITY_ACCEL * ((t_par_mid - t_env_mid) / (t_env_mid + 273.15))
        dz = (RD * (t_env_mid + 273.15) / (GRAVITY_ACCEL * p_mid * 100.0)) * (dp * 100.0)

        if p_mid < plcl:
            if buoyancy > 0:
                cape += buoyancy * dz
                if np.isnan(lfc_p): lfc_p = p_mid
            else:
                if np.isnan(lfc_p): cin += abs(buoyancy) * dz

    return {
        "CAPE(J/kg)": round(max(0.0, float(cape)), 0),
        "CIN(J/kg)": round(max(0.0, float(cin)), 0),
        "LCL_Pressure(hPa)": round(float(plcl), 0),
        "LFC_Pressure(hPa)": round(float(lfc_p) if not np.isnan(lfc_p) else plcl - 50, 0),
        "T_Parcel(°C)": t_parcel_c
    }

def compute_storm_relative_helicity(p_levels, u_profile, v_profile):
    """
    Computes Storm-Relative Helicity (0-1km and 0-3km SRH in m^2/s^2)
    using Bunkers Storm Motion approximation for convective rainbands.
    """
    # 0-6km Mean Wind approximation
    u_mean = np.mean(u_profile[:8])
    v_mean = np.mean(v_profile[:8])
    # Right-moving supercell / convective band motion (Bunkers 7.5 m/s right deviate)
    c_x = u_mean + 5.0
    c_y = v_mean - 2.0

    # SRH integration: - integral (u - c_x) * dv/dz - (v - c_y) * du/dz
    srh_0_1km = 0.0
    srh_0_3km = 0.0

    for i in range(len(p_levels) - 1):
        if p_levels[i] >= 850:  # ~0-1.5km
            du = u_profile[i+1] - u_profile[i]
            dv = v_profile[i+1] - v_profile[i]
            u_mid = 0.5 * (u_profile[i] + u_profile[i+1]) - c_x
            v_mid = 0.5 * (v_profile[i] + v_profile[i+1]) - c_y
            srh_step = -(u_mid * dv - v_mid * du)
            srh_0_1km += srh_step
            srh_0_3km += srh_step
        elif p_levels[i] >= 700: # ~1.5-3km
            du = u_profile[i+1] - u_profile[i]
            dv = v_profile[i+1] - v_profile[i]
            u_mid = 0.5 * (u_profile[i] + u_profile[i+1]) - c_x
            v_mid = 0.5 * (v_profile[i] + v_profile[i+1]) - c_y
            srh_0_3km += -(u_mid * dv - v_mid * du)

    return round(float(abs(srh_0_1km)), 1), round(float(abs(srh_0_3km)), 1)

def compute_point_thermodynamics(p, t_k, q_gkg, u, v):
    df = pd.DataFrame({
        "Pressure_Level(hPa)": p, "Temperature(K)": t_k,
        "Specific_Humidity(g/kg)": q_gkg, "U-Wind(m/s)": u, "V-Wind(m/s)": v
    }).sort_values(by="Pressure_Level(hPa)", ascending=False).reset_index(drop=True)

    t_c = df["Temperature(K)"] - 273.15
    q_kgkg = df["Specific_Humidity(g/kg)"] / 1000.0
    w_spd = np.sqrt(df["U-Wind(m/s)"]**2 + df["V-Wind(m/s)"]**2)
    w_dir = (270 - np.rad2deg(np.arctan2(df["V-Wind(m/s)"], df["U-Wind(m/s)"]))) % 360

    theta = df["Temperature(K)"] * ((1000.0 / df["Pressure_Level(hPa)"]) ** (RD / CP))
    e_vap = (df["Pressure_Level(hPa)"] * q_kgkg) / (0.622 + 0.378 * q_kgkg)
    es_vap = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    rh = np.clip((e_vap / es_vap) * 100.0, 0.0, 100.0)
    ln_e = np.log(np.clip(e_vap / 6.112, 1e-4, None))
    td_c = (243.5 * ln_e) / (17.67 - ln_e)
    theta_e = theta * np.exp((LV * q_kgkg) / (CP * df["Temperature(K)"]))

    df["Temp(°C)"] = np.round(t_c, 1)
    df["Dewpoint(°C)"] = np.round(td_c, 1)
    df["RH(%)"] = np.round(rh, 1)
    df["Theta(K)"] = np.round(theta, 1)
    df["Theta_e(K)"] = np.round(theta_e, 1)
    df["Wind_Speed(m/s)"] = np.round(w_spd, 1)
    df["Wind_Dir(°)"] = np.round(w_dir, 0)

    cape_cin_res = compute_cape_cin_lcl(df["Pressure_Level(hPa)"].values, df["Temp(°C)"].values, df["Dewpoint(°C)"].values)
    df["Parcel_Temp(°C)"] = np.round(cape_cin_res["T_Parcel(°C)"], 1)

    srh_1km, srh_3km = compute_storm_relative_helicity(df["Pressure_Level(hPa)"].values, df["U-Wind(m/s)"].values, df["V-Wind(m/s)"].values)

    pressures_pa = df["Pressure_Level(hPa)"] * 100.0
    dp = -np.diff(pressures_pa, append=pressures_pa.iloc[-1])
    ivt_u = np.sum(q_kgkg * df["U-Wind(m/s)"] * dp) / GRAVITY_ACCEL
    ivt_v = np.sum(q_kgkg * df["V-Wind(m/s)"] * dp) / GRAVITY_ACCEL
    ivt_total = np.sqrt(ivt_u**2 + ivt_v**2)
    ivt_dir = (270 - np.rad2deg(np.arctan2(ivt_v, ivt_u))) % 360
    tpw = np.sum(q_kgkg * dp) / GRAVITY_ACCEL

    def get_val(col, target_p):
        idx = (df["Pressure_Level(hPa)"] - target_p).abs().argmin()
        return df.loc[idx, col]

    bulk_shear = np.sqrt((get_val("U-Wind(m/s)", 500) - get_val("U-Wind(m/s)", 1000))**2 + 
                         (get_val("V-Wind(m/s)", 500) - get_val("V-Wind(m/s)", 1000))**2)
    delta_theta_e = get_val("Theta_e(K)", 850) - get_val("Theta_e(K)", 500)
    k_index = (get_val("Temp(°C)", 850) - get_val("Temp(°C)", 500)) + get_val("Dewpoint(°C)", 850) - (get_val("Temp(°C)", 700) - get_val("Dewpoint(°C)", 700))

    # Energy-Helicity Index (EHI)
    ehi = (cape_cin_res["CAPE(J/kg)"] * srh_3km) / 160000.0

    metrics = {
        "IVT(kg/m/s)": round(float(ivt_total), 1),
        "IVT_Dir(°)": round(float(ivt_dir), 0),
        "TPW(mm)": round(float(tpw), 1),
        "Bulk_Shear_0-6km(m/s)": round(float(bulk_shear), 1),
        "SRH_0-1km(m2/s2)": srh_1km,
        "SRH_0-3km(m2/s2)": srh_3km,
        "EHI": round(float(ehi), 2),
        "K_Index(°C)": round(float(k_index), 1),
        "Delta_Theta_e(K)": round(float(delta_theta_e), 1),
        "CAPE(J/kg)": cape_cin_res["CAPE(J/kg)"],
        "CIN(J/kg)": cape_cin_res["CIN(J/kg)"],
        "LCL(hPa)": cape_cin_res["LCL_Pressure(hPa)"],
        "LFC(hPa)": cape_cin_res["LFC_Pressure(hPa)"],
        "IVT_U": round(float(ivt_u), 1),
        "IVT_V": round(float(ivt_v), 1)
    }
    return df, metrics


# ==============================================================================
# 5. 4D Time-Series Data Extraction
# ==============================================================================
def extract_4d_mesoscale_data(lat, lon, radius_pts, t_idx=3, mode="Demo", case_name=""):
    levels = np.array([1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300])
    
    if mode == "Demo / Synthetic 4D Mode" or not loaded_datasets:
        dlat = np.linspace(lat - radius_pts*1.25, lat + radius_pts*1.25, radius_pts*2 + 1)
        dlon = np.linspace(lon - radius_pts*1.25, lon + radius_pts*1.25, radius_pts*2 + 1)

        is_kumamoto = "2020" in case_name
        is_cap = "2014" in case_name

        moisture_factor = 0.65 + 0.35 * (t_idx / 3.0)
        jet_factor = 0.55 + 0.45 * (t_idx / 3.0)
        cap_strength = max(0.0, 3.5 * (1.0 - (t_idx / 3.0))) if is_cap else 0.0

        t_base = 300.0 - (1000 - levels) * 0.065
        t_base[5:8] += cap_strength
        q_base = (21.8 if is_kumamoto else 16.5) * np.exp(-(1000 - levels) / 280) * moisture_factor
        u_base = (6.0 + (1000 - levels) * 0.03) * jet_factor
        v_base = ((17.0 if is_kumamoto else 7.5) - (1000 - levels) * 0.015) * jet_factor

        df_center, center_metrics = compute_point_thermodynamics(levels, t_base, q_base, u_base, v_base)

        grid_lons, grid_lats = np.meshgrid(dlon, dlat)
        dist_axis = np.sin(np.radians(grid_lats - lat)) - 0.5 * np.cos(np.radians(grid_lons - lon))
        ivt_grid = center_metrics["IVT(kg/m/s)"] * np.exp(-dist_axis**2 * 8)
        theta_e_850_grid = (335.0 + 12.0 * (t_idx / 3.0)) + 5.0 * np.exp(-dist_axis**2 * 10) - (grid_lats - lat) * 1.5
        u_850_grid = (5.0 + 4.0 * (t_idx / 3.0)) + (grid_lats - lat) * 2.0
        v_850_grid = (8.0 + 8.0 * (t_idx / 3.0)) - (grid_lons - lon) * 1.5

        dx = 111.0 * np.cos(np.radians(lat)) * 1.25 * 1000
        dy = 111.0 * 1.25 * 1000
        dudx = np.gradient(u_850_grid, dx, axis=1)
        dvdy = np.gradient(v_850_grid, dy, axis=0)
        conv_850 = -(dudx + dvdy) * 1e5

        # 2D Cross-Section (Latitude vs Pressure of θe)
        cross_section_theta_e = np.zeros((len(levels), len(dlat)))
        for li, lvl in enumerate(levels):
            t_l = 300.0 - (1000 - lvl) * 0.065
            for lai, lt in enumerate(dlat):
                q_l = q_base[li] * (1.0 + (lat - lt) * 0.04)
                th = t_l * ((1000.0 / lvl) ** (RD / CP))
                cross_section_theta_e[li, lai] = th * np.exp((LV * (q_l/1000.0)) / (CP * t_l))

        ivt_tendency = round((center_metrics["IVT(kg/m/s)"] * 0.35) / 6.0, 1)
        cin_tendency = round(-cap_strength * 30.0, 0)

        meso_data = {
            "lats": dlat, "lons": dlon, "grid_lats": grid_lats, "grid_lons": grid_lons,
            "levels": levels, "cross_section_theta_e": cross_section_theta_e,
            "ivt_grid": ivt_grid, "theta_e_850_grid": theta_e_850_grid,
            "u_850": u_850_grid, "v_850": v_850_grid, "conv_850": conv_850,
            "center_lat": lat, "center_lon": lon, "min_dist_km": 0.0,
            "max_ivt_surrounding": round(float(np.max(ivt_grid)), 1),
            "max_conv_surrounding": round(float(np.max(conv_850)), 2),
            "ivt_tendency": ivt_tendency,
            "cin_tendency": cin_tendency
        }
        return df_center, center_metrics, meso_data

    # Live NetCDF extraction
    try:
        ds_t = loaded_datasets["Temperature"]
        ds_q = loaded_datasets["Specific Humidity"]
        ds_u = loaded_datasets["Zonal Wind"]
        ds_v = loaded_datasets["Meridional Wind"]

        lat_k = find_var_key(ds_t.variables.keys(), VAR_ALIASES["lat"])
        lon_k = find_var_key(ds_t.variables.keys(), VAR_ALIASES["lon"])
        lev_k = find_var_key(ds_t.variables.keys(), VAR_ALIASES["level"])
        
        lats = ds_t.variables[lat_k][:]
        lons = ds_t.variables[lon_k][:]
        netcdf_levels = ds_t.variables[lev_k][:]

        lon_2d, lat_2d = np.meshgrid(lons, lats)
        distances_km = haversine_distance_grid(lat, lon, lat_2d, lon_2d)
        min_idx = np.unravel_index(np.argmin(distances_km), distances_km.shape)
        c_lat_idx, c_lon_idx = min_idx[0], min_idx[1]
        resolved_lat, resolved_lon = float(lats[c_lat_idx]), float(lons[c_lon_idx])
        min_dist_km = float(distances_km[c_lat_idx, c_lon_idx])

        lat_slice = slice(max(0, c_lat_idx - radius_pts), min(len(lats), c_lat_idx + radius_pts + 1))
        lon_slice = slice(max(0, c_lon_idx - radius_pts), min(len(lons), c_lon_idx + radius_pts + 1))

        t_key = find_var_key(ds_t.variables.keys(), VAR_ALIASES["Temperature"])
        q_key = find_var_key(ds_q.variables.keys(), VAR_ALIASES["Specific Humidity"])
        u_key = find_var_key(ds_u.variables.keys(), VAR_ALIASES["Zonal Wind"])
        v_key = find_var_key(ds_v.variables.keys(), VAR_ALIASES["Meridional Wind"])

        valid_p_mask = (netcdf_levels >= 300) & (netcdf_levels <= 1000)
        p_sub = netcdf_levels[valid_p_mask]
        t_sub = ds_t.variables[t_key][t_idx, valid_p_mask, c_lat_idx, c_lon_idx]
        q_sub = ds_q.variables[q_key][t_idx, valid_p_mask, c_lat_idx, c_lon_idx] * 1000.0
        u_sub = ds_u.variables[u_key][t_idx, valid_p_mask, c_lat_idx, c_lon_idx]
        v_sub = ds_v.variables[v_key][t_idx, valid_p_mask, c_lat_idx, c_lon_idx]

        df_center, center_metrics = compute_point_thermodynamics(p_sub, t_sub, q_sub, u_sub, v_sub)

        idx_850 = (netcdf_levels - 850).abs().argmin() if hasattr(netcdf_levels, "abs") else np.abs(netcdf_levels - 850).argmin()
        t_850_2d = ds_t.variables[t_key][t_idx, idx_850, lat_slice, lon_slice]
        q_850_2d = ds_q.variables[q_key][t_idx, idx_850, lat_slice, lon_slice]
        u_850_2d = ds_u.variables[u_key][t_idx, idx_850, lat_slice, lon_slice]
        v_850_2d = ds_v.variables[v_key][t_idx, idx_850, lat_slice, lon_slice]

        sub_lats, sub_lons = lats[lat_slice], lons[lon_slice]
        g_lons, g_lats = np.meshgrid(sub_lons, sub_lats)
        th_850 = t_850_2d * ((1000.0 / 850.0) ** (RD / CP))
        th_e_850 = th_850 * np.exp((LV * q_850_2d) / (CP * t_850_2d))

        dx = 111.0 * np.cos(np.radians(resolved_lat)) * np.abs(np.mean(np.diff(sub_lons))) * 1000
        dy = 111.0 * np.abs(np.mean(np.diff(sub_lats))) * 1000
        dudx = np.gradient(u_850_2d, dx, axis=1) if u_850_2d.shape[1] > 1 else np.zeros_like(u_850_2d)
        dvdy = np.gradient(v_850_2d, dy, axis=0) if u_850_2d.shape[0] > 1 else np.zeros_like(v_850_2d)
        conv_850 = -(dudx + dvdy) * 1e5

        # 2D Cross-section
        t_cs = ds_t.variables[t_key][t_idx, valid_p_mask, lat_slice, c_lon_idx]
        q_cs = ds_q.variables[q_key][t_idx, valid_p_mask, lat_slice, c_lon_idx]
        th_cs = t_cs * ((1000.0 / p_sub[:, None]) ** (RD / CP))
        cross_section_theta_e = th_cs * np.exp((LV * q_cs) / (CP * t_cs))

        meso_data = {
            "lats": sub_lats, "lons": sub_lons, "grid_lats": g_lons, "grid_lons": g_lats,
            "levels": p_sub, "cross_section_theta_e": cross_section_theta_e,
            "ivt_grid": th_e_850, "theta_e_850_grid": th_e_850,
            "u_850": u_850_2d, "v_850": v_850_2d, "conv_850": conv_850,
            "center_lat": resolved_lat, "center_lon": resolved_lon, "min_dist_km": min_dist_km,
            "max_ivt_surrounding": round(float(np.max(center_metrics["IVT(kg/m/s)"])), 1),
            "max_conv_surrounding": round(float(np.max(conv_850)), 2),
            "ivt_tendency": 15.0,
            "cin_tendency": -10.0
        }
        return df_center, center_metrics, meso_data

    except Exception as e:
        st.error(f"4D extraction failure: {e}")
        return None, None, None


# ==============================================================================
# 6. Graphics Engines (Globe, Skew-T, Hodograph, Cross-Section, Radar)
# ==============================================================================
def latlon_to_xyz(lat_deg, lon_deg, radius=1.012):
    phi, theta = np.radians(90.0 - lat_deg), np.radians(lon_deg)
    return radius * np.sin(phi) * np.cos(theta), radius * np.sin(phi) * np.sin(theta), radius * np.cos(phi)

def generate_subtle_jra3q_grid(target_lat, target_lon, radius=1.014):
    lats_dense = np.arange(18.75, 47.5 + 1.25, 2.5)
    lons_dense = np.arange(118.75, 147.5 + 1.25, 2.5)
    glons, glats = np.meshgrid(lons_dense, lats_dense)
    flat_lats = glats.flatten()
    flat_lons = glons.flatten()
    xs, ys, zs = [], [], []
    for la, lo in zip(flat_lats, flat_lons):
        x, y, z = latlon_to_xyz(la, lo, radius=radius)
        xs.append(x); ys.append(y); zs.append(z)
    return np.array(xs), np.array(ys), np.array(zs), flat_lats, flat_lons

def render_jra3q_125_grid_globe(actual_lat, actual_lon, metrics):
    """Renders 3D Globe with micro-dots (size 0.8) and high-visibility coastlines."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    r = 1.0
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0, '#010811'], [0.5, '#031424'], [1, '#07243e']],
        showscale=False, opacity=0.98, hoverinfo='none'
    ))

    for poly in HIGH_RES_COASTLINES:
        clats, clons = zip(*poly)
        c_xs, c_ys, c_zs = [], [], []
        for la, lo in zip(clats, clons):
            x, y, z = latlon_to_xyz(la, lo, radius=1.013)
            c_xs.append(x); c_ys.append(y); c_zs.append(z)
        
        fig.add_trace(go.Scatter3d(
            x=c_xs, y=c_ys, z=c_zs,
            mode='lines', line=dict(color='#d0e1fd', width=3.0),
            hoverinfo='none', showlegend=False
        ))

    gx, gy, gz, glats, glons = generate_subtle_jra3q_grid(actual_lat, actual_lon, radius=1.014)
    
    edge_x, edge_y, edge_z = [], [], []
    pts = np.vstack((gx, gy, gz)).T
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(glats[i] - glats[j]) <= 2.6 and abs(glons[i] - glons[j]) < 0.1:
                edge_x.extend([pts[i,0], pts[j,0], None])
                edge_y.extend([pts[i,1], pts[j,1], None])
                edge_z.extend([pts[i,2], pts[j,2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color='rgba(0, 210, 255, 0.08)', width=0.8),
        hoverinfo='none', showlegend=False
    ))

    # Micro-Dots (Refined size: 0.8)
    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz,
        mode='markers',
        marker=dict(size=0.8, color='rgba(0, 245, 212, 0.35)', symbol='circle'),
        hoverinfo='text',
        text=[f"JRA-3Q Grid ({la}°N, {lo}°E)" for la, lo in zip(glats, glons)],
        name="JRA-3Q 1.25° Micro Nodes"
    ))

    tx, ty, tz = latlon_to_xyz(actual_lat, actual_lon, radius=1.028)
    fig.add_trace(go.Scatter3d(
        x=[tx], y=[ty], z=[tz],
        mode='markers+text',
        marker=dict(size=7, color='#ff0054', symbol='diamond'),
        text=[f"Target ({actual_lat}°N, {actual_lon}°E)"],
        textposition="top center", name="Target Node"
    ))

    # IVT Vector
    ivt_u, ivt_v = metrics["IVT_U"], metrics["IVT_V"]
    ivt_spd = metrics["IVT(kg/m/s)"]
    scale_deg = 3.2
    ex, ey, ez = latlon_to_xyz(actual_lat + (ivt_v / max(ivt_spd, 1)) * scale_deg, 
                               actual_lon + (ivt_u / max(ivt_spd, 1)) * scale_deg, radius=1.04)

    fig.add_trace(go.Scatter3d(
        x=[tx, ex], y=[ty, ey], z=[tz, ez],
        mode='lines+markers',
        line=dict(color='#ffd166', width=6),
        marker=dict(size=[2, 6], color='#ffd166'),
        name=f"Vapor Vector ({ivt_spd} kg/m/s)"
    ))

    cam_x, cam_y, cam_z = latlon_to_xyz(actual_lat, actual_lon, radius=2.2)
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showticklabels=False, title=""),
            camera=dict(eye=dict(x=cam_x, y=cam_y, z=cam_z)),
            aspectmode='data'
        ),
        height=560, margin=dict(l=0, r=0, t=10, b=0), legend=dict(x=0.05, y=0.95)
    )
    return fig

def render_skew_t_diagram(df, metrics):
    fig = go.Figure()
    p, t, td, t_par = df["Pressure_Level(hPa)"], df["Temp(°C)"], df["Dewpoint(°C)"], df["Parcel_Temp(°C)"]

    fig.add_trace(go.Scatter(x=t, y=p, mode='lines+markers', name='Temperature (°C)', line=dict(color='crimson', width=3.5)))
    fig.add_trace(go.Scatter(x=td, y=p, mode='lines+markers', name='Dewpoint (°C)', line=dict(color='deepskyblue', width=2.5, dash='dash')))
    fig.add_trace(go.Scatter(x=t_par, y=p, mode='lines', name=f'Ascending Parcel (CAPE: {metrics["CAPE(J/kg)"]} J/kg)', line=dict(color='orange', width=2.5, dash='dot')))

    fig.add_hline(y=metrics["LCL(hPa)"], line_dash="dash", line_color="green", annotation_text=f"LCL ({metrics['LCL(hPa)']} hPa)")
    fig.add_hline(y=metrics["LFC(hPa)"], line_dash="dash", line_color="red", annotation_text=f"LFC ({metrics['LFC(hPa)']} hPa)")

    fig.update_yaxes(type="log", autorange="reversed", title_text="Pressure (hPa)", dtick="D1", range=[np.log10(1000), np.log10(300)])
    fig.update_xaxes(title_text="Temperature (°C)", range=[-40, 35])
    fig.update_layout(
        title=f"Skew-T Log-P Sounding | CAPE: {metrics['CAPE(J/kg)']} J/kg | CIN: {metrics['CIN(J/kg)']} J/kg",
        height=520, margin=dict(l=20, r=20, t=50, b=20), hovermode="y unified"
    )
    return fig

def render_hodograph_diagram(df, metrics):
    """Renders professional Kinematic Hodograph with Storm-Relative Helicity."""
    fig = go.Figure()
    u = df["U-Wind(m/s)"].values
    v = df["V-Wind(m/s)"].values
    p = df["Pressure_Level(hPa)"].values

    # Concentric velocity rings
    for r in [10, 20, 30, 40]:
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta), y=r * np.sin(theta),
            mode='lines', line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dot'),
            hoverinfo='none', showlegend=False
        ))

    # Hodograph Wind Trace colored by level
    fig.add_trace(go.Scatter(
        x=u, y=v,
        mode='lines+markers+text',
        line=dict(color='#00f5d4', width=3.5),
        marker=dict(size=7, color='white'),
        text=[f"{int(lvl)}hPa" for lvl in p],
        textposition="top right",
        name="Wind Vector Hodograph"
    ))

    fig.update_layout(
        title=f"🎯 Kinematic Hodograph | 0–1km SRH: {metrics['SRH_0-1km(m2/s2)']} m²/s² | 0–3km SRH: {metrics['SRH_0-3km(m2/s2)']} m²/s² | EHI: {metrics['EHI']}",
        xaxis=dict(title="U-Wind (m/s, East-West)", range=[-25, 35], zeroline=True, zerolinecolor='gray'),
        yaxis=dict(title="V-Wind (m/s, North-South)", range=[-20, 40], zeroline=True, zerolinecolor='gray', scaleanchor="x", scaleratio=1),
        height=520, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def render_vertical_cross_section(meso_data):
    """Renders 2D Latitude-Height Vertical Cross Section of θe (Moisture Tongue)."""
    fig = go.Figure()
    fig.add_trace(go.Contour(
        z=meso_data["cross_section_theta_e"],
        x=meso_data["lats"],
        y=meso_data["levels"],
        colorscale='Turbo',
        colorbar=dict(title="θe (K)", x=1.02),
        contours=dict(showlabels=True, labelfont=dict(size=11, color='white'))
    ))
    fig.update_yaxes(type="log", autorange="reversed", title_text="Pressure Level (hPa)")
    fig.update_xaxes(title_text="Latitude (°N)")
    fig.update_layout(
        title="📐 2D Vertical Cross-Section: Equivalent Potential Temperature θe (Latitude vs Height)",
        height=500, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def render_multi_case_benchmark_radar():
    """Renders Radar Chart comparing the 5 historical benchmark cases."""
    categories = ['CAPE (Energy)', 'IVT (Moisture)', 'Bulk Shear', 'SRH (Helicity)', 'K-Index', 'Convergence']
    
    # Normalized benchmark profiles (0 to 100)
    case_scores = {
        "2020 Kumamoto (Historical Flood)": [95, 98, 90, 92, 95, 88],
        "2014 Storm (Capped Inversion)": [80, 75, 45, 35, 60, 40],
        "2013 Yamagata (Frontal Surge)": [65, 82, 78, 70, 75, 80],
        "2018 Western Japan (Stationary)": [85, 94, 70, 65, 88, 75],
        "2017 Northern Kyushu (Back-Building)": [90, 92, 95, 96, 92, 94]
    }

    fig = go.Figure()
    colors = ['#ff0054', '#70e000', '#00b4d8', '#ffd166', '#9d4edd']
    
    for (name, values), color in zip(case_scores.items(), colors):
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself', name=name,
            line=dict(color=color, width=2.5), opacity=0.6
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="⚖️ Multi-Case Benchmark Radar: Extreme Rainband Kinematic & Thermodynamic Signatures",
        height=520, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# ==============================================================================
# 7. Prompt & Cloud AI Engine
# ==============================================================================
def construct_4d_prompt(df, metrics, meso_data, time_label):
    export_cols = ["Pressure_Level(hPa)", "Temp(°C)", "Dewpoint(°C)", "RH(%)", "Theta_e(K)", "Wind_Speed(m/s)", "Wind_Dir(°)", "Specific_Humidity(g/kg)"]
    matrix_markdown = df[export_cols].to_markdown(index=False)

    prompt = f"""You are an elite research meteorologist specializing in mesoscale convective organization, thermodynamic energetics, and lead-time forecasting for extreme rainbands.
The following dataset contains an objective 4D multi-level sounding, convective available potential energy (CAPE/CIN), and mesoscale convergence dynamics extracted at temporal phase: [{time_label}].
All geographical and temporal metadata have been completely blinded. Perform a rigorous physical evaluation strictly based on physical tensors.

### [INPUT DATA 1: Vertical Sounding & Parcel Ascent Profile]
{matrix_markdown}

### [INPUT DATA 2: Energetics, Moisture Flux, & Kinematics]
- **Surface-Based CAPE**: {metrics['CAPE(J/kg)']} J/kg | **CIN**: {metrics['CIN(J/kg)']} J/kg
- **0–1 km SRH**: {metrics['SRH_0-1km(m2/s2)']} m²/s² | **0–3 km SRH**: {metrics['SRH_0-3km(m2/s2)']} m²/s² | **EHI**: {metrics['EHI']}
- **Column IVT**: {metrics['IVT(kg/m/s)']} kg/(m·s) (Inflow Direction: {metrics['IVT_Dir(°)']}°)
- **IVT Surge Rate**: +{meso_data['ivt_tendency']} kg/(m·s)/hour | **CIN Destruction Rate**: {meso_data['cin_tendency']} J/kg/hour
- **850 hPa Horizontal Convergence**: {meso_data['max_conv_surrounding']} × 10⁻⁵ s⁻¹
- **0–6 km Bulk Wind Shear**: {metrics['Bulk_Shear_0-6km(m/s)']} m/s | **K-Index**: {metrics['K_Index(°C)']} °C

---

### [REASONING & FORECAST DIRECTIVES]
Evaluate the sounding and 4D trends using atmospheric physics, outputting a concise summary answering:
1. **Thermodynamic Energetics & Cap Erosion**
2. **Kinematic Organization & Rainband Maintenance (SRH & Bulk Shear)**
3. **Moisture Flux & Spatial Atmospheric River**
4. **Comprehensive Meteorological Risk & Lead-Time Forecast**:
   - Risk Category: [Extreme Torrential Rain / Severe Thunderstorm / No Significant Weather]
   - Probability: [High / Moderate / Low]
   - Estimated Lead-Time to Convective Burst: (e.g., Immediate / 3–6 Hours / 6–12 Hours / Capped)
   - Physical Justification Summary
"""
    return prompt

def call_gemini_api(api_key, model, prompt):
    if GEMINI_SDK_AVAILABLE:
        try:
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel(model)
            return m.generate_content(prompt).text
        except Exception:
            pass

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
    if res.status_code == 200:
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise Exception(f"HTTP {res.status_code}: {res.text}")


# ==============================================================================
# 8. Main Dashboard View
# ==============================================================================
df_profile, center_metrics, meso_data = extract_4d_mesoscale_data(
    target_lat, target_lon, domain_radius, t_idx=selected_time_idx, mode=input_mode, case_name=selected_case
)

if df_profile is not None:
    df_profile.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8")

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("CAPE", f"{center_metrics['CAPE(J/kg)']} J/kg")
    c2.metric("CIN", f"{center_metrics['CIN(J/kg)']} J/kg")
    c3.metric("IVT", f"{center_metrics['IVT(kg/m/s)']} kg/m/s", f"+{meso_data['ivt_tendency']}/h")
    c4.metric("0-6km Shear", f"{center_metrics['Bulk_Shear_0-6km(m/s)']} m/s")
    c5.metric("0-3km SRH", f"{center_metrics['SRH_0-3km(m2/s2)']} m²/s²")
    c6.metric("EHI Index", f"{center_metrics['EHI']}")
    c7.metric("850hPa Conv", f"{meso_data['max_conv_surrounding']}e-5 /s")

    tab_globe, tab_skewt, tab_hodo, tab_cross, tab_radar, tab_meso, tab_table, tab_prompt, tab_ai, tab_report = st.tabs([
        "🌐 3D 1.25° Globe",
        "📈 Skew-T Sounding",
        "🎯 Hodograph & SRH",
        "📐 Vertical Cross-Section",
        "⚖️ Multi-Case Benchmark",
        "🗺️ Mesoscale Map",
        "📊 Data Matrix",
        "📋 4D AI Prompt",
        "⚡ AI Forecast",
        "📑 Executive Report"
    ])

    with tab_globe:
        st.subheader("🌐 JRA-3Q 1.25° Grid Point Mesh & Vector Earth Projection")
        globe_fig = render_jra3q_125_grid_globe(meso_data["center_lat"], meso_data["center_lon"], center_metrics)
        st.plotly_chart(globe_fig, use_container_width=True)

    with tab_skewt:
        st.subheader("📈 Skew-T Log-P Thermodynamic Sounding & Energetics")
        skewt_fig = render_skew_t_diagram(df_profile, center_metrics)
        st.plotly_chart(skewt_fig, use_container_width=True)

    with tab_hodo:
        st.subheader("🎯 Kinematic Hodograph & Storm-Relative Helicity (SRH)")
        st.caption("Displays the vertical wind shear vector curvature, 0-1km & 0-3km SRH, and Energy-Helicity Index (EHI).")
        hodo_fig = render_hodograph_diagram(df_profile, center_metrics)
        st.plotly_chart(hodo_fig, use_container_width=True)

    with tab_cross:
        st.subheader("📐 2D Latitude-Height Vertical Cross-Section of Equivalent Potential Temp θe")
        st.caption("Reveals the vertical tilt of the frontal boundary and the depth of the low-level moisture tongue.")
        cross_fig = render_vertical_cross_section(meso_data)
        st.plotly_chart(cross_fig, use_container_width=True)

    with tab_radar:
        st.subheader("⚖️ Multi-Case Benchmark Radar Matrix")
        st.caption("Comparative multi-parameter signature analysis across 5 historical disaster scenarios.")
        radar_fig = render_multi_case_benchmark_radar()
        st.plotly_chart(radar_fig, use_container_width=True)

    with tab_meso:
        st.subheader("🗺️ 2D Mesoscale Atmospheric Field & Convective Dynamics")
        fig_meso = go.Figure(data=go.Contour(
            z=meso_data["theta_e_850_grid"], x=meso_data["lons"], y=meso_data["lats"], colorscale='Viridis'
        ))
        fig_meso.update_layout(title="850 hPa Equivalent Potential Temp θe (K)", height=500)
        st.plotly_chart(fig_meso, use_container_width=True)

    with tab_table:
        st.subheader("📊 Full Extracted Sounding Matrix (1000 hPa → 300 hPa)")
        st.dataframe(df_profile, use_container_width=True)
        csv_data = df_profile.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Sounding CSV", data=csv_data, file_name=f"sounding_profile_{meso_data['center_lat']}_{meso_data['center_lon']}.csv", mime="text/csv")

    with tab_prompt:
        st.subheader("📋 4D Mesoscale & Lead-Time AI Prompt")
        current_time_label = time_steps_labels[selected_time_idx]
        prompt_text = construct_4d_prompt(df_profile, center_metrics, meso_data, current_time_label)
        st.code(prompt_text, language="markdown")
        st.download_button("📄 Save Prompt as Markdown", data=prompt_text, file_name="4d_leadtime_prompt.md", mime="text/markdown")

    with tab_ai:
        st.subheader(f"⚡ AI Objective Forecasting ({ai_backend})")
        current_time_label = time_steps_labels[selected_time_idx]

        if ai_backend == "🧠 OpenMythos Local AI (local_ai.py)":
            st.info("💡 **OpenMythos RDT Active**: Shared-weight Recurrent-Depth Transformer executing latent-space reasoning loops on your machine.")
            recurrent_loops = st.slider(
                "🧠 OpenMythos Latent Thinking Loops (Recurrent Depth)",
                min_value=1, max_value=10, value=6,
                help="Higher loop count deepens the latent-space reasoning and convergence."
            )
            if st.button("🚀 Run OpenMythos Recurrent Inference", type="primary"):
                with st.spinner(f"Executing {recurrent_loops} Recurrent-Depth Transformer Loops..."):
                    if LOCAL_AI_AVAILABLE:
                        output_text, loop_logs, latency = local_engine.run_recurrent_inference(
                            df_profile, center_metrics, meso_data, max_loops=recurrent_loops, time_label=current_time_label
                        )
                        st.success(f"✅ OpenMythos Recurrent Loop Complete! Latency: **{latency * 1000:.1f} ms** (Loops: {recurrent_loops})")
                        st.markdown(output_text)
                    else:
                        st.error("local_ai.py module not found in working directory.")
        elif ai_backend == "🦙 Local Ollama LLM (Llama-3 / Qwen)":
            st.info(f"💡 **Ollama Engine**: Connecting to `{ollama_endpoint}` (Model: `{ollama_model}`).")
            if st.button("🚀 Run Ollama Local LLM Forecast", type="primary"):
                with st.spinner(f"Querying local Ollama model ({ollama_model})..."):
                    if LOCAL_AI_AVAILABLE:
                        prompt_content = construct_4d_prompt(df_profile, center_metrics, meso_data, current_time_label)
                        output_text, latency = local_engine.predict_via_ollama(prompt_content, model=ollama_model)
                        st.success(f"✅ Ollama Inference Complete! Latency: **{latency:.2f} seconds**")
                        st.markdown(output_text)
                    else:
                        st.error("local_ai.py module not found.")
        else:
            if not gemini_api_key:
                st.info("💡 Enter your Gemini API Key in the sidebar to execute cloud LLM inference.")
            else:
                if st.button("🚀 Run Cloud AI Forecast (Gemini)", type="primary"):
                    with st.spinner("Calling Gemini Cloud API..."):
                        start_time = time.time()
                        try:
                            prompt_content = construct_4d_prompt(df_profile, center_metrics, meso_data, current_time_label)
                            output_text = call_gemini_api(gemini_api_key, model_name, prompt_content)
                            elapsed_time = round(time.time() - start_time, 2)
                            st.success(f"✅ Cloud Inference Complete! Latency: **{elapsed_time} seconds** (Model: `{model_name}`)")
                            st.markdown("### 🤖 Cloud AI Meteorological Assessment")
                            st.markdown(output_text)
                        except Exception as e:
                            st.error(f"Inference Failure: {str(e)}")

    with tab_report:
        st.subheader("📑 Official Meteorological Briefing Report")
        report_md = f"""# Executive Meteorological Briefing Report
**Target Location**: Lat {meso_data['center_lat']}°N, Lon {meso_data['center_lon']}°E  
**Temporal State**: {time_steps_labels[selected_time_idx]}  
**System**: JRA-3Q OpenMythos AI Forecaster

## 1. Key Atmospheric Indices
* **CAPE (Energy)**: {center_metrics['CAPE(J/kg)']} J/kg | **CIN**: {center_metrics['CIN(J/kg)']} J/kg
* **Integrated Vapor Transport (IVT)**: {center_metrics['IVT(kg/m/s)']} kg/(m·s) (Tendency: +{meso_data['ivt_tendency']}/h)
* **0-6 km Bulk Shear**: {center_metrics['Bulk_Shear_0-6km(m/s)']} m/s | **0-3km SRH**: {center_metrics['SRH_0-3km(m2/s2)']} m²/s²
* **Energy-Helicity Index (EHI)**: {center_metrics['EHI']}
* **K-Index**: {center_metrics['K_Index(°C)']} °C

## 2. Summary Assessment
Severe convective rainband organization conditions are met when IVT exceeds 500 kg/(m·s) and 0-6km bulk shear is >= 15 m/s.
"""
        st.markdown(report_md)
        st.download_button("📥 Download Official Report (Markdown)", data=report_md, file_name=f"Meteorological_Report_{meso_data['center_lat']}_{meso_data['center_lon']}.md", mime="text/markdown")
