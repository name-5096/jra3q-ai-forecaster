import streamlit as st
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import random
import string

# Initialize Page Configuration
st.set_page_config(page_title="JRA-3Q AI Forecaster Data Converter", layout="wide")
st.title("⚡ JRA-3Q Vertical Profile Matrix Converter (v1.1.0)")

# Session State Cache-Busting Mechanism for NumPy 2.5 Safety
if "nc_filename" not in st.session_state:
    rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    st.session_state.nc_filename = f"tmp_{rand_str}.nc"

filename = st.session_state.nc_filename

# Sub-Routine: Generate Clean NetCDF4 Mock Data (Prevents Admin/Local File Errors)
def create_dummy_nc(target_file):
    if not os.path.exists(target_file):
        rootgrp = nc.Dataset(target_file, "w", format="NETCDF4")
        rootgrp.createDimension("time", None)
        rootgrp.createDimension("level", 5)
        rootgrp.createDimension("lat", 8)
        rootgrp.createDimension("lon", 8)
        
        times = rootgrp.createVariable("time","f8",("time",))
        levels = rootgrp.createVariable("level","f4",("level",))
        lats = rootgrp.createVariable("lat","f4",("lat",))
        lons = rootgrp.createVariable("lon","f4",("lon",))
        
        # Explicit Array Assignment (100% Fixed Syntax)
        levels[:] = np.array([1000, 850, 700, 500, 300], dtype=np.float32)
        lats[:] = np.arange(30.0, 40.0, 1.25, dtype=np.float32)
        lons[:] = np.arange(130.0, 140.0, 1.25, dtype=np.float32)
        
        tmp = rootgrp.createVariable("tmp","f4",("time","level","lat","lon"))
        spfh = rootgrp.createVariable("spfh","f4",("time","level","lat","lon"))
        ugrd = rootgrp.createVariable("ugrd","f4",("time","level","lat","lon"))
        vgrd = rootgrp.createVariable("vgrd","f4",("time","level","lat","lon"))
        
        # NumPy 2.5 Broadcast Compliance Layer via Type-Safe np.full
        tmp[0, :, :, :] = np.full((5, 8, 8), 280.0, dtype=np.float32)
        spfh[0, :, :, :] = np.full((5, 8, 8), 0.012, dtype=np.float32)
        ugrd[0, :, :, :] = np.full((5, 8, 8), 5.0, dtype=np.float32)
        vgrd[0, :, :, :] = np.full((5, 8, 8), -2.0, dtype=np.float32)
        rootgrp.close()

create_dummy_nc(filename)

# Sidebar UI: Mode Configuration
st.sidebar.header("🕹️ Extraction Control")
mode = st.sidebar.radio("Select Processing Mode:", ("Pinpoint (Single Grid)", "Area (Spatial Bounding Box)"))

# Core Data Ingestion Layer
try:
    dataset = nc.Dataset(filename, "r")
    available_lats = dataset.variables['lat'][:]
    available_lons = dataset.variables['lon'][:]
    available_levels = dataset.variables['level'][:]
except Exception as e:
    st.error(f"Missing or corrupted data file: {e}")
    st.stop()

# Execution Constants
latest_time_idx = 0
extracted_records = []

# Scope Isolation Block: Mode-Specific Logic Routing
if mode == "Pinpoint (Single Grid)":
    st.sidebar.header("📍 Coordinate Target Settings")
    target_lat = st.sidebar.number_input("Target Latitude (Lat)", value=32.5, min_value=float(available_lats.min()), max_value=float(available_lats.max()), step=1.25)
    target_lon = st.sidebar.number_input("Target Longitude (Lon)", value=130.0, min_value=float(available_lons.min()), max_value=float(available_lons.max()), step=1.25)

    p_lat_idx = int((np.abs(available_lats - target_lat)).argmin())
    p_lon_idx = int((np.abs(available_lons - target_lon)).argmin())
    actual_lat = available_lats[p_lat_idx]
    actual_lon = available_lons[p_lon_idx]
    
    st.subheader(f"📊 Extraction Output: Single Grid Coordinate (Lat: {actual_lat}°N, Lon: {actual_lon}°E)")
    
    for lvl_idx, lvl in enumerate(available_levels):
        extracted_records.append({
            "Level(hPa)": int(lvl),
            "Temperature(K)": round(float(dataset.variables['tmp'][latest_time_idx, lvl_idx, p_lat_idx, p_lon_idx]), 3),
            "Specific_Humidity(g/kg)": round(float(dataset.variables['spfh'][latest_time_idx, lvl_idx, p_lat_idx, p_lon_idx]) * 1000, 3),
            "U_Wind(m/s)": round(float(dataset.variables['ugrd'][latest_time_idx, lvl_idx, p_lat_idx, p_lon_idx]), 3),
            "V_Wind(m/s)": round(float(dataset.variables['vgrd'][latest_time_idx, lvl_idx, p_lat_idx, p_lon_idx]), 3),
        })

else:
    st.sidebar.subheader("📐 Spatial Bounding Box Parameters")
    lat_min = st.sidebar.number_input("Minimum Latitude (Min Lat)", value=32.5, min_value=float(available_lats.min()), max_value=float(available_lats.max()), step=1.25)
    lat_max = st.sidebar.number_input("Maximum Latitude (Max Lat)", value=35.0, min_value=float(available_lats.min()), max_value=float(available_lats.max()), step=1.25)
    st.sidebar.markdown("---")
    lon_min = st.sidebar.number_input("Minimum Longitude (Min Lon)", value=130.0, min_value=float(available_lons.min()), max_value=float(available_lons.max()), step=1.25)
    lon_max = st.sidebar.number_input("Maximum Longitude (Max Lon)", value=132.5, min_value=float(available_lons.min()), max_value=float(available_lons.max()), step=1.25)

    st.subheader(f"🗺️ Extraction Output: Area Matrix Block (Lat: {lat_min}°N to {lat_max}°N / Lon: {lon_min}°E to {lon_max}°E)")
    
    for lat_val in np.arange(lat_min, lat_max + 0.1, 1.25):
        for lon_val in np.arange(lon_min, lon_max + 0.1, 1.25):
            
            l_idx = int((np.abs(available_lats - lat_val)).argmin())
            o_idx = int((np.abs(available_lons - lon_val)).argmin())
            
            current_lat = available_lats[l_idx]
            current_lon = available_lons[o_idx]
            
            for lvl_idx, lvl in enumerate(available_levels):
                extracted_records.append({
                    "Grid_Lat": round(float(current_lat), 2),
                    "Grid_Lon": round(float(current_lon), 2),
                    "Level(hPa)": int(lvl),
                    "Temperature(K)": round(float(dataset.variables['tmp'][latest_time_idx, lvl_idx, l_idx, o_idx]), 3),
                    "Specific_Humidity(g/kg)": round(float(dataset.variables['spfh'][latest_time_idx, lvl_idx, l_idx, o_idx]) * 1000, 3),
                    "U_Wind(m/s)": round(float(dataset.variables['ugrd'][latest_time_idx, lvl_idx, l_idx, o_idx]), 3),
                    "V_Wind(m/s)": round(float(dataset.variables['vgrd'][latest_time_idx, lvl_idx, l_idx, o_idx]), 3),
                })

# Data Pipeline Integration & Rendering
df_output = pd.DataFrame(extracted_records)

if not df_output.empty:
    st.dataframe(df_output, use_container_width=True)
    st.markdown("### 📋 AI Prompt Ingestion Matrix (Bias-Free Markdown Source)")
    st.caption("Copy the entire raw markdown string below and feed it directly into your LLM context window.")
    st.code(df_output.to_markdown(index=False), language="markdown")
else:
    st.warning("No matching spatial coordinate matrices were found within the specified boundary. Re-verify coordinate inputs.")

dataset.close()
