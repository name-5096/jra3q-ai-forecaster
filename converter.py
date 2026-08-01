import streamlit as st
import netCDF4 as nc
import numpy as np
import pandas as pd
import os

# ==========================================
# Web App UI Configuration (Streamlit)
# ==========================================
st.set_page_config(page_title="JRA-3Q AI Forecaster Payload Generator", layout="wide")
st.title("JRA-3Q AI Forecaster: 3D Vertical Profile Extractor")
st.write("Extract bias-free atmospheric matrix data for objective LLM inference from JRA-3Q netCDF4 files.")

# --- File Layout Setup in Sidebar ---
st.sidebar.header("1. Dataset Configuration")
st.sidebar.write("Ensure the following 4 netCDF4 files are placed in the application directory:")

FILES = {
    "Temperature": "tmp.nc",
    "Specific Humidity": "spfh.nc",
    "Zonal Wind": "ugrd.nc",
    "Meridional Wind": "vgrd.nc"
}
OUTPUT_CSV_FILE = "gemini_data.csv"

# Real-time check for netCDF4 files existence
missing_files = [path for name, path in FILES.items() if not os.path.exists(path)]

if missing_files:
    st.sidebar.error(f"Missing files: {', '.join(missing_files)}")
    st.error("Please place the required JRA-3Q files (tmp.nc, spfh.nc, ugrd.nc, vgrd.nc) in the same directory.")
    st.stop()
else:
    st.sidebar.success("All 4 target netCDF4 datasets detected successfully.")

# --- Coordinate Input ---
st.sidebar.header("2. Target Coordinates")
# Default coordinates preset to Kumamoto area (32.5°N, 130.0°E) for the case study
target_lat = st.sidebar.number_input("Target Latitude (degrees_north)", min_value=-90.0, max_value=90.0, value=32.5, step=1.25)
target_lon = st.sidebar.number_input("Target Longitude (degrees_east)", min_value=0.0, max_value=360.0, value=130.0, step=1.25)


# ==========================================
# 3. Data Processing Core (Vertical Extractor)
# ==========================================
@st.cache_data
def extract_vertical_profile(lat, lon):
    try:
        # Open files securely using the exact original dictionary structure
        datasets = {key: nc.Dataset(path, mode="r") for key, path in FILES.items()}

        # Extract dimension coordinate arrays
        lats = datasets["Temperature"].variables["lat"][:]
        lons = datasets["Temperature"].variables["lon"][:]
        levels = datasets["Temperature"].variables["pressure_level"][:]

        # Nearest neighbor indexing logic (0-360 deg support)
        lat_idx = np.abs(lats - lat).argmin()
        lon_idx = np.abs(lons - lon).argmin()

        # Exact JRA-3Q specific isobaric variable keys from the original script
        t_var = "tmp-pres-an-ll125"
        q_var = "spfh-pres-an-ll125"
        u_var = "ugrd-pres-an-ll125"
        v_var = "vgrd-pres-an-ll125"

        latest_time_idx = -1
        records = []

        # Slice 4D array into a clean 1D vertical profile (1000hPa down to 300hPa)
        for k, level in enumerate(levels):
            if 300 <= level <= 1000:
                t_val = datasets["Temperature"].variables[t_var][latest_time_idx, k, lat_idx, lon_idx]
                q_val = datasets["Specific Humidity"].variables[q_var][latest_time_idx, k, lat_idx, lon_idx]
                u_val = datasets["Zonal Wind"].variables[u_var][latest_time_idx, k, lat_idx, lon_idx]
                v_val = datasets["Meridional Wind"].variables[v_var][latest_time_idx, k, lat_idx, lon_idx]
                
                records.append({
                    "Pressure_Level(hPa)": float(level),
                    "Temperature(K)": round(float(t_val), 2),
                    "U-Wind(m/s)": round(float(u_val), 2),
                    "V-Wind(m/s)": round(float(v_val), 2),
                    "Specific_Humidity(g/kg)": round(float(q_val) * 1000, 3) # Converted to g/kg unit
                })

        # Close all dataset hooks safely
        for ds in datasets.values():
            ds.close()

        # Build DataFrame and sort descending (from surface 1000hPa going up to 300hPa)
        df = pd.DataFrame(records)
        df_output = df.sort_values(by="Pressure_Level(hPa)", ascending=False)
        
        return df_output, lats[lat_idx], lons[lon_idx]

    except Exception as e:
        st.error(f"Processing Matrix Error: {str(e)}")
        return None, None, None


# ==========================================
# 4. Render App View & Output Elements
# ==========================================
# Execute extraction matrix pipeline
df_profile, actual_lat, actual_lon = extract_vertical_profile(target_lat, target_lon)

if df_profile is not None:
    st.subheader("Extracted Bias-Free Vertical Grid Matrix")
    st.write(f"Nearest resolved grid point coordinates: Lat {actual_lat} N, Lon {actual_lon} E")
    
    # Interactive UI Data Table
    st.dataframe(df_profile, use_container_width=True)

    # Automatically save a background copy as CSV (inheriting original script's logic)
    df_profile.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8")

    # --- LLM Plain-Text Clipboard Section ---
    st.subheader("Copy Payload for Bias-Free LLM Input")
    st.write("This matrix represents the vertical atmospheric profile (1000hPa to 300hPa) exactly above the observed point. Copy and paste it straight into your LLM prompt.")
    
    # Generate clean markdown layout for copy-pasting
    markdown_payload = df_profile.to_markdown(index=False)
    st.code(markdown_payload, language="markdown")
    st.success(f"Background update complete: {OUTPUT_CSV_FILE} successfully dumped to directory root.")
