import os
import tempfile

import netCDF4 as nc
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# JRA-3Q AI Forecaster Data Converter
# Release v2.1.1
# ============================================================

APP_VERSION = "v2.1.1"
APP_RELEASE = "Release"


# 1. Initialize Page Configuration
st.set_page_config(
    page_title="JRA-3Q AI Forecaster Data Converter",
    layout="wide",
)

st.title(
    f"⚡ JRA-3Q Multi-Variable Profile Matrix Converter ({APP_VERSION})"
)
st.caption(f"{APP_RELEASE} {APP_VERSION}")


# ============================================================
# 2. Sidebar UI: Configuration & Multi-File Upload
# ============================================================

st.sidebar.header("📁 Data Source Ingestion")

uploaded_files = st.sidebar.file_uploader(
    "Upload JRA-3Q NetCDF Files (.nc)",
    type=["nc", "nc4", "cdf"],
    accept_multiple_files=True,
)

st.sidebar.markdown("---")
st.sidebar.header("🕹️ Extraction Control")

mode = st.sidebar.radio(
    "Select Processing Mode:",
    (
        "Pinpoint (Single Grid)",
        "Area (Spatial Bounding Box)",
    ),
)


# ============================================================
# Helper Functions
# ============================================================

def detect_variable_type(filename):
    """
    Detect the weather parameter type from JRA-3Q filename patterns.

    Returns
    -------
    tuple
        (variable_key, display_label)
    """

    fn_lower = filename.lower()

    if "tmp" in fn_lower or "temperature" in fn_lower:
        return "tmp", "Temperature(K)"

    if (
        "ugrd" in fn_lower
        or "u-wind" in fn_lower
        or "u_wind" in fn_lower
    ):
        return "ugrd", "U_Wind(m/s)"

    if (
        "vgrd" in fn_lower
        or "v-wind" in fn_lower
        or "v_wind" in fn_lower
    ):
        return "vgrd", "V_Wind(m/s)"

    if "spfh" in fn_lower or "humidity" in fn_lower:
        return "spfh", "Specific_Humidity(g/kg)"

    return None, None


def find_level_variable_name_flexible(dataset):
    """
    Dynamically locate the vertical pressure coordinate variable.
    """

    all_vars = list(dataset.variables.keys())

    target_keywords = [
        "level",
        "lev",
        "pres",
        "isobaric",
        "hpa",
        "plev",
        "pressure_level",
    ]

    # First pass:
    # Search common pressure-level coordinate names.
    for var in all_vars:
        var_lower = var.lower()

        if any(keyword in var_lower for keyword in target_keywords):
            if len(dataset.variables[var].dimensions) == 1:
                return var

    # Second pass:
    # Search other one-dimensional coordinates while excluding
    # standard time/latitude/longitude coordinates.
    excluded_names = {
        "lat",
        "latitude",
        "lon",
        "longitude",
        "time",
        "t",
    }

    for var in all_vars:
        if len(dataset.variables[var].dimensions) == 1:
            if var.lower() not in excluded_names:
                return var

    return None


def find_coordinate_variable(dataset, candidates):
    """
    Find a coordinate variable from multiple possible names.
    """

    variables = dataset.variables

    for candidate in candidates:
        if candidate in variables:
            return candidate

    lower_map = {
        variable.lower(): variable
        for variable in variables.keys()
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def safe_float(value):
    """
    Safely convert NetCDF / masked-array values to float.

    Missing values are returned as NaN.
    """

    if np.ma.is_masked(value):
        return np.nan

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(numeric_value):
        return np.nan

    return numeric_value


# ============================================================
# 3. Structural Containers
# ============================================================

loaded_matrices = {}

available_lats = None
available_lons = None
available_levels = None

reference_lat_name = None
reference_lon_name = None
reference_level_name = None

time_labels = []
time_idx = 0


# ============================================================
# 4. Secure Timeline Loader
# ============================================================

if uploaded_files:
    tmp_t_path = None

    try:
        ref_file = uploaded_files[0]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".nc",
        ) as tmp_t:
            tmp_t.write(ref_file.getvalue())
            tmp_t_path = tmp_t.name

        with nc.Dataset(tmp_t_path, "r") as ds_time:
            if "time" in ds_time.variables:
                t_var_ref = ds_time.variables["time"]

                if hasattr(t_var_ref, "units"):
                    time_dates = nc.num2date(
                        t_var_ref[:],
                        units=t_var_ref.units,
                        calendar=getattr(
                            t_var_ref,
                            "calendar",
                            "standard",
                        ),
                    )

                    time_labels = [
                        date.strftime("%Y-%m-%d %H:%M UTC")
                        for date in time_dates
                    ]

        if time_labels:
            st.sidebar.markdown("---")
            st.sidebar.header("📅 Temporal Control")

            selected_time = st.sidebar.selectbox(
                "Select Target Date/Time:",
                time_labels,
            )

            time_idx = time_labels.index(selected_time)

    except Exception as e:
        st.sidebar.error(
            f"Timeline parsing failed: {e}"
        )

    finally:
        if tmp_t_path and os.path.exists(tmp_t_path):
            os.remove(tmp_t_path)


# ============================================================
# 5. Dynamic Multi-Data Loading & Ingestion Layer
# ============================================================

if uploaded_files:
    detected_labels = []

    for up_file in uploaded_files:
        var_key, var_label = detect_variable_type(up_file.name)

        if var_key is None:
            st.sidebar.warning(
                f"Skipped unknown file: '{up_file.name}'"
            )
            continue

        tmp_file_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".nc",
            ) as tmp_file:
                tmp_file.write(up_file.getvalue())
                tmp_file_path = tmp_file.name

            with nc.Dataset(tmp_file_path, "r") as dataset:

                # ------------------------------------------------
                # Detect coordinate variables
                # ------------------------------------------------

                lat_name = find_coordinate_variable(
                    dataset,
                    ["lat", "latitude"],
                )

                lon_name = find_coordinate_variable(
                    dataset,
                    ["lon", "longitude"],
                )

                level_var_name = (
                    find_level_variable_name_flexible(dataset)
                )

                if lat_name is None:
                    raise ValueError(
                        "Latitude coordinate could not be detected."
                    )

                if lon_name is None:
                    raise ValueError(
                        "Longitude coordinate could not be detected."
                    )

                if level_var_name is None:
                    raise ValueError(
                        "Vertical pressure axis could not be detected."
                    )

                # ------------------------------------------------
                # Initialize reference grid from first valid file
                # ------------------------------------------------

                if available_lats is None:
                    reference_lat_name = lat_name
                    reference_lon_name = lon_name
                    reference_level_name = level_var_name

                    available_lats = np.asarray(
                        dataset.variables[lat_name][:]
                    )

                    available_lons = np.asarray(
                        dataset.variables[lon_name][:]
                    )

                    available_levels = np.asarray(
                        dataset.variables[level_var_name][:]
                    )

                # ------------------------------------------------
                # Find target meteorological variable
                # ------------------------------------------------

                actual_key_in_file = None

                for key in dataset.variables.keys():
                    if var_key in key.lower():
                        actual_key_in_file = key
                        break

                # ------------------------------------------------
                # Conservative fallback
                # ------------------------------------------------

                if actual_key_in_file is None:
                    possible_variables = []

                    excluded_variables = {
                        "time",
                        "level",
                        "lev",
                        "lat",
                        "latitude",
                        "lon",
                        "longitude",
                        "pressure_level",
                    }

                    for key in dataset.variables.keys():
                        variable = dataset.variables[key]

                        if (
                            len(variable.dimensions) == 4
                            and key.lower()
                            not in excluded_variables
                        ):
                            possible_variables.append(key)

                    # Only fallback when there is exactly one
                    # unambiguous four-dimensional data variable.
                    if len(possible_variables) == 1:
                        actual_key_in_file = possible_variables[0]

                if actual_key_in_file is None:
                    st.sidebar.warning(
                        "Could not locate parameter variable "
                        f"'{var_key}' in '{up_file.name}'."
                    )
                    continue

                data_variable = dataset.variables[
                    actual_key_in_file
                ]

                # ------------------------------------------------
                # Supported input:
                # (time, level, lat, lon)
                # ------------------------------------------------

                if len(data_variable.dimensions) == 4:
                    if time_idx >= data_variable.shape[0]:
                        raise IndexError(
                            f"Selected time index {time_idx} "
                            f"is unavailable in '{up_file.name}'."
                        )

                    matrix_data = np.ma.asarray(
                        data_variable[
                            time_idx,
                            :,
                            :,
                            :,
                        ]
                    )

                # Optional support for already time-sliced
                # (level, lat, lon) data.
                elif len(data_variable.dimensions) == 3:
                    matrix_data = np.ma.asarray(
                        data_variable[:, :, :]
                    )

                else:
                    raise ValueError(
                        f"Variable '{actual_key_in_file}' "
                        "must be three- or four-dimensional. "
                        f"Detected dimensions: "
                        f"{data_variable.dimensions}"
                    )

                loaded_matrices[var_key] = (
                    var_label,
                    matrix_data,
                )

                detected_labels.append(var_label)

        except Exception as e:
            st.error(
                f"Failed to parse file '{up_file.name}': {e}"
            )
            st.stop()

        finally:
            if (
                tmp_file_path
                and os.path.exists(tmp_file_path)
            ):
                os.remove(tmp_file_path)

    if loaded_matrices:
        st.sidebar.success(
            "Merged Variables: "
            + ", ".join(detected_labels)
        )

    else:
        st.error(
            "No valid JRA-3Q parameter variables could "
            "be mapped from the uploaded files."
        )
        st.stop()

else:
    st.info(
        "💡 Please upload one or more JRA-3Q NetCDF (.nc) "
        "files via the sidebar to begin processing."
    )
    st.stop()


# ============================================================
# 6. Extraction
# ============================================================

extracted_records = []


# ============================================================
# 6-A. Pinpoint Mode
# ============================================================

if mode == "Pinpoint (Single Grid)":

    st.sidebar.header(
        "📍 Coordinate Target Settings"
    )

    target_lat = st.sidebar.number_input(
        "Target Latitude (Lat)",
        value=float(np.mean(available_lats)),
        min_value=float(np.min(available_lats)),
        max_value=float(np.max(available_lats)),
        step=1.25,
    )

    target_lon = st.sidebar.number_input(
        "Target Longitude (Lon)",
        value=float(np.mean(available_lons)),
        min_value=float(np.min(available_lons)),
        max_value=float(np.max(available_lons)),
        step=1.25,
    )

    p_lat_idx = int(
        np.abs(
            available_lats - target_lat
        ).argmin()
    )

    p_lon_idx = int(
        np.abs(
            available_lons - target_lon
        ).argmin()
    )

    actual_lat = float(
        available_lats[p_lat_idx]
    )

    actual_lon = float(
        available_lons[p_lon_idx]
    )

    st.subheader(
        "📊 Extraction Output: Single Grid Coordinate "
        f"(Lat: {actual_lat}°N, "
        f"Lon: {actual_lon}°E)"
    )

    for lvl_idx, lvl in enumerate(
        available_levels
    ):
        row_data = {
            "Level(hPa)": int(lvl),
        }

        for (
            v_key,
            (v_label, matrix),
        ) in loaded_matrices.items():

            raw_val = safe_float(
                matrix[
                    lvl_idx,
                    p_lat_idx,
                    p_lon_idx,
                ]
            )

            if np.isnan(raw_val):
                display_val = np.nan

            elif v_key == "spfh":
                # JRA-3Q specific humidity:
                # kg/kg -> g/kg
                display_val = round(
                    raw_val * 1000.0,
                    3,
                )

            else:
                display_val = round(
                    raw_val,
                    3,
                )

            row_data[v_label] = display_val

        extracted_records.append(
            row_data
        )


# ============================================================
# 6-B. Area Mode
# ============================================================

else:

    st.sidebar.subheader(
        "📐 Spatial Bounding Box Parameters"
    )

    lat_min = st.sidebar.number_input(
        "Minimum Latitude (Min Lat)",
        value=float(np.min(available_lats)),
        min_value=float(np.min(available_lats)),
        max_value=float(np.max(available_lats)),
        step=1.25,
    )

    lat_max = st.sidebar.number_input(
        "Maximum Latitude (Max Lat)",
        value=float(np.max(available_lats)),
        min_value=float(np.min(available_lats)),
        max_value=float(np.max(available_lats)),
        step=1.25,
    )

    st.sidebar.markdown("---")

    lon_min = st.sidebar.number_input(
        "Minimum Longitude (Min Lon)",
        value=float(np.min(available_lons)),
        min_value=float(np.min(available_lons)),
        max_value=float(np.max(available_lons)),
        step=1.25,
    )

    lon_max = st.sidebar.number_input(
        "Maximum Longitude (Max Lon)",
        value=float(np.max(available_lons)),
        min_value=float(np.min(available_lons)),
        max_value=float(np.max(available_lons)),
        step=1.25,
    )

    # Prevent reversed bounding boxes.
    if lat_min > lat_max:
        st.error(
            "Minimum Latitude must be less than "
            "or equal to Maximum Latitude."
        )
        st.stop()

    if lon_min > lon_max:
        st.error(
            "Minimum Longitude must be less than "
            "or equal to Maximum Longitude."
        )
        st.stop()

    st.subheader(
        "🗺️ Extraction Output: Area Matrix Block "
        f"(Lat: {lat_min}°N to {lat_max}°N / "
        f"Lon: {lon_min}°E to {lon_max}°E)"
    )

    lat_indices = np.where(
        (available_lats >= lat_min)
        & (available_lats <= lat_max)
    )[0]

    lon_indices = np.where(
        (available_lons >= lon_min)
        & (available_lons <= lon_max)
    )[0]

    if (
        len(lat_indices) == 0
        or len(lon_indices) == 0
    ):
        st.warning(
            "No grid points exist inside the "
            "selected bounding box."
        )
        st.stop()

    for l_idx in lat_indices:

        for o_idx in lon_indices:

            current_lat = float(
                available_lats[l_idx]
            )

            current_lon = float(
                available_lons[o_idx]
            )

            for lvl_idx, lvl in enumerate(
                available_levels
            ):

                row_data = {
                    "Grid_Lat": round(
                        current_lat,
                        2,
                    ),
                    "Grid_Lon": round(
                        current_lon,
                        2,
                    ),
                    "Level(hPa)": int(lvl),
                }

                for (
                    v_key,
                    (v_label, matrix),
                ) in loaded_matrices.items():

                    raw_val = safe_float(
                        matrix[
                            lvl_idx,
                            l_idx,
                            o_idx,
                        ]
                    )

                    if np.isnan(raw_val):
                        display_val = np.nan

                    elif v_key == "spfh":
                        display_val = round(
                            raw_val * 1000.0,
                            3,
                        )

                    else:
                        display_val = round(
                            raw_val,
                            3,
                        )

                    row_data[
                        v_label
                    ] = display_val

                extracted_records.append(
                    row_data
                )


# ============================================================
# 7. UI Presentation & Markdown Output
# ============================================================

if extracted_records:

    df_output = pd.DataFrame(
        extracted_records
    )

    st.write(
        f"Total Rows Extracted: "
        f"`{len(df_output)}`"
    )

    st.dataframe(
        df_output,
        width="stretch",
        hide_index=True,
    )

    st.subheader(
        "📋 AI Prompt Ingestion Matrix "
        "(Bias-Free Markdown Source)"
    )

    st.write(
        "Copy the entire raw markdown string below "
        "and feed it directly into your "
        "LLM context window."
    )

    markdown_payload = (
        df_output.to_markdown(
            index=False
        )
    )

    st.code(
        markdown_payload,
        language="markdown",
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption(
    f"JRA-3Q AI Forecaster Data Converter — "
    f"{APP_RELEASE} {APP_VERSION}"
)