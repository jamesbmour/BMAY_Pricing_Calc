import streamlit as st
import pandas as pd
import numpy as np
import pgeocode
import io

st.set_page_config(layout="wide")
st.title("Zip Code Distance Calculator")
st.write(
    """
    Upload an Excel file that contains your data.
    The app will calculate the distance from each zip code (using only the first 5 digits if needed)
    to the origin zip code you specify, then display the results and provide a download link.
    """
)

# Sidebar options
st.sidebar.header("Options")
origin_zip = st.sidebar.text_input("Enter Origin Zip Code", "45241")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance (in miles) between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956  # Earth's radius in miles
    return c * r


def get_distance(zip_code, origin_lat, origin_lon, nomi):
    """Extract the first 5 digits of the zip code, look up its location, and calculate the distance."""
    zip_str = str(zip_code)
    zip_clean = zip_str.split("-")[0].strip()  # Use first 5 digits
    info = nomi.query_postal_code(zip_clean)
    if np.isnan(info.latitude) or np.isnan(info.longitude):
        return np.nan
    return haversine(origin_lat, origin_lon, info.latitude, info.longitude)


if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        # Replace blank strings (including spaces) with NaN and then drop rows that are completely empty
        df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
        df.dropna(how="all", inplace=True)
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
    else:
        # Allow selection of the column that contains the zip codes from available columns
        default_zip_col = "Zip" if "Zip" in df.columns else df.columns[0]
        zip_input_col = st.sidebar.selectbox(
            "Select the column for zip codes",
            options=df.columns,
            index=list(df.columns).index(default_zip_col),
        )

        # Allow selection for the output column for distances; include existing columns and add "Distance" if needed.
        output_options = list(df.columns)
        if "Distance" not in output_options:
            output_options.append("Distance")
        default_distance_index = (
            output_options.index("Distance") if "Distance" in output_options else 0
        )
        distance_output_col = st.sidebar.selectbox(
            "Select the output column for distances",
            options=output_options,
            index=default_distance_index,
        )

        # Initialize pgeocode for US zip codes
        nomi = pgeocode.Nominatim("us")
        origin_info = nomi.query_postal_code(origin_zip)
        origin_lat = origin_info.latitude
        origin_lon = origin_info.longitude

        if np.isnan(origin_lat) or np.isnan(origin_lon):
            st.error(
                "Invalid origin zip code. Please enter a valid US zip code in the sidebar."
            )
        else:
            # Calculate the distance for each zip code using the selected column
            df[distance_output_col] = df[zip_input_col].apply(
                lambda z: get_distance(z, origin_lat, origin_lon, nomi)
            )

            # Count and display the number of NaN values in the output distance column
            nan_count = df[distance_output_col].isna().sum()
            st.write(
                f"Number of NaN values in '{distance_output_col}' column: {nan_count}"
            )
            st.dataframe(df)
            # Prepare the DataFrame for download as an Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Results")
            processed_data = output.getvalue()

            st.download_button(
                label="Download Results as Excel",
                data=processed_data,
                file_name="output_with_distances.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            # Display a warning section for rows with missing (NaN) distance values,
            # with the zip code and distance columns moved to the front.
            if nan_count > 0:
                st.warning("The following rows have missing (NaN) distance values:")
                nan_df = df[df[distance_output_col].isna()]
                cols_order = [zip_input_col, distance_output_col] + [
                    col
                    for col in nan_df.columns
                    if col not in [zip_input_col, distance_output_col]
                ]
                st.dataframe(nan_df[cols_order])


# thisFile
