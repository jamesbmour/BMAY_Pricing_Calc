import streamlit as st
import pandas as pd
import numpy as np
import pgeocode
import io

st.title("Zip Code Distance Calculator")
st.write(
    """
    Upload an Excel file that contains a **Zip** column.
    The app will calculate the distance from each zip code (using only the first 5 digits if needed)
    to the origin zip code **45241**, then display the results and provide a download link.
    """
)

# File uploader widget
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

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
    # Ensure zip code is string and split on '-' if present
    zip_str = str(zip_code)
    zip_clean = zip_str.split('-')[0].strip()  # use first 5 digits

    # Query location info for the cleaned zip code
    info = nomi.query_postal_code(zip_clean)
    # If location is not found, return NaN
    if np.isnan(info.latitude) or np.isnan(info.longitude):
        return np.nan

    return haversine(origin_lat, origin_lon, info.latitude, info.longitude)

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
    else:
        # Initialize pgeocode for US zip codes
        nomi = pgeocode.Nominatim('us')

        # Get latitude and longitude for the origin zip code (45241)
        origin_info = nomi.query_postal_code("45241")
        origin_lat = origin_info.latitude
        origin_lon = origin_info.longitude

        # Check if 'Zip' column exists
        if 'Zip' not in df.columns:
            st.error("The uploaded Excel file does not contain a 'Zip' column.")
        else:
            # Calculate the distance for each zip code in the DataFrame
            df['Distance'] = df['Zip'].apply(lambda z: get_distance(z, origin_lat, origin_lon, nomi))

            # Count the number of NaN values in the 'Distance' column
            nan_count = df['Distance'].isna().sum()

            st.write(f"Number of NaN values in 'Distance' column: {nan_count}")
            st.dataframe(df)

            # Prepare the DataFrame for download as an Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
            processed_data = output.getvalue()

            st.download_button(
                label="Download Results as Excel",
                data=processed_data,
                file_name="output_with_distances.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
