import streamlit as st
import pandas as pd
import numpy as np
import pgeocode
import io
from typing import Tuple, Optional

# Constants
EARTH_RADIUS_MILES = 3956
DISTANCE_COL_NAME = "Distance"


class GeoLocator:
    """Handles geolocation lookups using pgeocode."""

    def __init__(self, country: str = "us"):
        self.nomi = pgeocode.Nominatim(country)

    def get_coordinates(self, zip_code: str) -> Tuple[float, float]:
        """Get latitude and longitude for a given zip code."""
        location = self.nomi.query_postal_code(str(zip_code))
        return location.latitude, location.longitude


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance (in miles) between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def get_sidebar_inputs() -> Tuple[str, Optional[io.BytesIO]]:
    """Create sidebar inputs and return values."""
    st.sidebar.header("Options")
    return (
        st.sidebar.text_input("Enter Origin Zip Code", "45241"),
        st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"]),
    )


def read_excel_data(file: io.BytesIO) -> pd.DataFrame:
    """Read and clean Excel data."""
    df = pd.read_excel(file)
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    return df.dropna(how="all")


def select_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Column selection widgets for zip codes and output column."""
    default_zip = "Zip" if "Zip" in df.columns else df.columns[0]
    zip_col = st.sidebar.selectbox(
        "Select the column for zip codes",
        df.columns,
        index=list(df.columns).index(default_zip),
    )

    output_cols = list(df.columns)
    if DISTANCE_COL_NAME not in output_cols:
        output_cols.append(DISTANCE_COL_NAME)

    return (
        zip_col,
        st.sidebar.selectbox(
            "Select the output column for distances",
            output_cols,
            index=output_cols.index(DISTANCE_COL_NAME),
        ),
    )


def calculate_distances(
    df: pd.DataFrame,
    zip_col: str,
    output_col: str,
    origin_lat: float,
    origin_lon: float,
    geo_locator: GeoLocator,
) -> pd.DataFrame:
    """Calculate distances for all zip codes."""

    def get_distance(zip_code: str) -> float:
        clean_zip = str(zip_code).split("-")[0].strip()
        lat, lon = geo_locator.get_coordinates(clean_zip)
        return np.nan if np.isnan(lat) else haversine(origin_lat, origin_lon, lat, lon)

    df[output_col] = df[zip_col].apply(get_distance)
    return df


def display_results(df: pd.DataFrame, output_col: str) -> int:
    """Display results and return count of missing values."""
    nan_count = df[output_col].isna().sum()
    st.write(f"Number of NaN values in '{output_col}' column: {nan_count}")
    st.dataframe(df)
    return nan_count


def create_download_link(df: pd.DataFrame) -> None:
    """Create Excel download link."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        label="Download Results as Excel",
        data=output.getvalue(),
        file_name="output_with_distances.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def show_missing_values(df: pd.DataFrame, output_col: str, zip_col: str) -> None:
    """Display rows with missing distance values."""
    nan_df = df[df[output_col].isna()]
    if not nan_df.empty:
        st.warning("The following rows have missing (NaN) distance values:")
        cols = [zip_col, output_col] + [c for c in df if c not in {zip_col, output_col}]
        st.dataframe(nan_df[cols])


# Streamlit UI Configuration
st.set_page_config(layout="wide")
st.title("Zip Code Distance Calculator")
st.write(
    """
    Upload an Excel file containing zip codes. The app will calculate distances
    from each zip code to the origin zip code you specify. Results can be downloaded
    as an Excel file.
"""
)


def main():
    """Main application logic."""
    origin_zip, uploaded_file = get_sidebar_inputs()
    geo_locator = GeoLocator()

    if not uploaded_file:
        return

    try:
        df = read_excel_data(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return

    zip_col, output_col = select_columns(df)

    try:
        origin_lat, origin_lon = geo_locator.get_coordinates(origin_zip)
        if np.isnan(origin_lat):
            raise ValueError("Invalid origin zip code")
    except ValueError as e:
        st.error(str(e))
        return

    df = calculate_distances(
        df, zip_col, output_col, origin_lat, origin_lon, geo_locator
    )
    nan_count = display_results(df, output_col)
    create_download_link(df)

    if nan_count > 0:
        show_missing_values(df, output_col, zip_col)


if __name__ == "__main__":
    main()
