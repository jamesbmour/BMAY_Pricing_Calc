import streamlit as st
import pandas as pd
import numpy as np
import pgeocode
import io
from typing import Tuple, Optional

# Constants
EARTH_RADIUS_MILES = 3956
DISTANCE_COL_NAME = "Distance"
ZONE_COL_NAME = "Zone"

# Define the rates for each zone with numeric keys
ZONE_RATES = {
    1: 249.04,  # Zone 1
    2: 298.70,  # Zone 2
    3: 379.34,  # Zone 3
    4: 439.51,  # Zone 4
    5: 497.19,  # Zone 5
    6: 583.61,  # Zone 6
    7: 672.34,  # Zone 7
    8: 697.09,  # Zone 8
    9: 810.00,  # Canada
    10: 750.00,  # Hawaii
}

# Mapping between zone numbers and descriptions
ZONE_MAPPING = {
    1: "Zone 1 (0-100 miles)",
    2: "Zone 2 (101-200 miles)",
    3: "Zone 3 (201-400 miles)",
    4: "Zone 4 (401-600 miles)",
    5: "Zone 5 (601-1000 miles)",
    6: "Zone 6 (1001-1400 miles)",
    7: "Zone 7 (1401-1800 miles)",
    8: "Zone 8 (1801-2500 miles)",
    9: "Canada",
    10: "Hawaii (Oahu island only)",
}

# Define Install Types that should have zone costs applied
INSTALL_TYPES_WITH_ZONE_COSTS = [
    "CoreNew Install",
    "HubPremier New Install",
    "DobbyLockersPremier New Install",
    "CoreReinstall",
    "HubReinstall",
]

# Define base costs for other install types
BASE_COSTS = {
    "CoreDownsize": 150.00,
    "CorePermanent Removal": 100.00,
    "CoreSwap": 200.00,
    "CoreTemporary Removal": 75.00,
    "HubSwap": 200.00,
    "CoreSvc Call": 125.00,
    "CoreUpsize": 150.00,
    "CorePermanent Relocation": 200.00,
    "HubUpsize": 150.00,
    "HubPermanent Removal": 100.00,
    "DobbyLockersTemporary Removal": 75.00,
    "HubDownsize": 150.00,
    "HubPermanent Relocation": 200.00,
    "HubMaintenance": 100.00,
    "HubTemporary Removal": 75.00,
    "DobbyLockersUpsize": 150.00,
    "HubSvc Call": 125.00,
    " ": 0.00,
}


def calculate_cost(zone: int, install_type: str) -> float:
    """
    Calculate the cost based on the zone and install type.

    Args:
        zone: numeric zone value
        install_type: type of installation

    Returns:
        float: calculated cost
    """
    if pd.isna(zone):
        return 0.0

    # If install type is in the list that should have zone costs
    if install_type in INSTALL_TYPES_WITH_ZONE_COSTS:
        return ZONE_RATES.get(zone, 0.0)

    # Otherwise, return the base cost for the install type
    return BASE_COSTS.get(install_type, 0.0)


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


def assign_zone(distance: float, zip_code: str, geo_locator: GeoLocator) -> int:
    """Assign a numeric zone based on the distance and special cases."""
    clean_zip = str(zip_code).split("-")[0].strip()
    postal_info = geo_locator.nomi.query_postal_code(clean_zip)

    # Special case: Hawaii
    if pd.notna(postal_info.state_code) and postal_info.state_code == "HI":
        return 10

    # Special case: Canada
    if not clean_zip.isdigit() and len(clean_zip) >= 3:
        return 9

    if pd.isna(distance):
        return np.nan

    # Assign numeric zone based on distance ranges
    if 0 <= distance <= 100:
        return 1
    elif 101 <= distance <= 200:
        return 2
    elif 201 <= distance <= 400:
        return 3
    elif 401 <= distance <= 600:
        return 4
    elif 601 <= distance <= 1000:
        return 5
    elif 1001 <= distance <= 1400:
        return 6
    elif 1401 <= distance <= 1800:
        return 7
    elif 1801 <= distance <= 2500:
        return 8
    else:
        return np.nan


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


def display_results(df: pd.DataFrame, priority_columns: list) -> int:
    """Display results with specified columns first."""
    valid_priority_cols = [col for col in priority_columns if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in valid_priority_cols]
    ordered_cols = valid_priority_cols + remaining_cols
    df_display = df[ordered_cols]

    nan_count = df[DISTANCE_COL_NAME].isna().sum()
    st.write(f"Number of NaN values in '{DISTANCE_COL_NAME}' column: {nan_count}")
    st.dataframe(df_display)
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


def show_missing_values(df: pd.DataFrame, priority_columns: list) -> None:
    """Display rows with missing distance values."""
    nan_df = df[df[DISTANCE_COL_NAME].isna()]
    if not nan_df.empty:
        st.warning("The following rows have missing (NaN) distance values:")
        valid_priority_cols = [col for col in priority_columns if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in valid_priority_cols]
        st.dataframe(nan_df[valid_priority_cols + remaining_cols])


# Streamlit UI Configuration
st.set_page_config(layout="wide")
st.title("Zip Code Distance Calculator")
st.write(
    """
    Upload an Excel file containing zip codes. The app will calculate distances
    from each zip code to the origin zip code you specify and assign a zone based on:

    - Zone 1 (0-100 miles): $249.04
    - Zone 2 (101-200 miles): $298.70
    - Zone 3 (201-400 miles): $379.34
    - Zone 4 (401-600 miles): $439.51
    - Zone 5 (601-1000 miles): $497.19
    - Zone 6 (1001-1400 miles): $583.61
    - Zone 7 (1401-1800 miles): $672.34
    - Zone 8 (1801-2500 miles): $697.09
    - Zone 9 (Canada): $810.00
    - Zone 10 (Hawaii/Oahu): $750.00
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

    # Assign zones based on the calculated distance and zip code information
    df[ZONE_COL_NAME] = df.apply(
        lambda row: assign_zone(row[output_col], row[zip_col], geo_locator), axis=1
    )

    # Calculate the total cost based on the zone and install type
    df["Total Cost"] = df.apply(
        lambda row: calculate_cost(row[ZONE_COL_NAME], row["Install Type"]), axis=1
    )

    # Display total cost summary
    total_cost = df["Total Cost"].sum()
    st.write(f"Total Cost: ${total_cost:,.2f}")

    # Display detailed breakdown by zone and install type
    st.write("### Cost Breakdown by Zone and Install Type")
    zone_summary = (
        df.groupby([ZONE_COL_NAME, "Install Type"])
        .agg({"Total Cost": ["count", "sum"]})
        .reset_index()
    )
    zone_summary.columns = ["Zone", "Install Type", "Count", "Total Cost"]
    zone_summary["Zone Description"] = zone_summary["Zone"].map(ZONE_MAPPING)
    st.dataframe(zone_summary)

    # Additional summary by Install Type
    st.write("### Summary by Install Type")
    install_summary = (
        df.groupby("Install Type").agg({"Total Cost": ["count", "sum"]}).reset_index()
    )
    install_summary.columns = ["Install Type", "Count", "Total Cost"]
    st.dataframe(install_summary)

    # Define priority columns for display
    priority_columns = [
        zip_col,
        DISTANCE_COL_NAME,
        ZONE_COL_NAME,
        "Install Type",
        "Total Cost",
    ]

    # Display results with priority columns
    nan_count = display_results(df, priority_columns)
    create_download_link(df)

    if nan_count > 0:
        show_missing_values(df, priority_columns)


if __name__ == "__main__":
    main()
