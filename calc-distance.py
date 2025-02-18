import pandas as pd
import pgeocode
import numpy as np
from geopy.distance import geodesic

nomi = pgeocode.Nominatim('us')


def get_lat_long(zip_code):
    location = nomi.query_postal_code(str(zip_code))
    return location.latitude, location.longitude


def get_distance(zip1, zip2):
    lat1, long1 = get_lat_long(zip1)
    lat2, long2 = get_lat_long(zip2)
    if pd.isna(lat1) or pd.isna(long1) or pd.isna(lat2) or pd.isna(long2):
        return np.nan  # Return NaN if any coordinate is not finite

    return geodesic((lat1, long1), (lat2, long2)).miles


if __name__ == "__main__":
    fp = "Septemer And October Comparison.xlsx"
    df = pd.read_excel(fp)
    # df = df.dropna(subset=['Zip'])
    print(df.shape)
    # get first 5 rows
    # df = df.head()

    # Convert ZIP codes to 5-digit format
    df['Zip'] = df['Zip'].apply(lambda x: str(x).split('-')[0] if '-' in str(x) else str(x))

    # Calculate distances from zipcode 45241
    base_zip = "45241"

    df['Miles'] = df['Zip'].apply(lambda x: get_distance(base_zip, x))
    print(df[['Zip', 'Miles']])

    # Save the output to a new Excel file
    file_name = fp.split(".")[0]
    df.to_excel(f"{file_name}-JDB.xlsx", index=False)
