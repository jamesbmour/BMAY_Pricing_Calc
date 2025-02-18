#%%
import pandas as pd
import numpy as np
import pgeocode

#%%
# Read the Excel file
df = pd.read_excel('data/New Install Billing Worksheet January.xlsx')
print(df.columns)

#%%
# Initialize pgeocode for US zip codes
nomi = pgeocode.Nominatim('us')

# Get latitude and longitude for the origin zip code (45241)
origin_info = nomi.query_postal_code("45241")
origin_lat = origin_info.latitude
origin_lon = origin_info.longitude
print(origin_lat, origin_lon)

#%%
# Define a function to compute the Haversine distance (in miles)
def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956  # Earth's radius in miles
    return c * r

#%%
# Function to get the distance from the origin to a given zip code
def get_distance(zip_code):
    # Ensure the zip code is a string and split on '-' if present, using only the first 5 digits
    zip_str = str(zip_code)
    zip_clean = zip_str.split('-')[0].strip()  # Take the first part and remove any extra spaces

    # Query the location information for the cleaned zip code
    info = nomi.query_postal_code(zip_clean)
    # If the zip code is not found, return NaN
    if np.isnan(info.latitude) or np.isnan(info.longitude):
        return np.nan

    # Calculate the distance using the Haversine formula
    return haversine(origin_lat, origin_lon, info.latitude, info.longitude)

#%%
# Apply the function to each zip code in the 'Zip' column
df['Distance'] = df['Zip'].apply(get_distance)
print(df[['Zip', 'Distance']])
#%%
# count the number of NaN values in the 'Distance' column
nan_count = df['Distance'].isna().sum()
print(f"Number of NaN values in 'Distance' column: {nan_count}")

#%%
# Optionally, save the result to a new Excel file
df.to_excel('output_with_distances.xlsx', index=False)

# Print the updated DataFrame
print(df)
