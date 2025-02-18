#%%
import pandas as pd
import pgeocode
import numpy as np
import os
import dotenv
from geopy.distance import geodesic

ORIGIN = '45241'

#%%
# Load environment variables from .env file
dotenv.load_dotenv()
# print working directory
print(os.getcwd())

#%%
#%%
# Load the data
df = pd.read_excel('data/New Install Billing Worksheet January.xlsx', sheet_name='Sheet1')
print(df.columns)

#%%
nomi = pgeocode.Nominatim('US')

#%%
# Calculate distances from zipcode df['Zip'} to ORIGIN
df['Distance'] = df['Zip'].apply(lambda x: nomi.distance_between_postal_codes(ORIGIN, str(x)) if pd.notna(x) else np.nan)
df['Distance'] = df['Distance'].apply(lambda x: x * 0.621371 if pd.notna(x) else np.nan)
# Convert distance to miles
