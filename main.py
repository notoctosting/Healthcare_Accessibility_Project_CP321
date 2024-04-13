import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import seaborn as sns

# Load and clean population data
column_names = [
    "Geographic name", "Geographic area type abbreviation", "Population, 2021", "Population, 2016",
    "Population percentage change, 2016 to 2021", "Total private dwellings, 2021",
    "Total private dwellings, 2016", "Total private dwellings percentage change, 2016 to 2021",
    "Private dwellings occupied by usual residents, 2021", "Private dwellings occupied by usual residents, 2016",
    "Private dwellings occupied by usual residents percentage change, 2016 to 2021",
    "Land area in square kilometres, 2021", "Population density per square kilometre, 2021",
    "National population rank, 2021", "Province/territory population rank, 2021"
]
def clean_and_convert_population(data, column_name):
    if data[column_name].dtype == 'object':
        # Remove commas and convert to numeric, setting errors to coerce turns non-convertible values into NaN
        data[column_name] = pd.to_numeric(data[column_name].str.replace(',', ''), errors='coerce')
    return data


population_data = pd.read_csv("datasets/Ontario_Population_and_Dwelling_Counts.csv", names=column_names, header=0)
population_data = clean_and_convert_population(population_data, 'Population, 2021')


# Load and preprocess hospital data
hospital_data = pd.read_csv("datasets/Ontario_Hospital_Locations.csv")
hospital_data = hospital_data[['ENGLISH_NA', 'COMMUNITY', 'ADDRESS_LI', 'POSTAL_COD', 'X', 'Y']]
hospital_data.rename(columns={'X': 'Latitude', 'Y': 'Longitude'}, inplace=True)
hospital_data['COMMUNITY'] = hospital_data['COMMUNITY'].str.title()
# Check if 'Latitude' and 'Longitude' columns are available
print(hospital_data.columns)


population_data = population_data[['Geographic name', 'Population, 2021']].dropna()
population_data['Geographic name'] = population_data['Geographic name'].str.title()
# Merge data on city
merged_data = pd.merge(hospital_data, population_data, left_on='COMMUNITY', right_on='Geographic name', how='left')
merged_data.dropna(subset=['Population, 2021'], inplace=True)
print(merged_data['Population, 2021'].dtype)  # Should confirm the type is numeric
print(merged_data.columns)

# Calculate Facilities Per Capita
merged_data['Facilities_Per_Capita'] = merged_data.groupby('COMMUNITY')['ENGLISH_NA'].transform('count') / merged_data['Population, 2021']
print(merged_data.head(100))
merged_data.to_csv("datasets/merged_data.csv", index=False)


## Bar chart of facilities per capita
plt.figure(figsize=(20, 6))
sns.barplot(x='COMMUNITY', y='Facilities_Per_Capita', data=merged_data)
plt.xticks(rotation=90)
plt.title('Healthcare Facilities Per Capita by Region')
plt.xlabel('Region')
plt.ylabel('Facilities Per Capita')
plt.tight_layout()
plt.show()



