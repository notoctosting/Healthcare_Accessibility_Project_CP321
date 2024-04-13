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
hospital_data = hospital_data.drop_duplicates(subset=['ENGLISH_NA'])

def extract_facility_name(hospital_name):
    # Split the hospital name by '-' and take the first part
    return hospital_name.split('-')[0].strip()
hospital_data['Facility_Name'] = hospital_data['ENGLISH_NA'].apply(extract_facility_name)
grouped_hospitals = hospital_data.groupby('Facility_Name').first().reset_index()


population_data = population_data[['Geographic name', 'Population, 2021']].dropna()
population_data['Geographic name'] = population_data['Geographic name'].str.title()

aggregated_population = population_data.groupby('Geographic name')['Population, 2021'].sum().reset_index()

# update the population 2021 column data to be the aggregated population so that theres no duplicates for each area type
population_data = population_data.drop_duplicates(subset=['Geographic name'])
population_data = population_data.merge(aggregated_population, on='Geographic name', how='inner')
population_data.drop(columns=['Population, 2021_x'], inplace=True)
population_data.rename(columns={'Population, 2021_y': 'Population, 2021'}, inplace=True)


# Merge data on city
merged_data = pd.merge(grouped_hospitals, population_data[['Geographic name', 'Population, 2021']], left_on='COMMUNITY', right_on='Geographic name', how='inner')
merged_data.dropna(subset=['Population, 2021'], inplace=True)
merged_data.drop(columns=['Geographic name'], inplace=True)
print(merged_data.columns)

# Calculate Facilities Per Capita
merged_data['Facilities_Per_Capita'] = merged_data.groupby('COMMUNITY').transform('count')['Facility_Name'] / merged_data['Population, 2021']
print(merged_data.tail(10))
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


# # Convert geographical data for plotting
# gdf = gpd.GeoDataFrame(
#     merged_data, geometry=gpd.points_from_xy(merged_data.Longitude, merged_data.Latitude)
# )

# # Load a more detailed base map
# ontario_map = gpd.read_file("datasets/ontario_base_map.shp")  # Specify the path to your Ontario shapefile

# # Plot using GeoPandas
# fig, ax = plt.subplots(figsize=(10, 10))
# ontario_map.plot(ax=ax, color='white', edgecolor='black')  # Plot Ontario base map
# gdf.plot(ax=ax, marker='o', color='red', markersize=5)  # Plot hospital locations
# plt.title('Healthcare Facilities in Ontario')
# plt.xlim([-95, -74])  # Adjust these values based on the longitude of Ontario
# plt.ylim([41, 57])   # Adjust these values based on the latitude of Ontario
# plt.show()





