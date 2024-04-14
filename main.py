import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
import matplotlib.colors as colors
from adjustText import adjust_text
import numpy as np

tqdm.pandas()

def load_population_data(filepath="datasets/Ontario_Population_and_Dwelling_Counts.csv"):
    column_names = [
        "Geographic name", "Geographic area type abbreviation", "Population, 2021", "Population, 2016",
        "Population percentage change, 2016 to 2021", "Total private dwellings, 2021",
        "Total private dwellings, 2016", "Total private dwellings percentage change, 2016 to 2021",
        "Private dwellings occupied by usual residents, 2021", "Private dwellings occupied by usual residents, 2016",
        "Private dwellings occupied by usual residents percentage change, 2016 to 2021",
        "Land area in square kilometres, 2021", "Population density per square kilometre, 2021",
        "National population rank, 2021", "Province/territory population rank, 2021"
    ]
    population_data = pd.read_csv(filepath, names=column_names, header=0)
    population_data = clean_population_data(population_data)
    return population_data

def clean_population_data(data):
    numeric_columns = [
        "Population, 2021", "Population, 2016",
        "Total private dwellings, 2021", "Total private dwellings, 2016",
        "Private dwellings occupied by usual residents, 2021", "Private dwellings occupied by usual residents, 2016"
    ]
    for column in numeric_columns:
        if data[column].dtype == 'object':
            # Remove commas and convert to numeric
            data[column] = pd.to_numeric(data[column].str.replace(',', ''), errors='coerce')
    data = data[['Geographic name', 'Population, 2021']].dropna()
    data['Geographic name'] = data['Geographic name'].str.title()
    return data

def load_hospital_data(filepath="datasets/Ontario_Hospital_Locations.csv"):
    hospital_data = pd.read_csv(filepath)
    hospital_data = clean_hospital_data(hospital_data)
    return hospital_data

def clean_hospital_data(data):
    data = data[['ENGLISH_NA', 'COMMUNITY', 'ADDRESS_LI', 'POSTAL_COD']]
    data['COMMUNITY'] = data['COMMUNITY'].str.title()
    data['Facility_Name'] = data['ENGLISH_NA'].apply(lambda x: x.split('-')[0].strip())
    data = data.drop_duplicates(subset=['Facility_Name'])
    return data

def merge_datasets(hospital_data, population_data):
    # Preprocessing steps if any additional are needed
    population_data['Geographic name'] = population_data['Geographic name'].str.title()
    hospital_data['COMMUNITY'] = hospital_data['COMMUNITY'].str.title()
    # Merge the data on the community name
    merged_data = pd.merge(hospital_data, population_data, left_on='COMMUNITY', right_on='Geographic name', how='inner')
    merged_data.dropna(subset=['Population, 2021'], inplace=True)
    merged_data.drop(columns=['Geographic name'], inplace=True)
    return merged_data

# function to get the coordinates from the address using geopy api - takes around 2.5 hours to run so use pre-collected data if possible
def fetch_geopy_coordinates(merged_data):
    # Define a function to parse the location into point geometry
    def geocode_address(address):
        try:
            location = geocode(address)
            if location:
                return Point(location.longitude, location.latitude)
        except Exception as e:
            print(f"Error geocoding address {address}: {e}")
        return None
    # Apply geocoding to the address column
    merged_data['address'] = merged_data['ADDRESS_LI'] + ', ' + merged_data['COMMUNITY'] + ', Ontario, Canada'
    merged_data['point'] = merged_data['address'].head(10).progress_apply(geocode_address)
    # Drop rows where geocoding failed
    merged_data = merged_data.dropna(subset=['point'])
    # Convert the DataFrame to a GeoDataFrame
    hospital_gdf = gpd.GeoDataFrame(merged_data, geometry='point')
    # Save the data to a CSV file including longitude and latitude
    hospital_gdf['latitude'] = hospital_gdf.geometry.y
    hospital_gdf['longitude'] = hospital_gdf.geometry.x
    hospital_gdf.to_csv("datasets/merged_data_with_locations_v2.csv", index=False)
    return hospital_gdf

def load_precollected_data(filepath = "datasets/merged_data_with_location.csv"):
    data = pd.read_csv(filepath)
    # Ensure columns containing longitude and latitude are correctly typed
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    data.dropna(subset=['latitude', 'longitude'], inplace=True)
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf

def load_geonames_data(filepath):
    # Use GeoPandas to read shapefile data
    geonames_data = gpd.read_file(filepath)
    return geonames_data

def load_and_simplify_shapefile(filepath, tolerance=0.1):
    gdf = gpd.read_file(filepath)
    gdf['geometry'] = gdf['geometry'].simplify(tolerance)
    return gdf

def normalize_marker_sizes(series, min_size=85, max_size=750):
    # Normalize series to have a minimum of min_size and a maximum of max_size
    series_normalized = (series - series.min()) / (series.max() - series.min())
    return series_normalized * (max_size - min_size) + min_size

# Function to identify highest and lowest facilities per capita
def identify_extreme_cities(merged_data):
    # Sort by facilities per capita
    sorted_data = merged_data.sort_values('Facilities_Per_Capita', ascending=False)
    # Get the highest and lowest
    highest = sorted_data.head(1)
    lowest = sorted_data.tail(1)
    # Return their community names
    return highest['COMMUNITY'].values[0], lowest['COMMUNITY'].values[0]

def plot_ontario_map(hospital_gdf, base_map_path, upper_tier_path ,geonames_path, highest_FPC, lowest_FPC):
    ontario_map = load_and_simplify_shapefile(base_map_path, tolerance=0.0001)
    upper_tier_boundaries = load_and_simplify_shapefile(upper_tier_path, tolerance=0.001)
    # Load geonames shapefile and filter for cities
    geonames_gdf = load_geonames_data(geonames_path)

    # Ensure the coordinate reference systems match
    if ontario_map.crs != hospital_gdf.crs:
        hospital_gdf = hospital_gdf.to_crs(ontario_map.crs)
    if ontario_map.crs != upper_tier_boundaries.crs:
        upper_tier_boundaries = upper_tier_boundaries.to_crs(ontario_map.crs)

    # Normalize the total population for the marker size
    hospital_gdf['marker_size'] = normalize_marker_sizes(hospital_gdf['Population, 2021'])

    # Start plotting
    fig, ax = plt.subplots(figsize=(15, 15))
    ontario_map.plot(ax=ax, color='white', edgecolor='lightgrey')
    upper_tier_boundaries.plot(ax=ax, edgecolor='black', linewidth=0.1, alpha=1, linestyle='--', facecolor="none")
    major_cities = ['Toronto', 'Ottawa', 'Hamilton', 'London', 'Kitchener', 'Windsor', 'Barrie', 'Kingston', 'Guelph', 'St. Catharines']
    
    texts = []
    for idx, row in geonames_gdf.iterrows():
        centroid = row['geometry'].centroid
        if row['LABEL'] in major_cities:
            texts.append(ax.text(centroid.x, centroid.y, row['LABEL'], fontsize=8, ha='center', va='center',
                             bbox=dict(facecolor='yellow', alpha=0.25, edgecolor='black', boxstyle='round,pad=0.5')))

    # add the highest and lowest cities to the plot
    highest_city = hospital_gdf[hospital_gdf['COMMUNITY'] == highest_FPC]
    lowest_city = hospital_gdf[hospital_gdf['COMMUNITY'] == lowest_FPC]
    ax.plot(highest_city.geometry.x, highest_city.geometry.y, 'x', color='green', markersize=10)
    ax.plot(lowest_city.geometry.x, lowest_city.geometry.y, 'x', color='red', markersize=10)
    texts.append(ax.text(highest_city.geometry.x, highest_city.geometry.y, highest_FPC, fontsize=8, ha='center', va='center',
                             bbox=dict(facecolor='green', alpha=0.25, edgecolor='black', boxstyle='round,pad=0.5')))
    texts.append(ax.text(lowest_city.geometry.x, lowest_city.geometry.y, lowest_FPC, fontsize=8, ha='center', va='center',
                                bbox=dict(facecolor='red', alpha=0.25, edgecolor='black', boxstyle='round,pad=0.5')))

    hospital_gdf.plot(ax=ax, column='Facilities_Per_Capita', cmap='viridis', alpha=0.6,
                          markersize=hospital_gdf['marker_size'], legend=True,
                          legend_kwds={'label': "Facilities Per Capita", 'orientation': "horizontal"})
    adjust_text(texts, expand=(5, 3.25), # expand text bounding boxes by 1.2 fold in x direction and 2 fold in y direction
                arrowprops=dict(arrowstyle='->', color='red') # ensure the labeling is clear by adding arrows
                )
    ax.set_title('Healthcare Facilities Per Capita in Ontario', fontsize=20, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Adjust visual elements
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    fig.savefig('datasets/ontario_healthcare_facilities_per_capita.png', dpi=300, bbox_inches='tight')
    plt.show()

population_data = load_population_data("datasets/Ontario_Population_and_Dwelling_Counts.csv")   
hospital_data = load_hospital_data("datasets/Ontario_Hospital_Locations.csv")

# Group the hospital data by facility name
grouped_hospitals = hospital_data.groupby('Facility_Name').first().reset_index()

# Aggregate the population data by geographic name
aggregated_population = population_data.groupby('Geographic name')['Population, 2021'].sum().reset_index()

# update the population 2021 column data to be the aggregated population so that theres no duplicates for each area type
population_data = population_data.merge(aggregated_population, on='Geographic name', how='inner')
population_data.drop(columns=['Population, 2021_x'], inplace=True)
population_data.rename(columns={'Population, 2021_y': 'Population, 2021'}, inplace=True)

# Merge data on city name
merged_data = merge_datasets(hospital_data, population_data)
# Calculate Facilities Per Capita
merged_data['Facilities_Per_Capita'] = merged_data.groupby('COMMUNITY').transform('count')['Facility_Name'] / merged_data['Population, 2021']
print(merged_data.tail(10))
merged_data.to_csv("datasets/merged_data.csv", index=False)

# identify the highest and lowest facilities per capita
highest_FPC, lowest_FPC = identify_extreme_cities(merged_data)

# Initialize the geolocator
geolocator = Nominatim(user_agent="geoapiExercises")
# To prevent spamming the service with too many requests, use RateLimiter
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# # run the api to get the coordinates of the hospitals - takes around 2.5 hours to run so use pre-collected data if possible
# hospital_gdf = fetch_geopy_coordinates(merged_data)

# read the geocoded data from the csv file
hospital_gdf = load_precollected_data("datasets/merged_data_with_location.csv")

base_map_path = "datasets/ontario_shapefiles/base_map/OBM_INDEX.shp"
upper_tier_path = "datasets/ontario_shapefiles/upper_tier_boundaries/Municipal_Boundary_-_Upper_Tier_and_District.shp"
geonames_path = "datasets/ontario_shapefiles/names/Geographic_Names_Ontario.shp"

plot_ontario_map(hospital_gdf, base_map_path, upper_tier_path, geonames_path, highest_FPC, lowest_FPC)


# Additional visualization function definitions
def plot_bar_chart_of_facilities_per_capita(data):
    # Sort the data by 'Facilities_Per_Capita' in descending order
    sorted_data = data.sort_values('Facilities_Per_Capita', ascending=False)
    plt.figure(figsize=(20, 6))
    # Ensure that the barplot uses the sorted data
    sns.barplot(x='COMMUNITY', y='Facilities_Per_Capita', data=sorted_data, palette='viridis')
    plt.xticks(rotation=90)
    plt.title('Healthcare Facilities Per Capita by Region')
    plt.xlabel('Region')
    plt.ylabel('Facilities Per Capita')
    plt.tight_layout()
    plt.savefig('datasets/bar_chart_facilities_per_capita_sorted.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_population_vs_facilities(merged_data):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x=np.log1p(merged_data['Population, 2021']), y='Facilities_Per_Capita', c='Facilities_Per_Capita', cmap='viridis', data=merged_data)
    plt.colorbar(scatter, label='Facilities Per Capita')
    plt.title('Log Population vs. Healthcare Facilities Per Capita')
    plt.xlabel('Log of Population')
    plt.ylabel('Facilities Per Capita')
    plt.tight_layout()
    plt.savefig('datasets/scatter_population_vs_facilities_per_capita.png', dpi=300, bbox_inches='tight')
    plt.show()  

def plot_facilities_boxplot(merged_data):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Facilities_Per_Capita', data=merged_data)
    plt.title('Boxplot of Healthcare Facilities Per Capita')
    plt.xlabel('Facilities Per Capita')

    # Calculate the upper whisker
    Q1 = merged_data['Facilities_Per_Capita'].quantile(0.25)
    Q3 = merged_data['Facilities_Per_Capita'].quantile(0.75)
    IQR = Q3 - Q1
    upper_whisker = Q3 + 1.5 * IQR

    # Draw upper whisker line
    plt.axvline(x=upper_whisker, color='r', linestyle='--')
    plt.text(upper_whisker, plt.ylim()[1], 'Upper Whisker', color='r', ha='left', va='top', 
             bbox=dict(facecolor='white', alpha=0.5))
    # Identify and annotate the farthest outlier, if any
    outliers = merged_data[merged_data['Facilities_Per_Capita'] > upper_whisker]['Facilities_Per_Capita']
    if not outliers.empty:
        farthest_outlier_value = outliers.max()
        plt.annotate('Farthest Outlier', xy=(farthest_outlier_value, 0), xytext=(farthest_outlier_value, -0.05),
                     arrowprops=dict(facecolor='red', shrink=0.05), ha='center', va='bottom')
    plt.show()

def plot_scatter_trend(merged_data):
    plt.figure(figsize=(10, 6))
    sns.regplot(x=np.log1p(merged_data['Population, 2021']), y='Facilities_Per_Capita', data=merged_data, scatter_kws={'alpha':0.5})
    plt.title('Log Population vs. Trend of Facilities Per Capita')
    plt.xlabel('Log of Population')
    plt.ylabel('Facilities Per Capita')
    plt.savefig('datasets/scatter_trend_facilities_per_capita.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_bar_chart_of_facilities_per_capita(merged_data)
plot_population_vs_facilities(merged_data)
plot_scatter_trend(merged_data)
plot_facilities_boxplot(merged_data)
