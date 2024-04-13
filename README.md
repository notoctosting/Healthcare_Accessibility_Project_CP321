# Healthcare Accessibility Across Ontario

## Project Overview
This project aims to analyze healthcare accessibility across different regions in Ontario by visualizing the distribution and density of healthcare facilities in relation to the population. This allows us to identify areas with potentially insufficient healthcare services.

## Data Sources
- **Hospital Locations**: Extracted from Ontario's open data portal. This dataset includes information on the names, types, and locations of healthcare facilities in Ontario.
  - Data URL: [Ontario Hospital Locations](https://data.ontario.ca/dataset/hospital-locations)
- **Population Data**: Provides population statistics for municipalities in Ontario, essential for per capita calculations.
  - Data URL: [Ontario Population and Dwelling Counts](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=9810000202)

## Prerequisites
To run this project, you will need Python installed on your machine, along with the following libraries:
- Pandas
- Matplotlib
- Seaborn
- Geopandas (for geographical data handling)
- Folium (for interactive maps)

## Installation
Clone this repository or download the files into your local machine. To install the required Python libraries, run:
```bash
pip install pandas matplotlib seaborn geopandas folium
