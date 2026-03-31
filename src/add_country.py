import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

print("Loading ADS-B data...")
DATA_FILE = "data/opensky_labeled.csv"
df = pd.read_csv(DATA_FILE)

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

print("Loading country boundaries...")
WORLD_SHP = "data/world/ne_110m_admin_0_countries.shp"
world = gpd.read_file(WORLD_SHP)[["ADMIN", "geometry"]]

# Spatial join
print("Assigning country to each flight point...")
gdf = gpd.sjoin(gdf, world, how="left", predicate="within")

# Rename column
gdf.rename(columns={"ADMIN": "country"}, inplace=True)

# Drop geometry for CSV
gdf.drop(columns=["geometry"], inplace=True)

# Save
OUTPUT_FILE = "data/opensky_with_country.csv"
gdf.to_csv(OUTPUT_FILE, index=False)

print("Country mapping completed ✅")
print("Saved:", OUTPUT_FILE)
print("Columns:", list(gdf.columns))


