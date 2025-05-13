#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined workflow:
 1) Generate cluster centroids (plant locations) within Bayern, constrained to be:
    - In LAU polygons with POP_DENS_2023 < 250 (low population density),
    - Close to gas pipelines (within 3 km),
 2) Save locations with alphabetical labels in geographic coordinates (EPSG:4326),
 3) [OPTIONAL] Compute a distance matrix to feedstock supply nodes,
 4) Plot LAU units with POP_DENS_2023 < 250 in a separate map,
 5) Plot a combined map with low-density LAUs, Bavaria-only gas pipelines, and plant locations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import geopandas as gpd

# ---------------------------------------------------------------------------
# 1) Location Generation Functions
# ---------------------------------------------------------------------------

def load_low_density_polygons(lau_geojson_path, density_threshold=500):
    """
    Load polygons from the LAU GeoJSON file where POP_DENS_2023 < density_threshold.
    Returns a shapely MultiPolygon (union of valid polygons).
    """
    gdf = gpd.read_file(lau_geojson_path)
    low_density_gdf = gdf[gdf["POP_DENS_2023"] < density_threshold]
    if low_density_gdf.empty:
        raise ValueError(f"No polygons found with POP_DENS_2023 < {density_threshold}")
    print(f"Found {len(low_density_gdf)} polygons with POP_DENS_2023 < {density_threshold}")
    low_density_polygons = unary_union(low_density_gdf.geometry.tolist())
    return low_density_polygons

def load_bavaria_boundary(lau_geojson_path):
    """
    Load all LAU polygons from the GeoJSON file and return their union as the Bavaria boundary.
    Returns a shapely MultiPolygon.
    """
    gdf = gpd.read_file(lau_geojson_path)
    bavaria_boundary = unary_union(gdf.geometry.tolist())
    return bavaria_boundary

def random_points_near_gas_in_polygon(polygon, gas_lines, num_points, max_distance_km=5):
    """
    Randomly sample points inside the polygon that are within a certain distance (in km) of gas pipelines.
    Assumes all geometries are in a projected CRS (meters).
    """
    max_distance_m = max_distance_km * 1000
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        pt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pt) and gas_lines.distance(pt).min() <= max_distance_m:
            points.append(pt)
    return points

def generate_cluster_centroids(projected_polygon, projected_gas_lines, num_clusters=30, sample_size=20000):
    """
    Uses k-means clustering to generate centroids from sampled points near pipelines.
    Verifies that all centroids are within the low-density polygon.
    """
    max_attempts = 5
    attempt = 0
    while attempt < max_attempts:
        random_pts = random_points_near_gas_in_polygon(projected_polygon, projected_gas_lines, sample_size, max_distance_km=5)
        coords = np.array([[pt.x, pt.y] for pt in random_pts])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42 + attempt).fit(coords)
        centroids = kmeans.cluster_centers_
        
        # Verify all centroids are within the low-density polygon
        all_valid = True
        for x, y in centroids:
            if not projected_polygon.contains(Point(x, y)):
                all_valid = False
                break
        
        if all_valid:
            return centroids
        attempt += 1
        print(f"Warning: Attempt {attempt} produced invalid centroids. Retrying...")
    
    raise ValueError(f"Failed to generate valid centroids after {max_attempts} attempts. Consider increasing sample_size or relaxing constraints.")

def get_location_label(n):
    """Generate alphabetical labels (A, B, ..., Z, AA, AB, ...)."""
    label = ""
    while n >= 0:
        remainder = n % 26
        label = chr(65 + remainder) + label
        n = n // 26 - 1
    return label

def save_centroids_to_csv(centroids, output_path):
    """
    Save centroids to CSV after converting from projected CRS (EPSG:25832) to geographic CRS (EPSG:4326).
    """
    points = [Point(x, y) for x, y in centroids]
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:25832")
    gdf = gdf.to_crs("EPSG:4326")
    df = pd.DataFrame({
        "Longitude": [point.x for point in gdf.geometry],
        "Latitude": [point.y for point in gdf.geometry],
        "Location": [get_location_label(i) for i in range(len(gdf))]
    })
    df.to_csv(output_path, index=False)
    print(f"Saved centroids to {output_path}")

def plot_low_density_laus(lau_geojson_path, density_threshold=500):
    """
    Plot all LAU units, coloring those with POP_DENS_2023 < density_threshold in green.
    Returns fig, ax for reuse in combined plot.
    """
    gdf = gpd.read_file(lau_geojson_path)
    gdf = gdf.to_crs("EPSG:25832")  # Project for accurate visualization
    gdf["is_low_density"] = gdf["POP_DENS_2023"] < density_threshold

    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot all LAUs in gray (background)
    gdf[gdf["is_low_density"] == False].plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5, alpha=0.5)
    # Plot low-density LAUs in green
    gdf[gdf["is_low_density"] == True].plot(ax=ax, color="green", edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_title(f"LAU Units in Bavaria with Population Density < {density_threshold} inhabitants/km² (2023)")
    ax.set_axis_off()  # Remove axes for cleaner map
    plt.tight_layout()
    plt.savefig("low_density_lau_map.png", dpi=300)
    plt.close()  # Close to avoid display in combined plot
    return fig, ax

def plot_combined_map(lau_geojson_path, centroids, gas_lines_gdf, density_threshold=500):
    """
    Plot a combined map with:
    - LAU units (green for POP_DENS_2023 < 250, gray for others),
    - Gas pipelines within Bavaria (blue),
    - Plant location centroids (red).
    """
    # Replot LAUs
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    gdf = gpd.read_file(lau_geojson_path)
    gdf = gdf.to_crs("EPSG:25832")
    gdf["is_low_density"] = gdf["POP_DENS_2023"] < density_threshold
    gdf[gdf["is_low_density"] == False].plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5, alpha=0.5)
    gdf[gdf["is_low_density"] == True].plot(ax=ax, color="lightgreen", edgecolor="black", linewidth=0.5, alpha=0.8)

    # Plot Bavaria-clipped gas pipelines
    gas_lines_gdf.plot(ax=ax, color="blue", linewidth=0.7, label="Gas Pipelines")

    # Plot centroids
    xs, ys = zip(*centroids)
    ax.scatter(xs, ys, color="red", s=50, label="Cluster Centroids")

    ax.set_title(f"Plant Locations in Low-Density Areas (POP_DENS_2023 < {density_threshold}) with Bavaria Gas Grid")
    ax.legend()
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("combined_plant_locations.png", dpi=300)
    plt.show()

# ---------------------------------------------------------------------------
# 2) Distance Matrix Calculation
# ---------------------------------------------------------------------------

import pandas as pd
from geopy.distance import geodesic

def compute_distance_matrix(feedstock_csv, plant_csv, output_distance_csv, valid_plant_locs=None):
    # Load input data
    feedstock_df = pd.read_csv(feedstock_csv)
    plant_df = pd.read_csv(plant_csv)
    
    # Validate coordinates for NaN and invalid ranges
    for df_name, df, lat_col, lon_col in [
        ("Feedstock", feedstock_df, "Centroid_Lat", "Centroid_Lon"),
        ("Plant", plant_df, "Latitude", "Longitude")
    ]:
        # Check for NaN values
        nan_lat = df[lat_col].isna()
        nan_lon = df[lon_col].isna()
        if nan_lat.any() or nan_lon.any():
            print(f"Found NaN coordinates in {df_name} data:")
            nan_rows = df[nan_lat | nan_lon][['GISCO_ID' if df_name == "Feedstock" else 'Location', lat_col, lon_col]]
            print(nan_rows)
            # Remove rows with NaN coordinates
            initial_len = len(df)
            df = df.dropna(subset=[lat_col, lon_col])
            print(f"Removed {initial_len - len(df)} rows with NaN coordinates from {df_name} data.")
            if df.empty:
                raise ValueError(f"No valid {df_name} locations remain after removing NaN coordinates.")
            # Update the dataframe in the calling scope
            if df_name == "Feedstock":
                feedstock_df = df
            else:
                plant_df = df
        
        # Check for invalid ranges
        invalid_lat = df[lat_col][(df[lat_col] < -90) | (df[lat_col] > 90)]
        invalid_lon = df[lon_col][(df[lon_col] < -180) | (df[lon_col] > 180)]
        if not invalid_lat.empty:
            print(f"Invalid latitudes in {df_name}: {invalid_lat.values}")
            raise ValueError(f"Invalid latitude values found in {df_name} data.")
        if not invalid_lon.empty:
            print(f"Invalid longitudes in {df_name}: {invalid_lon.values}")
            raise ValueError(f"Invalid longitude values found in {df_name} data.")
    
    # Extract unique GISCO_ID centroids from feedstock_df
    initial_feedstock_count = len(feedstock_df)
    unique_feedstock_df = feedstock_df[['GISCO_ID', 'Centroid_Lat', 'Centroid_Lon']].drop_duplicates(subset=['GISCO_ID'])
    if len(unique_feedstock_df) < initial_feedstock_count:
        print(f"Reduced {initial_feedstock_count} feedstock rows to {len(unique_feedstock_df)} unique GISCO_ID entries.")
    
    # Remove duplicates in plant_df based on Location
    initial_plant_count = len(plant_df)
    plant_df = plant_df.drop_duplicates(subset=['Location'])
    if len(plant_df) < initial_plant_count:
        print(f"Removed {initial_plant_count - len(plant_df)} duplicate Location entries from plant_df.")
    
    # Filter plant_df to valid_plant_locs if provided
    if valid_plant_locs is not None:
        invalid_locs = set(plant_df['Location']) - set(valid_plant_locs)
        if invalid_locs:
            print(f"Warning: {len(invalid_locs)} plant locations not in valid_plant_locs: {invalid_locs}")
        plant_df = plant_df[plant_df['Location'].isin(valid_plant_locs)]
        if plant_df.empty:
            raise ValueError("No valid plant locations remain after filtering.")
    
    # Log input sizes
    print(f"Processing {len(unique_feedstock_df)} unique feedstock locations and {len(plant_df)} plant locations.")
    
    # Compute distances
    distance_matrix = []
    for _, f_row in unique_feedstock_df.iterrows():
        f_id = f_row["GISCO_ID"]
        f_coords = (f_row["Centroid_Lat"], f_row["Centroid_Lon"])
        for _, p_row in plant_df.iterrows():
            p_id = p_row["Location"]
            p_coords = (p_row["Latitude"], p_row["Longitude"])
            dist_km = geodesic(f_coords, p_coords).kilometers
            distance_matrix.append([f_id, p_id, dist_km])
    
    # Create DataFrame
    distance_df = pd.DataFrame(distance_matrix, columns=["Feedstock_LAU", "Location", "Distance_km"])
    
    # Check for duplicates in output
    initial_distance_count = len(distance_df)
    distance_df = distance_df.drop_duplicates(subset=['Feedstock_LAU', 'Location'])
    if len(distance_df) < initial_distance_count:
        print(f"Removed {initial_distance_count - len(distance_df)} duplicate (Feedstock_LAU, Location) pairs from distance matrix.")
    
    # Validate expected size
    expected_rows = len(unique_feedstock_df) * len(plant_df)
    if len(distance_df) != expected_rows:
        raise ValueError(f"Distance matrix has {len(distance_df)} rows, expected {expected_rows}.")
    
    # Save to CSV
    distance_df.to_csv(output_distance_csv, index=False)
    print(f"Saved distance matrix to {output_distance_csv} with {len(distance_df)} rows.")
    
    return distance_df
# ---------------------------------------------------------------------------
# 3) MAIN WORKFLOW
# ---------------------------------------------------------------------------

def main():
    # -----------------------------
    # Define simulation parameters
    # -----------------------------
    params = {
        "lau_geojson_path": "C:/Master_Python/bavaria_lau_clean.geojson",
        "gas_shapefile": r"C:/Users/fredr/OneDrive - NTNU/NTNU_semester/MasterArbeid/durchleitungsabschnitte/durchleitungsabschnitte.shp",
        "num_clusters": 5,
        "sample_size": 5000,
        "locations_csv": "C:/Clone/Master/equally_spaced_locations.csv",
        "feedstock_csv": "C:/Master_Python/processed_biomass_data.csv",
        "plant_csv": "C:/Clone/Master/equally_spaced_locations.csv",
        "output_distance_csv": "C:/Clone/Master/Distance_Matrix.csv",
        "density_threshold": 500
    }

    # -----------------------------
    # Plot low-density LAU units
    # -----------------------------
    print("Plotting LAU units with POP_DENS_2023 < 500...")
    plot_low_density_laus(params["lau_geojson_path"], params["density_threshold"])

    # -----------------------------
    # Load and reproject spatial data
    # -----------------------------
    print("Loading low-density polygons (POP_DENS_2023 < 500)...")
    low_density_polygon = load_low_density_polygons(params["lau_geojson_path"], params["density_threshold"])
    print("Loading Bavaria boundary...")
    bavaria_boundary = load_bavaria_boundary(params["lau_geojson_path"])
    
    polygon_gdf = gpd.GeoDataFrame(geometry=[low_density_polygon], crs="EPSG:4326").to_crs("EPSG:25832")
    projected_polygon = polygon_gdf.geometry.iloc[0]
    bavaria_gdf = gpd.GeoDataFrame(geometry=[bavaria_boundary], crs="EPSG:4326").to_crs("EPSG:25832")
    projected_bavaria = bavaria_gdf.geometry.iloc[0]

    print("Loading gas pipelines...")
    gas_lines_gdf = gpd.read_file(params["gas_shapefile"])
    gas_lines_gdf = gas_lines_gdf.to_crs("EPSG:25832")

    # Clip gas pipelines to Bavaria for plotting
    gas_lines_bavaria = gpd.clip(gas_lines_gdf, projected_bavaria)
    gas_lines_bavaria = gas_lines_bavaria[gas_lines_bavaria.geometry.notnull()]
    gas_lines_bavaria = gas_lines_bavaria[gas_lines_bavaria.is_valid & (~gas_lines_bavaria.is_empty)]

    # Clip gas pipelines to low-density polygons for centroid generation
    gas_lines_low_density = gpd.clip(gas_lines_gdf, projected_polygon)
    gas_lines_low_density = gas_lines_low_density[gas_lines_low_density.geometry.notnull()]
    gas_lines_low_density = gas_lines_low_density[gas_lines_low_density.is_valid & (~gas_lines_low_density.is_empty)]
    projected_gas_lines = gas_lines_low_density.geometry

    # -----------------------------
    # Generate plant locations
    # -----------------------------
    print("Generating cluster centroids...")
    centroids = generate_cluster_centroids(projected_polygon,
                                          projected_gas_lines,
                                          num_clusters=params["num_clusters"],
                                          sample_size=params["sample_size"])

    # -----------------------------
    # Plot combined map
    # -----------------------------
    print("Plotting combined map with LAUs, Bavaria gas pipelines, and centroids...")
    plot_combined_map(params["lau_geojson_path"], centroids, gas_lines_bavaria, params["density_threshold"])

    # -----------------------------
    # Save centroids
    # -----------------------------
    print("Saving centroids to CSV...")
    save_centroids_to_csv(centroids, params["locations_csv"])

    # -----------------------------
    # Compute distance matrix
    # -----------------------------
    print("Computing distance matrix...")
    distance_df = compute_distance_matrix(params["feedstock_csv"],
                                         params["plant_csv"],
                                         params["output_distance_csv"])
    print("Distance matrix sample:")
    print(distance_df.head())

if __name__ == "__main__":
    main()