#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Workflow:
1) Generate 100 plant locations in Bavaria (low-density LAUs, near gas pipelines).
2) Cluster feedstock data into 1100 clusters (if not already clustered).
3) Calculate total biogas potential using aggregated feedstock data.
4) Compute distance matrix from plant locations to feedstock cluster centroids.
5) Plot feedstock nodes, feedstock cluster centroids, and plant locations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.cluster import KMeans, AgglomerativeClustering
from geopy.distance import geodesic
import geopandas as gpd

# Parameters
BASE_DIR = "C:/Clone/Master/"
LAU_GEOJSON = "C:/Master_Python/bavaria_lau_clean.geojson"
GAS_SHAPEFILE = "C:/Users/fredr/OneDrive - NTNU/NTNU_semester/MasterArbeid/durchleitungsabschnitte/durchleitungsabschnitte.shp"
FEEDSTOCK_CSV = f"{BASE_DIR}aggregated_bavaria_supply_nodes.csv"
YIELDS_CSV = f"{BASE_DIR}Feedstock_yields.csv"
PLANT_CSV = f"{BASE_DIR}equally_spaced_locations.csv"
DISTANCE_CSV = f"{BASE_DIR}Distance_Matrix.csv"
PLOT_OUTPUT = f"{BASE_DIR}bavaria_feedstock_plants.png"
NUM_PLANT_CLUSTERS = 15  # Restored to original distribution
NUM_FEEDSTOCK_CLUSTERS = 750
SAMPLE_SIZE = 10000
DENSITY_THRESHOLD = 500
MAX_DISTANCE_KM = 5

# Location Generation Functions
def load_low_density_polygons(lau_geojson_path, density_threshold=500):
    gdf = gpd.read_file(lau_geojson_path)
    low_density_gdf = gdf[gdf["POP_DENS_2023"] < density_threshold]
    if low_density_gdf.empty:
        raise ValueError(f"No polygons with POP_DENS_2023 < {density_threshold}")
    print(f"Found {len(low_density_gdf)} low-density polygons")
    return unary_union(low_density_gdf.geometry.tolist())

def load_bavaria_boundary(lau_geojson_path):
    gdf = gpd.read_file(lau_geojson_path)
    return unary_union(gdf.geometry.tolist())

def random_points_near_gas_in_polygon(polygon, gas_lines, num_points, max_distance_km=5):
    max_distance_m = max_distance_km * 1000
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        pt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pt) and gas_lines.distance(pt).min() <= max_distance_m:
            points.append(pt)
    return points

def generate_cluster_centroids(projected_polygon, projected_gas_lines, num_clusters=100, sample_size=10000):
    max_attempts = 5
    for attempt in range(max_attempts):
        random_pts = random_points_near_gas_in_polygon(projected_polygon, projected_gas_lines, sample_size, max_distance_km=MAX_DISTANCE_KM)
        coords = np.array([[pt.x, pt.y] for pt in random_pts])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42 + attempt).fit(coords)
        centroids = kmeans.cluster_centers_
        if all(projected_polygon.contains(Point(x, y)) for x, y in centroids):
            return centroids
        print(f"Attempt {attempt + 1} failed. Retrying...")
    raise ValueError(f"Failed to generate valid centroids after {max_attempts} attempts")

def get_location_label(n):
    label = ""
    while n >= 0:
        remainder = n % 26
        label = chr(65 + remainder) + label
        n = n // 26 - 1
    return label

def save_centroids_to_csv(centroids, output_path):
    points = [Point(x, y) for x, y in centroids]
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:25832").to_crs("EPSG:4326")
    df = pd.DataFrame({
        "Longitude": [point.x for point in gdf.geometry],
        "Latitude": [point.y for point in gdf.geometry],
        "Location": [get_location_label(i) for i in range(len(gdf))]
    })
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} plant locations to {output_path}")

# Distance Matrix Function (Plant to Feedstock Cluster Centroids)
def compute_distance_matrix_to_cluster_centroids(cluster_centroids_df, plant_csv, output_distance_csv):
    plant_df = pd.read_csv(plant_csv)
    
    # Validate plant coordinates
    nan_rows = plant_df[plant_df["Latitude"].isna() | plant_df["Longitude"].isna()]
    if not nan_rows.empty:
        print(f"Removing {len(nan_rows)} rows with NaN coordinates from Plant data")
        plant_df = plant_df.dropna(subset=["Latitude", "Longitude"])
    invalid_rows = plant_df[(plant_df["Latitude"] < -90) | (plant_df["Latitude"] > 90) | 
                            (plant_df["Longitude"] < -180) | (plant_df["Longitude"] > 180)]
    if not invalid_rows.empty:
        raise ValueError("Invalid coordinates in Plant data")
    
    # Compute distances
    distance_matrix = []
    for _, f_row in cluster_centroids_df.iterrows():
        f_coords = (f_row["Centroid_Lat"], f_row["Centroid_Lon"])
        for _, p_row in plant_df.iterrows():
            p_coords = (p_row["Latitude"], p_row["Longitude"])
            dist_km = geodesic(f_coords, p_coords).kilometers
            distance_matrix.append([f_row["Cluster_ID"], p_row["Location"], dist_km])
    
    distance_df = pd.DataFrame(distance_matrix, columns=["Feedstock_Cluster", "Plant_Location", "Distance_km"])
    distance_df.to_csv(output_distance_csv, index=False)
    print(f"Saved distance matrix with {len(distance_df)} rows to {output_distance_csv}")
    return distance_df

# Main Workflow
def main():
    # Load and reproject spatial data
    try:
        low_density_polygon = load_low_density_polygons(LAU_GEOJSON, DENSITY_THRESHOLD)
        bavaria_boundary = load_bavaria_boundary(LAU_GEOJSON)
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure {LAU_GEOJSON} exists")
        return
    
    polygon_gdf = gpd.GeoDataFrame(geometry=[low_density_polygon], crs="EPSG:4326").to_crs("EPSG:25832")
    projected_polygon = polygon_gdf.geometry.iloc[0]
    bavaria_gdf = gpd.GeoDataFrame(geometry=[bavaria_boundary], crs="EPSG:4326").to_crs("EPSG:25832")
    projected_bavaria = bavaria_gdf.geometry.iloc[0]
    
    try:
        gas_lines_gdf = gpd.read_file(GAS_SHAPEFILE).to_crs("EPSG:25832")
    except FileNotFoundError:
        print(f"Error: {GAS_SHAPEFILE} not found")
        return
    
    gas_lines_low_density = gpd.clip(gas_lines_gdf, projected_polygon)
    gas_lines_low_density = gas_lines_low_density[gas_lines_low_density.is_valid & (~gas_lines_low_density.is_empty)]
    if gas_lines_low_density.empty:
        print("Error: No valid gas pipelines in low-density areas")
        return
    projected_gas_lines = gas_lines_low_density.geometry

    # Generate plant locations
    try:
        centroids = generate_cluster_centroids(projected_polygon, projected_gas_lines, NUM_PLANT_CLUSTERS, SAMPLE_SIZE)
        save_centroids_to_csv(centroids, PLANT_CSV)
    except ValueError as e:
        print(f"Error generating plant centroids: {e}")
        return

    # Load aggregated feedstock data
    try:
        feedstock_df = pd.read_csv(FEEDSTOCK_CSV)
        print(f"Loaded feedstock data with {len(feedstock_df)} rows")
        print(f"Columns: {feedstock_df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: {FEEDSTOCK_CSV} not found")
        return
    
    # Check for required columns
    required_columns = ['GISCO_ID', 'substrat_ENG', 'nutz_pot_tFM', 'Centroid_Lat', 'Centroid_Lon']
    missing_columns = [col for col in required_columns if col not in feedstock_df.columns]
    if missing_columns:
        print(f"Error: Missing columns in feedstock data: {missing_columns}")
        return
    
    # Use all data (already Bavaria-specific)
    df = feedstock_df.copy()
    print(f"Using all feedstock data: {len(df)} rows")
    print(f"Sample GISCO_IDs: {df['GISCO_ID'].head().tolist()}")
    
    # Load yields data
    try:
        yields_df = pd.read_csv(YIELDS_CSV)
        print(f"Loaded yields data with substrates: {yields_df['substrat_ENG'].unique()}")
    except FileNotFoundError:
        print(f"Warning: {YIELDS_CSV} not found. Using default yields")
        yields_df = pd.DataFrame({
            'substrat_ENG': df['substrat_ENG'].unique(),
            'Biogas_Yield_m3_ton': [500] * df['substrat_ENG'].nunique(),
            'Methane_Content_%': [50] * df['substrat_ENG'].nunique()
        })
    
    # Check substrate mismatch
    feedstock_substrates = set(df['substrat_ENG'].unique())
    yields_substrates = set(yields_df['substrat_ENG'].unique())
    unmatched_substrates = feedstock_substrates - yields_substrates
    if unmatched_substrates:
        print(f"Warning: Substrates in feedstock not found in yields: {unmatched_substrates}")
    
    # Merge yield info
    df = df.merge(yields_df[["substrat_ENG", "Biogas_Yield_m3_ton", "Methane_Content_%"]], on="substrat_ENG", how="left")
    print(f"After merge: {len(df)} rows")
    
    # Check for NaNs before cleaning
    nan_counts = df[["Centroid_Lat", "Centroid_Lon", "nutz_pot_tFM", "Biogas_Yield_m3_ton", "Methane_Content_%"]].isna().sum()
    print(f"NaN counts before cleaning:\n{nan_counts}")
    
    # Clean data
    df = df.dropna(subset=["Centroid_Lat", "Centroid_Lon", "nutz_pot_tFM", "Biogas_Yield_m3_ton", "Methane_Content_%"])
    print(f"After dropna: {len(df)} rows")
    
    # Check coordinates
    invalid_coords = df[~(df["Centroid_Lat"].between(-90, 90)) | ~(df["Centroid_Lon"].between(-180, 180))]
    if not invalid_coords.empty:
        print(f"Found {len(invalid_coords)} rows with invalid coordinates:\n{invalid_coords[['GISCO_ID', 'Centroid_Lat', 'Centroid_Lon']]}")
    df = df[(df["Centroid_Lat"].between(-90, 90)) & (df["Centroid_Lon"].between(-180, 180))]
    print(f"After coordinate check: {len(df)} rows")
    
    if df.empty:
        print("Error: No valid feedstock data after cleaning. Check NaNs, substrates, or coordinates.")
        return
    
    # Compute biogas potential
    df["Biogas_Potential_m3"] = df["nutz_pot_tFM"] * df["Biogas_Yield_m3_ton"] * (df["Methane_Content_%"])
    
    # Calculate total biogas potential
    total_biogas_m3 = df["Biogas_Potential_m3"].sum()
    print(f"Total biogas potential in Bavaria: {total_biogas_m3:,.2f} m³")
    
    # Check if feedstock data is clustered
    is_clustered = df['GISCO_ID'].str.startswith('CLUSTER_').any()
    if is_clustered:
        print("Feedstock data is already clustered with CLUSTER_... IDs")
        # Extract unique cluster centroids
        cluster_centroids_df = df[['GISCO_ID', 'Centroid_Lat', 'Centroid_Lon']].drop_duplicates('GISCO_ID')
        cluster_centroids_df = cluster_centroids_df.rename(columns={'GISCO_ID': 'Cluster_ID'})
    else:
        print("Feedstock data is not clustered. Applying Agglomerative Clustering...")
        # Aggregate by GISCO_ID for clustering
        agg = df.groupby(['GISCO_ID', 'Centroid_Lat', 'Centroid_Lon']).agg(
            Total_Biogas_Potential=('Biogas_Potential_m3', 'sum')
        ).reset_index()
        
        # Create GeoDataFrame and project to UTM
        gdf = gpd.GeoDataFrame(
            agg,
            geometry=gpd.points_from_xy(agg["Centroid_Lon"], agg["Centroid_Lat"]),
            crs="EPSG:4326"
        ).to_crs("EPSG:25832")
        
        # Perform Agglomerative Clustering
        coords = np.stack([gdf.geometry.x, gdf.geometry.y], axis=1)
        clustering = AgglomerativeClustering(n_clusters=NUM_FEEDSTOCK_CLUSTERS, linkage="ward")
        gdf["Cluster_ID"] = clustering.fit_predict(coords)
        
        # Compute weighted centroids
        df = df.merge(gdf[['GISCO_ID', 'Cluster_ID']], on='GISCO_ID', how='left')
        cluster_centroids = df.groupby('Cluster_ID').apply(
            lambda x: pd.Series({
                'Centroid_Lat': np.average(x['Centroid_Lat'], weights=x['Biogas_Potential_m3']),
                'Centroid_Lon': np.average(x['Centroid_Lon'], weights=x['Biogas_Potential_m3'])
            }), include_groups=False
        ).reset_index()
        cluster_centroids['Cluster_ID'] = cluster_centroids['Cluster_ID'].apply(lambda x: f'CLUSTER_{int(x)}')
        cluster_centroids_df = cluster_centroids
    
    # Compute distance matrix to cluster centroids
    distance_df = compute_distance_matrix_to_cluster_centroids(cluster_centroids_df, PLANT_CSV, DISTANCE_CSV)
    
    # Plot feedstock nodes, cluster centroids, and plant locations
    plant_df = pd.read_csv(PLANT_CSV)
    feedstock_centroids = df[['GISCO_ID', 'Centroid_Lat', 'Centroid_Lon']].drop_duplicates('GISCO_ID')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.scatter(feedstock_centroids["Centroid_Lon"], feedstock_centroids["Centroid_Lat"], 
                c="blue", s=5, alpha=0.8, label="Feedstock Nodes")
    plt.scatter(cluster_centroids_df["Centroid_Lon"], cluster_centroids_df["Centroid_Lat"], 
                c="green", s=20, alpha=0.8, label="Feedstock Cluster Centroids")
    plt.scatter(plant_df["Longitude"], plant_df["Latitude"], 
                c="red", s=50, alpha=0.8, label="Plant Locations")
    
    ax.set_xlim(9, 13.8)
    ax.set_ylim(47.3, 50.6)
    ax.set_title("Feedstock Nodes, Cluster Centroids, and Plant Locations in Bavaria", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=300)
    print(f"Saved plot to {PLOT_OUTPUT}")
    plt.show()

if __name__ == "__main__":
    main()