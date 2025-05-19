#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Workflow:
1) Generate 10 plant locations in Bavaria (low-density LAUs, near gas pipelines).
2) Cluster feedstock data into 1100 clusters (if not already clustered).
3) Calculate total biogas potential and analyze clustering effects.
4) Compute distance matrix and visualize cluster mapping for 5 clusters with GISCO_ID and biogas potential.
5) Plot feedstock nodes, cluster centroids, plant locations, distance distributions, and subset visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.cluster import KMeans, AgglomerativeClustering
from geopy.distance import geodesic
import geopandas as gpd
import seaborn as sns
from scipy.stats import skew, kurtosis

# Parameters
BASE_DIR = "C:/Clone/Master/"
LAU_GEOJSON = "C:/Master_Python/bavaria_lau_clean.geojson"
GAS_SHAPEFILE = "C:/Users/fredr/OneDrive - NTNU/NTNU_semester/MasterArbeid/durchleitungsabschnitte/durchleitungsabschnitte.shp"
FEEDSTOCK_CSV = f"{BASE_DIR}aggregated_bavaria_supply_nodes.csv"
ORIGINAL_FEEDSTOCK_CSV = f"{BASE_DIR}processed_biomass_data.csv"
YIELDS_CSV = f"{BASE_DIR}Feedstock_yields.csv"
PLANT_CSV = f"{BASE_DIR}equally_spaced_locations_100.csv"
DISTANCE_CSV = f"{BASE_DIR}Distance_Matrix_100.csv"
CLUSTER_MAPPING_CSV = f"{BASE_DIR}cluster_mapping.csv"
DISTANCE_PLOT = f"{BASE_DIR}distance_distribution.png"
CLUSTER_MAP_PLOT = f"{BASE_DIR}cluster_mapping_5_clusters.png"
SUBSET_PLOT = f"{BASE_DIR}subset_centroid_placement.png"
PLOT_OUTPUT = f"{BASE_DIR}bavaria_feedstock_plants.png"
NUM_PLANT_CLUSTERS = 100  # Corrected to match expected distance counts
NUM_FEEDSTOCK_CLUSTERS = 550
SAMPLE_SIZE = 20000
DENSITY_THRESHOLD = 500
MAX_DISTANCE_KM = 5
SUBSET_CLUSTERS = 5  # Number of clusters for subset visualization
SELECTED_CLUSTERS = [f"CLUSTER_{i}" for i in range(5)]  # Clusters for mapping plot

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

def generate_cluster_centroids(projected_polygon, projected_gas_lines, num_clusters, sample_size):
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

def save_centroids_to_csv(centroids, output_path):
    points = [Point(x, y) for x, y in centroids]
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:25832").to_crs("EPSG:4326")
    df = pd.DataFrame({
        "Longitude": [point.x for point in gdf.geometry],
        "Latitude": [point.y for point in gdf.geometry],
        "Location": [f"Plant_{i}" for i in range(len(gdf))]
    })
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} plant locations to {output_path}")

# Distance Calculation Functions
def compute_distances_to_nodes(plant_df, node_df):
    print(f"Computing distances: {len(plant_df)} plants, {len(node_df)} nodes")
    distances = []
    for _, p_row in plant_df.iterrows():
        p_coords = (p_row["Latitude"], p_row["Longitude"])
        for _, n_row in node_df.iterrows():
            n_coords = (n_row["Centroid_Lat"], n_row["Centroid_Lon"])
            dist_km = geodesic(p_coords, n_coords).kilometers
            distances.append(dist_km)
    print(f"Computed {len(distances)} distances before clustering")
    return distances

def compute_distances_to_centroids(plant_df, centroid_df):
    print(f"Computing distances: {len(plant_df)} plants, {len(centroid_df)} centroids")
    distances = []
    for _, p_row in plant_df.iterrows():
        p_coords = (p_row["Latitude"], p_row["Longitude"])
        for _, c_row in centroid_df.iterrows():
            c_coords = (c_row["Centroid_Lat"], c_row["Centroid_Lon"])
            dist_km = geodesic(p_coords, c_coords).kilometers
            distances.append(dist_km)
    print(f"Computed {len(distances)} distances after clustering")
    return distances

def compute_distance_matrix_to_cluster_centroids(cluster_centroids_df, plant_csv, output_distance_csv):
    plant_df = pd.read_csv(plant_csv)
    distance_matrix = []
    for _, f_row in cluster_centroids_df.iterrows():
        f_coords = (f_row["Centroid_Lat"], f_row["Centroid_Lon"])
        for _, p_row in plant_df.iterrows():
            p_coords = (p_row["Latitude"], p_row["Longitude"])
            dist_km = geodesic(f_coords, p_coords).kilometers
            distance_matrix.append([f_row["Cluster_ID"], p_row["Location"], dist_km])
    distance_df = pd.DataFrame(distance_matrix, columns=["Feedstock_LAU", "Location", "Distance_km"])
    distance_df.to_csv(output_distance_csv, index=False)
    print(f"Saved distance matrix with {len(distance_df)} rows to {output_distance_csv}")
    return distance_df

def plot_cluster_mapping_5_clusters(cluster_mapping, cluster_centroids_df, output_path):
    # Calculate distances from centroids to a central point (Munich: 11.58, 48.14)
    central_point = (48.14, 11.58)
    cluster_centroids_df['Distance_to_center'] = cluster_centroids_df.apply(
        lambda row: geodesic((row['Centroid_Lat'], row['Centroid_Lon']), central_point).kilometers, axis=1)
    # Select 5 clusters closest to the central point
    nearby_clusters = cluster_centroids_df.nsmallest(5, 'Distance_to_center')['Cluster_ID'].tolist()
    
    subset_mapping = cluster_mapping[cluster_mapping['Cluster_ID'].isin(nearby_clusters)]
    subset_centroids = cluster_centroids_df[cluster_centroids_df['Cluster_ID'].isin(nearby_clusters)]
    
    # Calculate plot bounds based on subset data
    lon_min = subset_mapping['Centroid_Lon'].min() - 0.02
    lon_max = subset_mapping['Centroid_Lon'].max() + 0.04
    lat_min = subset_mapping['Centroid_Lat'].min() - 0.02
    lat_max = subset_mapping['Centroid_Lat'].max() + 0.02
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for idx, cluster_id in enumerate(nearby_clusters):
        cluster_nodes = subset_mapping[subset_mapping['Cluster_ID'] == cluster_id]
        centroid = subset_centroids[subset_centroids['Cluster_ID'] == cluster_id]
        if cluster_nodes.empty or centroid.empty:
            print(f"Warning: No data for {cluster_id}")
            continue
        # Plot nodes
        plt.scatter(cluster_nodes["Centroid_Lon"], cluster_nodes["Centroid_Lat"], 
                    c=colors[idx], s=125, alpha=0.6, label=f"Nodes {cluster_id}")
        # Plot centroid
        plt.scatter(centroid["Centroid_Lon"], centroid["Centroid_Lat"], 
                    c=colors[idx], s=150, marker='x', label=f"Centroid {cluster_id}")
        # Draw lines and add labels
        for _, node in cluster_nodes.iterrows():
            plt.plot([node["Centroid_Lon"], centroid["Centroid_Lon"].iloc[0]], 
                     [node["Centroid_Lat"], centroid["Centroid_Lat"].iloc[0]], 
                     c=colors[idx], linestyle='--', alpha=0.5)
            # Label full GISCO_ID
            #plt.text(node["Centroid_Lon"] + 0.005, node["Centroid_Lat"]+ 0.005, node["GISCO_ID"], fontsize=14, ha='left')
            # Label Biogas_Potential_m3 with m³
            biogas = node["Biogas_Potential_m3"]
            plt.text(node["Centroid_Lon"] - 0.005, node["Centroid_Lat"], 
                     f"{biogas:,.0f} m³", fontsize=16, ha='left', color='black')
    
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(f"Cluster Mapping for 5 Nearby Clusters in Bavaria", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 5-cluster mapping plot to {output_path}")
    plt.close()

def plot_subset_before_after(original_subset, centroid_subset, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for idx, cluster_id in enumerate(original_subset['Cluster_ID'].unique()):
        cluster_nodes = original_subset[original_subset['Cluster_ID'] == cluster_id]
        centroid = centroid_subset[centroid_subset['Cluster_ID'] == cluster_id]
        plt.scatter(cluster_nodes["Centroid_Lon"], cluster_nodes["Centroid_Lat"], 
                    c=colors[idx % len(colors)], s=30, alpha=0.6, 
                    label=f"Nodes (Cluster {cluster_id.split('_')[1]})")
        plt.scatter(centroid["Centroid_Lon"], centroid["Centroid_Lat"], 
                    c=colors[idx % len(colors)], s=100, marker='x', 
                    label=f"Centroid (Cluster {cluster_id.split('_')[1]})")
        for _, node in cluster_nodes.iterrows():
            plt.plot([node["Centroid_Lon"], centroid["Centroid_Lon"].iloc[0]], 
                     [node["Centroid_Lat"], centroid["Centroid_Lat"].iloc[0]], 
                     'k-', alpha=0.3)
    ax.set_title(f"Before and After Clustering (Subset of {SUBSET_CLUSTERS} Clusters)", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved subset centroid placement plot to {output_path}")
    plt.close()

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
    '''
    # Generate plant locations
    try:
        centroids = generate_cluster_centroids(projected_polygon, projected_gas_lines, NUM_PLANT_CLUSTERS, SAMPLE_SIZE)
        save_centroids_to_csv(centroids, PLANT_CSV)
    except ValueError as e:
        print(f"Error generating plant centroids: {e}")
        return
    '''
    # Load original feedstock data (before clustering)
    try:
        original_df = pd.read_csv(ORIGINAL_FEEDSTOCK_CSV)
        original_df = original_df[original_df['GISCO_ID'].str.startswith('DE_09')].copy()
        print(f"Loaded original feedstock data with {len(original_df)} rows")
    except FileNotFoundError:
        print(f"Error: {ORIGINAL_FEEDSTOCK_CSV} not found")
        return

    # Load aggregated (clustered) feedstock data
    try:
        feedstock_df = pd.read_csv(FEEDSTOCK_CSV)
        print(f"Loaded clustered feedstock data with {len(feedstock_df)} rows")
    except FileNotFoundError:
        print(f"Error: {FEEDSTOCK_CSV} not found")
        return
    
    # Load yields data
    try:
        yields_df = pd.read_csv(YIELDS_CSV)
        print(f"Loaded yields data with substrates: {yields_df['substrat_ENG'].unique()}")
    except FileNotFoundError:
        print(f"Warning: {YIELDS_CSV} not found. Using default yields")
        yields_df = pd.DataFrame({
            'substrat_ENG': feedstock_df['substrat_ENG'].unique(),
            'Biogas_Yield_m3_ton': [500] * feedstock_df['substrat_ENG'].nunique(),
            'Methane_Content_%': [50] * feedstock_df['substrat_ENG'].nunique()
        })
    
    # Merge yields and compute biogas potential for original data
    original_df = original_df.merge(yields_df[["substrat_ENG", "Biogas_Yield_m3_ton", "Methane_Content_%"]], 
                                    on="substrat_ENG", how="left")
    original_df = original_df.dropna(subset=["Centroid_Lat", "Centroid_Lon", "nutz_pot_tFM", "Biogas_Yield_m3_ton", "Methane_Content_%"])
    original_df["Biogas_Potential_m3"] = original_df["nutz_pot_tFM"] * original_df["Biogas_Yield_m3_ton"] * (original_df["Methane_Content_%"])

    # Merge yields and compute biogas potential for clustered data
    df = feedstock_df.merge(yields_df[["substrat_ENG", "Biogas_Yield_m3_ton", "Methane_Content_%"]], 
                            on="substrat_ENG", how="left")
    df = df.dropna(subset=["Centroid_Lat", "Centroid_Lon", "nutz_pot_tFM", "Biogas_Yield_m3_ton", "Methane_Content_%"])
    df["Biogas_Potential_m3"] = df["nutz_pot_tFM"] * df["Biogas_Yield_m3_ton"] * (df["Methane_Content_%"])

    # Calculate total biogas potential
    total_biogas_m3 = original_df["Biogas_Potential_m3"].sum()
    print(f"Total biogas potential in Bavaria: {total_biogas_m3:,.2f} m³")

    # Check if feedstock data is clustered and generate cluster mapping
    if df['GISCO_ID'].str.startswith('CLUSTER_').any():
        print("Feedstock data is already clustered")
        cluster_centroids_df = df[['GISCO_ID', 'Centroid_Lat', 'Centroid_Lon']].drop_duplicates('GISCO_ID')
        cluster_centroids_df = cluster_centroids_df.rename(columns={'GISCO_ID': 'Cluster_ID'})
        
        # Generate cluster mapping with biogas potential
        agg = original_df.groupby(['GISCO_ID', 'Centroid_Lat', 'Centroid_Lon']).agg(
            Biogas_Potential_m3=('Biogas_Potential_m3', 'sum')
        ).reset_index()
        gdf = gpd.GeoDataFrame(agg, geometry=gpd.points_from_xy(agg["Centroid_Lon"], agg["Centroid_Lat"]), crs="EPSG:4326").to_crs("EPSG:25832")
        coords = np.stack([gdf.geometry.x, gdf.geometry.y], axis=1)
        clustering = AgglomerativeClustering(n_clusters=NUM_FEEDSTOCK_CLUSTERS, linkage="ward")
        gdf["Cluster_ID"] = clustering.fit_predict(coords)
        gdf["Cluster_ID"] = gdf["Cluster_ID"].apply(lambda x: f"CLUSTER_{int(x)}")
        cluster_mapping = gdf[['GISCO_ID', 'Cluster_ID', 'Centroid_Lat', 'Centroid_Lon', 'Biogas_Potential_m3']]
        cluster_mapping.to_csv(CLUSTER_MAPPING_CSV, index=False)
        print(f"Saved cluster mapping to {CLUSTER_MAPPING_CSV}")

        # Calculate average nodes per cluster
        nodes_per_cluster = cluster_mapping.groupby('Cluster_ID')['GISCO_ID'].count()
        avg_nodes = nodes_per_cluster.mean()
        print(f"Average number of nodes per cluster: {avg_nodes:.2f}")
        print(f"Nodes per cluster stats:\n{nodes_per_cluster.describe()}")

        # Plot 5-cluster mapping with GISCO_ID and biogas potential
        plot_cluster_mapping_5_clusters(cluster_mapping, cluster_centroids_df, CLUSTER_MAP_PLOT)

        # Subset for before-and-after visualization
        subset_clusters = cluster_mapping['Cluster_ID'].unique()[:SUBSET_CLUSTERS]
        original_subset = cluster_mapping[cluster_mapping['Cluster_ID'].isin(subset_clusters)]
        centroid_subset = cluster_centroids_df[cluster_centroids_df['Cluster_ID'].isin(subset_clusters)]
        plot_subset_before_after(original_subset, centroid_subset, SUBSET_PLOT)
    else:
        print("Error: Feedstock data should be clustered for this analysis")
        return

    # Load plant locations
    plant_df = pd.read_csv(PLANT_CSV)

    # Compute distances before clustering (to individual nodes)
    unique_nodes = original_df[['GISCO_ID', 'Centroid_Lat', 'Centroid_Lon']].drop_duplicates('GISCO_ID')
    distances_before = compute_distances_to_nodes(plant_df, unique_nodes)
    
    # Compute distances after clustering (to cluster centroids)
    distances_after = compute_distances_to_centroids(plant_df, cluster_centroids_df)

     # Enhanced distribution analysis
    def compute_distribution_stats(distances, name):
        stats = {
            'Count': len(distances),
            'Mean (km)': np.mean(distances),
            'Median (km)': np.median(distances),
            'Variance (km²)': np.var(distances, ddof=1),
            'Std Dev (km)': np.std(distances, ddof=1),
            'Skewness': skew(distances),
            'Kurtosis': kurtosis(distances),
            'Min (km)': np.min(distances),
            '25th Percentile (km)': np.percentile(distances, 25),
            '75th Percentile (km)': np.percentile(distances, 75),
            'Max (km)': np.max(distances)
        }
        print(f"\nDistribution Statistics for {name}:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        return stats

    stats_before = compute_distribution_stats(distances_before, "Before Clustering")
    stats_after = compute_distribution_stats(distances_after, "After Clustering")

    # Plot distance distributions with statistical annotations
    plt.figure(figsize=(12, 8))
    sns.histplot(distances_before, bins=50, color='blue', label='Before Clustering', kde=True, alpha=0.5)
    sns.histplot(distances_after, bins=50, color='red', label='After Clustering', kde=True, alpha=0.5)
    
    # Add statistical annotations
    text_before = f"Before Clustering:\nCount: {stats_before['Count']:.0f}\nMean: {stats_before['Mean (km)']:.2f} km\nMedian: {stats_before['Median (km)']:.2f} km \nStd Dev: {stats_before['Std Dev (km)']:.2f}\nSkewness: {stats_before['Skewness']:.2f} \n Kurtosis: {stats_before['Kurtosis']:.2f}"
    text_after = f"After Clustering:\nCount: {stats_after['Count']:.0f}\nMean: {stats_after['Mean (km)']:.2f} km\nMedian: {stats_after['Median (km)']:.2f} km \nStd Dev: {stats_after['Std Dev (km)']:.2f} \nSkewness: {stats_after['Skewness']:.2f} \n Kurtosis: {stats_after['Kurtosis']:.2f} \n "
    plt.text(0.775, 0.8, text_before, transform=plt.gca().transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.775, 0.55, text_after, transform=plt.gca().transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('Distance Distribution Before and After Clustering', fontsize=14)
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(fontsize = 14)
    plt.tight_layout()
    plt.savefig(DISTANCE_PLOT, dpi=300)
    print(f"Saved distance distribution plot to {DISTANCE_PLOT}")
    plt.close()


    # Compute distance matrix
    compute_distance_matrix_to_cluster_centroids(cluster_centroids_df, PLANT_CSV, DISTANCE_CSV)

    # Plot feedstock nodes, cluster centroids, and plant locations
    feedstock_centroids = original_df[['GISCO_ID', 'Centroid_Lat', 'Centroid_Lon']].drop_duplicates('GISCO_ID')
    
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
    plt.close()

if __name__ == "__main__":
    main()