import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import geopandas as gpd
from shapely.geometry import Point

# Parameters
N_CLUSTERS = 750
BASE_DIR = "C:/Clone/Master/"
OUTPUT_CSV = "aggregated_bavaria_supply_nodes.csv"
PLOT_OUTPUT = "bavaria_clustered_centroids.png"

# Load data
try:
    feedstock_df = pd.read_csv(f"{BASE_DIR}processed_biomass_data.csv")
    yields_df = pd.read_csv(f"{BASE_DIR}Feedstock_yields.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure processed_biomass_data.csv and Feedstock_yields.csv are in {BASE_DIR}")
    exit(1)

# Filter for Bavaria (GISCO_ID starts with DE_09)
df = feedstock_df[feedstock_df['GISCO_ID'].str.startswith('DE_09')].copy()

# Merge yield info
df = df.merge(yields_df[["substrat_ENG", "Biogas_Yield_m3_ton", "Methane_Content_%"]],
              on="substrat_ENG", how="left")

# Clean data
df = df.dropna(subset=["Centroid_Lat", "Centroid_Lon", "nutz_pot_tFM", "Biogas_Yield_m3_ton", "Methane_Content_%"])
df = df[(df["Centroid_Lat"].between(-90, 90)) & (df["Centroid_Lon"].between(-180, 180))]

# Compute biogas potential
df["Biogas_Potential_m3"] = df["nutz_pot_tFM"] * df["Biogas_Yield_m3_ton"] * (df["Methane_Content_%"])

# Aggregate to GISCO_ID level for clustering
agg = df.groupby(["GISCO_ID", "Centroid_Lat", "Centroid_Lon"]).agg(
    Total_Biogas_Potential=("Biogas_Potential_m3", "sum")
).reset_index()

# Create GeoDataFrame and project to UTM (EPSG:25832) for clustering
gdf = gpd.GeoDataFrame(
    agg,
    geometry=gpd.points_from_xy(agg["Centroid_Lon"], agg["Centroid_Lat"]),
    crs="EPSG:4326"
).to_crs("EPSG:25832")

# Perform Agglomerative Clustering
coords = np.stack([gdf.geometry.x, gdf.geometry.y], axis=1)
clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
gdf["Cluster_ID"] = clustering.fit_predict(coords)

# Merge cluster assignments back to original data
gdf = gdf[["GISCO_ID", "Cluster_ID"]]
df = df.merge(gdf, on="GISCO_ID", how="left")

# Compute cluster centroids (weighted by Biogas_Potential_m3)
cluster_coords = (
    df.groupby("Cluster_ID")
      .apply(lambda x: pd.Series({
          "Centroid_Lat_cluster": np.average(x["Centroid_Lat"], weights=x["Biogas_Potential_m3"]),
          "Centroid_Lon_cluster": np.average(x["Centroid_Lon"], weights=x["Biogas_Potential_m3"]),
      }), include_groups=False)
      .reset_index()
)

# Merge cluster coordinates and update original coordinates
df = df.merge(cluster_coords, on="Cluster_ID", how="left")
df["Centroid_Lat"] = df["Centroid_Lat_cluster"]
df["Centroid_Lon"] = df["Centroid_Lon_cluster"]

# Update GISCO_ID to Cluster_ID and clean up
df["GISCO_ID"] = df["Cluster_ID"].apply(lambda x: f"CLUSTER_{int(x)}")
df = df.drop(columns=["Cluster_ID", "Centroid_Lat_cluster", "Centroid_Lon_cluster", "Biogas_Yield_m3_ton", 
                      "Methane_Content_%", "Biogas_Potential_m3"])

# Aggregate by Cluster_ID and substrat_ENG, summing nutz_pot_tFM
final_df = df.groupby(["GISCO_ID", "substrat_ENG", "Centroid_Lat", "Centroid_Lon"]).agg(
    nutz_pot_tFM=("nutz_pot_tFM", "sum"),
    gem_code=("gem_code", "first"),
    Modified_GEM_CODE=("Modified_GEM_CODE", "first"),
    LAU_NAME=("LAU_NAME", "first")
).reset_index()

# Ensure output matches original columns
final_df = final_df[["GISCO_ID", "substrat_ENG", "nutz_pot_tFM", "gem_code", "Modified_GEM_CODE", "LAU_NAME", 
                     "Centroid_Lon", "Centroid_Lat"]]

# Save to CSV
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved aggregated data to {OUTPUT_CSV}")

# Plot cluster centroids in Bavaria (simple scatter plot)
fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(cluster_coords["Centroid_Lon_cluster"], cluster_coords["Centroid_Lat_cluster"], 
            c="red", s=5, alpha=0.8)

# Set bounds to focus on Bavaria
ax.set_xlim(9, 13.8)  # Approximate longitude range for Bavaria
ax.set_ylim(47.3, 50.6)  # Approximate latitude range for Bavaria

# Customize plot
ax.set_title(f"Clustered Biomass Supply Nodes in Bavaria ({N_CLUSTERS} Clusters)", fontsize=14)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)

# Save and show plot
plt.tight_layout()
plt.savefig(PLOT_OUTPUT, dpi=300, bbox_inches="tight")
print(f"Saved plot to {PLOT_OUTPUT}")
plt.show()


# Load data
try:
    feedstock_df = pd.read_csv(f"{BASE_DIR}processed_biomass_data.csv")
except FileNotFoundError:
    print(f"Error: processed_biomass_data.csv not found in {BASE_DIR}")
    exit(1)

try:
    yields_df = pd.read_csv(f"{BASE_DIR}Feedstock_yields.csv")
except FileNotFoundError:
    print(f"Warning: Feedstock_yields.csv not found. Using default yield values.")
    # Create a placeholder yields_df based on typical substrates
    yields_df = pd.DataFrame({
        'substrat_ENG': feedstock_df['substrat_ENG'].unique(),
        'Biogas_Yield_m3_ton': [500] * feedstock_df['substrat_ENG'].nunique(),  # Typical value
        'Methane_Content_%': [50] * feedstock_df['substrat_ENG'].nunique()      # Typical value
    })

# Filter for Bavaria (GISCO_ID starts with DE_09)
df = feedstock_df[feedstock_df['GISCO_ID'].str.startswith('DE_09')].copy()

# Merge yield info
df = df.merge(yields_df[["substrat_ENG", "Biogas_Yield_m3_ton", "Methane_Content_%"]],
              on="substrat_ENG", how="left")

# Clean data
df = df.dropna(subset=["Centroid_Lat", "Centroid_Lon", "nutz_pot_tFM", "Biogas_Yield_m3_ton", "Methane_Content_%"])
df = df[(df["Centroid_Lat"].between(-90, 90)) & (df["Centroid_Lon"].between(-180, 180))]

# Compute biogas potential
df["Biogas_Potential_m3"] = df["nutz_pot_tFM"] * df["Biogas_Yield_m3_ton"] * (df["Methane_Content_%"])

# Calculate total biogas potential
total_biogas_m3 = df["Biogas_Potential_m3"].sum()

# Print result
print(f"Total biogas potential in Bavaria: {total_biogas_m3:,.2f} m³")