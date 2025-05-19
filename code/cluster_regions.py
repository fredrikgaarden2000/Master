import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import numpy as np
import os

# 1) Paths — adjust as needed
BASE_DIR        = "C:/Clone/Master/"
CENTROIDS_CSV   = os.path.join(BASE_DIR, "aggregated_bavaria_supply_nodes.csv")
BAVARIA_GEOJSON = os.path.join(BASE_DIR, "bavaria_lau_clean.geojson")
OUTPUT_REGIONS  = os.path.join(BASE_DIR, "bavaria_cluster_regions.geojson")

# 2) Load the centroids and reset index
centroids_df = (
    pd.read_csv(CENTROIDS_CSV)
      .drop_duplicates(subset=["GISCO_ID", "Centroid_Lon", "Centroid_Lat"])
      .reset_index(drop=True)            # <— important!
)
centroids_gdf = gpd.GeoDataFrame(
    centroids_df[["GISCO_ID"]],
    geometry=gpd.points_from_xy(
        centroids_df.Centroid_Lon, centroids_df.Centroid_Lat
    ),
    crs="EPSG:4326"
)

# 3) Read Bavaria outline & dissolve
bavaria = gpd.read_file(BAVARIA_GEOJSON).to_crs(epsg=4326)
bavaria_poly = unary_union(bavaria.geometry)

# 4) Compute Voronoi on the centroid coordinates
pts = np.column_stack([centroids_gdf.geometry.x, centroids_gdf.geometry.y])
vor = Voronoi(pts)

# 5) Build and clip each Voronoi cell
regions = []
for i in range(len(pts)):
    region_index = vor.point_region[i]
    vert_indices = vor.regions[region_index]
    if -1 in vert_indices or not vert_indices:
        # skip unbounded regions
        continue
    poly = Polygon(vor.vertices[vert_indices])
    clipped = poly.intersection(bavaria_poly)
    if not clipped.is_empty:
        regions.append({
            "GISCO_ID": centroids_gdf.iloc[i]["GISCO_ID"],
            "geometry": clipped
        })

# 6) Save as GeoJSON
regions_gdf = gpd.GeoDataFrame(regions, crs="EPSG:4326")
regions_gdf.to_file(OUTPUT_REGIONS, driver="GeoJSON")
print(f"Wrote cluster regions to {OUTPUT_REGIONS}")

# 7) Plot to verify
fig, ax = plt.subplots(figsize=(10,10))
# Plot the Bavaria boundary (no fill) in black
gpd.GeoSeries(bavaria_poly).plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    linewidth=1.0,
    zorder=1
)

regions_gdf.plot(
    ax=ax,
    column="GISCO_ID",
    cmap="tab20",
    alpha=0.6,
    edgecolor="black",     # <-- black outline
    linewidth=0.5          # <-- a bit bolder
)
centroids_gdf.plot(
    ax=ax,
    color="red",
    markersize=5,
    label="Centroids"
)
ax.set_title("Cluster-based Voronoi Regions in Bavaria", fontsize=16)
ax.set_axis_off()
ax.legend()
plt.tight_layout()
plt.show()
