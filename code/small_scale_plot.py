import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.ops import unary_union
from matplotlib import colors, cm

# --- CONFIGURATION ---
BASE_DIR = "C:/Clone/Master/"
PLANT_FILE = os.path.join(BASE_DIR, "equally_spaced_locations_100.csv")
IN_FLOW_FILE_THIRD = os.path.join(BASE_DIR, "results/small_scale_normal/Output_in_flow.csv")
GEOJSON_PATH = os.path.join(BASE_DIR, "bavaria_cluster_regions.geojson")
LAU_SHELL = os.path.join(BASE_DIR, "bavaria_lau_clean.geojson")

SCENARIOS = [
    {"label": "Base case",     "fin_file": "Output_financials_base.csv",    "color": "red"},
    {"label": "+15 ct/kWh",    "fin_file": "Output_financials_plus15.csv",  "color": "blue"},
    {"label": "+30 ct/kWh",    "fin_file": "Output_financials_plus30.csv",  "color": "green"},
]
# Replace the fin_file values above with the actual filenames.

# --- LOAD COMMON DATA ---
# Plant candidates
plant_df   = pd.read_csv(PLANT_FILE)
plant_coords = {row["Location"]:(row["Longitude"], row["Latitude"]) for _, row in plant_df.iterrows()}

# Cluster regions
clusters_gdf = gpd.read_file(GEOJSON_PATH).to_crs(epsg=4326)
lau_gdf      = gpd.read_file(LAU_SHELL).to_crs(epsg=4326)
bavaria_shell = unary_union(lau_gdf.geometry)

# Prepare plotting
fig, ax = plt.subplots(figsize=(12,10))
# plot clusters with empty fill
clusters_gdf.plot(ax=ax, facecolor="none", edgecolor="grey", linewidth=0.4, zorder=1)
# plot Bavaria boundary
gpd.GeoSeries([bavaria_shell],crs="EPSG:4326").plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.75, zorder=2)

# --- PLOT SCENARIOS ---
handles, labels = [], []
all_locations = set(plant_coords.keys())

for scen in SCENARIOS:
    # load financials
    fin_df = pd.read_csv(os.path.join(BASE_DIR, "results/small_scale_normal", scen["fin_file"]))
    built = set(fin_df["PlantLocation"])
    no_build = all_locations - built

    # 1) plot no-builds (only once for first scenario in legend)
    if scen is SCENARIOS[0]:
        hb = ax.scatter(
            [plant_coords[p][0] for p in no_build],
            [plant_coords[p][1] for p in no_build],
            marker='x', s=100, facecolor='none',
            edgecolor='grey', linewidth=1.2, alpha=0.7, zorder=3
        )
        handles.append(hb); labels.append("No-build")
    # 2) plot built
    hb = ax.scatter(
        [plant_coords[p][0] for p in built],
        [plant_coords[p][1] for p in built],
        marker='^', s=150,
        color=scen["color"], edgecolor="white",
        linewidth=0.5, alpha=0.9, zorder=4
    )
    handles.append(hb); labels.append(scen["label"])

# --- PLOT THIRD SCENARIO FLOWS ---
in_flow = pd.read_csv(IN_FLOW_FILE_THIRD)
# we assume supply_coords loaded similarly to original code
# here we build a quick supply_coords from feedstock file
feedstock_df = pd.read_csv(os.path.join(BASE_DIR, "aggregated_bavaria_supply_nodes.csv"))
supply_coords = {r["GISCO_ID"]:(r["Centroid_Lon"],r["Centroid_Lat"]) for _,r in feedstock_df.iterrows()}

lines = in_flow.groupby(["SupplyNode","PlantLocation"],as_index=False)["FlowTons"].sum()
for _, row in lines.iterrows():
    s, p = row["SupplyNode"], row["PlantLocation"]
    if s in supply_coords and p in plant_coords:
        x1,y1 = supply_coords[s]
        x2,y2 = plant_coords[p]
        ax.plot([x1,x2],[y1,y2], color="grey", linewidth=0.4, alpha=0.4, zorder=2)

# --- FINALIZE ---
ax.set_axis_off()
ax.legend(handles, labels, title="Scenario / Status", loc="upper left", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "scenario_comparison_map.png"), dpi=300)
plt.show()
