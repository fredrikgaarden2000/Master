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
IN_FLOW_FILE_30 = os.path.join(BASE_DIR, "results/small_scale/small_scale_30/Output_in_flow.csv")
GEOJSON_PATH = os.path.join(BASE_DIR, "bavaria_cluster_regions.geojson")
LAU_SHELL   = os.path.join(BASE_DIR, "bavaria_lau_clean.geojson")
FEEDSTOCK   = os.path.join(BASE_DIR, "aggregated_bavaria_supply_nodes.csv")

SCENARIOS = [
    {"label": "Base case",  "fin_file": "results/small_scale/small_scale_normal/Output_financials.csv", "color": "red"},
    {"label": "+15 ct/kWh", "fin_file": "results/small_scale/small_scale_15/Output_financials.csv",    "color": "blue"},
    {"label": "+30 ct/kWh", "fin_file": "results/small_scale/small_scale_30/Output_financials.csv",    "color": "green"},
]

# --- LOAD COMMON DATA ---
plant_df      = pd.read_csv(PLANT_FILE)
plant_coords  = {r["Location"]:(r["Longitude"],r["Latitude"]) for _,r in plant_df.iterrows()}

feedstock_df  = pd.read_csv(FEEDSTOCK)
supply_coords = {r["GISCO_ID"]:(r["Centroid_Lon"],r["Centroid_Lat"]) for _,r in feedstock_df.iterrows()}

clusters_gdf  = gpd.read_file(GEOJSON_PATH).to_crs(epsg=4326)
lau_gdf       = gpd.read_file(LAU_SHELL).to_crs(epsg=4326)
bavaria_shell = unary_union(lau_gdf.geometry)

# --- COMPUTE DeliveredMethane by SupplyNode for +30 scenario ---
in30 = (pd.read_csv(IN_FLOW_FILE_30)
        .merge(pd.read_csv(os.path.join(BASE_DIR, "Feedstock_yields.csv"))
               .rename(columns={"substrat_ENG":"Feedstock"}),
               on="Feedstock", how="left"))
in30["DeliveredMethane"] = in30["FlowTons"] * in30["Biogas_Yield_m3_ton"] * in30["Methane_Content_%"]
meth_by_supply = (in30.groupby("SupplyNode", as_index=False)["DeliveredMethane"]
                      .sum().rename(columns={"DeliveredMethane":"SupplyMethane"}))

# attach to clusters
clusters_gdf = clusters_gdf.merge(
    meth_by_supply, left_on="GISCO_ID", right_on="SupplyNode", how="left"
)
clusters_gdf["SupplyMethane"].fillna(0, inplace=True)
clusters_gdf["PlotMethane"] = clusters_gdf["SupplyMethane"].replace(0, np.nan)

# --- START PLOTTING ---
fig, ax = plt.subplots(figsize=(12,10))

# 1) cluster fill
clusters_gdf.plot(
    ax=ax, column="PlotMethane", cmap="OrRd", 
    edgecolor="grey", linewidth=0.4, alpha=0.8,
    missing_kwds={"color":"none","edgecolor":"black","label":"No Flow"},
    legend=False, zorder=1
)
# 2) bavaria boundary
gpd.GeoSeries([bavaria_shell], crs="EPSG:4326")\
   .plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.75, zorder=2)

# --- NO-BUILDS from +30 scenario only ---
fin30    = pd.read_csv(os.path.join(BASE_DIR, SCENARIOS[2]["fin_file"]))
built30  = set(fin30["PlantLocation"])
all_plants = set(plant_coords)
no_build30 = sorted(all_plants - built30)
ax.scatter(
    [plant_coords[p][0] for p in no_build30],
    [plant_coords[p][1] for p in no_build30],
    marker='x', s=125, facecolor='black',
    edgecolor='grey', linewidth=1.5, alpha=0.7, zorder=4,
    label="No‐build"
)

# --- PLOT SCENARIOS in back‐to‐front order (+30 → +15 → Base) ---
for scen in SCENARIOS[::-1]:
    fin   = pd.read_csv(os.path.join(BASE_DIR, scen["fin_file"]))
    built = set(fin["PlantLocation"])
    xs    = [plant_coords[p][0] for p in built]
    ys    = [plant_coords[p][1] for p in built]
    ax.scatter(
        xs, ys,
        marker='^', s=150,
        color=scen["color"], edgecolor="white",
        linewidth=0.5, alpha=1.0,
        zorder=5 if scen["label"]=="Base case" else 4,
        label=f"{scen['label']} ({len(built)})"
    )

# --- FLOW LINES for +30 only (greyscale fade + end-dot) ---
lines = in30.groupby(["SupplyNode","PlantLocation"], as_index=False)["FlowTons"].sum()
# normalize distances for color
def hav(lon1,lat1,lon2,lat2):
    R=6371.0; φ1,φ2=np.radians(lat1),np.radians(lat2)
    dφ,dλ=np.radians(lat2-lat1),np.radians(lon2-lon1)
    a=np.sin(dφ/2)**2+np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
lines["Dist"] = lines.apply(lambda r: hav(*supply_coords[r.SupplyNode], *plant_coords[r.PlantLocation]), axis=1)
vmin,vmax = lines["Dist"].min(), lines["Dist"].max()
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.get_cmap("Greys_r")

for _,r in lines.iterrows():
    s,p = r.SupplyNode, r.PlantLocation
    lon1,lat1 = supply_coords[s]
    lon2,lat2 = plant_coords[p]
    col = cmap(norm(r.Dist))
    lw  = 1.2*(1-norm(r.Dist))
    ax.plot([lon1,lon2],[lat1,lat2], color=col, linewidth=lw, alpha=0.8, zorder=3)

seen_sources = set()          # avoid double plotting
for _, row in lines.iterrows():
    s, p = row["SupplyNode"], row["PlantLocation"]

    if s in supply_coords and p in plant_coords:
        x1, y1 = supply_coords[s]
        x2, y2 = plant_coords[p]

        # the line itself
        ax.plot([x1, x2], [y1, y2],
                color="grey", linewidth=0.4, alpha=0.5, zorder=2)

        # ----------  NEW: mark the supply centroid  --------------
        if s not in seen_sources:               # plot each only once
            ax.scatter(x1, y1,
                        s=12,                    # small filled circle
                        color="grey",
                        edgecolor="white",
                        linewidth=0.3,
                        zorder=3)
            seen_sources.add(s)

# --- FINALIZE ---
ax.set_axis_off()
ax.legend(title="Effects on tariff increase", bbox_to_anchor=(0.75,0.75), fontsize = 12, title_fontsize = 14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "scenario_comparison_map.png"), dpi=300)
plt.show()
