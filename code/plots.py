import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import LineString, Point
from collections import defaultdict
import pickle
import os
from shapely.ops import unary_union

BASE_DIR = "C:/Clone/Master/"
FILES = {
    "in_flow": os.path.join(BASE_DIR, "Output_in_flow_35_large.csv"),
    #"out_flow": os.path.join(BASE_DIR, "/Output_out_flow.csv"),
    "financials": os.path.join(BASE_DIR, "Output_financials_35_large.csv"),
    #"feedstock": os.path.join(BASE_DIR, "processed_biomass_data.csv"),
    "feedstock": os.path.join(BASE_DIR, "aggregated_bavaria_supply_nodes.csv"),
    "plant": os.path.join(BASE_DIR, "equally_spaced_locations.csv"),
    #"plant": os.path.join(BASE_DIR, "equally_space_locations_10.csv"),
    "yields": os.path.join(BASE_DIR, "Feedstock_yields.csv"),
    "bavaria_geojson": os.path.join(BASE_DIR, "bavaria_cluster_regions.geojson"),
    "supply_coords": os.path.join(BASE_DIR, "supply_coords.csv")
}

# Load data
in_flow_df = pd.read_csv(FILES["in_flow"])
#out_flow_df = pd.read_csv(FILES["out_flow"])
fin_df = pd.read_csv(FILES["financials"])
yields_df = pd.read_csv(FILES["yields"])
feedstock_df = pd.read_csv(FILES["feedstock"])
plant_df = pd.read_csv(FILES["plant"])

# Prepare coordinates
supply_coords = {row["GISCO_ID"]: (row["Centroid_Lon"], row["Centroid_Lat"]) 
                 for _, row in feedstock_df.iterrows()}
plant_coords = {row["Location"]: (row["Longitude"], row["Latitude"]) 
                for _, row in plant_df.iterrows()}
iPrime_coords = supply_coords.copy()

# Feedstock types
feedstock_types = yields_df["substrat_ENG"].unique().tolist()
avail_mass = {(row["GISCO_ID"], row["substrat_ENG"]): row["nutz_pot_tFM"] 
              for _, row in feedstock_df.iterrows()}

# System methane average (from script)
total_methane = sum(avail_mass[i, f] * yields_df[yields_df["substrat_ENG"] == f]["Methane_Content_%"].iloc[0] 
                    for i, f in avail_mass)
total_mass = sum(avail_mass[i, f] for i, f in avail_mass)
system_methane_average = total_methane / total_mass

def plot_methane_fraction(fin_df, system_methane_average):
    methane_fractions = []
    valid_plants = []
    for _, row in fin_df.iterrows():
        j = row["PlantLocation"]
        omega_val = row["Omega"]
        n_ch4_val = row["N_CH4"]
        if omega_val > 1e-6:
            fraction = n_ch4_val / omega_val if omega_val > 0 else 0
            methane_fractions.append(fraction)
            valid_plants.append(j)
    if not valid_plants:
        print("No plants with non-zero production for methane fraction plot.")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(valid_plants, methane_fractions, color="blue", s=100, label="Plant Methane Fraction")
    ax.axhline(y=system_methane_average, color="red", linestyle="--", linewidth=2, 
               label=f"System Average ({system_methane_average:.3f})")
    for j, frac in zip(valid_plants, methane_fractions):
        deviation = ((frac - system_methane_average) / system_methane_average) * 100
        ax.text(j, frac, f"{deviation:+.1f}%", fontsize=8, ha="center", 
                va="bottom" if frac < system_methane_average else "top")
    ax.set_xlabel("Plant Location")
    ax.set_ylabel("Methane Fraction (N_CH4 / Omega)")
    ax.set_title("Methane Fraction by Plant Location vs. System Average")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    min_frac = min(methane_fractions + [system_methane_average]) * 0.95
    max_frac = max(methane_fractions + [system_methane_average]) * 1.05
    ax.set_ylim(min_frac, max_frac)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "methane_fraction_plot.png"))
    plt.show()

def plot_feedstock_stacked_chart(in_flow_df, feedstock_types, color_map):
    flow_data = []
    for _, row in in_flow_df.iterrows():
        j, f, flow_val = row["PlantLocation"], row["Feedstock"], row["FlowTons"]
        if flow_val > 1e-6:
            flow_data.append({"Plant": j, "Feedstock": f, "FlowTons": flow_val})

    df = pd.DataFrame(flow_data)

    if df.empty:
        print("No feedstock flows to plot.")
        return

    pivot_df = df.pivot_table(index="Plant", columns="Feedstock", values="FlowTons", fill_value=0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100  # Convert to percentages

    for f in feedstock_types:
        if f not in pivot_df.columns:
            pivot_df[f] = 0.0

    pivot_df = pivot_df[feedstock_types]

    fig, ax = plt.subplots(figsize=(12, 8))
    plants = pivot_df.index
    bottoms = np.zeros(len(plants))

    for feedstock in feedstock_types:
        values = pivot_df[feedstock].values
        color = color_map.get(feedstock, None)  # Fallback to default if not specified
        ax.bar(plants, values, bottom=bottoms, label=feedstock, color=color)
        bottoms += values

    ax.set_xlabel("Plant Location")
    ax.set_ylabel("Percentage of Feedstock (%)")
    ax.set_title("Feedstock Composition per Plant (100% Stacked)")
    ax.legend(title="Feedstock Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "feedstock_stacked_chart.png"))
    plt.show()

def plot_cluster_heatmap(in_flow_df, yields_df, fin_df, plant_coords, supply_coords, geojson_path, output_png):
    # Load cluster polygons
    clusters_gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)
    
    BASE_DIR = "C:/Clone/Master/"
    gdf = gpd.read_file(os.path.join(BASE_DIR, "bavaria_lau_clean.geojson")).to_crs(epsg=4326)

    # Compute delivered methane per cluster
    merged = in_flow_df.merge(
        yields_df, left_on="Feedstock", right_on="substrat_ENG", how="left"
    )
    merged["DeliveredMethane"] = (
        merged["FlowTons"] *
        merged["Biogas_Yield_m3_ton"] *
        merged["Methane_Content_%"]
    )
    methane_sum = merged.groupby("SupplyNode", as_index=False)["DeliveredMethane"].sum()
    
    clusters_gdf = clusters_gdf.merge(
        methane_sum, left_on="GISCO_ID", right_on="SupplyNode", how="left"
    )
    clusters_gdf["DeliveredMethane"].fillna(0, inplace=True)
    clusters_gdf["Methane_for_plot"] = clusters_gdf["DeliveredMethane"].replace(0, np.nan)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    # Cluster regions with black border
    clusters_gdf.plot(
        ax=ax,
        column="Methane_for_plot",
        cmap="OrRd",
        edgecolor="grey",
        linewidth=0.4,
        alpha=0.5,
        legend=False,
        missing_kwds={"color": "lightgrey", "edgecolor":"black", "label":"No Flow"},
        zorder=1
    )
    ax.set_title("Delivered Methane by Cluster Region", fontsize=16)
        # 2) Dissolve into one geometry (the outer shell)
    bavaria_shell = unary_union(gdf.geometry)

    # wrap in a GeoSeries so GeoPandas can plot it
    gpd.GeoSeries([bavaria_shell], crs="EPSG:4326").plot(
        ax=ax,
        facecolor="none",     # transparent fill
        edgecolor="black",    # black outer border
        linewidth=0.75
    )
    # Colorbar
    vmin = clusters_gdf["Methane_for_plot"].min()
    vmax = clusters_gdf["Methane_for_plot"].max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="OrRd")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Delivered Methane (m³)", size=12)
    
    # Flow lines
    lines = in_flow_df.groupby(["SupplyNode","PlantLocation"], as_index=False)["FlowTons"].sum()
    for _, row in lines.iterrows():
        s, p = row["SupplyNode"], row["PlantLocation"]
        if s in supply_coords and p in plant_coords:
            x1, y1 = supply_coords[s]
            x2, y2 = plant_coords[p]
            ax.plot([x1, x2], [y1, y2], color="grey", linewidth=0.3, alpha=0.5, zorder=2)
    
    # Plant markers scaled by capacity
    caps = fin_df["Capacity"]
    min_c, max_c = caps.min(), caps.max()
    def size_scale(c): return 50 + 150*(c-min_c)/(max_c-min_c)
    
    alt_colors = {
        "FlexEEG_biogas":"blue",
        "Upgrading_tech1":"purple",
        # add other categories as needed
    }
    # Plot markers
    for _, r in fin_df.iterrows():
        if r.Alternative != "no_build":
            lon, lat = plant_coords[r.PlantLocation]
            ax.scatter(
                lon, lat, marker="^",
                color=alt_colors.get(r.Alternative, "black"),
                s=size_scale(r.Capacity),
                edgecolor="white", linewidth=0.5,
                zorder=3
            )
            ax.annotate(
                str(r.PlantLocation),
                xy=(lon, lat), xytext=(3,3),
                textcoords="offset points",
                fontsize=8, zorder=4
            )
    
    # Build legend handles
    handles, labels = [], []
    # Alternative legend
    for alt, col in alt_colors.items():
        handles.append(plt.Line2D([], [], marker="^", color=col, linestyle="",
                                  markersize=8, label=alt))
        labels.append(alt)
    # Capacity size legend
    for cap in [min_c, (min_c+max_c)/2, max_c]:
        handles.append(plt.Line2D([], [], marker="^", color="gray", linestyle="",
                                  markersize=np.sqrt(size_scale(cap)), label=f"{int(cap)} m³"))
        labels.append(f"{int(cap)} m³")
    
    ax.legend(handles, labels, title="Alternatives & Capacities",
              loc="upper left", bbox_to_anchor=(0.7, 1))
    
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.show()

def plot_bavaria_lau_highlight_with_labels(gisco_ids):
    BASE_DIR = "C:/Master_Python/"
    GEOJSON_PATH = os.path.join(BASE_DIR, "bavaria_lau_clean.geojson")
    
    if not os.path.exists(GEOJSON_PATH):
        print("GeoJSON file not found. Skipping LAU highlight plot.")
        return
    
    bavaria_gdf = gpd.read_file(GEOJSON_PATH)
    bavaria_gdf = bavaria_gdf.to_crs(epsg=4326)
    
    # Create column to indicate highlighted LAUs
    bavaria_gdf["Highlight"] = bavaria_gdf["GISCO_ID"].isin(gisco_ids)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    bavaria_gdf.plot(
        ax=ax,
        color=bavaria_gdf["Highlight"].map({True: "red", False: "lightgray"}),
        edgecolor="black",
        alpha=0.6
    )
    
    # Add GISCO_ID labels at LAU centroids
    for idx, row in bavaria_gdf.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(
            text=row["GISCO_ID"],
            xy=(centroid.x, centroid.y),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=6,
            color="black",
            ha="center",
            va="center"
        )
    
    ax.set_title("Bavaria LAU Regions with Highlighted GISCO_IDs")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "bavaria_lau_highlight_labeled_plot.png"))
    plt.show()

gisco_ids = ["DE_0967113"]
# Generate plots
color_map = {
"cattle_man": "sienna",
"cattle_slu": "chocolate",
"horse_man": "rosybrown",
"pig_slu": "lightpink",
"pig_man": "pink",
"cereal_str": "gold",
"clover_alfalfa_grass": "seagreen",
"perm_grass": "lawngreen",
"maize_str": "olive",
"beet_leaf": "purple",
"rape_str": "teal",  # reuse or adjust if you run out of distinct colors
"legume_str": "blue"  # reuse or adjust
}


#plot_methane_fraction(fin_df, system_methane_average)
#plot_feedstock_stacked_chart(in_flow_df, feedstock_types, color_map)
plot_cluster_heatmap(in_flow_df, yields_df, fin_df, plant_coords, supply_coords,FILES["bavaria_geojson"], os.path.join(BASE_DIR, "cluster_heatmap.png"))
#plot_bavaria_lau_highlight_with_labels(gisco_ids)