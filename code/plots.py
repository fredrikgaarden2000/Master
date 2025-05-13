import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import LineString, Point
from collections import defaultdict
import pickle
import os

BASE_DIR = "C:/Clone/Master/"
FILES = {
    "in_flow": os.path.join(BASE_DIR, "Solutions/10/Output_in_flow_warm_start.csv"),
    "out_flow": os.path.join(BASE_DIR, "Solutions/10/Output_out_flow_warm_start.csv"),
    "financials": os.path.join(BASE_DIR, "Solutions/10/Output_financials_warm_start.csv"),
    "feedstock": os.path.join(BASE_DIR, "processed_biomass_data.csv"),
    "plant": os.path.join(BASE_DIR, "equally_spaced_locations.csv"),
    #"plant": os.path.join(BASE_DIR, "equally_space_locations_10.csv"),
    "yields": os.path.join(BASE_DIR, "Feedstock_yields.csv"),
    "bavaria_geojson": os.path.join(BASE_DIR, "bavaria_lau_clean.geojson"),
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


def plot_geojson_map(in_flow_df, yields_df, fin_df, plant_coords, supply_coords):
    BASE_DIR = "C:/Master_Python/"
    GEOJSON_PATH = os.path.join(BASE_DIR, "bavaria_lau_clean.geojson")
    
    if not os.path.exists(GEOJSON_PATH):
        print("GeoJSON file not found. Skipping GeoJSON map plot.")
        return
    
    bavaria_gdf = gpd.read_file(GEOJSON_PATH)
    bavaria_gdf = bavaria_gdf.to_crs(epsg=4326)
    
    merged_df = in_flow_df.merge(
        yields_df,
        left_on="Feedstock",
        right_on="substrat_ENG",
        how="left"
    )
    merged_df["DeliveredMethane_m3"] = (
        merged_df["FlowTons"] *
        merged_df["Biogas_Yield_m3_ton"] *
        merged_df["Methane_Content_%"]
    )
    lau_methane_df = merged_df.groupby("SupplyNode", as_index=False)["DeliveredMethane_m3"].sum()
    lau_methane_df.rename(columns={"DeliveredMethane_m3": "CalculatedMethane", "SupplyNode": "GISCO_ID"}, inplace=True)
    
    bavaria_gdf = bavaria_gdf.merge(lau_methane_df, on="GISCO_ID", how="left")
    bavaria_gdf["CalculatedMethane"] = bavaria_gdf["CalculatedMethane"].fillna(0)
    bavaria_gdf["Methane_for_plot"] = bavaria_gdf["CalculatedMethane"].replace(0, np.nan)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = "OrRd"
    bavaria_gdf.plot(
        ax=ax,
        column="Methane_for_plot",
        cmap=cmap,
        edgecolor="black",
        alpha=0.6,
        legend=False,
        missing_kwds={
            "color": "white",
            "edgecolor": "black",
            "label": "No Delivered Methane"
        }
    )
    ax.set_title("Delivered Methane by LAU (Feedstock in_flow)")
    
    data_col = bavaria_gdf["Methane_for_plot"].dropna()
    if not data_col.empty:
        vmin, vmax = data_col.min(), data_col.max()
    else:
        vmin, vmax = 0, 1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Delivered Methane (m³)")
    
    lines_agg = in_flow_df.groupby(["SupplyNode", "PlantLocation"], as_index=False)["FlowTons"].sum()
    for _, row in lines_agg.iterrows():
        s_node = row["SupplyNode"]
        p_loc = row["PlantLocation"]
        if s_node not in supply_coords or p_loc not in plant_coords:
            continue
        (lon1, lat1) = supply_coords[s_node]
        (lon2, lat2) = plant_coords[p_loc]
        line = LineString([(lon1, lat1), (lon2, lat2)])
        ax.plot(*line.xy, color="black", linewidth=0.5, alpha=0.8)
    
    alt_to_color = {
        "boiler": "red",
        "nonEEG_CHP": "blue",
        "EEG_CHP_small1": "lightgreen",
        "EEG_CHP_small2": "lightgreen",
        "EEG_CHP_large1": "green",
        "EEG_CHP_large2": "green",
        "FlexEEG_biogas": "magenta",
        "FlexEEG_biomethane_tech1": "pink",
        "FlexEEG_biomethane_tech2": "pink",
        "FlexEEG_biomethane_tech3": "pink",
        "FlexEEG_biomethane_tech4": "pink",
        "FlexEEG_biomethane_tech5": "pink",
        "Upgrading_tech1": "purple",
        "Upgrading_tech2": "purple",
        "Upgrading_tech3": "purple",
        "Upgrading_tech4": "purple",
        "Upgrading_tech5": "purple",
        "no_build": None
    }
    
    capacity_levels = fin_df["Capacity"].unique()
    min_capacity = min(capacity_levels)
    max_capacity = max(capacity_levels)
    min_size = 50
    max_size = 200
    def scale_size(capacity):
        if max_capacity == min_capacity:
            return min_size
        return min_size + (max_size - min_size) * (capacity - min_capacity) / (max_capacity - min_capacity)
    
    plant_points = []
    for _, row in fin_df.iterrows():
        j = row["PlantLocation"]
        alt = row["Alternative"]
        c = row["Capacity"]
        if alt != "no_build" and j in plant_coords:
            lon, lat = plant_coords[j]
            plant_points.append({
                "PlantLocation": j,
                "geometry": Point(lon, lat),
                "Alternative": alt,
                "Capacity": c
            })
    
    if plant_points:
        plant_gdf = gpd.GeoDataFrame(plant_points, crs="EPSG:4326")
        for _, row in plant_gdf.iterrows():
            ax.scatter(
                row.geometry.x,
                row.geometry.y,
                marker="^",
                color=alt_to_color[row["Alternative"]],
                s=scale_size(row["Capacity"]),
                label=row["Alternative"],
                alpha=0.8,
                zorder=10
            )
            ax.annotate(
                text=row["PlantLocation"],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                color="black",
                zorder=11
            )
    
        handles = []
        labels = []
        plotted_alts = set()
        for _, row in plant_gdf.iterrows():
            alt = row["Alternative"]
            if alt not in plotted_alts and alt_to_color[alt] is not None:
                handles.append(plt.scatter([], [], color=alt_to_color[alt], marker="^", s=min_size, label=alt))
                labels.append(alt)
                plotted_alts.add(alt)
        for cap in [min_capacity, sum(capacity_levels)/len(capacity_levels), max_capacity]:
            size = scale_size(cap)
            handles.append(plt.scatter([], [], color="gray", marker="^", s=size, label=f"{int(cap)} m³/year"))
            labels.append(f"Capacity: {int(cap)} m³/year")
        ax.legend(handles, labels, title="Alternatives & Capacities", loc="upper left", bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "methane_map_plot.png"))
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
plot_feedstock_stacked_chart(in_flow_df, feedstock_types, color_map)
plot_geojson_map(in_flow_df, yields_df, fin_df, plant_coords, supply_coords)
#plot_bavaria_lau_highlight_with_labels(gisco_ids)