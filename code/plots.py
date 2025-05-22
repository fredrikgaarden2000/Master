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
from matplotlib import cm, colors

BASE_DIR = "C:/Clone/Master/"
FILES = {
    "in_flow": os.path.join(BASE_DIR, "results/small_scale/small_scale_30/Output_in_flow.csv"),
    #"out_flow": os.path.join(BASE_DIR, "/Output_out_flow.csv"),
    "financials": os.path.join(BASE_DIR, "results/small_scale/small_scale_30/Output_financials.csv"),
    #"feedstock": os.path.join(BASE_DIR, "processed_biomass_data.csv"),
    "feedstock": os.path.join(BASE_DIR, "aggregated_bavaria_supply_nodes.csv"),
    "plant": os.path.join(BASE_DIR, "equally_spaced_locations_100.csv"),
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
        ax.text(j, frac, f"{deviation:+.1f}%", fontsize=12, ha="center", 
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

def plot_cluster_heatmap(in_flow_df, yields_df, fin_df,
                         plant_coords, supply_coords,
                         geojson_path, output_png):
        # -----------------------------------------------------------
    # build legend handles
    # -----------------------------------------------------------
    # ----------------  styling dictionaries  ------------------
    alt_colors = {
        "FlexEEG_biogas"       : "blue",
        "Upgrading_tech1"      : "purple",
        "nonEEG_CHP"           : "orange",
        "FlexEEG_biomethane"   : "green",
        "EEG_CHP_large1"    : "red",
        "EEG_CHP_large2" : "pink",
        "EEG_CHP_small1"   : "magenta",
        "EEG_CHP_small2"    : "lightblue",
        "boiler" : "black",


    }

    # NEW: map internal names → pretty legend labels
    alt_labels = {
        "FlexEEG_biogas"      : "Flex-EEG (biogas)",
        "Upgrading_tech1"     : "Upgrading",
        "nonEEG_CHP"          : "CHP (no EEG)",
        "FlexEEG_biomethane"  : "Flex-EEG (biomethane)",
        "EEG_CHP_large1"    : "150kw EEG Manure",
        "EEG_CHP_large2" : "150kw EEG Manure + Clover",        
        "EEG_CHP_small1"   : "75kw EEG Manure",
        "EEG_CHP_small2"    : "75kw EEG Manure + Clover",
        "boiler" : "Boiler", 
    }

    # -----------------------------------------------------------
    # 0)  set up the figure *first*
    # -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))

    # -----------------------------------------------------------
    # 1)  build & colour the flow-lines  (needs ax)
    # -----------------------------------------------------------
    lines = (in_flow_df.groupby(["SupplyNode", "PlantLocation"], as_index=False)
                         .agg({"FlowTons": "sum"}))

    # Great-circle length [km]
    def haversine(lon1, lat1, lon2, lat2):
        R = 6371.0
        φ1, φ2 = np.radians(lat1), np.radians(lat2)
        dφ, dλ = φ2 - φ1, np.radians(lon2 - lon1)
        a = (np.sin(dφ/2)**2 +
             np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2)
        return 2*R*np.arcsin(np.sqrt(a))

    lengths = []
    for _, row in lines.iterrows():
        s, p = row["SupplyNode"], row["PlantLocation"]
        if s in supply_coords and p in plant_coords:
            lon1, lat1 = supply_coords[s]
            lon2, lat2 = plant_coords[p]
            lengths.append(haversine(lon1, lat1, lon2, lat2))
        else:
            lengths.append(np.nan)
    lines["SegLen_km"] = lengths
    lines.dropna(subset=["SegLen_km"], inplace=True)

    # normalise & colour
    vmin, vmax = lines["SegLen_km"].min(), lines["SegLen_km"].max()
    norm  = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap  = mpl.colormaps["Greys_r"]          # light-grey → black

    seen_sources = set()
    for _, row in lines.iterrows():
        s, p = row["SupplyNode"], row["PlantLocation"]
        lon1, lat1 = supply_coords[s]
        lon2, lat2 = plant_coords[p]

        col = cmap(norm(row["SegLen_km"]))
        lw  = 1.2*(1 - norm(row["SegLen_km"]))   # thicker if shorter

        ax.plot([lon1, lon2], [lat1, lat2],
                color=col, linewidth=lw, alpha=0.9, zorder=2)

        if s not in seen_sources:               # centroid marker once
            ax.scatter(lon1, lat1, s=6, color=col,
                       edgecolor="white", linewidth=0.3, zorder=3)
            seen_sources.add(s)

    # -----------------------------------------------------------
    # 2)  draw cluster polygons & rest of the map  (uses same ax)
    # -----------------------------------------------------------
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
    
    lines = (in_flow_df.groupby(["SupplyNode", "PlantLocation"],
                                as_index=False)["FlowTons"]
                            .sum())

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

    # Plant markers scaled by capacity
    caps = fin_df["Capacity"]
    min_c, max_c = caps.min(), caps.max()
    if max_c == min_c:
        # all capacities equal → use constant marker size
        def size_scale(c):
            return 200
    else:
        def size_scale(c):
            return 150 + 150 * (c - min_c) / (max_c - min_c)


    # 1) All 75 candidate locations
    all_plants = set(plant_df['Location'])
    #print(f"DEBUG: total candidates = {len(all_plants)}")

    # 2) Those that actually got built
    built_plants = set(fin_df['PlantLocation'])
    #print(f"DEBUG: built plants = {built_plants}")

    # 3) The remainder are “no build”
    no_builds = sorted(all_plants - built_plants)
    #print(f"DEBUG: no-build count = {len(no_builds)}, list = {no_builds}")

    # 4) Plot them as transparent grey X’s
    for loc in no_builds:
        if loc not in plant_coords:
            #print(f"WARNING: {loc} missing from plant_coords!")
            continue
        lon, lat = plant_coords[loc]
        #print(f"  plotting no-build at {loc}: ({lon:.3f}, {lat:.3f})")
        ax.scatter(
            lon, lat,
            marker='x',
            s=200,
            facecolor='black',
            edgecolor='grey',
            linewidth=2,
            alpha=0.7,
            zorder=3
        )

    # Now plot built plants and annotate
    for _, r in fin_df.iterrows():
        lon, lat = plant_coords[r.PlantLocation]
        if r.Alternative != "no_build":
            ax.scatter(
                lon, lat,
                marker="^",
                color=alt_colors.get(r.Alternative, "black"),
                s=size_scale(r.Capacity),
                edgecolor="white",
                linewidth=0.5,
                zorder=4
            )
        ax.annotate(
            str(r.PlantLocation),
            xy=(lon, lat), xytext=(4,4),
            textcoords="offset points",
            fontsize=8, zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.8
            )
        )

    # -----------------------------------------------------------
    # build legend handles
    # -----------------------------------------------------------
    handles, labels = [], []

    # no-build legend entry
    handles.append( plt.Line2D([], [], marker='x', color='black',
                               linestyle='', markersize=8,
                               markeredgewidth=1.2) )
    labels.append("No‐build location")

    # 1) alternative-type legend  (colour only, fixed size)
    for alt_key, col in alt_colors.items():
        label = alt_labels.get(alt_key, alt_key)          # fallback: show key
        h = plt.Line2D([], [], marker="^", linestyle="",
                    color=col, markersize=8)
        handles.append(h)
        labels.append(label)

    # 2) capacity-size legend  (same scatter proxies you draw on the map)
    cap_ticks = [min_c, (min_c+max_c)/2, max_c]
    for cap in cap_ticks:
        h = plt.scatter([], [], marker="^", color="grey",
                        s=size_scale(cap),
                        edgecolor="white", linewidth=0.5)
        handles.append(h)
        #labels.append(f"{int(cap):,} m³")

    ax.legend(handles, labels,
            title="Alternatives & Capacities",
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


import seaborn as sns
from scipy.stats import skew, kurtosis

def plot_irr_vs_rate(fin_df, interest_rate=0.042, output_png="irr_summary.png"):
    """
    Scatter‐plot of plant IRRs with two horizontal lines:
      • the financing rate r
      • the average IRR across all built plants
    """
    # 1) pull out only the plants with a valid IRR
    df = fin_df.dropna(subset=["Plant_IRR"])
    plants = df["PlantLocation"].astype(str)
    irr    = df["Plant_IRR"].astype(float)

    # 2) compute average IRR
    avg_irr = irr.mean()

    # 3) plot
    fig, ax = plt.subplots(figsize=(12,6))
    ax.scatter(plants, irr, s=100, color="teal", label="Plant IRR")
    ax.axhline(y=interest_rate, color="red", linestyle="--", linewidth=2,
               label=f"Financing Rate (r={interest_rate:.3f})")
    ax.axhline(y=avg_irr, color="blue", linestyle="--", linewidth=2,
               label=f"Average IRR ({avg_irr:.3f})")

    ax.set_xlabel("Plant Location", fontsize=12)
    ax.set_ylabel("Internal Rate of Return (IRR)", fontsize=12)
    ax.set_title("Plant IRRs vs. Financing Rate", fontsize=14)
    plt.xticks(rotation=45, ha="right")

    # annotate the average IRR
    ax.text(0.02, 0.85,
            f"Avg. IRR = {avg_irr:.3f}",
            transform=ax.transAxes,
            fontsize=14,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.show()
    print(f"Saved IRR summary plot to {output_png}")

import seaborn as sns

def plot_distance_summary(in_flow_df, supply_coords, plant_coords, output_png="distance_summary.png"):
    """
    Plot a histogram of transport distances and annotate only:
      • min
      • mean
      • 75th percentile
      • max
    """
    # 1) compute distances
    distances = []
    R = 6371.0
    for _, row in in_flow_df.iterrows():
        s, p = row["SupplyNode"], row["PlantLocation"]
        if s in supply_coords and p in plant_coords:
            lon1, lat1 = supply_coords[s]
            lon2, lat2 = plant_coords[p]
            φ1, φ2 = np.radians(lat1), np.radians(lat2)
            dφ, dλ = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
            a = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
            distances.append(2*R*np.arcsin(np.sqrt(a)))
    distances = np.array(distances)

    # 2) compute summary stats
    mn   = distances.min()
    mx   = distances.max()
    mean = distances.mean()
    q75  = np.percentile(distances, 75)

    print(f"Distance summary (km): min={mn:.2f}, mean={mean:.2f}, 75th%={q75:.2f}, max={mx:.2f}")

    # 3) plot
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(distances, bins=40, kde=False, color="steelblue", alpha=0.7, ax=ax)
    ax.set_title("Transport Distance Distribution", fontsize=14)
    ax.set_xlabel("Distance (km)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    # annotate
    txt = (
        f"Min: {mn:.2f} km\n"
        f"Mean: {mean:.2f} km\n"
        f"75th %ile: {q75:.2f} km\n"
        f"Max: {mx:.2f} km"
    )
    ax.text(0.70, 0.75, txt, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=16)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.show()
    print(f"Saved distance summary plot to {output_png}")




#plot_methane_fraction(fin_df, system_methane_average)
plot_feedstock_stacked_chart(in_flow_df, feedstock_types, color_map)
#plot_cluster_heatmap(in_flow_df, yields_df, fin_df, plant_coords, supply_coords,FILES["bavaria_geojson"], os.path.join(BASE_DIR, "cluster_heatmap.png"))
#plot_bavaria_lau_highlight_with_labels(gisco_ids)
plot_distance_summary(in_flow_df, supply_coords, plant_coords,
                               output_png="distance_distribution.png")
plot_irr_vs_rate(fin_df, interest_rate=0.042, output_png="irr_summary.png")