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
import seaborn as sns


BASE_DIR = "C:/Clone/Master/"
FILES = {
    "in_flow": os.path.join(BASE_DIR, "results/large_scale/30_runs/Flows_30_optimal.csv"),
    #"out_flow": os.path.join(BASE_DIR, "/Output_out_flow.csv"),
    "financials": os.path.join(BASE_DIR,"results/large_scale/30_runs/Financials_30_optimal.csv"),
    #"feedstock": os.path.join(BASE_DIR, "processed_biomass_data.csv"),
    "feedstock": os.path.join(BASE_DIR, "aggregated_bavaria_supply_nodes.csv"),
    "plant": os.path.join(BASE_DIR, "equally_spaced_locations_30.csv"),
    #"plant": os.path.join(BASE_DIR, "equally_space_locations_10.csv"),
    "yields": os.path.join(BASE_DIR, "Feedstock_yields.csv"),
    "bavaria_geojson": os.path.join(BASE_DIR, "bavaria_cluster_regions.geojson"),
    "supply_coords": os.path.join(BASE_DIR, "supply_coords.csv")
}

color_map = {
    "cattle_man":            "tan",
    "cattle_slu":            "chocolate",
    "horse_man":             "saddlebrown",
    "pig_slu":               "pink",
    "pig_man":               "palevioletred",
    "cereal_str":            "gold",
    "clover_alfalfa_grass":  "seagreen",
    "perm_grass":            "lawngreen",
    "maize_str":             "olive",
    "beet_leaf":             "mediumorchid",
    "rape_str":              "orange",     # adjust as needed
    "legume_str":            "slategrey",  # string literal
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

def plot_feedstock_costs(yields_df, output_filename="feedstock_cost_plot.png", 
                        capacity_dig=27, loading_cost_dig=37, cost_ton_km_dig=0.104):

    try:
        # Ensure columns are stripped of whitespace and quotes
        yields_df.columns = yields_df.columns.str.strip().str.replace('"', '')
        
        # Distance range (0 to 150 km) for plotting
        distances = np.linspace(0, 150, 100)
        
        # Specific distances for printing and annotations
        print_distances = [0, 100]
        annotation_distances = [20, 100]
        
        # Initialize plot
        plt.figure(figsize=(12, 8))
        
        # Store costs for sorting (for annotations)
        costs_at_distances = {20: [], 100: []}  # List of (feedstock, cost) tuples
        
        # Print header
        print("\nFeedstock Costs (€/m³ CH4) at Distances 0 km and 100 km:")
        print("-" * 50)
        
        # Process each feedstock - SINGLE LOOP ONLY
        for index, row in yields_df.iterrows():
            try:
                # Extract data, checking for valid numeric values
                feedstock = row['substrat_ENG']
                methane_yield = float(row['Methane_Yield_m3_ton'])
                digestate_yield = float(row['Digestate_Yield_%']) / 100  # Convert percentage to fraction
                price = float(row['Price'])
                capacity_load = float(row['Capacity_load'])
                loading_cost = float(row['Loading_cost'])
                cost_ton_km = float(row['€_ton_km'])
                
                # Skip if methane yield is zero or invalid to avoid division by zero
                if methane_yield <= 0 or np.isnan(methane_yield):
                    print(f"Skipping feedstock {feedstock} due to invalid methane yield")
                    continue
                
                # Calculate costs per ton
                # Feedstock transport: loading cost per ton + distance-dependent cost
                feedstock_loading_cost = loading_cost / capacity_load
                feedstock_transport_cost = feedstock_loading_cost + cost_ton_km * distances
                
                # Digestate transport: same logic, using digestate yield and fixed constants
                digestate_mass = digestate_yield  # Mass of digestate per ton of feedstock
                digestate_loading_cost = loading_cost_dig / capacity_dig
                digestate_transport_cost = digestate_mass * (digestate_loading_cost + cost_ton_km_dig * distances)
                
                # Total cost per ton: feedstock price + feedstock transport + digestate transport
                total_cost_per_ton = price + feedstock_transport_cost + digestate_transport_cost
                
                # Cost per m³ CH4
                cost_per_m3_ch4 = total_cost_per_ton / methane_yield
                
                # Plot the line - MOVED INSIDE THE MAIN LOOP
                plt.plot(
                    distances,
                    cost_per_m3_ch4,
                    label=feedstock,
                    color=color_map[feedstock],
                    linewidth=2
                )
                    
                # Calculate and print costs at specific distances (0 km and 100 km)
                for dist in print_distances:
                    feedstock_cost = feedstock_loading_cost + cost_ton_km * dist
                    digestate_cost = digestate_mass * (digestate_loading_cost + cost_ton_km_dig * dist)
                    total_cost = price + feedstock_cost + digestate_cost
                    cost_ch4 = total_cost / methane_yield
                    print(f"Feedstock: {feedstock}, Distance: {dist} km, Cost: {cost_ch4:.2f} €/m³ CH4")
                
                # Store costs for annotations at 20 km and 100 km
                for dist in annotation_distances:
                    feedstock_cost = feedstock_loading_cost + cost_ton_km * dist
                    digestate_cost = digestate_mass * (digestate_loading_cost + cost_ton_km_dig * dist)
                    total_cost = price + feedstock_cost + digestate_cost
                    cost_ch4 = total_cost / methane_yield
                    costs_at_distances[dist].append((feedstock, cost_ch4))
                
            except (ValueError, TypeError, KeyError) as e:
                # Skip rows with invalid data
                print(f"Skipping feedstock {row.get('substrat_ENG', 'unknown')} due to error: {str(e)}")
                continue
        
        # Add annotation boxes for top 3 cheapest and most expensive at 20 km and 100 km
        for dist in annotation_distances:
            # Sort by cost
            sorted_costs = sorted(costs_at_distances[dist], key=lambda x: x[1])
            cheapest = sorted_costs[:3]  # Top 3 cheapest
            most_expensive = sorted_costs[-3:][::-1]  # Top 3 most expensive (reversed)
            
            # 1) Make the "Distance: XX km" header bold via TeX
            header = r"$\bf{Distance:\ %d\ km}$" % dist

            # 2) Insert a blank line between sections
            text_lines = [
                header,
                "",               # ← blank line after the header
                "Cheapest:",
            ]
            for feedstock, cost in cheapest:
                text_lines.append(f"  {feedstock}: {cost:.2f} €/m³")
            text_lines.append("")  # ← blank line before the "Most Expensive"
            text_lines.append("Most Expensive:")
            for feedstock, cost in most_expensive:
                text_lines.append(f"  {feedstock}: {cost:.2f} €/m³")

            annotation_text = "\n".join(text_lines)

            plt.text(
                dist, 0.95, annotation_text,
                transform=plt.gca().get_xaxis_transform(),
                fontsize=10, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
                usetex=False  # matplotlib understands basic TeX by default
            )
        
        # Customize plot
        plt.xlabel('Distance (km)', fontsize=13)
        plt.ylabel('Cost (€/m³ CH4)', fontsize=13)
        plt.title('Cost of Energy Transported per Feedstock', fontsize=14)
        
        # Create legend
        leg = plt.legend(
            title="Feedstock Type",
            title_fontsize=14,
            fontsize=12,
            handlelength=2,
            handletextpad=1.0,
            labelspacing=0.5,
            bbox_to_anchor=(1.05, 1), 
            loc='upper left'
        )

        # Thicken the line samples in the legend
        for legline in leg.get_lines():
            legline.set_linewidth(3)

        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(BASE_DIR, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved feedstock cost plot to {output_path}")
        
    except Exception as e:
        print(f"Error processing yields_df: {str(e)}")

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
    ax.set_xlabel("Plant Location", fontsize = 14)
    ax.set_ylabel("Methane Fraction (N_CH4 / Omega)", fontsize = 14)
    ax.set_title("Methane Fraction by Plant Location vs. System Average", fontsize = 14)
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
        # force the bar color to come from your global map
        if feedstock not in color_map:
            raise KeyError(f"No color defined for feedstock '{feedstock}' in color_map")
        c = color_map[feedstock]
        ax.bar(plants, pivot_df[feedstock].values,
               bottom=bottoms,
               label=feedstock,
               color=c)    # <-- HERE
        bottoms += pivot_df[feedstock].values

    ax.set_xlabel("Plant Location", fontsize =20)
    ax.set_ylabel("Percentage of Feedstock (%)", fontsize =20)
    ax.set_title("Feedstock Composition per Plant (100% Stacked)", fontsize =24)
    ax.legend(
        title="Feedstock Type",
        title_fontsize=22,           # Ensures title is same as items
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=20                  # Font size of legend entries
    )
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right", fontsize=18)
    plt.yticks(fontsize=18)          # <-- Set y-axis tick label font size
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
            f"{r.PlantLocation}, {int(r.Capacity // 1_000_000)}",
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

from matplotlib.ticker import FuncFormatter
in_flow_df.columns = in_flow_df.columns.str.strip().str.replace('"', '')
yields_df.columns = yields_df.columns.str.strip().str.replace('"', '')
# Defining color map for feedstocks


# Defining numerical abbreviation function
def abbreviate_number(value, pos):
    if np.isnan(value):
        return '0'
    if abs(value) < 1000:
        return f'{int(value)}'
    units = ['K', 'M', 'B', 'T']
    unit_index = -1
    scaled = abs(value)
    while scaled >= 1000 and unit_index < len(units) - 1:
        scaled /= 1000
        unit_index += 1
    return f'{int(value/abs(value)) if value < 0 else ""}{scaled:.1f}{units[unit_index]}'

# Cleaning column names
in_flow_df.columns = in_flow_df.columns.str.strip().str.replace('"', '')
yields_df.columns = yields_df.columns.str.strip().str.replace('"', '')
def energy_transported(in_flow_df, yields_df, color_map):
    # Converting numeric columns, handling NaN/invalid values
    in_flow_df['FlowTons'] = pd.to_numeric(in_flow_df['FlowTons'], errors='coerce').fillna(0)
    in_flow_df['Distance_km'] = pd.to_numeric(in_flow_df['Distance_km'], errors='coerce').fillna(0)
    yields_df['Biogas_Yield_m3_ton'] = pd.to_numeric(yields_df['Biogas_Yield_m3_ton'], errors='coerce').fillna(0)
    yields_df['Methane_Content_%'] = pd.to_numeric(yields_df['Methane_Content_%'], errors='coerce').fillna(0)

    # Merging data
    yields_map = yields_df.set_index('substrat_ENG')[['Biogas_Yield_m3_ton', 'Methane_Content_%']].to_dict('index')
    in_flow_df['Biogas_Yield_m3_ton'] = in_flow_df['Feedstock'].map(lambda x: yields_map.get(x, {}).get('Biogas_Yield_m3_ton', 0))
    in_flow_df['Methane_Content_percentage'] = in_flow_df['Feedstock'].map(lambda x: yields_map.get(x, {}).get('Methane_Content_%', 0))
    alphaHV = 9.97  # kWh per Nm³ CH₄
    in_flow_df['Energy_MWh'] = (in_flow_df['FlowTons'] * in_flow_df['Biogas_Yield_m3_ton'] * 
                                in_flow_df['Methane_Content_percentage'] * alphaHV) / 1000.0

    # Filtering valid data
    in_flow_df = in_flow_df[(in_flow_df['FlowTons'] > 0) & (in_flow_df['Distance_km'] >= 0)]

    # Calculating feedstock utilization grade
    total_tons = in_flow_df['FlowTons'].sum()
    feedstock_tons = in_flow_df.groupby('Feedstock')['FlowTons'].sum()
    utilization_grades = (feedstock_tons / total_tons * 100).round(2).reset_index()
    utilization_grades.columns = ['Feedstock', 'Utilization_%']
    utilization_grades['Total_Tons'] = feedstock_tons.values

    # Finding feedstock with highest energy contribution
    feedstock_energy = in_flow_df.groupby('Feedstock')['Energy_MWh'].sum()
    top_feedstock = feedstock_energy.idxmax()
    print(f"Interesting Fact: The feedstock '{top_feedstock}' contributes the most to total energy transported.")

    # Setting up distance bins
    dist_min = in_flow_df['Distance_km'].min()
    dist_max = in_flow_df['Distance_km'].max()
    num_bins = 20
    bins = np.linspace(dist_min, dist_max, num_bins + 1)
    bin_width = (dist_max - dist_min) / num_bins
    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(num_bins)]

    # Grouping data by bins - KEY FIX: Remove observed=True to include empty bins
    in_flow_df['Bin'] = pd.cut(in_flow_df['Distance_km'], bins=bins, include_lowest=True, labels=bin_labels)
    bin_data = in_flow_df.groupby(['Bin', 'Feedstock'], observed=False)['FlowTons'].sum().unstack(fill_value=0)  # observed=False
    bin_energy = in_flow_df.groupby('Bin', observed=False)['Energy_MWh'].sum().reindex(bin_labels, fill_value=0)  # observed=False
    
    # ADDITIONAL FIX: Ensure bin_data has all bin_labels as index
    bin_data = bin_data.reindex(bin_labels, fill_value=0)
    
    # Debug information
    print(f"DEBUG: bin_labels length = {len(bin_labels)}")
    print(f"DEBUG: bin_data shape = {bin_data.shape}")
    print(f"DEBUG: bin_energy length = {len(bin_energy)}")
    
    # Verify shapes match
    for feedstock in color_map.keys():
        if feedstock in bin_data.columns:
            print(f"DEBUG: {feedstock} length = {len(bin_data[feedstock])}")
            assert len(bin_data[feedstock]) == len(bin_labels), f"Shape mismatch for {feedstock}"

    # Creating the plot
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plotting stacked bars for feedstock tons
    feedstock_order = list(color_map.keys())
    bottom = np.zeros(num_bins)
    for feedstock in feedstock_order:
        if feedstock in bin_data.columns:
            # Now bin_data[feedstock] should have exactly num_bins elements
            ax1.bar(bin_labels, bin_data[feedstock], bottom=bottom, label=feedstock, color=color_map[feedstock])
            bottom += bin_data[feedstock]

    # Setting up left y-axis (Tons)
    ax1.set_xlabel('Distance (km)', fontsize=13)
    ax1.set_ylabel('Transported Tons', fontsize=13)
    ax1.yaxis.set_major_formatter(FuncFormatter(abbreviate_number))
    ax1.tick_params(axis='x', rotation=45)

    # Creating second y-axis for energy
    ax2 = ax1.twinx()
    ax2.plot(bin_labels, bin_energy, color='#000000', marker='o', linestyle='-', linewidth=1.5, markersize=6, label='Total Energy (MWh)')
    ax2.set_ylabel('Transported Energy (MWh)', fontsize=13)
    ax2.yaxis.set_major_formatter(FuncFormatter(abbreviate_number))

    # Synchronize zero points
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()

    # Calculate the ratio of the ranges
    ratio = (y2_max - y2_min) / (y1_max - y1_min)

    # Set ax2's limits to maintain the same zero point
    ax2.set_ylim(y1_min * ratio, y1_max * ratio)

    # Adding legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.65, 0.9), ncol=3, fontsize=13)

    # Adjusting layout to prevent overlap
    plt.tight_layout()

    # Saving the plot
    plt.savefig('combined_feedstock_energy_plot.png', bbox_inches='tight', dpi=300)
    plt.plot()


#plot_methane_fraction(fin_df, system_methane_average)
plot_feedstock_stacked_chart(in_flow_df, feedstock_types, color_map)
#plot_cluster_heatmap(in_flow_df, yields_df, fin_df, plant_coords, supply_coords,FILES["bavaria_geojson"], os.path.join(BASE_DIR, "cluster_heatmap.png"))
#plot_bavaria_lau_highlight_with_labels(gisco_ids)
#plot_distance_summary(in_flow_df, supply_coords, plant_coords,output_png="distance_distribution.png")
#plot_irr_vs_rate(fin_df, interest_rate=0.042, output_png="irr_summary.png")
#plot_feedstock_costs(yields_df, output_filename="feedstock_cost_plot.png", capacity_dig=27, loading_cost_dig=37, cost_ton_km_dig=0.104)
#energy_transported(in_flow_df, yields_df, color_map)