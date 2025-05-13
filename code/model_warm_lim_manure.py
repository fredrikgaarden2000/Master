import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import LineString, Point
from collections import defaultdict
import uuid
from geopy.distance import geodesic
from scipy.spatial import Delaunay
import pickle
import time
from multiprocessing import Pool
import os

script_start_time = time.time()

###############################################################################
# 1) LOAD DATA
###############################################################################
#BASE_DIR = "/home/fredrgaa/Master/"
BASE_DIR = "C:/Clone/Master/"
feedstock_df = pd.read_csv(f"{BASE_DIR}processed_biomass_data.csv")
plant_df = pd.read_csv(f"{BASE_DIR}equally_spaced_locations.csv")
distance_df = pd.read_csv(f"{BASE_DIR}Distance_Matrix.csv")
yields_df = pd.read_csv(f"{BASE_DIR}Feedstock_yields.csv")

# Filter feedstock_df to exclude nodes with < 20 tons
feedstock_df = feedstock_df[
    (feedstock_df["GISCO_ID"].notna()) &
    (feedstock_df["Centroid_Lon"].notna()) &
    (feedstock_df["Centroid_Lat"].notna()) &
    (feedstock_df["nutz_pot_tFM"] >= 20)
]

# Log the number of removed nodes
original_rows = len(pd.read_csv(f"{BASE_DIR}processed_biomass_data.csv"))
filtered_rows = len(feedstock_df)

# Verify distance matrix columns
expected_columns = ['Feedstock_LAU', 'Location', 'Distance_km']
for col in expected_columns:
    if col not in distance_df.columns:
        raise ValueError(f"Column '{col}' not found in Distance_Matrix.csv. Available columns: {distance_df.columns}")

# Update distance_df to only include GISCO_IDs present in filtered feedstock_df
valid_gisco_ids = set(feedstock_df['GISCO_ID'].unique())
distance_df = distance_df[distance_df['Feedstock_LAU'].isin(valid_gisco_ids)]

# Supply coordinates (Longitude, Latitude)
supply_coords = {row['GISCO_ID']: (row['Centroid_Lon'], row['Centroid_Lat']) 
                 for _, row in feedstock_df.iterrows()}

# Plant coordinates (Longitude, Latitude)
plant_coords = {row['Location']: (row['Longitude'], row['Latitude']) 
                for _, row in plant_df.iterrows()}

# Digestate coordinates (same as supply for LAU)
iPrime_coords = supply_coords.copy()

# Verify GISCO_ID alignment
feedstock_gisco = set(feedstock_df['GISCO_ID'].unique())
distance_gisco = set(distance_df['Feedstock_LAU'].unique())
if not distance_gisco.issubset(feedstock_gisco):
    missing = distance_gisco - feedstock_gisco
    raise ValueError(f"GISCO_IDs in Distance_Matrix.csv not found in processed_biomass_data.csv: {missing}")

###############################################################################
# 2) SETS & DICTIONARIES
###############################################################################
supply_nodes = feedstock_df['GISCO_ID'].unique().tolist()
iPrime_nodes = supply_nodes[:]
feedstock_types = yields_df['substrat_ENG'].unique().tolist()
plant_locs = plant_df['Location'].unique().tolist()
capacity_levels = (250_000, 500_000)  #, 40_000_000, 75_000_000
FLH_max = 8000
alphaHV = 9.97
CN_min = 20.0
CN_max = 30.0
heat_price = 20
boiler_eff = 0.9
electricity_spot_price = 60
chp_elec_eff = 0.4
chp_heat_eff = 0.4
r = 0.042
years = 25
kappa = sum(1/(1+r)**t for t in range(1, years+1))
EEG_price_small = 210.0
EEG_price_med = 190.0
EEG_skip_chp_price = 194.3
EEG_skip_upg_price = 210.4
gas_price_mwh = 30
gas_price_m3 = gas_price_mwh * (alphaHV / 1000)
co2_price_ton = 50
co2_price = co2_price_ton / 556.2
Cap_biogas = 0.45
Cap_biomethane = 0.10
variable_upg_cost = 0.05
alpha_GHG_comp = 94.0
alpha_GHG_lim = 0.35 * alpha_GHG_comp
GHG_certificate_price = 60.0
avail_mass = {(row['GISCO_ID'], row['substrat_ENG']): row['nutz_pot_tFM'] for _, row in feedstock_df.iterrows()}
dist_ik = {(row['Feedstock_LAU'], row['Location']): row['Distance_km'] for _, row in distance_df.iterrows()}
dist_pl_iprime = {(ploc, iP): dist_ik.get((iP, ploc), 0.0) for ploc in plant_locs for iP in iPrime_nodes}
feed_yield = {
    row['substrat_ENG']: {
        'biogas_m3_per_ton': row['Biogas_Yield_m3_ton'],
        'ch4_content': row['Methane_Content_%'],
        'digestate_frac': row['Digestate_Yield_%'] / 100.0,
        'CN': row['C/N_average'],
        'price': row['Price'],
        'GHG_intensity': row['GHG_intensity_gCO2eMJ'],
        'loading': row['Loading_cost'],
        'capacity_load': row['Capacity_load'],
        'cost_ton_km': row['€_ton_km']
    } for _, row in yields_df.iterrows()
}

# Digestate hard values
capacity_dig = 27
loading_cost_dig = 37
cost_ton_km_dig = 0.104

# Perform Delaunay triangulation to find adjacent plants
points = plant_df[["Longitude", "Latitude"]].values
tri = Delaunay(points)

# Extract adjacent pairs from triangulation
adjacent_pairs = set()
for simplex in tri.simplices:
    pairs = [(simplex[i], simplex[j]) for i, j in [(0, 1), (1, 2), (0, 2)]]
    for i, j in pairs:
        pair = tuple(sorted([i, j]))
        adjacent_pairs.add(pair)

# Compute pairwise distances for adjacent plants, filtering out > 75 km
distances = []
filtered_pairs = set()
max_distance_threshold = 500
for i, j in adjacent_pairs:
    p1_id = plant_df.iloc[i]["Location"]
    p1_coords = (plant_df.iloc[i]["Latitude"], plant_df.iloc[i]["Longitude"])
    p2_id = plant_df.iloc[j]["Location"]
    p2_coords = (plant_df.iloc[j]["Latitude"], plant_df.iloc[j]["Longitude"])
    dist_km = geodesic(p1_coords, p2_coords).kilometers
    if dist_km <= max_distance_threshold:
        distances.append({
            "Location1": p1_id,
            "Location2": p2_id,
            "Distance_km": dist_km
        })
        filtered_pairs.add((i, j))

# Convert distances to DataFrame
dist_df = pd.DataFrame(distances)
max_distance = dist_df["Distance_km"].max() * 2 if not dist_df.empty else 150

total_biogas = {}
for j in plant_locs:
    total_biogas[j] = sum(avail_mass[(i, f)] * feed_yield[f]['biogas_m3_per_ton'] for (i, f) in avail_mass)

def is_manure(ftype):
    return 'man' in ftype.lower() or 'slu' in ftype.lower()

def is_clover(ftype):
    return 'clover' in ftype.lower()

def is_maize_cereal(ftype):
    return 'maize' in ftype.lower() or 'cereal' in ftype.lower()

total_methane = sum(avail_mass[i, f] * feed_yield[f]['ch4_content'] for i, f in avail_mass)
total_mass = sum(avail_mass[i, f] for i, f in avail_mass)
system_methane_average = total_methane / total_mass
EEG_small_m3 = (75 * FLH_max) / (chp_elec_eff * system_methane_average * alphaHV)
EEG_med_m3 = (150 * FLH_max) / (chp_elec_eff * system_methane_average * alphaHV)
auction_chp_limit = 225000 * FLH_max / alphaHV / system_methane_average
auction_bm_limit = 125000 * FLH_max / alphaHV / system_methane_average
alternative_configs = [
    {"name": "EEG_CHP_small1", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_small_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_small},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
     "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_small2", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_small_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_small},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 2,
     "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_large1", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_med_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_med},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
     "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_large2", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_med_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_med},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 2,
     "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
]
###############################################################################
# 3) GLOBAL PARAMETERS
###############################################################################
premium = {f: max(0, (alpha_GHG_comp - feed_yield[f]['GHG_intensity'])) * (alphaHV * 3.6) * GHG_certificate_price / 1e6 for f in feedstock_types}
threshold_m3 = (100 * FLH_max) / (chp_elec_eff * system_methane_average * alphaHV)
FLH_min_limit = 1000
M_large = 1e9
avg_discount = sum(0.99**t for t in range(1, years+1)) / years

# Precompute bounds
M_j = {j: sum(avail_mass[i, f] for i, f in avail_mass if dist_ik.get((i, j), float('inf')) <= max_distance) for j in plant_locs}
M_NCH4 = {j: total_biogas[j] * 0.7 for j in plant_locs}

###############################################################################
# 4) CONSTRAINT FUNCTIONS
###############################################################################
def add_eeg_constraints(m, total_feed, manure_feed, clover_feed, Y, plant_locs, alternative_configs, capacity_levels):
    for j in plant_locs:
        delta1 = gp.quicksum(Y[j, a, c] for a, alt in enumerate(alternative_configs) if alt["category"].startswith("EEG_CHP") and alt.get("feed_constraint", 0) == 1 for c in capacity_levels)
        delta2 = gp.quicksum(Y[j, a, c] for a, alt in enumerate(alternative_configs) if alt["category"].startswith("EEG_CHP") and alt.get("feed_constraint", 0) == 2 for c in capacity_levels)
        aux_manure1 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"aux_manure1_{j}")
        aux_manure2 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"aux_manure2_{j}")
        m.addConstr(aux_manure1 <= 0.80 * total_feed[j], name=f"aux_manure1_upper_{j}")
        m.addConstr(aux_manure1 <= M_j[j] * delta1, name=f"aux_manure1_bound_{j}")
        m.addConstr(aux_manure1 >= 0.80 * total_feed[j] - M_j[j] * (1 - delta1), name=f"aux_manure1_lower_{j}")
        m.addConstr(aux_manure2 <= 0.70 * total_feed[j], name=f"aux_manure2_upper_{j}")
        m.addConstr(aux_manure2 <= M_j[j] * delta2, name=f"aux_manure2_bound_{j}")
        m.addConstr(aux_manure2 >= 0.70 * total_feed[j] - M_j[j] * (1 - delta2), name=f"aux_manure2_lower_{j}")
        m.addConstr(manure_feed[j] >= aux_manure1 + aux_manure2, name=f"EEG_manure_{j}")
        aux_clover = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"aux_clover_{j}")
        m.addConstr(aux_clover <= 0.10 * total_feed[j], name=f"aux_clover_upper_{j}")
        m.addConstr(aux_clover <= M_j[j] * delta2, name=f"aux_clover_bound_{j}")
        m.addConstr(aux_clover >= 0.10 * total_feed[j] - M_j[j] * (1 - delta2), name=f"aux_clover_lower_{j}")
        m.addConstr(clover_feed[j] >= aux_clover, name=f"EEG_clover_{j}")

def add_supply_constraints(m, avail_mass, x, plant_locs, max_distance, dist_ik):
    for (i, f), amt in avail_mass.items():
        m.addConstr(gp.quicksum(x[i, f, j] for j in plant_locs if dist_ik.get((i, j), float('inf')) <= max_distance) <= amt, name=f"Supply_{i}_{f}")

def add_digestate_constraints(m, x, digestate_return, supply_nodes, plant_locs, avail_mass, feed_yield, dist_pl_iprime, max_distance, return_frac=0.99):
    for i in supply_nodes:
        contributed = gp.quicksum(x[i, f, j] * feed_yield[f]['digestate_frac'] for f in feedstock_types for j in plant_locs if (i, f) in avail_mass and dist_ik.get((i, j), float('inf')) <= max_distance)
        returned = gp.quicksum(digestate_return[j, i] for j in plant_locs if dist_pl_iprime.get((j, i), float('inf')) <= max_distance)
        m.addConstr(returned >= return_frac * contributed, name=f"Digestate_{i}")
    for j in plant_locs:
        total_prod = gp.quicksum(x[i, f, j] * feed_yield[f]['digestate_frac'] for i, f in avail_mass if dist_ik.get((i, j), float('inf')) <= max_distance)
        outflow = gp.quicksum(digestate_return[j, i] for i in supply_nodes if dist_pl_iprime.get((j, i), float('inf')) <= max_distance)
        m.addConstr(outflow <= total_prod, name=f"DigestateOut_{j}")

def add_cn_constraints(m, x, avail_mass, plant_locs, feed_yield, cn_min=20.0, cn_max=30.0):
    for j in plant_locs:
        total_feed = gp.quicksum(x[i, f, j] for i, f in avail_mass)
        total_CN = gp.quicksum(x[i, f, j] * feed_yield[f]['CN'] for i, f in avail_mass)
        m.addConstr(total_CN >= cn_min * total_feed, name=f"CN_min_{j}")
        m.addConstr(total_CN <= cn_max * total_feed, name=f"CN_max_{j}")

def add_ghg_constraints(m, x, avail_mass, plant_locs, feed_yield, alpha_ghg_lim):
    for j in plant_locs:
        total_feed_j = gp.quicksum(x[i, f, j] for i, f in avail_mass)
        total_GHG_j = gp.quicksum(x[i, f, j] * feed_yield[f]['GHG_intensity'] for i, f in avail_mass)
        m.addConstr(total_GHG_j <= alpha_ghg_lim * total_feed_j, name=f"GHG_average_{j}")

def add_auction_constraints(m, Y, plant_locs, alternative_configs, capacity_levels):
    total_EEG_capacity = gp.quicksum(Y[j, a, c] * c for j in plant_locs for a, alt in enumerate(alternative_configs) if alt["EEG_flag"] for c in capacity_levels)
    m.addConstr(total_EEG_capacity <= auction_chp_limit, name="EEG_Auction_Limit")
    total_biogas_capacity = gp.quicksum(Y[j, a, c] * c for j in plant_locs for a, alt in enumerate(alternative_configs) if alt["EEG_flag"] and alt["category"] != "FlexEEG_biomethane" for c in capacity_levels)
    m.addConstr(total_biogas_capacity <= 225000 * FLH_max / alphaHV / system_methane_average, name="EEG_Biogas_Auction_Limit")
    total_biomethane_capacity = gp.quicksum(Y[j, a, c] * c for j in plant_locs for a, alt in enumerate(alternative_configs) if alt["category"] == "FlexEEG_biomethane" for c in capacity_levels)
    m.addConstr(total_biomethane_capacity <= 125000 * FLH_max / alphaHV / system_methane_average, name="EEG_Biomethane_Auction_Limit")

def add_flh_constraints(m, Omega, Y, plant_locs, capacity_levels, N_CH4):
    for j in plant_locs:
        cap_expr = gp.quicksum(c * Y[j, a, c] for a in range(len(alternative_configs)) for c in capacity_levels)
        m.addConstr(Omega[j] <= (FLH_max / 8760.0) * cap_expr, name=f"FLH_limit_{j}")
        m.addConstr(N_CH4[j] <= (FLH_max / 8760.0) * Omega[j], name=f"FLH_limit_NCH4{j}")

###############################################################################
# 5) MODEL FUNCTION
###############################################################################
def build_model(config, fixed_capacity=None):
    m = gp.Model("ShadowPlant_Biogas_Model")
    m.setParam("Heuristics", 0.3)
    m.setParam("NoRelHeurTime", 100)
    m.setParam("Presolve", 2)
    m.setParam("Cuts", 3)

    # Define capacity levels for this run
    caps = [fixed_capacity] if fixed_capacity is not None else capacity_levels

    # Define variables
    Omega = m.addVars(plant_locs, lb=0, ub=total_biogas, vtype=GRB.CONTINUOUS, name="Omega")
    N_CH4 = m.addVars(plant_locs, lb=0, vtype=GRB.CONTINUOUS, name="N_CH4")
    x = m.addVars(
        supply_nodes, feedstock_types, plant_locs,
        lb=0,
        ub={(i, f, j): avail_mass.get((i, f), 0) if dist_ik.get((i, j), float('inf')) <= max_distance else 0 for i in supply_nodes for f in feedstock_types for j in plant_locs},
        vtype=GRB.CONTINUOUS,
        name="x"
    )
    for i in supply_nodes:
        for f in feedstock_types:
            if (i, f) not in avail_mass:
                for j in plant_locs:
                    m.addConstr(x[i, f, j] == 0, name=f"ZeroFlow_{i}_{f}_{j}")
            else:
                for j in plant_locs:
                    if dist_ik.get((i, j), float('inf')) > max_distance:
                        m.addConstr(x[i, f, j] == 0, name=f"Max_Distance_x_{i}_{f}_{j}")
    
    digestate_return = m.addVars(
        plant_locs, supply_nodes,
        lb=0,
        ub={(j, i): sum(avail_mass.get((iP, f), 0) * feed_yield[f]['digestate_frac'] for iP, f in avail_mass if iP == i and dist_ik.get((iP, j), float('inf')) <= max_distance) for j in plant_locs for i in supply_nodes},
        vtype=GRB.CONTINUOUS,
        name="digestate_return"
    )
    for j in plant_locs:
        for i in supply_nodes:
            if dist_pl_iprime.get((j, i), float('inf')) > max_distance:
                m.addConstr(digestate_return[j, i] == 0, name=f"Max_Distance_digestate_{j}_{i}")
    Y = {(j, a, c): m.addVar(vtype=GRB.BINARY, name=f"Y_{j}_{a}_{c}") for j in plant_locs for a in range(len(alternative_configs)) for c in caps}

    m_up = m.addVars(plant_locs, feedstock_types, lb=0, vtype=GRB.CONTINUOUS, name="m_up")
    Rev_loc = m.addVars(plant_locs, lb=0, vtype=GRB.CONTINUOUS, name="Rev_loc")
    Cost_loc = m.addVars(plant_locs, lb=0, vtype=GRB.CONTINUOUS, name="Cost_loc")
    Rev_alt_selected = m.addVars(
        plant_locs, range(len(alternative_configs)), caps,
        lb=0, vtype=GRB.CONTINUOUS, name="Rev_alt_selected"
    )
    Cost_alt_selected = m.addVars(
        plant_locs, range(len(alternative_configs)), caps,
        lb=0, vtype=GRB.CONTINUOUS, name="Cost_alt_selected"
    )

    # Define bonus_dict
    bonus_dict = {}
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            for c in caps:
                if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"] and c > threshold_m3:
                    bonus_dict[j, a, c] = 100 * (c * system_methane_average * chp_elec_eff * alphaHV) / FLH_max
                else:
                    bonus_dict[j, a, c] = 0

    # Add constraints
    for j in plant_locs:
        m.addConstr(Omega[j] <= gp.quicksum(c * Y[j, a, c] for a in range(len(alternative_configs)) for c in caps), name=f"Omega_Link_{j}")
        m.addConstr(gp.quicksum(Y[j, a, c] for a in range(len(alternative_configs)) for c in caps) <= 1, name=f"OneAlt_{j}")
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            if alt.get("max_cap_m3_year") is not None:
                for c in caps:
                    m.addConstr(Omega[j] <= alt["max_cap_m3_year"] + M_large * (1 - Y[j, a, c]), name=f"MaxCap_{j}_{a}_{c}")

    for j in plant_locs:
        m.addConstr(Omega[j] == gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] for i, f in avail_mass if dist_ik.get((i, j), float('inf')) <= max_distance), name=f"Omega_Feed_{j}")
        m.addConstr(N_CH4[j] == gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] * feed_yield[f]['ch4_content'] for i, f in avail_mass if dist_ik.get((i, j), float('inf')) <= max_distance), name=f"N_CH4_Feed_{j}")

    total_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if dist_ik.get((i, j), float('inf')) <= max_distance) for j in plant_locs}
    manure_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if is_manure(f) and dist_ik.get((i, j), float('inf')) <= max_distance) for j in plant_locs}
    clover_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if is_clover(f) and dist_ik.get((i, j), float('inf')) <= max_distance) for j in plant_locs}

    # Enforce FLH_min_limit for FlexEEG alternatives
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                for c in caps:
                    if c > threshold_m3:
                        m.addConstr(Omega[j] >= (FLH_min_limit / 8760.0) * c * Y[j, a, c], name=f"FLH_min_limit_{j}_{a}_{c}")

    add_eeg_constraints(m, total_feed, manure_feed, clover_feed, Y, plant_locs, alternative_configs, caps)
    add_supply_constraints(m, avail_mass, x, plant_locs, max_distance, dist_ik)
    add_cn_constraints(m, x, avail_mass, plant_locs, feed_yield, CN_min, CN_max)
    #add_ghg_constraints(m, x, avail_mass, plant_locs, feed_yield, alpha_GHG_lim)
    #add_auction_constraints(m, Y, plant_locs, alternative_configs, caps)
    add_flh_constraints(m, Omega, Y, plant_locs, caps, N_CH4)

    for j in plant_locs:
        for f in feedstock_types:
            production_f = gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] * feed_yield[f]['ch4_content'] for i in supply_nodes if dist_ik.get((i, j), float('inf')) <= max_distance)
            m.addConstr(m_up[j, f] <= production_f, name=f"m_up_upper_{j}_{f}")
        upgrading_flag = gp.quicksum(Y[j, a, c] for a, alt in enumerate(alternative_configs) if alt["category"] == "Upgrading" for c in caps)
        aux_upgrading = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"aux_upgrading_{j}")
        m.addConstr(aux_upgrading <= N_CH4[j], name=f"aux_upgrading_upper_{j}")
        m.addConstr(aux_upgrading <= M_NCH4[j] * upgrading_flag, name=f"aux_upgrading_bound_{j}")
        m.addConstr(aux_upgrading >= N_CH4[j] - M_NCH4[j] * (1 - upgrading_flag), name=f"aux_upgrading_lower_{j}")
        m.addConstr(gp.quicksum(m_up[j, f] for f in feedstock_types) == aux_upgrading, name=f"m_up_sum_{j}")

    M_rev = {}
    for j in plant_locs:
        max_elec_price = max(200, electricity_spot_price)
        max_heat_price = heat_price
        max_bonus = max(100 * (c * system_methane_average * chp_elec_eff * alphaHV) / FLH_max 
                        for c in caps if c > threshold_m3)
        M_rev[j] = (M_NCH4[j] * (chp_elec_eff * alphaHV / 1000.0) * max_elec_price + 
                    M_NCH4[j] * (chp_heat_eff * alphaHV / 1000.0) * max_heat_price + 
                    max_bonus) * 1.1

    M_cost = {}
    for j in plant_locs:
        max_cost = 0
        for a, alt in enumerate(alternative_configs):
            for c in caps:
                if alt["opex_type"] == "fixed_variable_MW":
                    MW = c * system_methane_average * chp_heat_eff * alphaHV / (FLH_max * 1000)
                    fixed_opex = alt["opex_coeff"] * MW
                    variable_opex = 0.5 * M_NCH4[j] * alphaHV * chp_heat_eff
                    cost_val = fixed_opex + variable_opex
                else:
                    cost_val = alt["opex_coeff"] * (c ** alt["opex_exp"])
                max_cost = 1e9
        M_cost[j] = max_cost * 1.1

    Rev_alt = {}
    Cost_alt = {}
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            for c in caps:
                cost_val = alt["opex_coeff"] * (c ** alt["opex_exp"])                
                effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                rev_val = N_CH4[j] * (alphaHV / 1000) * (chp_elec_eff * effective_EEG + chp_heat_eff * heat_price)

                Rev_alt[j, a, c] = rev_val
                Cost_alt[j, a, c] = cost_val
                m.addConstr(Rev_alt_selected[j, a, c] <= rev_val, name=f"Rev_alt_sel_upper1_{j}_{a}_{c}")
                m.addConstr(Rev_alt_selected[j, a, c] <= M_rev[j] * Y[j, a, c], name=f"Rev_alt_sel_upper2_{j}_{a}_{c}")
                m.addConstr(Rev_alt_selected[j, a, c] >= rev_val - M_rev[j] * (1 - Y[j, a, c]), name=f"Rev_alt_sel_lower_{j}_{a}_{c}")
                m.addConstr(Rev_alt_selected[j, a, c] >= 0, name=f"Rev_alt_sel_nonneg_{j}_{a}_{c}")
                m.addConstr(Cost_alt_selected[j, a, c] <= cost_val, name=f"Opex_alt_sel_upper1_{j}_{a}_{c}")
                m.addConstr(Cost_alt_selected[j, a, c] <= M_cost[j] * Y[j, a, c], name=f"Opex_alt_sel_upper2_{j}_{a}_{c}")
                m.addConstr(Cost_alt_selected[j, a, c] >= cost_val - M_cost[j] * (1 - Y[j, a, c]), name=f"Opex_alt_sel_lower_{j}_{a}_{c}")
                m.addConstr(Cost_alt_selected[j, a, c] >= 0, name=f"Opex_alt_sel_nonneg_{j}_{a}_{c}")

    for j in plant_locs:
        m.addConstr(Rev_loc[j] == gp.quicksum(Rev_alt_selected[j, a, c] for a in range(len(alternative_configs)) for c in caps), name=f"Rev_link_{j}")
        m.addConstr(Cost_loc[j] == gp.quicksum(Cost_alt_selected[j, a, c] for a in range(len(alternative_configs)) for c in caps), name=f"Cost_link_{j}")

    Capex = {}
    for j in plant_locs:
        capex_expr = gp.LinExpr()
        for a, alt in enumerate(alternative_configs):
            for c in caps:
                capex_expr = c * alt["capex_coeff"] * (c ** alt["capex_exp"]) if alt["category"] != "no_build" else 0
        Capex[j] = m.addVar(lb=0, name=f"Capex_{j}")
        m.addConstr(Capex[j] == capex_expr, name=f"Capex_link_{j}")


    FeedstockCost = gp.LinExpr()
    DigestateFlows = {}  # Optional: for tracking digestate flows
    for i, f in avail_mass:
        for j in plant_locs:
            if dist_ik.get((i, j), float('inf')) <= max_distance:
                flow = x[i, f, j]
                dist_val = dist_ik.get((i, j), 0.0)

                # Feedstock cost
                FeedstockCost.add(flow * feed_yield[f]['price'])
                if dist_val > 0:
                    FeedstockCost.add(flow * (feed_yield[f]['loading'] / feed_yield[f]['capacity_load'] +
                                            dist_val * feed_yield[f]['cost_ton_km']))

                # Digestate flow = input flow * digestate yield
                digestate_flow = flow * feed_yield[f]['digestate_frac']
                DigestateFlows[(i, f, j)] = digestate_flow  # Optional storage

                # Digestate cost
                if dist_val > 0:
                    FeedstockCost.add(digestate_flow * (loading_cost_dig / capacity_dig +
                                                        dist_val * cost_ton_km_dig))


    TotalRev = gp.quicksum(Rev_loc[j] for j in plant_locs)
    TotalCost = FeedstockCost + gp.quicksum(Cost_loc[j] for j in plant_locs)
    TotalCapex = gp.quicksum(Capex[j] for j in plant_locs)
    GHGRevenue = gp.LinExpr()
    for j in plant_locs:
        for f in feedstock_types:
            GHGRevenue.add(premium[f] * m_up[j, f])
    NPV_expr = -TotalCapex
    for t in range(1, years + 1):
        discount_factor = 1 / (1 + r) ** t
        NPV_expr += discount_factor * (TotalRev + GHGRevenue - TotalCost)
    m.setObjective(NPV_expr, GRB.MAXIMIZE)

    return m, Omega, N_CH4, x,digestate_return, Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, GHGRevenue, TotalCapex, bonus_dict, Rev_alt_selected, Cost_alt_selected
###############################################################################
# 6) RUN MODEL
###############################################################################
config = {
    "name": "Baseline",
    "eeg_enabled": True,
    "supply_enabled": True,
    "digestate_enabled": False,
    "digestate_return_frac": 0.99,
    "cn_enabled": True,
    "maize_enabled": False,
    "ghg_enabled": True,
    "auction_enabled": True,
    "flh_enabled": True
}

def load_solution(capacity):
    """Load a solution from a pickle file if it exists."""
    file_path = f"{BASE_DIR}/Solutions/{len(plant_locs)}/solution_{capacity}.pkl"
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                solution = pickle.load(f)
            print(f"Loaded existing solution for capacity {capacity:,} m³/year.")
            return solution
        except Exception as e:
            print(f"Error loading solution for capacity {capacity:,}: {e}")
            return None
    return None

def run_single_capacity(capacity):
    """Run the model for a single capacity or load existing solution."""
    # Check for existing solution
    solution = load_solution(capacity)
    if solution is not None:
        return solution

    # Run the model if no solution is found
    print(f"Running model for capacity {capacity:,} m³/year...")
    m, Omega, N_CH4, x, digestate_return, Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, GHGRevenue, TotalCapex, bonus_dict, Rev_alt_selected, Cost_alt_selected = build_model(config, fixed_capacity=capacity)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        solution = {
            'capacity': capacity,
            'npv': m.objVal,
            'Y': {(j, a, capacity): Y[j, a, capacity].X for j in plant_locs for a in range(len(alternative_configs))},
            'Omega': {j: Omega[j].X for j in plant_locs},
            'N_CH4': {j: N_CH4[j].X for j in plant_locs},
            'x': {(i, f, j): x[i, f, j].X for i in supply_nodes for f in feedstock_types for j in plant_locs},
            'digestate_return': {(j, i): digestate_return[j, i].X for j in plant_locs for i in supply_nodes},
            'm_up': {(j, f): m_up[j, f].X for j in plant_locs for f in feedstock_types},
            'Rev_loc': {j: Rev_loc[j].X for j in plant_locs},
            'Cost_loc': {j: Cost_loc[j].X for j in plant_locs},
            'Capex': {j: Capex[j].X for j in plant_locs},
            'Rev_alt_selected': {(j, a, capacity): Rev_alt_selected[j, a, capacity].X for j in plant_locs for a in range(len(alternative_configs))},
            'Cost_alt_selected': {(j, a, capacity): Cost_alt_selected[j, a, capacity].X for j in plant_locs for a in range(len(alternative_configs))}
        }
        print(f"Capacity {capacity:,}: NPV = {m.objVal:,.2f} €, Solve time = {m.Runtime:.2f} s, MIP Gap = {m.MIPGap:.4f}")
        with open(f"{BASE_DIR}/Solutions/{len(plant_locs)}/solution_{capacity}.pkl", 'wb') as f:
            pickle.dump(solution, f)
        return solution
    else:
        print(f"Capacity {capacity:,}: No optimal solution found (status: {m.status}).")
        return None

# Run single-capacity models in parallel
if __name__ == '__main__':
    print("Starting single-capacity runs...")
    results = []
    capacities_to_run = []

    # Check for existing solutions
    for capacity in capacity_levels:
        solution = load_solution(capacity)
        if solution is not None:
            results.append(solution)
        else:
            capacities_to_run.append(capacity)

    # Run models for missing capacities in parallel
    if capacities_to_run:
        with Pool(processes=4) as pool:
            new_results = pool.map(run_single_capacity, capacities_to_run)
            results.extend(new_results)
    
    # Filter valid solutions and find the best
    valid_solutions = [res for res in results if res is not None]
    if not valid_solutions:
        raise ValueError("No optimal solutions found for any capacity.")
    
    best_solution = max(valid_solutions, key=lambda s: s['npv'])
    best_capacity = best_solution['capacity']
    print(f"\nBest capacity: {best_capacity:,} m³/year, NPV: {best_solution['npv']:,.2f} €")
    
    # Run full model with warm start
    print("\nRunning full model with warm start...")
    m, Omega, N_CH4, x, digestate_return,Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, GHGRevenue, TotalCapex, bonus_dict, Rev_alt_selected, Cost_alt_selected = build_model(config)
    m.update()
    
    # Apply warm start
    for j in plant_locs:
        for a in range(len(alternative_configs)):
            for c in capacity_levels:
                if c == best_capacity and best_solution['Y'].get((j, a, c), 0) > 0.5:
                    Y[j, a, c].Start = 1
                else:
                    Y[j, a, c].Start = 0
    for j in plant_locs:
        Omega[j].Start = best_solution['Omega'].get(j, 0)
        N_CH4[j].Start = best_solution['N_CH4'].get(j, 0)
    for i in supply_nodes:
        for f in feedstock_types:
            for j in plant_locs:
                x[i, f, j].Start = best_solution['x'].get((i, f, j), 0)
    for j in plant_locs:
        for i in supply_nodes:
            digestate_return[j, i].Start = best_solution['digestate_return'].get((j, i), 0)
    
    print("Warm start applied. Optimizing full model...")
    opt_start_time = time.time()
    m.optimize()
    opt_end_time = time.time()
    
    if m.status == GRB.OPTIMAL:
        print(f"\nFull model: NPV = {m.objVal:,.2f} €, Solve time = {m.Runtime:.2f} s, MIP Gap = {m.MIPGap:.4f}")
        # Validate N_CH4
        for j in plant_locs:
            if Omega[j].X > 1e-6:
                print(f"Plant {j}: N_CH4 = {N_CH4[j].X:,.0f}, Omega = {Omega[j].X:,.0f}, CH4 Fraction = {N_CH4[j].X/Omega[j].X:.3f}")
    else:
        print(f"Full model: No optimal solution found (status: {m.status}).")

    # Save inflow output
    inflow_rows = []
    for j in plant_locs:
        for i, f in avail_mass:
            flow_val = x[i, f, j].X
            if flow_val > 1e-6:
                # Get distance from dist_ik; default to 0 if not found
                distance = dist_ik.get((i, j), 0.0)
                inflow_rows.append({
                    "SupplyNode": i,
                    "PlantLocation": j,
                    "Feedstock": f,
                    "FlowTons": flow_val,
                    "Distance_km": distance
                })
    in_flow_df = pd.DataFrame(inflow_rows)
    in_flow_df.to_csv(f"{BASE_DIR}/Solutions/{len(plant_locs)}/Output_in_flow.csv", index=False)
    
    outflow_rows = []
    for j in plant_locs:
        for i in supply_nodes:
            digest_val = digestate_return[j, i].X
            if digest_val > 1e-6:
                outflow_rows.append({
                    "PlantLocation": j,
                    "SupplyNode": i,
                    "DigestateTons": digest_val
                })
    out_flow_df = pd.DataFrame(outflow_rows)
    out_flow_df.to_csv(f"{BASE_DIR}/Solutions/{len(plant_locs)}/Output_out_flow_warm_start.csv", index=False)
    
    merged_rows = []
    for j in plant_locs:
        for a in range(len(alternative_configs)):
            for c in capacity_levels:
                if Y[j, a, c].X > 0.5:
                    alt = alternative_configs[a]
                    alt_name = alt["name"]
                    cap_fraction = Cap_biogas if alt["category"] == "FlexEEG_biogas" else Cap_biomethane if alt["category"] == "FlexEEG_biomethane" else None
                    
                    row_data = {
                        "PlantLocation": j,
                        "Alternative": alt_name,
                        "Capacity": c,
                        "Omega": Omega[j].X,
                        "N_CH4": N_CH4[j].X,
                        "CO2_Production": Omega[j].X - N_CH4[j].X,
                        "Revenue": Rev_loc[j].X,
                        "Cost": Cost_loc[j].X,
                        "Capex": Capex[j].X,
                        "GHG": sum(premium[f] * m_up[j, f].X for f in feedstock_types),
                        "FLH": (Omega[j].X / c) * 8760 if c > 0 else 0,
                        "PlantLatitude": plant_coords.get(j, (None, None))[1],
                        "PlantLongitude": plant_coords.get(j, (None, None))[0]
                    }
                    if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                        effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                        E_actual = N_CH4[j].X * (chp_elec_eff * alphaHV / 1000.0)
                        EEG_rev = cap_fraction * E_actual * effective_EEG
                        spot_rev = (E_actual - (cap_fraction * E_actual)) * electricity_spot_price
                        heat_rev = heat_price * (N_CH4[j].X * chp_heat_eff * alphaHV / 1000.0)
                        row_data.update({
                            "EEG_Revenue": EEG_rev,
                            "Spot_Revenue": spot_rev,
                            "Heat_Revenue": heat_rev,
                            "Bonus": bonus_dict.get((j, a, c), 0)
                        })
                    else:
                        row_data.update({
                            "EEG_Revenue": 0,
                            "Spot_Revenue": 0,
                            "Heat_Revenue": 0,
                            "Bonus": 0
                        })
                    merged_rows.append(row_data)

    fin_df = pd.DataFrame(merged_rows)
    fin_df.to_csv(f"{BASE_DIR}/Solutions/{len(plant_locs)}/Output_financials_warm_start.csv", index=False)

    warm_start = {
        'Omega': {j: Omega[j].X for j in plant_locs},
        'N_CH4': {j: N_CH4[j].X for j in plant_locs},
        'x': {(i, f, j): x[i, f, j].X for i in supply_nodes for f in feedstock_types for j in plant_locs},
        'Y': {(j, a, c): Y[j, a, c].X for j in plant_locs for a in range(len(alternative_configs)) for c in capacity_levels},
        'm_up': {(j, f): m_up[j, f].X for j in plant_locs for f in feedstock_types},
        'Rev_loc': {j: Rev_loc[j].X for j in plant_locs},
        'Cost_loc': {j: Cost_loc[j].X for j in plant_locs},
        'Capex': {j: Capex[j].X for j in plant_locs},
        'Rev_alt_selected': {(j, a, c): Rev_alt_selected[j, a, c].X for j in plant_locs for a in range(len(alternative_configs)) for c in capacity_levels},
        'Cost_alt_selected': {(j, a, c): Cost_alt_selected[j, a, c].X for j in plant_locs for a in range(len(alternative_configs)) for c in capacity_levels},
    }
    with open(f'{BASE_DIR}/Solutions/{len(plant_locs)}/warm_start.pkl', 'wb') as f:
        pickle.dump(warm_start, f)

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    opt_time = opt_end_time - opt_start_time
    with open(f'{BASE_DIR}execution_times_warm_start.txt', 'a') as f:
        f.write(f"Total script execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)\n")
        f.write(f"Optimization time: {opt_time:.2f} seconds ({opt_time/60:.2f} minutes)\n")