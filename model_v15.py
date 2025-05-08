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

###############################################################################
# 1) LOAD DATA
###############################################################################
feedstock_df = pd.read_csv("C:/Master_Python/processed_biomass_data.csv")
plant_df = pd.read_csv("C:/Master_Python/equally_spaced_locations.csv")
distance_df = pd.read_csv("C:/Master_Python/Distance_Matrix.csv")
yields_df = pd.read_csv("C:/Master_Python/Feedstock_yields.csv")

# Filter feedstock_df to exclude nodes with < 30 tons
feedstock_df = feedstock_df[
    (feedstock_df["GISCO_ID"].notna()) &
    (feedstock_df["Centroid_Lon"].notna()) &
    (feedstock_df["Centroid_Lat"].notna()) &
    (feedstock_df["nutz_pot_tFM"] >= 20)  # New condition to filter < 30 tons
]

# Log the number of removed nodes
original_rows = len(pd.read_csv("C:/Master_Python/processed_biomass_data.csv"))
filtered_rows = len(feedstock_df)
print(f"Removed {original_rows - filtered_rows} supply nodes with < 30 tons of feedstock.")

# Verify distance matrix columns
expected_columns = ['Feedstock_LAU', 'Location', 'Distance_km']
for col in expected_columns:
    if col not in distance_df.columns:
        raise ValueError(f"Column '{col}' not found in Distance_Matrix.csv. Available columns: {distance_df.columns}")

# Update distance_df to only include GISCO_IDs present in filtered feedstock_df
valid_gisco_ids = set(feedstock_df['GISCO_ID'].unique())
distance_df = distance_df[distance_df['Feedstock_LAU'].isin(valid_gisco_ids)]
print(f"Distance matrix reduced to {len(distance_df)} rows after filtering invalid GISCO_IDs.")

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
capacity_levels = (250_000, 500_000)
FLH_max = 8000
alphaHV = 9.97
CN_min = 20.0
CN_max = 30.0
transport_cost_per_tkm = 1
digestate_transport_cost = 1
heat_price = 20
boiler_eff = 0.9
electricity_spot_price = 90
chp_elec_eff = 0.4
chp_heat_eff = 0.4
r = 0.05
years = 20
kappa = sum(1/(1+r)**t for t in range(1, years+1))
EEG_price_small = 210.0
EEG_price_med = 190.0
EEG_skip_chp_price = 194.3 * 10
EEG_skip_upg_price = 210.4
gas_price_mwh = 30
gas_price_m3 = gas_price_mwh * (alphaHV / 1000)
co2_price_ton = 0
co2_price = co2_price_ton / 556.2
Cap_biogas = 0.45
Cap_biomethane = 0.10
variable_upg_cost = 0.1
alphaMz = 0.3
alpha_GHG_comp = 94.0
alpha_GHG_lim = 0.35 * alpha_GHG_comp
GHG_certificate_price = 65.0
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

#Digestate hard values
capacity_dig = 27
loading_cost_dig = 37
cost_ton_km_dig = 0.104

######################### DISTANCE BETWEEN PLANTS #########################################

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
max_distance_threshold = 75  # km  (Set after inspection to avoid "cross border adjacent plants") (Set to 75km)
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
# Calculate statistics
max_distance = dist_df["Distance_km"].max() * 1.5

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
auction_chp_limit = 500000 * FLH_max / alphaHV / system_methane_average
auction_bm_limit = 300000 * FLH_max / alphaHV / system_methane_average
alternative_configs = [
    {"name": "nonEEG_CHP", "category": "CHP_nonEEG", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"spot": electricity_spot_price, "heat": heat_price},
     "EEG_flag": False, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_small1", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_small_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_small},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_large1", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_med_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_med},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "FlexEEG_biogas", "category": "FlexEEG_biogas", "prod_cap_factor": Cap_biogas, "max_cap_m3_year": None,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_skip_chp_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "FlexEEG_biomethane_tech1", "category": "FlexEEG_biomethane", "prod_cap_factor": Cap_biomethane, "max_cap_m3_year": None,
     "upg_cost_coeff": 980693, "upg_cost_exp": -0.991, "rev_price": {"EEG": EEG_skip_upg_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "Upgrading_tech1", "category": "Upgrading", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 980693, "upg_cost_exp": -0.991, "rev_price": {"gas": gas_price_m3, "co2": co2_price},
     "EEG_flag": False, "GHG_eligible": True, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "no_build", "category": "no_build", "prod_cap_factor": 0, "max_cap_m3_year": 0,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": 0, "spot": 0, "heat": 0},
     "EEG_flag": False, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff": 0, "capex_exp": 1, "capex_type": "standard",
     "opex_coeff": 0, "opex_exp": 1, "opex_type": "standard"}
]

'''alternative_configs = [
    {"name": "boiler", "category": "boiler", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"heat": heat_price},
     "EEG_flag": False, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff": 110000, "capex_exp": 1, "capex_type": "linear_MW",
     "opex_coeff": 3000, "opex_exp": 1, "opex_type": "fixed_variable_MW"},
    {"name": "nonEEG_CHP", "category": "CHP_nonEEG", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"spot": electricity_spot_price, "heat": heat_price},
     "EEG_flag": False, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_small1", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_small_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_small},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_small2", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_small_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_small},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 2,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_large1", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_med_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_med},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_large2", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_med_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_med},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 2,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "FlexEEG_biogas", "category": "FlexEEG_biogas", "prod_cap_factor": Cap_biogas, "max_cap_m3_year": None,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_skip_chp_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "FlexEEG_biomethane_tech1", "category": "FlexEEG_biomethane", "prod_cap_factor": Cap_biomethane, "max_cap_m3_year": None,
     "upg_cost_coeff": 980693, "upg_cost_exp": -0.991, "rev_price": {"EEG": EEG_skip_upg_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "FlexEEG_biomethane_tech2", "category": "FlexEEG_biomethane", "prod_cap_factor": Cap_biomethane, "max_cap_m3_year": None,
     "upg_cost_coeff": 239254, "upg_cost_exp": -0.696, "rev_price": {"EEG": EEG_skip_upg_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "FlexEEG_biomethane_tech3", "category": "FlexEEG_biomethane", "prod_cap_factor": Cap_biomethane, "max_cap_m3_year": None,
     "upg_cost_coeff": 81046, "upg_cost_exp": -0.534, "rev_price": {"EEG": EEG_skip_upg_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "FlexEEG_biomethane_tech4", "category": "FlexEEG_biomethane", "prod_cap_factor": Cap_biomethane, "max_cap_m3_year": None,
     "upg_cost_coeff": 980693, "upg_cost_exp": -0.991, "rev_price": {"EEG": EEG_skip_upg_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "FlexEEG_biomethane_tech5", "category": "FlexEEG_biomethane", "prod_cap_factor": Cap_biomethane, "max_cap_m3_year": None,
     "upg_cost_coeff": 185034, "upg_cost_exp": -0.67, "rev_price": {"EEG": EEG_skip_upg_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "Upgrading_tech1", "category": "Upgrading", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 980693, "upg_cost_exp": -0.991, "rev_price": {"gas": gas_price_m3, "co2": co2_price},
     "EEG_flag": False, "GHG_eligible": True, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "Upgrading_tech2", "category": "Upgrading", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 239254, "upg_cost_exp": -0.696, "rev_price": {"gas": gas_price_m3, "co2": co2_price},
     "EEG_flag": False, "GHG_eligible": True, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "Upgrading_tech3", "category": "Upgrading", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 81046, "upg_cost_exp": -0.534, "rev_price": {"gas": gas_price_m3, "co2": co2_price},
     "EEG_flag": False, "GHG_eligible": True, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "Upgrading_tech4", "category": "Upgrading", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 980693, "upg_cost_exp": -0.991, "rev_price": {"gas": gas_price_m3, "co2": co2_price},
     "EEG_flag": False, "GHG_eligible": True, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "Upgrading_tech5", "category": "Upgrading", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 185034, "upg_cost_exp": -0.67, "rev_price": {"gas": gas_price_m3, "co2": co2_price},
     "EEG_flag": False, "GHG_eligible": True, "feed_constraint": None,
     "capex_coeff":135.44, "capex_exp": -0.304, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "no_build", "category": "no_build", "prod_cap_factor": 0, "max_cap_m3_year": 0,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": 0, "spot": 0, "heat": 0},
     "EEG_flag": False, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff": 0, "capex_exp": 1, "capex_type": "standard",
     "opex_coeff": 0, "opex_exp": 1, "opex_type": "standard"}
]
'''

###############################################################################
# 3) GLOBAL PARAMETERS
###############################################################################
premium = {f: max(0, (alpha_GHG_comp - feed_yield[f]['GHG_intensity'])) * (alphaHV * 3.6) * GHG_certificate_price / 1e6 for f in feedstock_types}
threshold_m3 = (100 * FLH_max) / (chp_elec_eff * system_methane_average * alphaHV)
FLH_min_limit = 1000
M_large = max(capacity_levels) * 1.05
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

def add_maize_constraints(m, x, avail_mass, plant_locs, Y, alternative_configs, capacity_levels, alpha_mz=0.3):
    for j in plant_locs:
        total_feed = gp.quicksum(x[i, f, j] for i, f in avail_mass)
        maizeF = gp.quicksum(x[i, f, j] for i, f in avail_mass if is_maize_cereal(f))
        m.addConstr(maizeF <= alpha_mz * total_feed + M_large * (1 - gp.quicksum(Y[j, a, c] for a, alt in enumerate(alternative_configs) if alt["EEG_flag"] for c in capacity_levels)), name=f"MaizeCap_{j}")

def add_ghg_constraints(m, x, avail_mass, plant_locs, feed_yield, alpha_ghg_lim):
    for j in plant_locs:
        total_feed_j = gp.quicksum(x[i, f, j] for i, f in avail_mass)
        total_GHG_j = gp.quicksum(x[i, f, j] * feed_yield[f]['GHG_intensity'] for i, f in avail_mass)
        m.addConstr(total_GHG_j <= alpha_ghg_lim * total_feed_j, name=f"GHG_average_{j}")

def add_auction_constraints(m, Y, plant_locs, alternative_configs, capacity_levels):
    total_EEG_capacity = gp.quicksum(Y[j, a, c] * c for j in plant_locs for a, alt in enumerate(alternative_configs) if alt["EEG_flag"] for c in capacity_levels)
    m.addConstr(total_EEG_capacity <= auction_chp_limit, name="EEG_Auction_Limit")
    total_biogas_capacity = gp.quicksum(Y[j, a, c] * c for j in plant_locs for a, alt in enumerate(alternative_configs) if alt["EEG_flag"] and alt["category"] != "FlexEEG_biomethane" for c in capacity_levels)
    m.addConstr(total_biogas_capacity <= 500000 * FLH_max / alphaHV / system_methane_average, name="EEG_Biogas_Auction_Limit")
    total_biomethane_capacity = gp.quicksum(Y[j, a, c] * c for j in plant_locs for a, alt in enumerate(alternative_configs) if alt["category"] == "FlexEEG_biomethane" for c in capacity_levels)
    m.addConstr(total_biomethane_capacity <= 300000 * FLH_max / alphaHV / system_methane_average, name="EEG_Biomethane_Auction_Limit")

def add_flh_constraints(m, Omega, Y, plant_locs, capacity_levels, N_CH4):
    for j in plant_locs:
        cap_expr = gp.quicksum(c * Y[j, a, c] for a in range(len(alternative_configs)) for c in capacity_levels)
        m.addConstr(Omega[j] <= (FLH_max / 8760.0) * cap_expr, name=f"FLH_limit_{j}")
        m.addConstr(N_CH4[j] <= (FLH_max / 8760.0) * Omega[j], name=f"FLH_limit_NCH4{j}")

###############################################################################
# 5) MODEL FUNCTION
###############################################################################
def build_model(config):
    m = gp.Model("ShadowPlant_Biogas_Model")
    m.setParam("MIPFocus", 3)
    m.setParam("Threads", 12)
    m.setParam("NoRelHeurTime", 10)
    m.setParam("Presolve", 1)
    m.setParam("Heuristics", 0.2)
    m.setParam("NumericFocus", 1)
    m.setParam("Cuts", 2)

    # Define variables
    Omega = m.addVars(plant_locs, lb=0, ub = total_biogas, vtype=GRB.CONTINUOUS, name="Omega")
    N_CH4 = m.addVars(plant_locs, lb=0, ub=M_NCH4, vtype=GRB.CONTINUOUS, name="N_CH4")
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
    Y = {(j, a, c): m.addVar(vtype=GRB.BINARY, name=f"Y_{j}_{a}_{c}") for j in plant_locs for a in range(len(alternative_configs)) for c in capacity_levels}
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            for c in capacity_levels:
                if alt["category"] == "EEG_CHP_small" and c >= EEG_small_m3 + 1:
                    Y[j, a, c].ub = 0
                elif alt["category"] == "EEG_CHP_large" and (c <= EEG_small_m3 + 1 or c >= EEG_med_m3 + 1):
                    Y[j, a, c].ub = 0
    m_up = m.addVars(plant_locs, feedstock_types, lb=0, vtype=GRB.CONTINUOUS, name="m_up")
    Rev_loc = m.addVars(plant_locs, lb=0, vtype=GRB.CONTINUOUS, name="Rev_loc")
    Cost_loc = m.addVars(plant_locs, lb=0, vtype=GRB.CONTINUOUS, name="Cost_loc")
    # Auxiliary variable for linearizing Rev_alt * Y
    Rev_alt_selected = m.addVars(
        plant_locs, range(len(alternative_configs)), capacity_levels,
        lb=0, vtype=GRB.CONTINUOUS, name="Rev_alt_selected"
    )

    Cost_alt_selected = m.addVars(
        plant_locs, range(len(alternative_configs)), capacity_levels,
        lb=0, vtype=GRB.CONTINUOUS, name="Opex_alt_selected"
    )

    # Define bonus_dict for FlexEEG categories
    bonus_dict = {}
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            for c in capacity_levels:
                if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"] and c > threshold_m3:
                    bonus_dict[j, a, c] = 100 * (c * system_methane_average * chp_elec_eff * alphaHV) / FLH_max
                else:
                    bonus_dict[j, a, c] = 0

    # Add constraints
    for j in plant_locs:
        m.addConstr(Omega[j] <= gp.quicksum(c * Y[j, a, c] for a in range(len(alternative_configs)) for c in capacity_levels), name=f"Omega_Link_{j}")
        m.addConstr(gp.quicksum(Y[j, a, c] for a in range(len(alternative_configs)) for c in capacity_levels) <= 1, name=f"OneAlt_{j}")
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            if alt.get("max_cap_m3_year") is not None:
                for c in capacity_levels:
                    m.addConstr(Omega[j] <= alt["max_cap_m3_year"] + M_large * (1 - Y[j, a, c]), name=f"MaxCap_{j}_{a}_{c}")

    for j in plant_locs:
        m.addConstr(Omega[j] == gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] for i, f in avail_mass if dist_ik.get((i, j), float('inf')) <= max_distance), name=f"Omega_Feed_{j}")
        m.addConstr(N_CH4[j] == gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] * feed_yield[f]['ch4_content'] for i, f in avail_mass if dist_ik.get((i, j), float('inf')) <= max_distance), name=f"N_CH4_Feed_{j}")

    total_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if dist_ik.get((i, j), float('inf')) <= max_distance) for j in plant_locs}
    total_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if dist_ik.get((i, j), float('inf')) <= max_distance) for j in plant_locs}
    manure_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if is_manure(f) and dist_ik.get((i, j), float('inf')) <= max_distance) for j in plant_locs}
    clover_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if is_clover(f) and dist_ik.get((i, j), float('inf')) <= max_distance) for j in plant_locs}

    # Enforce FLH_min_limit for FlexEEG alternatives
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                for c in capacity_levels:
                    if c > threshold_m3:
                        m.addConstr(Omega[j] >= (FLH_min_limit / 8760.0) * c * Y[j, a, c],
                                    name=f"FLH_min_limit_{j}_{a}_{c}")

    add_eeg_constraints(m, total_feed, manure_feed, clover_feed, Y, plant_locs, alternative_configs, capacity_levels)
    add_supply_constraints(m, avail_mass, x, plant_locs, max_distance, dist_ik)
    add_digestate_constraints(m, x, digestate_return, supply_nodes, plant_locs, avail_mass, feed_yield, dist_pl_iprime, max_distance, config.get("digestate_return_frac", 0.99))
    add_cn_constraints(m, x, avail_mass, plant_locs, feed_yield, CN_min, CN_max)
    add_maize_constraints(m, x, avail_mass, plant_locs, Y, alternative_configs, capacity_levels, alphaMz)
    add_ghg_constraints(m, x, avail_mass, plant_locs, feed_yield, alpha_GHG_lim)
    add_auction_constraints(m, Y, plant_locs, alternative_configs, capacity_levels)
    add_flh_constraints(m, Omega, Y, plant_locs, capacity_levels, N_CH4)

    for j in plant_locs:
        for f in feedstock_types:
            production_f = gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] * feed_yield[f]['ch4_content'] for i in supply_nodes if dist_ik.get((i, j), float('inf')) <= max_distance)
            m.addConstr(m_up[j, f] <= production_f, name=f"m_up_upper_{j}_{f}")
        upgrading_flag = gp.quicksum(Y[j, a, c] for a, alt in enumerate(alternative_configs) if alt["category"] == "Upgrading" for c in capacity_levels)
        aux_upgrading = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"aux_upgrading_{j}")
        m.addConstr(aux_upgrading <= N_CH4[j], name=f"aux_upgrading_upper_{j}")
        m.addConstr(aux_upgrading <= M_NCH4[j] * upgrading_flag, name=f"aux_upgrading_bound_{j}")
        m.addConstr(aux_upgrading >= N_CH4[j] - M_NCH4[j] * (1 - upgrading_flag), name=f"aux_upgrading_lower_{j}")
        m.addConstr(gp.quicksum(m_up[j, f] for f in feedstock_types) == aux_upgrading, name=f"m_up_sum_{j}")
    
    M_rev = {}
    for j in plant_locs:
        max_elec_price = max(200, electricity_spot_price)  # Max of EEG, spot price (€/MWh)
        max_heat_price = heat_price  # €/MWh
        max_bonus = max(100 * (c * system_methane_average * chp_elec_eff * alphaHV) / FLH_max 
                        for c in capacity_levels if c > threshold_m3)
        M_rev[j] = (M_NCH4[j] * (chp_elec_eff * alphaHV / 1000.0) * max_elec_price + 
                    M_NCH4[j] * (chp_heat_eff * alphaHV / 1000.0) * max_heat_price + 
                max_bonus) * 1.1
        
    # Define M_cost for opex linearization
    M_cost = {}
    for j in plant_locs:
        max_cost = 0
        for a, alt in enumerate(alternative_configs):
            for c in capacity_levels:
                if alt["opex_type"] == "fixed_variable_MW":
                    MW = c * system_methane_average * chp_heat_eff * alphaHV / (FLH_max * 1000)
                    fixed_opex = alt["opex_coeff"] * MW
                    variable_opex = 0.5 * M_NCH4[j] * alphaHV * chp_heat_eff
                    cost_val = fixed_opex + variable_opex
                else:
                    cost_val = alt["opex_coeff"] * (c ** alt["opex_exp"]) if alt["category"] != "no_build" else 0
                max_cost = max(max_cost, cost_val)
        M_cost[j] = max_cost * 1.1  # Add 10% buffer

    # Revenue and cost computation
    Rev_alt = {}
    Cost_alt = {}
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            for c in capacity_levels:
                # Compute opex for cost_val
                if alt["opex_type"] == "fixed_variable_MW":
                    MW = c * system_methane_average * chp_heat_eff * alphaHV / (FLH_max * 1000)
                    fixed_opex = alt["opex_coeff"] * MW
                    variable_opex = 0.5 * N_CH4[j] * alphaHV * chp_heat_eff
                    cost_val = fixed_opex + variable_opex
                else:
                    cost_val = alt["opex_coeff"] * (c ** alt["opex_exp"]) if alt["category"] != "no_build" else 0
                
                if alt["category"] in ["Upgrading", "FlexEEG_biomethane"]:
                    cost_val += variable_upg_cost * N_CH4[j]

                if not alt["EEG_flag"]:
                    if alt["category"] == "CHP_nonEEG":
                        rev_val = N_CH4[j] * (alphaHV / 1000) * (chp_elec_eff * alt["rev_price"]["spot"] + chp_heat_eff * heat_price)
                    elif alt["category"] == "Upgrading":
                        rev_val = N_CH4[j] * alt["rev_price"]["gas"]
                    else:  # boiler or no_build
                        rev_val = N_CH4[j] * (alphaHV / 1000) * chp_heat_eff * alt["rev_price"]["heat"] if alt["category"] == "boiler" else 0

                else:
                    effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                    if alt["category"] in ["EEG_CHP_small", "EEG_CHP_large"]:
                        rev_val = N_CH4[j] * (alphaHV / 1000) * (chp_elec_eff * effective_EEG + chp_heat_eff * heat_price)
                    elif alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                        cap_fraction = Cap_biogas if a == 4 else Cap_biomethane
                        effective_EEG = alt["rev_price"]["EEG"] * avg_discount

                        # 1. Theoretical max electricity production (methane-unconstrained)
                        U_elec = c * (FLH_max / 8760) * chp_elec_eff * alphaHV / 1000.0
                        cap_production_elec = cap_fraction * U_elec  # EEG-eligible portion (45%)

                        # 2. Actual production (constrained by methane availability)
                        E_actual_var = m.addVar(lb=0, ub=U_elec, name=f"E_actual_{j}_{a}_{c}")
                        m.addConstr(E_actual_var == N_CH4[j] * (chp_elec_eff * alphaHV / 1000.0))

                        # 3. Split production into EEG and spot portions
                        E_EEG = m.addVar(lb=0, ub=cap_production_elec, name=f"E_EEG_{j}_{a}_{c}")
                        E_spot = m.addVar(lb=0, ub=U_elec, name=f"E_spot_{j}_{a}_{c}")

                        # 4. Link variables
                        m.addConstr(E_actual_var == E_EEG + E_spot)

                        # 5. Force EEG cap to be fully used before spot production
                        z = m.addVar(vtype=GRB.BINARY, name=f"z_spot_{j}_{a}_{c}")
                        m.addConstr(E_EEG >= cap_production_elec - M_rev[j] * (1 - z))  # If z=1, EEG >= cap
                        m.addConstr(E_EEG <= cap_production_elec)                        # EEG cannot exceed cap
                        m.addConstr(E_spot <= U_elec * z)                               # Spot only active if z=1

                        # 6. Revenue calculation
                        EEG_rev = E_EEG * effective_EEG
                        spot_rev = E_spot * electricity_spot_price
                        heat_rev = heat_price * E_actual_var
                        rev_val = EEG_rev + spot_rev + heat_rev + bonus_dict[j, a, c]

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


    # Link Rev_loc and Cost_loc
    for j in plant_locs:
        m.addConstr(Rev_loc[j] == gp.quicksum(Rev_alt_selected[j, a, c] for a in range(len(alternative_configs)) for c in capacity_levels), name=f"Rev_link_{j}")
        m.addConstr(Cost_loc[j] == gp.quicksum(Cost_alt_selected[j, a, c] for a in range(len(alternative_configs)) for c in capacity_levels), name=f"Cost_link_{j}")

    # Capex computation (in €)
    Capex = {}
    for j in plant_locs:
        capex_expr = gp.LinExpr()
        for a, alt in enumerate(alternative_configs):
            for c in capacity_levels:
                if alt["capex_type"] == "linear_MW":
                    MW = c * system_methane_average * chp_heat_eff * alphaHV / (FLH_max * 1000)
                    base_capex = alt["capex_coeff"] * MW
                else:
                    base_capex =  c * alt["capex_coeff"] * (c ** alt["capex_exp"]) if alt["category"] != "no_build" else 0
                extra_upg_cost = (alt["upg_cost_coeff"] * ((c / FLH_max) ** alt["upg_cost_exp"])) * (c / FLH_max) if alt["category"] in ["Upgrading", "FlexEEG_biomethane"] else 0
                capex_expr += Y[j, a, c] * (base_capex + extra_upg_cost)
        Capex[j] = m.addVar(lb=0, name=f"Capex_{j}")
        m.addConstr(Capex[j] == capex_expr, name=f"Capex_link_{j}")

    # Cost computations
    FeedstockCost = gp.LinExpr()
    for i, f in avail_mass:
        for j in plant_locs:
            if dist_ik.get((i, j), float('inf')) <= max_distance:
                flow = x[i, f, j]
                FeedstockCost.add(flow * feed_yield[f]['price'])
                dist_val = dist_ik.get((i, j), 0.0)
                if dist_val > 0:
                    FeedstockCost.add(flow * (feed_yield[f]['loading'] / feed_yield[f]['capacity_load'] +
                                             dist_val * feed_yield[f]['cost_ton_km']))
    DigestateCost = gp.LinExpr()
    for j in plant_locs:
        for i in supply_nodes:
            if dist_pl_iprime.get((j, i), float('inf')) <= max_distance:
                flow_d = digestate_return[j, i]
                dist_val = dist_pl_iprime.get((j, i), 0.0)
                DigestateCost.add(flow_d * (loading_cost_dig / capacity_dig +
                                           dist_val * cost_ton_km_dig))

    # Revenue and objective
    TotalRev = gp.quicksum(Rev_loc[j] for j in plant_locs)
    TotalCost = FeedstockCost + DigestateCost + gp.quicksum(Cost_loc[j] for j in plant_locs)
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

    return m, Omega, N_CH4, x, digestate_return, Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, DigestateCost, GHGRevenue, TotalCapex, bonus_dict, E_EEG, E_spot

###############################################################################
# 6) RUN MODEL
###############################################################################
config = {
    "name": "Baseline",
    "eeg_enabled": True,
    "supply_enabled": True,
    "digestate_enabled": True,
    "digestate_return_frac": 0.99,
    "cn_enabled": True,
    "maize_enabled": True,
    "ghg_enabled": True,
    "auction_enabled": True,
    "flh_enabled": True
}

m, Omega, N_CH4, x, digestate_return, Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, DigestateCost, GHGRevenue, TotalCapex, bonus_dict, E_EEG, E_spot= build_model(config)
m.update()  # Ensure model is up-to-date

print("Number of quadratic constraints:", m.NumQConstrs)  # Should print 15 or 30
for constr in m.getQConstrs():
    print(f"Quadratic constraint: {constr.QCName}")

m.optimize()

###############################################################################
# 16) PRINT RESULTS & PLOTTING
###############################################################################

# Sum the extra EEG payout for all plants and alternatives where capacity exceeds the threshold.
total_extra_EEG = 0.0
for j in plant_locs:
    for a, alt in enumerate(alternative_configs):
        if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
            for c in capacity_levels:
                if Y[j, a, c].X > 0.5 and c > threshold_m3:
                    total_extra_EEG += bonus_dict[j, a, c]


###############################################################################
# 7) PRINT RESULTS & PLOTTING
###############################################################################
def plot_biogas_flows_2x1(feed_arcs_, dig_arcs_, supply_coords_, plant_coords_, iPrime_coords_):
    fig, (ax_feed, ax_dig) = plt.subplots(2, 1, figsize=(8, 12))
    offset_factor = 0.02
    vertical_spacing = 0.25
    feed_label_counts = defaultdict(int)
    dig_label_counts = defaultdict(int)
    
    for iP, (lon, lat) in supply_coords_.items():
        ax_feed.scatter(lon, lat, color='blue')
        ax_feed.text(lon, lat, f"{iP}", color='blue', ha='right', va='center')
    for pl, (lon, lat) in plant_coords_.items():
        ax_feed.scatter(lon, lat, color='green', marker='^')
        ax_feed.text(lon, lat, f"{pl}", color='green', ha='left', va='bottom')
    for (iP, pl, flow_val, f) in feed_arcs_:
        (x1, y1) = supply_coords_[iP]
        (x2, y2) = plant_coords_[pl]
        lw = 1 + flow_val / 100
        ax_feed.plot([x1, x2], [y1, y2], color='blue', linewidth=lw, alpha=0.7)
        dx, dy = x2 - x1, y2 - y1
        norm = np.sqrt(dx**2 + dy**2)
        if norm != 0:
            px, py = -dy / norm, dx / norm
            label_offset = feed_label_counts[(iP, pl)] * vertical_spacing
            lx = x1 + 0.3 * (x2 - x1) + offset_factor * px
            ly = y1 + 0.3 * (y2 - y1) + offset_factor * py + label_offset
        else:
            lx, ly = x1, y1
        ax_feed.text(lx, ly, f"{flow_val:.1f}t\n{f}", color='blue', fontsize=8)
        feed_label_counts[(iP, pl)] += 1
    ax_feed.set_title("Feedstock Flows")
    ax_feed.set_xlabel("Longitude")
    ax_feed.set_ylabel("Latitude")
    
    for pl, (lon, lat) in plant_coords_.items():
        ax_dig.scatter(lon, lat, color='green', marker='^')
        ax_dig.text(lon, lat, f"{pl}", color='green', ha='right', va='bottom')
    for iP, (lon, lat) in iPrime_coords_.items():
        ax_dig.scatter(lon, lat, color='orange')
        ax_dig.text(lon, lat, f"{iP}", color='orange', ha='left', va='center')
    for (pl, iP, flow_val) in dig_arcs_:
        (x1, y1) = plant_coords_[pl]
        (x2, y2) = iPrime_coords_[iP]
        lw = 1 + flow_val / 100
        ax_dig.plot([x1, x2], [y1, y2], color='orange', linewidth=lw, alpha=0.7)
        dx, dy = x2 - x1, y2 - y1
        norm = np.sqrt(dx**2 + dy**2)
        if norm != 0:
            px, py = -dy / norm, dx / norm
            label_offset = dig_label_counts[(pl, iP)] * vertical_spacing
            lx = x1 + 0.3 * (x2 - x1) + offset_factor * px
            ly = y1 + 0.3 * (y2 - y1) + offset_factor * py + label_offset
        else:
            lx, ly = x1, y1
        ax_dig.text(lx, ly, f"{flow_val:.1f}t", color='orange', fontsize=8)
        dig_label_counts[(pl, iP)] += 1
    ax_dig.set_title("Digestate Flows")
    ax_dig.set_xlabel("Longitude")
    ax_dig.set_ylabel("Latitude")
    plt.tight_layout()
    plt.show()

def plot_methane_fraction(plant_locs, N_CH4, Omega, system_methane_average):
    methane_fractions = []
    valid_plants = []
    for j in plant_locs:
        omega_val = Omega[j].X
        if omega_val > 1e-6:
            n_ch4_val = N_CH4[j].X
            fraction = n_ch4_val / omega_val if omega_val > 0 else 0
            methane_fractions.append(fraction)
            valid_plants.append(j)
    if not valid_plants:
        print("No plants with non-zero production for methane fraction plot.")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(valid_plants, methane_fractions, color='blue', s=100, label='Plant Methane Fraction')
    ax.axhline(y=system_methane_average, color='red', linestyle='--', linewidth=2, label=f'System Average ({system_methane_average:.3f})')
    for j, frac in zip(valid_plants, methane_fractions):
        deviation = ((frac - system_methane_average) / system_methane_average) * 100
        ax.text(j, frac, f'{deviation:+.1f}%', fontsize=8, ha='center', va='bottom' if frac < system_methane_average else 'top')
    ax.set_xlabel('Plant Location')
    ax.set_ylabel('Methane Fraction (N_CH4 / Omega)')
    ax.set_title('Methane Fraction by Plant Location vs. System Average')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    min_frac = min(methane_fractions + [system_methane_average]) * 0.95
    max_frac = max(methane_fractions + [system_methane_average]) * 1.05
    ax.set_ylim(min_frac, max_frac)
    plt.tight_layout()
    plt.savefig('methane_fraction_plot.png')
    plt.show()

def plot_feedstock_stacked_chart(x, plant_locs, feedstock_types, avail_mass, output_file='feedstock_stacked_chart.png'):
    flow_data = []
    for j in plant_locs:
        total_flow = 0.0
        for i, f in avail_mass:
            flow_val = x[i, f, j].X
            if flow_val > 1e-6:
                flow_data.append({'Plant': j, 'Feedstock': f, 'FlowTons': flow_val})
                total_flow += flow_val
        if total_flow <= 1e-6:
            flow_data = [d for d in flow_data if d['Plant'] != j]
    df = pd.DataFrame(flow_data)
    if df.empty:
        print("No feedstock flows to plot.")
        return
    pivot_df = df.pivot_table(index='Plant', columns='Feedstock', values='FlowTons', fill_value=0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    for f in feedstock_types:
        if f not in pivot_df.columns:
            pivot_df[f] = 0.0
    pivot_df = pivot_df[feedstock_types]
    fig, ax = plt.subplots(figsize=(12, 8))
    plants = pivot_df.index
    bottoms = np.zeros(len(plants))
    for feedstock in feedstock_types:
        values = pivot_df[feedstock].values
        ax.bar(plants, values, bottom=bottoms, label=feedstock)
        bottoms += values
    ax.set_xlabel('Plant Location')
    ax.set_ylabel('Percentage of Feedstock (%)')
    ax.set_title('Feedstock Composition per Plant (100% Stacked)')
    ax.legend(title='Feedstock Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

if m.status == GRB.OPTIMAL:
    inflow_rows = []
    for j in plant_locs:
        for i, f in avail_mass:
            flow_val = x[i, f, j].X
            if flow_val > 1e-6:
                inflow_rows.append({
                    "SupplyNode": i,
                    "PlantLocation": j,
                    "Feedstock": f,
                    "FlowTons": flow_val
                })
    in_flow_df = pd.DataFrame(inflow_rows)
    in_flow_df.to_csv("Output_in_flow.csv", index=False)

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
    out_flow_df.to_csv("Output_out_flow.csv", index=False)

    merged_rows = []
    for j in plant_locs:
        for a in range(len(alternative_configs)):
            for c in capacity_levels:
                if Y[j, a, c].X > 0.5:
                    alt_name = alternative_configs[a]["name"]
                    merged_rows.append({
                        "PlantLocation": j,
                        "Alternative": alt_name,
                        "Capacity": c,
                        "Omega": Omega[j].X,
                        "N_CH4": N_CH4[j].X,
                        "Revenue": Rev_loc[j].X,
                        "Cost": Cost_loc[j].X,
                        "Capex": Capex[j].X
                    })
    fin_df = pd.DataFrame(merged_rows)
    fin_df.to_csv("Output_financials.csv", index=False)

    annual_net_cash_flow = TotalRev.getValue() + GHGRevenue.getValue() - TotalCost.getValue()
    print("Annual net cash flow: {:.2f} €/year".format(annual_net_cash_flow))
    print("Total Feedstock + Transport Cost: {:.2f} €/year".format(FeedstockCost.getValue()))
    print("Total Digestate Transport Cost: {:.2f} €/year".format(DigestateCost.getValue()))
    print(f"Optimal objective for all plants (NPV): {m.objVal:.2f} €")

    print("\n--- Alternative Selection and Production per Location ---")
    for j in plant_locs:
        for a in range(len(alternative_configs)):
            for c in capacity_levels:
                if Y[j, a, c].X > 0.5:
                    alt_name = alternative_configs[a]['name']
                    print(f"Location {j}:")
                    print(f"  Alternative: {alt_name}")
                    print(f"  Capacity: {c} m³/year")
                    print(f"  Production (Omega): {Omega[j].X:.2f} m³/year")
                    print(f"  Methane (N_CH4): {N_CH4[j].X:.2f} m³/year")
                    print(f"  CO₂ Production: {Omega[j].X - N_CH4[j].X:.2f} m³/year")
                    print(f"  Selling Prices: {alternative_configs[a]['rev_price']}")
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                for c in capacity_levels:
                    spot_val = E_spot[j, a, c].X
                    if spot_val > 0:
                        print(f"Spot production at {j}: {spot_val:.2f} kWh")

    '''
    #print("\n--- Feedstock Flows per Location ---")
    for j in plant_locs:
        #print(f"Location {j}:")
        for i, f in avail_mass:
            flow_val = x[i, f, j].X
            if flow_val > 1e-6:
                print(f"  Feedstock from {i} ({f}): {flow_val:.2f} tons/year")

    print("\n--- Digestate Flows per Location ---")
    for j in plant_locs:
        print(f"Location {j}:")
        total_digestate = 0.0
        for i in supply_nodes:
            flow_d = digestate_return[j, i].X
            if flow_d > 1e-6:
                print(f"  Digestate to {i}: {flow_d:.2f} tons/year")
                total_digestate += flow_d
        print(f"  Total Digestate Out: {total_digestate:.2f} tons/year")
    '''
    print("\n--- Upgrading Methane Flows (m_up) per Location ---")
    for j in plant_locs:
        print(f"Location {j}:")
        for f in feedstock_types:
            m_up_val = m_up[j, f].X
            if m_up_val > 1e-6:
                print(f"  Upgrading Methane for {f}: {m_up_val:.2f} m³/year")

    print("\n--- Global Revenue, Cost, and CAPEX per Location ---")
    for j in plant_locs:
        print(f"Location {j}: Rev = {Rev_loc[j].X:.2f} €, Cost = {Cost_loc[j].X:.2f} €, Capex = {Capex[j].X:.2f} €")

    print("\n--- GHG Premium Revenue (Global) ---")
    print(f"Total GHG Premium Revenue = {GHGRevenue.getValue():.2f} €")

    print("\n--- Flexibility Bonus Revenue (Global) ---")
    print("Total 100€/kW payout: {:.2f} €/year".format(total_extra_EEG))

    print("\n--- Upgrading CAPEX per Plant ---")
    for j in plant_locs:
        upg_capex = 0.0
        for a, alt in enumerate(alternative_configs):
            if alt["category"] in ["Upgrading", "FlexEEG_biomethane"]:
                for c in capacity_levels:
                    if Y[j, a, c].X > 0.5:
                        upg_capex += (alt["upg_cost_coeff"] * ((c / FLH_max) ** alt["upg_cost_exp"])) * (c / FLH_max)
        print(f"Upgrading CAPEX for plant {j}: {upg_capex:.2f} €")
    '''
    print("\nRevenue Components for Selected FlexEEG Configurations:")
    print("-" * 60)
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                for c in capacity_levels:
                    if Y[j, a, c].x > 0.5:  # Check if configuration is selected
                        effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                        heat_rev = heat_price * (N_CH4[j].x * chp_heat_eff * alphaHV / 1000)
                        bonus = bonus_dict[j, a, c]
                        print(f"Plant: {j}, Config: {alt['name']}, Capacity: {c} m³/year")
                        print(f"  EEG Revenue: €{EEG_rev:.2f}")
                        print(f"  Heat Revenue: €{heat_rev:.2f}")
                        print(f"  Bonus: €{bonus:.2f}")
                        print("-" * 60)
    '''
    params = {
        "feedstock_csv": r"C:/Master_Python/processed_biomass_data.csv",
        "plant_csv": r"C:/Master_Python/equally_spaced_locations.csv",
        "distance_csv": r"C:/Master_Python/Distance_Matrix.csv",
        "yields_csv": r"C:/Master_Python/Feedstock_yields.csv",
        "bavaria_geojson": r"C:/Master_Python/bavaria_lau_clean.geojson",
        "output_in_flow_csv": r"C:/Master_Python/Output_in_flow.csv",
        "supply_coords_csv": r"C:/Master_Python/supply_coords.csv",
        "merged_geojson_out": r"C:/Master_Python/methane_by_lau.geojson"
    }

    bavaria_geojson_path = params["bavaria_geojson"]
    bavaria_gdf = gpd.read_file(bavaria_geojson_path)
    bavaria_gdf = bavaria_gdf.to_crs(epsg=4326)

    inflow_csv = params["output_in_flow_csv"]
    yields_csv = params["yields_csv"]
    inflow_df = pd.read_csv(inflow_csv)
    yields_df = pd.read_csv(yields_csv)

    merged_df = inflow_df.merge(
        yields_df,
        left_on="Feedstock",
        right_on="substrat_ENG",
        how="left"
    )
    merged_df["DeliveredMethane_m3"] = (
        merged_df["FlowTons"] *
        merged_df["Biogas_Yield_m3_ton"] *
        (merged_df["Methane_Content_%"])
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

    supply_coords_df = pd.read_csv(params["supply_coords_csv"])
    supply_dict = {row["SupplyNode"]: (row["Lon"], row["Lat"]) for _, row in supply_coords_df.iterrows()}

    plant_coords_df = pd.read_csv(params["plant_csv"])
    plant_dict = {row["Location"]: (row["Longitude"], row["Latitude"]) for _, row in plant_coords_df.iterrows()}

    lines_agg = inflow_df.groupby(["SupplyNode", "PlantLocation"], as_index=False)["FlowTons"].sum()
    for _, row in lines_agg.iterrows():
        s_node = row["SupplyNode"]
        p_loc = row["PlantLocation"]
        if s_node not in supply_dict or p_loc not in plant_dict:
            continue
        (lon1, lat1) = supply_dict[s_node]
        (lon2, lat2) = plant_dict[p_loc]
        line = LineString([(lon1, lat1), (lon2, lat2)])
        ax.plot(*line.xy, color="black", linewidth=1.5, alpha=0.8)

        # Assign colors to alternatives
    unique_alternatives = sorted(set(alternative_configs[a]['name'] for j in plant_locs for a in range(len(alternative_configs)) for c in capacity_levels if Y[j, a, c].X > 0.5))

    # Hardcoded color mapping for alternatives
    alt_to_color = {
        'boiler': 'red',
        'nonEEG_CHP': 'blue',
        'EEG_CHP_small1': 'lightgreen',
        'EEG_CHP_small2': 'lightgreen',
        'EEG_CHP_large1': 'green',
        'EEG_CHP_large2': 'green',
        'FlexEEG_biogas': 'magenta',
        'FlexEEG_biomethane_tech1': 'pink',
        'FlexEEG_biomethane_tech2': 'pink',
        'FlexEEG_biomethane_tech3': 'pink',
        'FlexEEG_biomethane_tech4': 'pink',
        'FlexEEG_biomethane_tech5': 'pink',
        'Upgrading_tech1': 'purple',
        'Upgrading_tech2': 'purple',
        'Upgrading_tech3': 'purple',
        'Upgrading_tech4': 'purple',
        'Upgrading_tech5': 'purple',
        'no_build': None  # Excluded from plotting
    }

    # Scale marker sizes by capacity
    min_capacity = min(capacity_levels)
    max_capacity = max(capacity_levels)
    min_size = 50
    max_size = 200
    def scale_size(capacity):
        if max_capacity == min_capacity:
            return min_size
        return min_size + (max_size - min_size) * (capacity - min_capacity) / (max_capacity - min_capacity)

    # Collect plant data for plotting
    plant_points = []
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            for c in capacity_levels:
                if Y[j, a, c].X > 0.5 and alt["name"] != "no_build":
                    lon, lat = plant_dict[j]
                    plant_points.append({
                        "PlantLocation": j,
                        "geometry": Point(lon, lat),
                        "Alternative": alt["name"],
                        "Capacity": c
                    })

    if plant_points:
        plant_gdf = gpd.GeoDataFrame(plant_points, crs="EPSG:4326")
        for _, row in plant_gdf.iterrows():
            ax.scatter(
                row.geometry.x,
                row.geometry.y,
                marker='^',
                color=alt_to_color[row["Alternative"]],
                s=scale_size(row["Capacity"]),
                label=row["Alternative"],
                alpha=0.8,
                zorder=10  # Ensure markers are above lines
            )
            ax.annotate(
                text=row["PlantLocation"],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                color="black",
                zorder=11  # Ensure annotations are above markers and lines
            )

        # Create legend for alternatives
        handles = []
        labels = []
        plotted_alts = set()
        for _, row in plant_gdf.iterrows():
            alt = row["Alternative"]
            if alt not in plotted_alts and alt_to_color[alt] is not None:
                handles.append(plt.scatter([], [], color=alt_to_color[alt], marker='^', s=min_size, label=alt))
                labels.append(alt)
                plotted_alts.add(alt)

        # Create legend for capacity sizes
        representative_capacities = [min(capacity_levels), sum(capacity_levels)/2, max(capacity_levels)]
        for cap in representative_capacities:
            size = scale_size(cap)
            handles.append(plt.scatter([], [], color='gray', marker='^', s=size, label=f'{int(cap)} m³/year'))
            labels.append(f'Capacity: {int(cap)} m³/year')

        ax.legend(handles, labels, title="Alternatives & Capacities", loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()



    #plot_methane_fraction(plant_locs, N_CH4, Omega, system_methane_average)

    #plot_feedstock_stacked_chart(x, plant_locs, feedstock_types, avail_mass)

else:
    print("No optimal solution found.")