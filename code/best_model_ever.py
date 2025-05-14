import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import pickle
import time
import os

script_start_time = time.time()

# 1) LOAD DATA
BASE_DIR = "C:/Clone/Master/"
feedstock_df = pd.read_csv(f"{BASE_DIR}processed_biomass_data.csv")
plant_df = pd.read_csv(f"{BASE_DIR}equally_spaced_locations.csv")
distance_df = pd.read_csv(f"{BASE_DIR}Distance_Matrix.csv")
yields_df = pd.read_csv(f"{BASE_DIR}Feedstock_yields.csv")

feedstock_df = feedstock_df[
    (feedstock_df["GISCO_ID"].notna()) &
    (feedstock_df["Centroid_Lon"].notna()) &
    (feedstock_df["Centroid_Lat"].notna()) &
    (feedstock_df["nutz_pot_tFM"] >= 20)
]

original_rows = len(pd.read_csv(f"{BASE_DIR}processed_biomass_data.csv"))
filtered_rows = len(feedstock_df)

expected_columns = ['Feedstock_LAU', 'Location', 'Distance_km']
for col in expected_columns:
    if col not in distance_df.columns:
        raise ValueError(f"Column '{col}' not found in Distance_Matrix.csv. Available columns: {distance_df.columns}")

valid_gisco_ids = set(feedstock_df['GISCO_ID'].unique())
distance_df = distance_df[distance_df['Feedstock_LAU'].isin(valid_gisco_ids)]

supply_coords = {row['GISCO_ID']: (row['Centroid_Lon'], row['Centroid_Lat']) 
                 for _, row in feedstock_df.iterrows()}
plant_coords = {row['Location']: (row['Longitude'], row['Latitude']) 
                for _, row in plant_df.iterrows()}
iPrime_coords = supply_coords.copy()

feedstock_gisco = set(feedstock_df['GISCO_ID'].unique())
distance_gisco = set(distance_df['Feedstock_LAU'].unique())
if not distance_gisco.issubset(feedstock_gisco):
    missing = distance_gisco - feedstock_gisco
    raise ValueError(f"GISCO_IDs in Distance_Matrix.csv not found in processed_biomass_data.csv: {missing}")

# 2) SETS & DICTIONARIES
supply_nodes = feedstock_df['GISCO_ID'].unique().tolist()
iPrime_nodes = supply_nodes[:]
feedstock_types = yields_df['substrat_ENG'].unique().tolist()
plant_locs = plant_df['Location'].unique().tolist()
capacity_levels = (10_000_000, 20_000_000, 40_000_000, 75_000_000)
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

capacity_dig = 27
loading_cost_dig = 37
cost_ton_km_dig = 0.104

total_biogas = {}
for j in plant_locs:
    total_biogas[j] = sum(avail_mass[(i, f)] * feed_yield[f]['biogas_m3_per_ton'] for (i, f) in avail_mass) / 1e6  # Scale to millions

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
auction_chp_limit = 225000 * FLH_max / alphaHV / system_methane_average / 1e6  # Scale
auction_bm_limit = 125000 * FLH_max / alphaHV / system_methane_average / 1e6  # Scale
alternative_configs = [
    {"name": "no_build", "category": "no_build", "prod_cap_factor": 0, "max_cap_m3_year": 0,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": 0, "spot": 0, "heat": 0},
     "EEG_flag": False, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff": 0, "capex_exp": 1, "capex_type": "standard",
     "opex_coeff": 0, "opex_exp": 1, "opex_type": "standard"},
    {"name": "boiler", "category": "boiler", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"heat": heat_price},
     "EEG_flag": False, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff": 110000, "capex_exp": 1, "capex_type": "linear_MW",
     "opex_coeff": 3000, "opex_exp": 1, "opex_type": "fixed_variable_MW"},
    {"name": "nonEEG_CHP", "category": "CHP_nonEEG", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"spot": electricity_spot_price, "heat": heat_price},
     "EEG_flag": False, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_small1", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_small_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_small},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
     "capex_coeff":150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_small2", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_small_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_small},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 2,
     "capex_coeff":150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_large1", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_med_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_med},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
     "capex_coeff":150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_large2", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": EEG_med_m3,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_price_med},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 2,
     "capex_coeff":150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "FlexEEG_biogas", "category": "FlexEEG_biogas", "prod_cap_factor": Cap_biogas, "max_cap_m3_year": None,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_skip_chp_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "FlexEEG_biomethane_tech1", "category": "FlexEEG_biomethane", "prod_cap_factor": Cap_biomethane, "max_cap_m3_year": None,
     "upg_cost_coeff": 47777, "upg_cost_exp": -0.421, "rev_price": {"EEG": EEG_skip_upg_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff":150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "Upgrading_tech1", "category": "Upgrading", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 47777, "upg_cost_exp": -0.421, "rev_price": {"gas": gas_price_m3, "co2": co2_price},
     "EEG_flag": False, "GHG_eligible": True, "feed_constraint": None,
     "capex_coeff":150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
]

premium = {f: max(0, (alpha_GHG_comp - feed_yield[f]['GHG_intensity'])) * (alphaHV * 3.6) * GHG_certificate_price / 1e6 for f in feedstock_types}
threshold_m3 = (100 * FLH_max) / (chp_elec_eff * system_methane_average * alphaHV) / 1e6  # Scale
FLH_min_limit = 1000
M_large = max(capacity_levels) * 1.01 / 1e6  # Scale
avg_discount = sum(0.99**t for t in range(1, years+1)) / years

M_j = {j: sum(avail_mass[i, f] for i, f in avail_mass) / 1e6 for j in plant_locs}  # Scale
M_NCH4 = {j: total_biogas[j] * 0.7 for j in plant_locs}  # Already scaled

# 4) CONSTRAINT FUNCTIONS
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

def add_supply_constraints(m, avail_mass, x, plant_locs):
    for (i, f), amt in avail_mass.items():
        m.addConstr(gp.quicksum(x[i, f, j] for j in plant_locs) <= amt / 1e6, name=f"Supply_{i}_{f}")  # Scale

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
    total_EEG_capacity = gp.quicksum(Y[j, a, c] * (c / 1e6) for j in plant_locs for a, alt in enumerate(alternative_configs) if alt["EEG_flag"] for c in capacity_levels)
    m.addConstr(total_EEG_capacity <= auction_chp_limit, name="EEG_Auction_Limit")
    total_biogas_capacity = gp.quicksum(Y[j, a, c] * (c / 1e6) for j in plant_locs for a, alt in enumerate(alternative_configs) if alt["EEG_flag"] and alt["category"] != "FlexEEG_biomethane" for c in capacity_levels)
    m.addConstr(total_biogas_capacity <= 225000 * FLH_max / alphaHV / system_methane_average / 1e6, name="EEG_Biogas_Auction_Limit")
    total_biomethane_capacity = gp.quicksum(Y[j, a, c] * (c / 1e6) for j in plant_locs for a, alt in enumerate(alternative_configs) if alt["category"] == "FlexEEG_biomethane" for c in capacity_levels)
    m.addConstr(total_biomethane_capacity <= 125000 * FLH_max / alphaHV / system_methane_average / 1e6, name="EEG_Biomethane_Auction_Limit")

def add_flh_constraints(m, Omega, Y, plant_locs, capacity_levels, N_CH4):
    for j in plant_locs:
        cap_expr = gp.quicksum((c / 1e6) * Y[j, a, c] for a in range(len(alternative_configs)) for c in capacity_levels)
        m.addConstr(Omega[j] <= (FLH_max / 8760.0) * cap_expr, name=f"FLH_limit_{j}")
        m.addConstr(N_CH4[j] <= (FLH_max / 8760.0) * Omega[j], name=f"FLH_limit_NCH4{j}")

# 5) MODEL FUNCTION
def build_model(config):
    m = gp.Model("ShadowPlant_Biogas_Model")
    m.setParam("Heuristics", 0.3)
    m.setParam("NoRelHeurTime", 10)
    m.setParam("Cuts", 3)
    m.setParam("NumericFocus", 3)

    caps = capacity_levels

    Omega = m.addVars(plant_locs, lb=0, ub={j: total_biogas[j] for j in plant_locs}, vtype=GRB.CONTINUOUS, name="Omega")
    N_CH4 = m.addVars(plant_locs, lb=0, vtype=GRB.CONTINUOUS, name="N_CH4")
    x = m.addVars(
        supply_nodes, feedstock_types, plant_locs,
        lb=0,
        ub={(i, f, j): avail_mass.get((i, f), 0) / 1e6 for i in supply_nodes for f in feedstock_types for j in plant_locs},
        vtype=GRB.CONTINUOUS,
        name="x"
    )
    for i in supply_nodes:
        for f in feedstock_types:
            if (i, f) not in avail_mass:
                for j in plant_locs:
                    m.addConstr(x[i, f, j] == 0, name=f"ZeroFlow_{i}_{f}_{j}")

    Y = {(j, a, c): m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"Y_{j}_{a}_{c}")
         for j in plant_locs for a in range(len(alternative_configs)) for c in caps}

# Add auxiliary binary variable to indicate active plant
    is_active = m.addVars(plant_locs, vtype=GRB.BINARY, name="is_active")

    for j in plant_locs:
        y_vars = [Y[j, a, c] for a in range(len(alternative_configs)) for c in caps]
        weights = [a * len(caps) + caps.index(c) + 1 for a in range(len(alternative_configs)) for c in caps]
        m.addSOS(GRB.SOS_TYPE1, y_vars, weights)
        m.addConstr(gp.quicksum(Y[j, a, c] for a in range(len(alternative_configs)) for c in caps) <= 1, name=f"OneAlt_{j}")
        # Link is_active to Omega
        m.addConstr(Omega[j] <= total_biogas[j] * is_active[j], name=f"ActiveOmegaUpper_{j}")
        m.addConstr(Omega[j] >= 1e-6 * is_active[j], name=f"ActiveOmegaLower_{j}")
        # Enforce sum(Y) == is_active
        m.addConstr(gp.quicksum(Y[j, a, c] for a in range(len(alternative_configs)) for c in caps) == is_active[j], name=f"ActivePlant_{j}")

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

    bonus_dict = {}
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            for c in caps:
                if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"] and c / 1e6 > threshold_m3:
                    bonus_dict[j, a, c] = 100 * (c / 1e6 * system_methane_average * chp_elec_eff * alphaHV) / FLH_max
                else:
                    bonus_dict[j, a, c] = 0

    for j in plant_locs:
        m.addConstr(Omega[j] <= gp.quicksum((c / 1e6) * Y[j, a, c] for a in range(len(alternative_configs)) for c in caps), name=f"Omega_Link_{j}")
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            if alt.get("max_cap_m3_year") is not None:
                for c in caps:
                    m.addConstr(Omega[j] <= alt["max_cap_m3_year"] / 1e6 + M_large * (1 - Y[j, a, c]), name=f"MaxCap_{j}_{a}_{c}")

    for j in plant_locs:
        m.addConstr(Omega[j] == gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] for i, f in avail_mass), name=f"Omega_Feed_{j}")
        m.addConstr(N_CH4[j] == gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] * feed_yield[f]['ch4_content'] for i, f in avail_mass), name=f"N_CH4_Feed_{j}")

    total_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass) for j in plant_locs}
    manure_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if is_manure(f)) for j in plant_locs}
    clover_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if is_clover(f)) for j in plant_locs}

    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                for c in caps:
                    if c / 1e6 > threshold_m3:
                        m.addConstr(Omega[j] >= (FLH_min_limit / 8760.0) * (c / 1e6) * Y[j, a, c], name=f"FLH_min_limit_{j}_{a}_{c}")

    if config["eeg_enabled"] and any(alt["category"].startswith("EEG_CHP") for alt in alternative_configs):
        add_eeg_constraints(m, total_feed, manure_feed, clover_feed, Y, plant_locs, alternative_configs, caps)
    add_supply_constraints(m, avail_mass, x, plant_locs)
    add_cn_constraints(m, x, avail_mass, plant_locs, feed_yield, CN_min, CN_max)
    add_ghg_constraints(m, x, avail_mass, plant_locs, feed_yield, alpha_GHG_lim)
    add_auction_constraints(m, Y, plant_locs, alternative_configs, caps)
    add_flh_constraints(m, Omega, Y, plant_locs, caps, N_CH4)

    for j in plant_locs:
        for f in feedstock_types:
            production_f = gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] * feed_yield[f]['ch4_content'] for i in supply_nodes)
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
        max_bonus = 2_500_000
        M_rev[j] = (M_NCH4[j] * (chp_elec_eff * alphaHV / 1000.0) * max_elec_price + 
                    M_NCH4[j] * (chp_heat_eff * alphaHV / 1000.0) * max_heat_price + 
                    max_bonus) * 1.1

    M_cost = {}
    for j in plant_locs:
        max_cost = 0
        for a, alt in enumerate(alternative_configs):
            for c in caps:
                if alt["opex_type"] == "fixed_variable_MW":
                    MW = (c / 1e6) * system_methane_average * chp_heat_eff * alphaHV / (FLH_max * 1000)
                    fixed_opex = alt["opex_coeff"] * MW
                    variable_opex = 0.5 * M_NCH4[j] * alphaHV * chp_heat_eff
                    cost_val = fixed_opex + variable_opex
                else:
                    cost_val = (alt["opex_coeff"] * ((c ) ** alt["opex_exp"]))/1e6
                max_cost = max(max_cost, cost_val)
        M_cost[j] = max_cost * 1.1

    Rev_alt = {}
    Cost_alt = {}
    Z_rev = m.addVars(plant_locs, range(len(alternative_configs)), caps, lb=0, vtype=GRB.CONTINUOUS, name="Z_rev")
    Z_cost = m.addVars(plant_locs, range(len(alternative_configs)), caps, lb=0, vtype=GRB.CONTINUOUS, name="Z_cost")

    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            for c in caps:
                if alt["opex_type"] == "fixed_variable_MW":
                    MW = c * system_methane_average * chp_heat_eff * alphaHV / (FLH_max * 1000)
                    fixed_opex = alt["opex_coeff"] * MW / 1e6
                    variable_opex = 0.5 * N_CH4[j] * alphaHV * chp_heat_eff
                    cost_val = fixed_opex + variable_opex                    
                else: 
                    cost_val = (alt["opex_coeff"] * c ** alt["opex_exp"]) / 1e6

                if alt["category"] in ["Upgrading", "FlexEEG_biomethane"]:
                    cost_val += variable_upg_cost * N_CH4[j] /1e6
                if not alt["EEG_flag"]:
                    if alt["category"] == "CHP_nonEEG":
                        rev_val = N_CH4[j] * (alphaHV / 1000) * (chp_elec_eff * alt["rev_price"]["spot"] + chp_heat_eff * heat_price)
                    elif alt["category"] == "Upgrading":
                        rev_val = N_CH4[j] * alt["rev_price"]["gas"] + (Omega[j] - N_CH4[j]) * alt["rev_price"]["co2"]
                    else:
                        rev_val = N_CH4[j] * (alphaHV / 1000) * (chp_heat_eff * heat_price) if alt["category"] == "boiler" else 0
                else:
                    effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                    if alt["category"] in ["EEG_CHP_small", "EEG_CHP_large"]:
                        rev_val = N_CH4[j] * (alphaHV / 1000) * (chp_elec_eff * effective_EEG + chp_heat_eff * heat_price)
                    elif alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                        if alt["category"] == "FlexEEG_biogas":
                            cap_fraction = Cap_biogas
                        else: 
                            cap_fraction = Cap_biomethane

                        effective_EEG = alt["rev_price"]["EEG"] * avg_discount

                        E_actual_var = m.addVar(lb=0, name=f"E_actual_{j}_{a}_{c}")
                        m.addConstr(
                            E_actual_var == N_CH4[j] * (chp_elec_eff * alphaHV / 1000.0),
                            name=f"E_actual_constr_{j}_{a}_{c}"
                        )
                        U_elec = (c / 1e6) * (FLH_max / 8760) * system_methane_average * chp_elec_eff * alphaHV / 1000.0
                        cap_production_elec = cap_fraction * U_elec
                        m.addConstr(
                            E_actual_var >= cap_production_elec,
                            name=f"MinProd_{j}_{a}_{c}"
                        )
                        EEG_rev = cap_production_elec * effective_EEG
                        spot_rev = (E_actual_var - cap_production_elec) * electricity_spot_price
                        heat_rev = heat_price * (N_CH4[j] * chp_heat_eff * alphaHV / 1000.0)
                        rev_val = EEG_rev + spot_rev + heat_rev + bonus_dict[j, a, c]
                    elif alt["category"] == "None":
                        rev_val = 0
                        cost_val = 0
                    else:
                        raise ValueError(f"Unexpected alternative category: {alt['category']}")
                    Rev_alt[j, a, c] = rev_val
                    Cost_alt[j, a, c] = cost_val
                    
                    m.addConstr(Z_rev[j, a, c] <= rev_val, name=f"Z_rev_upper1_{j}_{a}_{c}")
                    m.addConstr(Z_rev[j, a, c] <= M_rev[j] * Y[j, a, c], name=f"Z_rev_upper2_{j}_{a}_{c}")
                    m.addConstr(Z_rev[j, a, c] >= rev_val - M_rev[j] * (1 - Y[j, a, c]), name=f"Z_rev_lower_{j}_{a}_{c}")
                    m.addConstr(Z_rev[j, a, c] >= 0, name=f"Z_rev_nonneg_{j}_{a}_{c}")
                    m.addConstr(Rev_alt_selected[j, a, c] == Z_rev[j, a, c], name=f"Rev_alt_sel_{j}_{a}_{c}")

                    m.addConstr(Z_cost[j, a, c] <= cost_val, name=f"Z_cost_upper1_{j}_{a}_{c}")
                    m.addConstr(Z_cost[j, a, c] <= M_cost[j] * Y[j, a, c], name=f"Z_cost_upper2_{j}_{a}_{c}")
                    m.addConstr(Z_cost[j, a, c] >= cost_val - M_cost[j] * (1 - Y[j, a, c]), name=f"Z_cost_lower_{j}_{a}_{c}")
                    m.addConstr(Z_cost[j, a, c] >= 0, name=f"Z_cost_nonneg_{j}_{a}_{c}")
                    m.addConstr(Cost_alt_selected[j, a, c] == Z_cost[j, a, c], name=f"Cost_alt_sel_{j}_{a}_{c}")

    for j in plant_locs:
        m.addConstr(Rev_loc[j] == gp.quicksum(Rev_alt_selected[j, a, c] for a in range(len(alternative_configs)) for c in caps), name=f"Rev_link_{j}")
        m.addConstr(Cost_loc[j] == gp.quicksum(Cost_alt_selected[j, a, c] for a in range(len(alternative_configs)) for c in caps), name=f"Cost_link_{j}")

    Capex = {}
    for j in plant_locs:
        capex_expr = gp.LinExpr()
        for a, alt in enumerate(alternative_configs):
            for c in caps:
                if c == 0:
                    continue
                elif alt["capex_type"] == "linear_MW":
                    MW = c * system_methane_average * chp_heat_eff * alphaHV / (FLH_max * 1000)
                    base_capex = alt["capex_coeff"] * MW / 1e6
                else:
                    base_capex = (c * (alt["capex_coeff"] * (c ** alt["capex_exp"])))/1e6 if alt["category"] != "no_build" else 0
                extra_upg_cost = (
                    (alt["upg_cost_coeff"] * ((c / FLH_max) ** alt["upg_cost_exp"]) * (c / FLH_max))/1e6
                    if alt["category"] == "Upgrading"
                    else 0
                )
                capex_expr += Y[j, a, c] * (base_capex + extra_upg_cost)
        Capex[j] = m.addVar(lb=0, name=f"Capex_{j}")
        m.addConstr(Capex[j] == capex_expr, name=f"Capex_link_{j}")

    FeedstockCost = gp.LinExpr()
    DigestateFlows = {}  # Optional: for tracking digestate flows
    for i, f in avail_mass:
        for j in plant_locs:
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
    penalty = 1e-3 * gp.quicksum(Y[j, a, c] for j in plant_locs for a in range(len(alternative_configs)) for c in caps)
    NPV_expr -= penalty
    m.setObjective(NPV_expr, GRB.MAXIMIZE)

    return m, Omega, N_CH4, x, Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, GHGRevenue, TotalCapex, bonus_dict, Rev_alt_selected, Cost_alt_selected

# 6) RUN MODEL
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

if __name__ == '__main__':
    print("Running full model...")
    m, Omega, N_CH4, x, Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, GHGRevenue, TotalCapex, bonus_dict, Rev_alt_selected, Cost_alt_selected = build_model(config)
    m.update()
    print(f"  Quadratic constraints: {m.NumQConstrs}")
    print(f"  Quadratic objective terms (non-zeros): {m.NumQNZs}")
    opt_start_time = time.time()
    m.optimize()
    opt_end_time = time.time()
    
    if m.status == GRB.OPTIMAL:

        print(f"\nFull model: NPV = {m.objVal:,.2f} €, Solve time = {m.Runtime:.2f} s, MIP Gap = {m.MIPGap:.4f}")
        for j in plant_locs:
            if Omega[j].X > 1e-6:
                print(f"Plant {j}: N_CH4 = {N_CH4[j].X * 1e6:,.0f}, Omega = {Omega[j].X * 1e6:,.0f}, CH4 Fraction = {N_CH4[j].X/Omega[j].X:.3f}")
    else:
        print(f"Full model: No optimal solution found (status: {m.status}).")

    inflow_rows = []
    for j in plant_locs:
        for i, f in avail_mass:
            flow_val = x[i, f, j].X
            if flow_val > 1e-6:
                distance = dist_ik.get((i, j), 0.0)
                inflow_rows.append({
                    "SupplyNode": i,
                    "PlantLocation": j,
                    "Feedstock": f,
                    "FlowTons": flow_val * 1e6,
                    "Distance_km": distance
                })
    in_flow_df = pd.DataFrame(inflow_rows)
    in_flow_df.to_csv(f"{BASE_DIR}/Solutions/Debug/Output_in_flow.csv", index=False)

    print("\nDebugging Y[j, a, c].X values:")
    for j in plant_locs:
        if Omega[j].X > 1e-6:
            print(f"Plant {j}:")
            for a in range(len(alternative_configs)):
                for c in capacity_levels:
                    y_val = Y[j, a, c].X
                    if y_val > 1e-6:
                        alt_name = alternative_configs[a]["name"]
                        print(f"  Alternative {alt_name}, Capacity {c:,}: Y = {y_val:.6f}")

    for j in plant_locs:
        no_build_selected = False
        for a in range(len(alternative_configs)):
            if alternative_configs[a]["name"] == "No_build":
                for c in capacity_levels:
                    if Y[j, a, c].X > 0.1:
                        no_build_selected = True
                        print(f"Plant {j} selected No_build with Y[{j}, {a}, {c}] = {Y[j, a, c].X:.6f}")
        if no_build_selected and Omega[j].X > 1e-6:
            print(f"Warning: Plant {j} has Omega = {Omega[j].X * 1e6:,.0f} but selected No_build")

    merged_rows = []
    for j in plant_locs:
        for a in range(len(alternative_configs)):
            for c in capacity_levels:
                if Y[j, a, c].X > 0.1:
                    alt = alternative_configs[a]
                    alt_name = alt["name"]
                    cap_fraction = Cap_biogas if alt["category"] == "FlexEEG_biogas" else Cap_biomethane if alt["category"] == "FlexEEG_biomethane" else None
                    
                    row_data = {
                        "PlantLocation": j,
                        "Alternative": alt_name,
                        "Capacity": c,
                        "Omega": Omega[j].X * 1e6,
                        "N_CH4": N_CH4[j].X * 1e6,
                        "CO2_Production": (Omega[j].X - N_CH4[j].X) * 1e6,
                        "Revenue": Rev_loc[j].X,
                        "Cost": Cost_loc[j].X,
                        "Capex": Capex[j].X,
                        "GHG": sum(premium[f] * m_up[j, f].X for f in feedstock_types),
                        "FLH": (Omega[j].X / (c / 1e6)) * 8760 if c > 0 else 0,
                        "PlantLatitude": plant_coords.get(j, (None, None))[1],
                        "PlantLongitude": plant_coords.get(j, (None, None))[0]
                    }
                    if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                        effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                        E_actual = N_CH4[j].X * (chp_elec_eff * alphaHV / 1000.0)
                        EEG_rev = cap_fraction * E_actual * effective_EEG if cap_fraction else 0
                        spot_rev = (E_actual - (cap_fraction * E_actual if cap_fraction else 0)) * electricity_spot_price
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
    print(f"Saving financials with {len(merged_rows)} rows")
    fin_df.to_csv(f"{BASE_DIR}/Solutions/Debug/Output_financials.csv", index=False)

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    opt_time = opt_end_time - opt_start_time
    with open(f'{BASE_DIR}/Solutions/Debug/execution_times.txt', 'a') as f:
        f.write(f"Total script execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)\n")
        f.write(f"Optimization time: {opt_time:.2f} seconds ({opt_time/60:.2f} minutes)\n")