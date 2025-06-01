import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import pickle
import time
import os
import numpy_financial as nf
from itertools import combinations, product

script_start_time = time.time()

# 1) LOAD DATA
try:
    BASE_DIR = "/home/fredrgaa/Master/"
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError("Linux path does not exist")
except FileNotFoundError:
    BASE_DIR = "C:/Clone/Master/"
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError("Neither Linux nor Windows path exists")

# Use BASE_DIR in your script
output_dir = os.path.join(BASE_DIR, "results/large_scale_cont/10_greedy_with_alternatives/k_swap/")
os.makedirs(output_dir, exist_ok=True)

feedstock_df = pd.read_csv(f"{BASE_DIR}aggregated_bavaria_supply_nodes.csv")
plant_df = pd.read_csv(f"{BASE_DIR}equally_spaced_locations_50.csv")
distance_df = pd.read_csv(f"{BASE_DIR}Distance_Matrix_50.csv")
yields_df = pd.read_csv(f"{BASE_DIR}Feedstock_yields.csv")

feedstock_df = feedstock_df[
    (feedstock_df["GISCO_ID"].notna()) &
    (feedstock_df["Centroid_Lon"].notna()) &
    (feedstock_df["Centroid_Lat"].notna()) &
    (feedstock_df["nutz_pot_tFM"] >= 10)
]

original_rows = len(pd.read_csv(f"{BASE_DIR}aggregated_bavaria_supply_nodes.csv"))
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
capacity_levels = (20_000_000, 40_000_000, 60_000_000)
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
EEG_price_small = 220.0
EEG_price_med = 190.0
EEG_skip_chp_price = 194.3
EEG_skip_upg_price = 210.4
gas_price_mwh = 30
gas_price_m3 = gas_price_mwh * (alphaHV / 1000)
co2_price_ton = 50
co2_price = co2_price_ton / 556.2
Cap_biogas = 0.45
Cap_biomethane = 0.10
variable_upg_cost = 0.2
alpha_GHG_comp = 94.0
alpha_GHG_lim = 0.35 * alpha_GHG_comp
GHG_certificate_price = 70
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
auction_chp_limit = 225000 * FLH_max / alphaHV / system_methane_average / 1e6
auction_bm_limit = 125000 * FLH_max / alphaHV / system_methane_average / 1e6

alternative_configs = [
    {"name": "FlexEEG_biogas", "category": "FlexEEG_biogas", "prod_cap_factor": Cap_biogas, "max_cap_m3_year": None,
     "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": EEG_skip_chp_price},
     "EEG_flag": True, "GHG_eligible": False, "feed_constraint": None,
     "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "Upgrading_tech1", "category": "Upgrading", "prod_cap_factor": 1.0, "max_cap_m3_year": None,
     "upg_cost_coeff": 47777, "upg_cost_exp": -0.421, "rev_price": {"gas": gas_price_m3, "co2": co2_price},
     "EEG_flag": False, "GHG_eligible": True, "feed_constraint": None,
     "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
     "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_small1", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": 255870,
        "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": 220},
        "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
        "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
        "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_small2", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": 255870,
        "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": 220},
        "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 2,
        "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
        "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_large1", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": 511740,
        "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": 190},
        "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
        "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
        "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    {"name": "EEG_CHP_large2", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": 511740,
        "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": 190},
        "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 2,
        "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
        "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
]

premium = {f: max(0, (alpha_GHG_comp - feed_yield[f]['GHG_intensity'])) * (alphaHV * 3.6) * GHG_certificate_price / 1e6 for f in feedstock_types}
threshold_m3 = (100 * FLH_max) / (chp_elec_eff * system_methane_average * alphaHV) / 1e6
FLH_min_limit = 1000
M_large = max(capacity_levels) * 1.01 / 1e6
avg_discount = sum(0.99**t for t in range(1, years+1)) / years
M_j = {j: sum(avail_mass[i, f] for i, f in avail_mass) / 1e6 for j in plant_locs}

# 3) CONSTRAINT FUNCTIONS
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
    m.addConstrs(
        (gp.quicksum(x[i, f, j] for j in plant_locs) <= amt / 1e6
         for (i, f), amt in avail_mass.items()),
        name="Supply"
    )

def add_cn_constraints(m, x, avail_mass, plant_locs, feed_yield, cn_min=20.0, cn_max=30.0):
    m.addConstrs(
        (gp.quicksum(x[i, f, j] * feed_yield[f]['CN'] for i, f in avail_mass) >=
         cn_min * gp.quicksum(x[i, f, j] for i, f in avail_mass)
         for j in plant_locs),
        name="CN_min"
    )
    m.addConstrs(
        (gp.quicksum(x[i, f, j] * feed_yield[f]['CN'] for i, f in avail_mass) <=
         cn_max * gp.quicksum(x[i, f, j] for i, f in avail_mass)
         for j in plant_locs),
        name="CN_max"
    )

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
    cap_coeff = {(a, c): c / 1e6 for a in range(len(alternative_configs)) for c in capacity_levels}
    m.addConstrs(
        (Omega[j] <= (FLH_max / 8760.0) * gp.quicksum(cap_coeff[a, c] * Y[j, a, c] for a, c in cap_coeff)
         for j in plant_locs),
        name="FLH_limit"
    )
    m.addConstrs(
        (N_CH4[j] <= (FLH_max / 8760.0) * Omega[j]
         for j in plant_locs),
        name="FLH_limit_NCH4"
    )

config = {
    "name": "Baseline",
    "eeg_enabled": True,
    "supply_enabled": True,
    "digestate_enabled": False,
    "digestate_return_frac": 0.99,
    "cn_enabled": True,
    "maize_enabled": False,
    "ghg_enabled": False,
    "auction_enabled": True,
    "flh_enabled": True
}

# 4) MODEL FUNCTION
def build_model(config, fixed_assignments, warmstart_vars=None):
    m = gp.Model("ShadowPlant_Biogas_Model")
    m.setParam("NodefileStart", 50)
    m.setParam("TimeLimit", 1800)  # 30 minutes per optimization

    Omega = m.addVars(plant_locs, lb=0, ub=max(capacity_levels) / 1e6, name="Omega")
    caps = capacity_levels
    ub_ch4 = max(capacity_levels) * 0.7 / 1e6
    N_CH4 = m.addVars(plant_locs, lb=0, ub=ub_ch4, vtype=GRB.CONTINUOUS, name="N_CH4")
    m_up = m.addVars(plant_locs, feedstock_types, lb=0, ub=ub_ch4, vtype=GRB.CONTINUOUS, name="m_up")
    MAX_DIST = 150
    x = m.addVars(
        supply_nodes, feedstock_types, plant_locs,
        lb=0,
        ub={(i, f, j): (avail_mass.get((i, f), 0) / 1e6) if dist_ik.get((i, j), 9999) <= MAX_DIST else 0.0
            for i in supply_nodes for f in feedstock_types for j in plant_locs},
        vtype=GRB.CONTINUOUS,
        name="x"
    )
    zero_triplets = [(i, f, j) for i in supply_nodes for f in feedstock_types if (i, f) not in avail_mass for j in plant_locs]
    m.addConstrs((x[i, f, j] == 0 for i, f, j in zero_triplets), name="ZeroFlow")

    Y = {(j, a, c): m.addVar(vtype=GRB.BINARY, name=f"Y_{j}_{a}_{c}")
         for j in plant_locs for a in range(len(alternative_configs)) for c in caps}

    # Apply fixed assignments from k-swap or greedy solution
    alt_index = {cfg["name"]: i for i, cfg in enumerate(alternative_configs)}
    for j in plant_locs:
        assignment = fixed_assignments.get(j)
        if assignment is not None:
            a, c = assignment
            m.addConstr(Y[j, a, c] == 1,
                        name=f"FixAltCap_{j}_{alternative_configs[a]['name']}_{c}")
            # force all others off
            for a2 in range(len(alternative_configs)):
                for c2 in capacity_levels:
                    if (a2, c2) != (a, c):
                        m.addConstr(Y[j, a2, c2] == 0,
                                    name=f"Off_{j}_{a2}_{c2}")
            # cap on Omega
            m.addConstr(Omega[j] <= c / 1e6,
                        name=f"CapUpper_{j}")
        else:
            # no fixed assignment → allow any single choice
            m.addConstr(
                gp.quicksum(Y[j, a2, c2]
                                for a2 in range(len(alternative_configs))
                                for c2 in capacity_levels) <= 1,
                name=f"OneAlt_{j}"
            )
            m.addConstr(
                Omega[j] <= (max(capacity_levels) / 1e6)
                            * gp.quicksum(Y[j, a2, c2]
                                            for a2 in range(len(alternative_configs))
                                            for c2 in capacity_levels),
                name=f"OmegaActive_{j}"
            )

    UpgSel = {j: m.addVar(vtype=GRB.BINARY, name=f"UpgSel_{j}") for j in plant_locs}
    for j in plant_locs:
        m.addConstr(UpgSel[j] == gp.quicksum(Y[j, a, c] for a, alt in enumerate(alternative_configs) if alt["category"] == "Upgrading" for c in caps), name=f"LinkUpg_{j}")
        m.addGenConstrIndicator(UpgSel[j], True, gp.quicksum(m_up[j, f] for f in feedstock_types) == N_CH4[j], name=f"UpgBal_on_{j}")
        m.addGenConstrIndicator(UpgSel[j], False, gp.quicksum(m_up[j, f] for f in feedstock_types) == 0, name=f"UpgBal_off_{j}")

    Rev_loc = m.addVars(plant_locs, lb=0, vtype=GRB.CONTINUOUS, name="Rev_loc")
    Cost_loc = m.addVars(plant_locs, lb=0, vtype=GRB.CONTINUOUS, name="Cost_loc")
    Rev_alt_selected = m.addVars(plant_locs, range(len(alternative_configs)), caps, lb=0, vtype=GRB.CONTINUOUS, name="Rev_alt_selected")
    Cost_alt_selected = m.addVars(plant_locs, range(len(alternative_configs)), caps, lb=0, vtype=GRB.CONTINUOUS, name="Cost_alt_selected")
    coef = {(a, c): c / 1e6 for a in range(len(alternative_configs)) for c in capacity_levels}
    m.addConstrs((Omega[j] <= gp.quicksum(coef[a, c] * Y[j, a, c] for a, c in coef) for j in plant_locs), name="OmegaLink")

    for j in plant_locs:
        m.addConstr(Omega[j] == gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] for i, f in avail_mass), name=f"Omega_Feed_{j}")
        m.addConstr(N_CH4[j] == gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] * feed_yield[f]['ch4_content'] for i, f in avail_mass), name=f"N_CH4_Feed_{j}")

    total_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass) for j in plant_locs}
    manure_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if is_manure(f)) for j in plant_locs}
    clover_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if is_clover(f)) for j in plant_locs}

    if config["eeg_enabled"]:
        add_eeg_constraints(m, total_feed, manure_feed, clover_feed, Y, plant_locs, alternative_configs, caps)
    if config["supply_enabled"]:
        add_supply_constraints(m, avail_mass, x, plant_locs)
    if config["cn_enabled"]:
        add_cn_constraints(m, x, avail_mass, plant_locs, feed_yield, CN_min, CN_max)
    if config["ghg_enabled"]:
        add_ghg_constraints(m, x, avail_mass, plant_locs, feed_yield, alpha_GHG_lim)
    if config["auction_enabled"]:
        add_auction_constraints(m, Y, plant_locs, alternative_configs, caps)
    if config["flh_enabled"]:
        add_flh_constraints(m, Omega, Y, plant_locs, caps, N_CH4)

    for j in plant_locs:
        for f in feedstock_types:
            production_f = gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] * feed_yield[f]['ch4_content'] for i in supply_nodes)
            m.addConstr(m_up[j, f] <= production_f, name=f"m_up_upper_{j}_{f}")

    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            for c in caps:
                if alt["opex_type"] == "fixed_variable_MW":
                    MW = c * system_methane_average * chp_heat_eff * alphaHV / (FLH_max * 1000)
                    fixed_opex = alt["opex_coeff"] * MW / 1e6
                    variable_opex = 0.5 * N_CH4[j] * alphaHV * chp_heat_eff / 1000
                    cost_val = fixed_opex + variable_opex
                else:
                    cost_val = (alt["opex_coeff"] * c ** alt["opex_exp"]) / 1e6
                if alt["category"] in ["Upgrading", "FlexEEG_biomethane"]:
                    cost_val += variable_upg_cost * N_CH4[j]
                rev_val = gp.LinExpr(0)
                if not alt["EEG_flag"]:
                    if alt["category"] == "Upgrading":
                        rev_val = N_CH4[j] * alt["rev_price"]["gas"] + (Omega[j] - N_CH4[j]) * alt["rev_price"]["co2"]
                    else:
                        raise ValueError(f"Unexpected alternative category (non-EEG): {alt['category']}")
                else:
                    effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                    bonus = 0
                    if c / 1e6 > threshold_m3 and alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                        bonus = 100 * (c / 1e6 * system_methane_average * chp_elec_eff * alphaHV) / FLH_max
                    cap_fraction = Cap_biogas if alt["category"] == "FlexEEG_biogas" else Cap_biomethane if alt["category"] == "FlexEEG_biomethane" else 1.0
                    E_actual = N_CH4[j] * (chp_elec_eff * alphaHV / 1000.0)
                    U_elec = (c / 1e6) * (FLH_max / 8760) * system_methane_average * chp_elec_eff * alphaHV / 1000.0
                    cap_production_elec = cap_fraction * U_elec
                    m.addGenConstrIndicator(Y[j, a, c], True, E_actual >= cap_production_elec, name=f"MinProd_{j}_{a}_{c}")
                    EEG_rev = cap_production_elec * effective_EEG
                    spot_rev = (E_actual - cap_production_elec) * electricity_spot_price
                    heat_rev = heat_price * (N_CH4[j] * chp_heat_eff * alphaHV / 1000.0)
                    rev_val = EEG_rev + spot_rev + heat_rev + bonus
                m.addGenConstrIndicator(Y[j, a, c], True, Rev_alt_selected[j, a, c] == rev_val, name=f"Rev_on_{j}_{a}_{c}")
                m.addGenConstrIndicator(Y[j, a, c], False, Rev_alt_selected[j, a, c] == 0, name=f"Rev_off_{j}_{a}_{c}")
                m.addGenConstrIndicator(Y[j, a, c], True, Cost_alt_selected[j, a, c] == cost_val, name=f"Cost_on_{j}_{a}_{c}")
                m.addGenConstrIndicator(Y[j, a, c], False, Cost_alt_selected[j, a, c] == 0, name=f"Cost_off_{j}_{a}_{c}")

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
                    base_capex = (c * (alt["capex_coeff"] * (c ** alt["capex_exp"]))) / 1e6
                extra_upg_cost = (
                    (alt["upg_cost_coeff"] * ((c / FLH_max) ** alt["upg_cost_exp"]) * (c / FLH_max)) / 1e6
                    if alt["category"] in ["Upgrading", "FlexEEG_biomethane"]
                    else 0
                )
                capex_expr += Y[j, a, c] * (base_capex + extra_upg_cost)
        Capex[j] = m.addVar(lb=0, name=f"Capex_{j}")
        m.addConstr(Capex[j] == capex_expr, name=f"Capex_link_{j}")

    FeedstockCost = gp.LinExpr()
    FeedstockCostPerPlant = {j: gp.LinExpr() for j in plant_locs}
    BaseFeedstockCost = {j: gp.LinExpr() for j in plant_locs}
    LoadingCost = {j: gp.LinExpr() for j in plant_locs}
    TransportCost = {j: gp.LinExpr() for j in plant_locs}
    DigestateCost = {j: gp.LinExpr() for j in plant_locs}

    flows = [(i, f, j) for i, f in avail_mass for j in plant_locs]
    cost_df = pd.DataFrame(flows, columns=["i", "f", "j"])
    cost_df["flow"] = cost_df.apply(lambda r: x[r.i, r.f, r.j], axis=1)
    cost_df = (cost_df.merge(distance_df.rename(columns={"Feedstock_LAU": "i", "Location": "j"}), on=["i", "j"], how="left")
                     .merge(yields_df.rename(columns={"substrat_ENG": "f"}), on="f", how="left"))
    cost_df["base"] = cost_df.flow * cost_df.Price
    cost_df["load_trp"] = cost_df.flow * 1e6 * ((cost_df.Loading_cost / cost_df.Capacity_load) + cost_df.Distance_km * cost_df["€_ton_km"]) / 1e6
    cost_df["dig"] = cost_df["flow"] * (cost_df["Digestate_Yield_%"] / 100) * 1e6 * ((loading_cost_dig / capacity_dig) + cost_df["Distance_km"] * cost_ton_km_dig) / 1e6

    for j in plant_locs:
        subtotal = cost_df.loc[cost_df.j == j, ["base", "load_trp", "dig"]].sum()
        BaseFeedstockCost[j] += subtotal.base
        LoadingCost[j] += subtotal.load_trp
        TransportCost[j] += subtotal.load_trp
        DigestateCost[j] += subtotal.dig
        FeedstockCostPerPlant[j] += subtotal.sum()
        FeedstockCost += subtotal.sum()

    FeedstockCostPlantVars = {j: m.addVar(lb=0, name=f"FeedCost_{j}") for j in plant_locs}
    for j in plant_locs:
        m.addConstr(FeedstockCostPlantVars[j] == FeedstockCostPerPlant[j], name=f"FeedCostConstr_{j}")

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

    # Apply warmstart if provided
    if warmstart_vars:
        for v in m.getVars():
            if v.VarName in warmstart_vars:
                v.start = warmstart_vars[v.VarName]

    return m, Omega, N_CH4, x, Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, GHGRevenue, TotalCapex, Rev_alt_selected, Cost_alt_selected, FeedstockCostPerPlant, BaseFeedstockCost, LoadingCost, TransportCost, DigestateCost

# 5) K-SWAP FUNCTION
def generate_k_swap_neighbors(current_assignments, k, plant_locs, alternative_configs, capacity_levels):
    neighbors = []
    assigned_plants = [j for j in current_assignments if current_assignments[j] is not None]
    all_assignments = [(a, c) for a in range(len(alternative_configs)) for c in capacity_levels]
    for swap_count in range(1, k + 1):
        for plants_to_swap in combinations(assigned_plants, swap_count):
            for new_assignments in product(all_assignments, repeat=swap_count):
                neighbor = current_assignments.copy()
                for idx, plant in enumerate(plants_to_swap):
                    neighbor[plant] = new_assignments[idx]
                neighbors.append(neighbor)
    return neighbors

def evaluate_neighbor(fixed_assignments, config, warmstart_vars=None):
    m, Omega, N_CH4, x, Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, GHGRevenue, TotalCapex, Rev_alt_selected, Cost_alt_selected, FeedstockCostPerPlant, BaseFeedstockCost, LoadingCost, TransportCost, DigestateCost = build_model(config, fixed_assignments, warmstart_vars)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        return m, m.objVal, {
            'Omega': {j: Omega[j].X for j in plant_locs},
            'N_CH4': {j: N_CH4[j].X for j in plant_locs},
            'x': {(i, f, j): x[i, f, j].X for i in supply_nodes for f in feedstock_types for j in plant_locs},
            'Y': {(j, a, c): Y[j, a, c].X for j in plant_locs for a in range(len(alternative_configs)) for c in capacity_levels},
            'm_up': {(j, f): m_up[j, f].X for j in plant_locs for f in feedstock_types},
            'Rev_loc': {j: Rev_loc[j].X for j in plant_locs},
            'Cost_loc': {j: Cost_loc[j].X for j in plant_locs},
            'Capex': {j: Capex[j].X for j in plant_locs},
            'FeedstockCostPerPlant': {j: FeedstockCostPerPlant[j].getValue() for j in plant_locs}
        }
    return None, -float('inf'), None

# 6) MAIN EXECUTION
if __name__ == '__main__':
    print("Running k-swap model...")
    # Load greedy solution
    fin_path = f"{BASE_DIR}results/large_scale_cont/10_greedy_with_alternatives/greedy/Financials_50_greedy.csv"
    greedy_fin_df = pd.read_csv(fin_path)
    alt_index = {cfg["name"]: i for i, cfg in enumerate(alternative_configs)}
    cap_set = set(capacity_levels)
    current_assignments = {}
    for j in plant_locs:
        row = greedy_fin_df[greedy_fin_df["PlantLocation"] == j]
        if not row.empty:
            alt_name = row.iloc[0]["Alternative"]
            cap = int(row.iloc[0]["Capacity"])
            if alt_name in alt_index and cap in cap_set:
                current_assignments[j] = (alt_index[alt_name], cap)
            else:
                current_assignments[j] = None
        else:
            current_assignments[j] = None

    k = 2  # Number of swaps
    max_iterations = 10
    iteration = 0
    best_npv = -float('inf')
    best_solution = None
    best_model = None
    best_vars = None
    warmstart_vars = None

    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}")
        # Evaluate current solution
        m, npv, vars_dict = evaluate_neighbor(current_assignments, config, warmstart_vars)
        if m is None or m.status != GRB.OPTIMAL:
            print("Current solution infeasible or not optimal")
            break
        print(f"Current NPV: {npv:,.2f} €")
        if npv > best_npv:
            best_npv = npv
            best_solution = current_assignments.copy()
            best_model = m
            best_vars = vars_dict
            warmstart_vars = {v.VarName: v.X for v in m.getVars()}

        # Generate and evaluate neighbors
        neighbors = generate_k_swap_neighbors(current_assignments, k, plant_locs, alternative_configs, capacity_levels)
        print(f"Generated {len(neighbors)} neighbors")
        best_neighbor_npv = npv
        best_neighbor_assignments = current_assignments
        best_neighbor_vars = vars_dict

        for idx, neighbor in enumerate(neighbors):
            print(f"Evaluating neighbor {idx + 1}/{len(neighbors)}")
            m_neighbor, npv_neighbor, vars_neighbor = evaluate_neighbor(neighbor, config, warmstart_vars)
            if m_neighbor and m_neighbor.status == GRB.OPTIMAL and npv_neighbor > best_neighbor_npv:
                best_neighbor_npv = npv_neighbor
                best_neighbor_assignments = neighbor
                best_neighbor_vars = vars_neighbor
                warmstart_vars = {v.VarName: v.X for v in m_neighbor.getVars()}

        if best_neighbor_npv <= best_npv:
            print("No better neighbor found, stopping")
            break
        current_assignments = best_neighbor_assignments
        best_npv = best_neighbor_npv
        best_vars = best_neighbor_vars
        best_model = m_neighbor
        iteration += 1

    # Process best solution
    if best_model and best_vars:
        print(f"\nBest NPV found: {best_npv:,.2f} €")
        Omega = best_vars['Omega']
        N_CH4 = best_vars['N_CH4']
        x = best_vars['x']
        Y = best_vars['Y']
        m_up = best_vars['m_up']
        Rev_loc = best_vars['Rev_loc']
        Cost_loc = best_vars['Cost_loc']
        Capex = best_vars['Capex']
        FeedstockCostPerPlant = best_vars['FeedstockCostPerPlant']

        # Save flows
        inflow_rows = []
        for j in plant_locs:
            for i, f in avail_mass:
                flow_val = x.get((i, f, j), 0)
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
        in_flow_df.to_csv(os.path.join(output_dir, "Output_in_flow_50_kswap.csv"), index=False)

        # Calculate financials
        plant_npvs = {}
        plant_annual_cf = {}
        plant_irr = {}
        for j in plant_locs:
            discounted_operating = 0.0
            for t in range(1, years + 1):
                df = 1.0 / pow(1.0 + r, t)
                rev_j = Rev_loc[j]
                cost_j = Cost_loc[j] + FeedstockCostPerPlant[j]
                ghg_j = sum(premium[f] * m_up.get((j, f), 0) for f in feedstock_types)
                discounted_operating += df * (rev_j - cost_j + ghg_j)
            capex_j = Capex[j]
            plant_npvs[j] = -capex_j + discounted_operating
            annual_net = rev_j - (Cost_loc[j] + FeedstockCostPerPlant[j]) + ghg_j
            plant_annual_cf[j] = annual_net
            cf_series = [-capex_j] + [annual_net] * years
            irr_j = nf.irr(cf_series)
            plant_irr[j] = irr_j

        merged_rows = []
        for j in plant_locs:
            for a in range(len(alternative_configs)):
                for c in capacity_levels:
                    if Y.get((j, a, c), 0) > 0.1:
                        alt = alternative_configs[a]
                        alt_name = alt["name"]
                        cap_fraction = Cap_biogas if alt["category"] == "FlexEEG_biogas" else Cap_biomethane if alt["category"] == "FlexEEG_biomethane" else None
                        feed_cost_j = FeedstockCostPerPlant[j]
                        bonus = 100 * (c / 1e6 * system_methane_average * chp_elec_eff * alphaHV) / FLH_max if c / 1e6 > threshold_m3 and alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"] else 0
                        row_data = {
                            "PlantLocation": j,
                            "Alternative": alt_name,
                            "Capacity": c,
                            "Plant_NPV": plant_npvs[j],
                            "Plant_IRR": plant_irr[j],
                            "Omega": Omega[j] * 1e6,
                            "N_CH4": N_CH4[j] * 1e6,
                            "CO2_Production": (Omega[j] - N_CH4[j]) * 1e6,
                            "Revenue": Rev_loc[j],
                            "Cost": Cost_loc[j],
                            "Feed_Trans_Cost": feed_cost_j,
                            "Capex": Capex[j],
                            "GHG": sum(premium[f] * m_up.get((j, f), 0) for f in feedstock_types),
                            "FLH": (Omega[j] / (c / 1e6)) * 8760 if c > 0 else 0,
                            "PlantLatitude": plant_coords.get(j, (None, None))[1],
                            "PlantLongitude": plant_coords.get(j, (None, None))[0]
                        }
                        if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                            effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                            E_actual = N_CH4[j] * (chp_elec_eff * alphaHV / 1000.0)
                            EEG_rev = cap_fraction * E_actual * effective_EEG if cap_fraction else 0
                            spot_rev = (E_actual - (cap_fraction * E_actual if cap_fraction else 0)) * electricity_spot_price
                            heat_rev = heat_price * (N_CH4[j] * chp_heat_eff * alphaHV / 1000.0)
                            row_data.update({
                                "EEG_Revenue": EEG_rev,
                                "Spot_Revenue": spot_rev,
                                "Heat_Revenue": heat_rev,
                                "Bonus": bonus
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
        fin_df.to_csv(os.path.join(output_dir, "Output_financials_50_kswap.csv"), index=False)

        # Save warmstart
        warmstart_path = os.path.join(output_dir, "warmstart_kswap.sol")
        best_model.write(warmstart_path)
        print(f"Warm-start solution written to: {warmstart_path}")

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    with open(f'{BASE_DIR}Solutions/aggregated/execution_times_kswap.txt', 'a') as f:
        f.write(f"Total script execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)\n")