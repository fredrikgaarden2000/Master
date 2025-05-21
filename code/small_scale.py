import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import pickle
import time
import os
import numpy_financial as nf

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
output_dir = os.path.join(BASE_DIR, "results/small_scale_added")
os.makedirs(output_dir, exist_ok=True)

feedstock_df = pd.read_csv(f"{BASE_DIR}aggregated_bavaria_supply_nodes.csv")
plant_df = pd.read_csv(f"{BASE_DIR}equally_spaced_locations_100.csv")
distance_df = pd.read_csv(f"{BASE_DIR}Distance_Matrix_100.csv")
yields_df = pd.read_csv(f"{BASE_DIR}Feedstock_yields.csv")

def is_cereal(ft):
    return 'cereal_str' in ft.lower()

# Filter out any feedstock labeled 'cereal_str'
feedstock_df = feedstock_df.dropna(subset=['GISCO_ID', 'Centroid_Lon', 'Centroid_Lat'])
feedstock_df = feedstock_df[feedstock_df['nutz_pot_tFM'] >= 10]
feedstock_df = feedstock_df[~feedstock_df['substrat_ENG'].apply(is_cereal)]
yields_df = yields_df[~yields_df['substrat_ENG'].apply(is_cereal)]

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
capacity_levels = (500_000,)
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

premium = {f: max(0, (alpha_GHG_comp - feed_yield[f]['GHG_intensity'])) * (alphaHV * 3.6) * GHG_certificate_price / 1e6 for f in feedstock_types}
threshold_m3 = (100 * FLH_max) / (chp_elec_eff * system_methane_average * alphaHV) / 1e6  # Scale
FLH_min_limit = 1000
M_large = max(capacity_levels) * 1.01 / 1e6  # Scale
avg_discount = sum(0.99**t for t in range(1, years+1)) / years

M_j = {j: sum(avail_mass[i, f] for i, f in avail_mass) / 1e6 for j in plant_locs}  # Scale


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



config = {
    "name": "Baseline",
    "eeg_enabled": True,
    "supply_enabled": True,
    "digestate_enabled": False,
    "digestate_return_frac": 0.99,
    "cn_enabled": True,
    "maize_enabled": False,
    "ghg_enabled": False,
    "auction_enabled": False,
    "flh_enabled": True}

# 5) MODEL FUNCTION
def build_model(config):
    m = gp.Model("ShadowPlant_Biogas_Model")
    m.setParam("NodefileStart", 16)  # Start offloading node data to disk after 40 GB
    
    Omega = m.addVars(plant_locs,lb=0,ub=max(capacity_levels) / 1e6,   # ← NEW
       name="Omega")

    caps = capacity_levels

    
    ub_ch4 = max(capacity_levels) * 0.7 /1e6   # 80 Mm³ / 1 000 000
    N_CH4 = m.addVars(plant_locs, lb=0, ub=ub_ch4, vtype=GRB.CONTINUOUS,
                    name="N_CH4")
    m_up  = m.addVars(plant_locs, feedstock_types,
                    lb=0, ub=ub_ch4,
                    vtype=GRB.CONTINUOUS, name="m_up")
    
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

    # Changed Y to binary for indicator constraints compatibility
    Y = {(j, a, c): m.addVar(vtype=GRB.BINARY, name=f"Y_{j}_{a}_{c}")
         for j in plant_locs for a in range(len(alternative_configs)) for c in caps}
    upgrading_idx = next(a for a, alt in enumerate(alternative_configs) if alt["name"] == "EEG_CHP_large1")
    target_capacity = 500_000
    for j in plant_locs:
        for a in range(len(alternative_configs)):
            for c in capacity_levels:
                if a == upgrading_idx and c == target_capacity:
                    Y[j, a, c].Start = 1.0
                else:
                    Y[j, a, c].Start = 0.0

# Add auxiliary binary variable to indicate active plant
    is_active = m.addVars(plant_locs, vtype=GRB.BINARY, name="is_active")
    # ------------------------------------------------------------------
    #  UpgSel[j] = 1 ⇔ plant j has chosen (any) upgrading alternative
    # ------------------------------------------------------------------
    UpgSel = {j: m.addVar(vtype=GRB.BINARY, name=f"UpgSel_{j}")
            for j in plant_locs}

    for j in plant_locs:
        m.addConstr(gp.quicksum(Y[j, a, c] for a in range(len(alternative_configs)) for c in caps) <= 1, name=f"OneAlt_{j}")
        # Link is_active to Omega
        m.addConstr(Omega[j] <= max(capacity_levels) * is_active[j],
              name=f"ActiveOmegaUpper_{j}")
        m.addConstr(Omega[j] >= 1e-6 * is_active[j], name=f"ActiveOmegaLower_{j}")
        # Enforce sum(Y) == is_active
        m.addConstr(gp.quicksum(Y[j, a, c] for a in range(len(alternative_configs)) for c in caps) == is_active[j], name=f"ActivePlant_{j}")

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

    for j in plant_locs:
        m.addConstr(Omega[j] <= gp.quicksum((c / 1e6) * Y[j, a, c] for a in range(len(alternative_configs)) for c in caps), name=f"Omega_Link_{j}")
    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            if alt.get("max_cap_m3_year") is not None:
                for c in caps:
                    m.addGenConstrIndicator(
                        Y[j,a,c], True,
                        Omega[j] <= alt["max_cap_m3_year"]/1e6,
                        name=f"MaxCap_{j}_{a}_{c}"
                    )
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

                # -- link UpgSel to the alternative choice ------------------
        m.addConstr(
            UpgSel[j] == gp.quicksum(
                Y[j, a, c]
                for a, alt in enumerate(alternative_configs)
                if alt["category"] == "Upgrading"
                for c in caps),
            name=f"LinkUpg_{j}"
        )

        # -- methane balance only if upgrading is built --------------
        m.addGenConstrIndicator(
            UpgSel[j], True,
            gp.quicksum(m_up[j, f] for f in feedstock_types) == N_CH4[j],
            name=f"UpgBal_on_{j}"
        )
        m.addGenConstrIndicator(
            UpgSel[j], False,
            gp.quicksum(m_up[j, f] for f in feedstock_types) == 0,
            name=f"UpgBal_off_{j}"
        )

    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            for c in caps:
                # ---- Cost Expression ----
                if alt["opex_type"] == "fixed_variable_MW":
                    MW = c * system_methane_average * chp_heat_eff * alphaHV / (FLH_max * 1000)
                    fixed_opex = alt["opex_coeff"] * MW / 1e6
                    variable_opex = 0.5 * N_CH4[j] * alphaHV * chp_heat_eff / 1000
                    cost_val = fixed_opex + variable_opex
                else:
                    cost_val = (alt["opex_coeff"] * c ** alt["opex_exp"]) / 1e6

                if alt["category"] in ["Upgrading", "FlexEEG_biomethane"]:
                    cost_val += variable_upg_cost * N_CH4[j]

                # ---- Revenue Expression ----
                rev_val = gp.LinExpr(0)
                
                if not alt["EEG_flag"]:
                    if alt["category"] == "CHP_nonEEG":
                        rev_val = N_CH4[j] * (alphaHV / 1000) * (chp_elec_eff * alt["rev_price"]["spot"] + chp_heat_eff * heat_price)
                    elif alt["category"] == "Upgrading":
                        rev_val = N_CH4[j] * alt["rev_price"]["gas"] + (Omega[j] - N_CH4[j]) * alt["rev_price"]["co2"]
                    elif alt["category"] == "boiler":
                        rev_val = N_CH4[j] * (alphaHV / 1000) * chp_heat_eff * heat_price
                    elif alt["category"] == "None":
                        cost_val = rev_val = 0
                    else:
                        raise ValueError(f"Unexpected alternative category (non-EEG): {alt['category']}")
                else:
                    effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                    if alt["category"] in ["EEG_CHP_small", "EEG_CHP_large"]:
                        rev_val = N_CH4[j] * (alphaHV / 1000) * (chp_elec_eff * effective_EEG + chp_heat_eff * heat_price)
                    elif alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                        bonus = 0
                        if c/1e6 > threshold_m3: bonus = 100 * (c/1e6 * system_methane_average * chp_elec_eff * alphaHV) / FLH_max

                        cap_fraction = Cap_biogas if alt["category"] == "FlexEEG_biogas" else Cap_biomethane
                        E_actual = N_CH4[j]*(chp_elec_eff*alphaHV/1000.0)
                        U_elec = (c / 1e6) * (FLH_max / 8760) * system_methane_average * chp_elec_eff * alphaHV / 1000.0
                        cap_production_elec = cap_fraction * U_elec
                        m.addConstr(E_actual >= cap_production_elec,name=f"MinProd_{j}_{a}_{c}")
                        EEG_rev  = cap_production_elec * effective_EEG
                        spot_rev = (E_actual - cap_production_elec) * electricity_spot_price
                        heat_rev = heat_price * (N_CH4[j] * chp_heat_eff * alphaHV / 1000.0)
                        rev_val = EEG_rev + spot_rev + heat_rev + bonus
                    elif alt["category"] == "None":
                        cost_val = rev_val = 0
                    else:
                        raise ValueError(f"Unexpected EEG alternative category: {alt['category']}")
                    
                m.addGenConstrIndicator(Y[j,a,c], True,
                            Rev_alt_selected[j,a,c]  == rev_val,
                            name=f"Rev_on_{j}_{a}_{c}")
                m.addGenConstrIndicator(Y[j,a,c], False,
                                        Rev_alt_selected[j,a,c]  == 0,
                                        name=f"Rev_off_{j}_{a}_{c}")

                m.addGenConstrIndicator(Y[j,a,c], True,
                                        Cost_alt_selected[j,a,c] == cost_val,
                                        name=f"Cost_on_{j}_{a}_{c}")
                m.addGenConstrIndicator(Y[j,a,c], False,
                                        Cost_alt_selected[j,a,c] == 0,
                                        name=f"Cost_off_{j}_{a}_{c}")



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
                    if alt["category"] in ["Upgrading", "FlexEEG_biomethane"]
                    else 0
                )
                capex_expr += Y[j, a, c] * (base_capex + extra_upg_cost)
        Capex[j] = m.addVar(lb=0, name=f"Capex_{j}")
        m.addConstr(Capex[j] == capex_expr, name=f"Capex_link_{j}")

    FeedstockCost = gp.LinExpr()
    FeedstockCostPerPlant = {j: gp.LinExpr() for j in plant_locs}

    # NEW: For detailed breakdown
    BaseFeedstockCost = {j: gp.LinExpr() for j in plant_locs}
    LoadingCost = {j: gp.LinExpr() for j in plant_locs}
    TransportCost = {j: gp.LinExpr() for j in plant_locs}
    DigestateCost = {j: gp.LinExpr() for j in plant_locs}

    flows = [(i,f,j) for (i,f) in avail_mass for j in plant_locs]
    cost_df = pd.DataFrame(flows, columns=["i","f","j"])
    cost_df["flow"] = cost_df.apply(lambda r: x[r.i, r.f, r.j], axis=1)

    # merge static data once
    cost_df = (cost_df.merge(distance_df.rename(columns={"Feedstock_LAU":"i","Location":"j"}), on=["i","j"], how="left")
                        .merge(yields_df.rename(columns={"substrat_ENG":"f"}), on="f", how="left"))

    # base feed cost
    cost_df["base"] = cost_df.flow * cost_df.Price

    # loading & transport
    cost_df["load_trp"] = cost_df.flow * 1e6 * (
            (cost_df.Loading_cost/cost_df.Capacity_load) +
            cost_df.Distance_km*cost_df["€_ton_km"]) / 1e6

    cost_df["dig"] = cost_df["flow"] * (cost_df["Digestate_Yield_%"] / 100) * 1e6 * (
            (loading_cost_dig / capacity_dig) +
            cost_df["Distance_km"] * cost_ton_km_dig) / 1e6


    # group to build linear expressions quickly
    for j in plant_locs:
        subtotal = cost_df.loc[cost_df.j==j, ["base","load_trp","dig"]].sum()
        BaseFeedstockCost[j]  += subtotal.base
        LoadingCost[j]        += subtotal.load_trp
        TransportCost[j]      += subtotal.load_trp  # already split above
        DigestateCost[j]      += subtotal.dig
        FeedstockCostPerPlant[j] += subtotal.sum()
        FeedstockCost           += subtotal.sum()


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
    #m.addConstr(NPV_expr <= 3000, name = "upper bound")
    
    m.setObjective(NPV_expr, GRB.MAXIMIZE)

    return m, Omega, N_CH4, x, Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, GHGRevenue, TotalCapex, Rev_alt_selected, Cost_alt_selected, FeedstockCostPerPlant, BaseFeedstockCost, LoadingCost, TransportCost, DigestateCost# 6) RUN MODEL

if __name__ == '__main__':
    print("Running full model...")
    m, Omega, N_CH4, x, Y, m_up, Rev_loc, Cost_loc, Capex, TotalRev, TotalCost, FeedstockCost, GHGRevenue, TotalCapex, Rev_alt_selected, Cost_alt_selected, FeedstockCostPerPlant, BaseFeedstockCost, LoadingCost, TransportCost, DigestateCost = build_model(config)
    m.update()
    # –– Warm‐start if a solution exists
    warmstart_path = os.path.join(output_dir, "warmstart.sol")
    if os.path.isfile(warmstart_path):
        print(f"Loading warm‐start from {warmstart_path}")
        m.read(warmstart_path)

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
    elif m.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        m.computeIIS()
        m.write("model_infeasible.ilp")
        print("IIS written to 'model_infeasible.ilp'")
    else:
        print(f"Model status: {m.status}")

    for j in plant_locs:
        print(f"--- Plant {j} Feedstock Cost Breakdown ---")
        print(f"  Base feedstock: €{BaseFeedstockCost[j].getValue():,.2f}")
        print(f"  Loading:        €{LoadingCost[j].getValue():,.2f}")
        print(f"  Transport:      €{TransportCost[j].getValue():,.2f}")
        print(f"  Digestate:      €{DigestateCost[j].getValue():,.2f}")
        print(f"  TOTAL:          €{FeedstockCostPerPlant[j].getValue():,.2f}")

    print("\n--- Revenue Debug for EEG_CHP_large2 ---")

    for j in plant_locs:
        for a, alt in enumerate(alternative_configs):
            if alt["name"] == "EEG_CHP_small1":
                for c in capacity_levels:
                    y_val = Y[j, a, c].X 
                    if y_val > 1e-6:
                        N_val = N_CH4[j].X *1e6
                        omega_val = Omega[j].X *1e6
                        effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                        ch4_energy = N_val * (alphaHV / 1000)
                        electricity_revenue = ch4_energy * chp_elec_eff * effective_EEG
                        heat_revenue = ch4_energy * chp_heat_eff * heat_price
                        total_revenue = electricity_revenue + heat_revenue

                        print(f"\nPlant {j} selected EEG_CHP_large2 with capacity {c}:")
                        print(f"  Y = {y_val:.6f}")
                        print(f"  N_CH4 = {N_val:.6f} m³")
                        print(f"  Omega = {omega_val:.6f} m³")
                        print(f"  CH4 Energy Input = {ch4_energy:.6f} MWh")
                        print(f"  Electricity Revenue = €{electricity_revenue:.6f}")
                        print(f"  Heat Revenue = €{heat_revenue:.6f}")
                        print(f"  Total Calculated Revenue = €{total_revenue:.6f}")
                        print(f"  Model Revenue = €{Rev_loc[j].X:.6f}")
    

    '''
    print(f"--- Debug: Rev_alt_selected terms for {j} ---")
    for a in range(len(alternative_configs)):
        for c in capacity_levels:
            print(f"  Y[{j},{a},{c}] = {Y[j,a,c].X:.6f}, Rev_alt_selected = {Rev_alt_selected[j,a,c].X:.6f}")
    print(f"  ==> Rev_loc[{j}] = {Rev_loc[j].X:.6f}")

    
    print(f"--- Debug: Cost_alt_selected terms for {j} ---")
    for a in range(len(alternative_configs)):
        for c in capacity_levels:
            print(f"  Y[{j},{a},{c}] = {Y[j,a,c].X:.6f}, Cost_alt_selected = {Cost_alt_selected[j,a,c].X:.6f}")
    print(f"  ==> Cost_loc[{j}] = {Cost_loc[j].X:.6f}")
    '''
        # How much of each feedstock class is still unused?
    for f in feedstock_types:
        used = sum(x[i,f,j].X for i in supply_nodes for j in plant_locs)
        avail = sum(avail_mass.get((i, f), 0) for i in supply_nodes) / 1e6
        print(f"{f:20s}  {used:8.1f} / {avail:8.1f}  ({100*used/avail:5.1f}%)")



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
    in_flow_df.to_csv(f"{BASE_DIR}/results/small_scale_added/Output_in_flow.csv", index=False)
    '''
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
    '''
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
    plant_npvs = {}
    for j in plant_locs:
        # discounted sum of revenues minus variable costs + GHG revenues
        discounted_operating = 0.0
        for t in range(1, years+1):
            df = 1.0 / pow(1.0 + r, t)
            rev_j  = Rev_loc[j].X
            costj  = Cost_loc[j].X + FeedstockCostPerPlant[j].getValue()
            #ghg_j  = sum(premium[f] * m_up[j, f].X for f in feedstock_types)
            discounted_operating += df * (rev_j - costj)

        capex_j = Capex[j].X
        plant_npvs[j] = -capex_j + discounted_operating

    plant_annual_cf = {}
    for j in plant_locs:
        # Annual operating inflow = Rev_loc[j] – (Cost_loc[j] + FeedstockCostPerPlant[j]) + GHG revenue
        rev_j   = Rev_loc[j].X
        varcost = Cost_loc[j].X
        feedcost= FeedstockCostPerPlant[j].getValue()
        #ghg_j   = sum(premium[f] * m_up[j, f].X for f in feedstock_types)
        annual_net = rev_j - (varcost + feedcost)
        plant_annual_cf[j] = annual_net

    plant_irr = {}
    for j in plant_locs:
        capex_j = Capex[j].X
        # cash‐flow series: year 0 = –capex, years 1..T = annual_net
        cf_series = [-capex_j] + [plant_annual_cf[j]] * years
        irr_j = nf.irr(cf_series)
        plant_irr[j] = irr_j    

    merged_rows = []
    for j in plant_locs:
        for a in range(len(alternative_configs)):
            for c in capacity_levels:
                if Y[j, a, c].X > 0.1:
                    alt = alternative_configs[a]
                    alt_name = alt["name"]
                    cap_fraction = Cap_biogas if alt["category"] == "FlexEEG_biogas" else Cap_biomethane if alt["category"] == "FlexEEG_biomethane" else None
                    
                    feed_cost_j = FeedstockCostPerPlant[j].getValue()

                    row_data = {
                        "PlantLocation": j,
                        "Alternative": alt_name,
                        "Capacity": c,
                        "Plant_NPV": plant_npvs[j],
                        "Plant_IRR" : plant_irr[j],
                        "Omega": Omega[j].X * 1e6,
                        "N_CH4": N_CH4[j].X * 1e6,
                        "CO2_Production": (Omega[j].X - N_CH4[j].X) * 1e6,
                        "Revenue": Rev_loc[j].X,
                        "Cost": Cost_loc[j].X,
                        "Feed_Trans_Cost": feed_cost_j,
                        "Capex": Capex[j].X,
                        "GHG": 0,
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
                            "Bonus": 0               # << new column
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
    fin_df.to_csv(f"{BASE_DIR}/results/small_scale_added/Output_financials.csv", index=False)

    warmstart_path = os.path.join(output_dir, "warmstart.sol")
    m.write(warmstart_path)
    print(f"Warm-start solution written to: {warmstart_path}")

    # 2) (Optional) Also store Python‐side Start attributes, in case you rebuild the model in memory
    for v in m.getVars():
        v.start = v.X

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    opt_time = opt_end_time - opt_start_time
    with open(f'{BASE_DIR}/results/small_scale_added/execution_times.txt', 'a') as f:
        f.write(f"Total script execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)\n")
        f.write(f"Optimization time: {opt_time:.2f} seconds ({opt_time/60:.2f} minutes)\n")