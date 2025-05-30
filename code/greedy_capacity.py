import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
import time
import numpy_financial as nf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1) DATA LOADING WITH ERROR HANDLING
def load_data():
    try:
        BASE_DIR = "/home/fredrgaa/Master/"
        if not os.path.exists(BASE_DIR):
            raise FileNotFoundError("Linux path not found")
    except FileNotFoundError:
        BASE_DIR = "C:/Clone/Master/"
        if not os.path.exists(BASE_DIR):
            raise FileNotFoundError("No valid base directory found")

    def safe_load_csv(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.read_csv(path)

    try:
        feedstock_df = safe_load_csv(f"{BASE_DIR}aggregated_bavaria_supply_nodes.csv")
        plant_df = safe_load_csv(f"{BASE_DIR}Solutions/100/equally_spaced_locations_100.csv")
        distance_df = safe_load_csv(f"{BASE_DIR}Solutions/100/Distance_Matrix.csv")
        yields_df = safe_load_csv(f"{BASE_DIR}Feedstock_yields.csv")
    except FileNotFoundError as e:
        print(f"Critical error: {str(e)}")
        exit(1)

    feedstock_df = feedstock_df[
        (feedstock_df["GISCO_ID"].notna()) &
        (feedstock_df["Centroid_Lon"].notna()) &
        (feedstock_df["Centroid_Lat"].notna()) &
        (feedstock_df["nutz_pot_tFM"] >= 10)
    ].copy()

    required_columns = ['Feedstock_LAU', 'Location', 'Distance_km']
    if not all(col in distance_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in distance_df.columns]
        raise ValueError(f"Distance matrix missing columns: {missing}")

    return feedstock_df, plant_df, distance_df, yields_df

# 2) PARAMETER INITIALIZATION
def initialize_parameters():
    return {
        "FLH_max": 8000,
        "alphaHV": 9.97,
        "CN_min": 20.0,
        "CN_max": 30.0,
        "heat_price": 20,
        "chp_elec_eff": 0.4,
        "chp_heat_eff": 0.4,
        "electricity_spot_price": 60,
        "EEG_price_small": 220,
        "EEG_price_large": 190,
        "EEG_skip_chp_price": 194.3,
        "EEG_skip_upg_price": 210.4,
        "r": 0.042,
        "years": 25,
        "gas_price_mwh": 30,
        "co2_price_ton": 20,
        "variable_upg_cost": 0.2,
        "alpha_GHG_comp": 94.0,
        "GHG_certificate_price": 70,
        "Q_MAX": 0,
        "Q_MIN": 0.1,
        "cap_biogas": 0.45,
        "cap_biomethane": 0.10,
        "bonus_rate": 100,
        "loading_cost_dig": 27,
        "capacity_dig": 37,
        "cost_ton_km_dig": 0.104,
        "auction_bg_limit": 225000,
        "auction_bm_limit": 125000,
        "EEG_small_m3" : 255870,
        "EEG_large_m3" : 511740,
        "manure_percent_limit": 1,
        "boiler_eff": 0.9
    }

# Define CAPACITY LEVELS
capacity_levels = [500000,]

# Define ALTERNATIVE CONFIGURATIONS
# Define alternative configurations
def get_alternative_configs(params):
    return [
        {"name": "EEG_CHP_small1", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": params['EEG_small_m3'],
        "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": params['EEG_price_small']},
        "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
        "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
        "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},

        {"name": "EEG_CHP_small2", "category": "EEG_CHP_small", "prod_cap_factor": 1.0, "max_cap_m3_year": params['EEG_small_m3'],
        "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": params['EEG_price_small']},
        "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 2,
        "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
        "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},

        {"name": "EEG_CHP_large1", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": params['EEG_large_m3'],
        "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": params['EEG_price_large']},
        "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 1,
        "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
        "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},

        {"name": "EEG_CHP_large2", "category": "EEG_CHP_large", "prod_cap_factor": 1.0, "max_cap_m3_year": params['EEG_large_m3'],
        "upg_cost_coeff": 0, "upg_cost_exp": 0, "rev_price": {"EEG": params['EEG_price_large']},
        "EEG_flag": True, "GHG_eligible": False, "feed_constraint": 2,
        "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard",
        "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
    ]

# HELPER FUNCTIONS FOR FEEDSTOCK CLASSIFICATION
def is_manure(ftype):
    return 'man' in ftype.lower() or 'slu' in ftype.lower()

def is_clover(ftype):
    return 'clover' in ftype.lower()

def is_maize_cereal(ftype):
    return 'maize' in ftype.lower() or 'cereal' in ftype.lower()

# 3) PLANT MODEL BUILDER - UPDATED WITH DISCRETE CAPACITY LEVELS
def build_single_plant_model(j, avail_mass, supply_nodes, feedstock_types, feed_yield, 
                            params, Capex_params, Opex_params, Upg_params, premium, distances,
                            cumulative_eeg_bg=0, cumulative_eeg_bm=0, manure_used=0, total_feed_used=0):
    m = gp.Model(f"Plant_{j}")
    m.setParam('OutputFlag', 0)
    
    # Get alternative configurations
    alternative_configs = get_alternative_configs(params)
    
    # Variables
    x = m.addVars(supply_nodes, feedstock_types, lb=0, name="x")
    y = m.addVars(range(len(alternative_configs)), capacity_levels, vtype=GRB.BINARY, name="y")
    Omega = m.addVar(lb=0, ub=max(capacity_levels)/1e6, name="Omega")
    N_CH4 = m.addVar(lb=0, ub=max(capacity_levels)*0.7/1e6, name="N_CH4")
    
    total_methane = sum(avail_mass[i, f] * feed_yield[f]['ch4_content'] for i, f in avail_mass)
    total_mass = sum(avail_mass[i, f] for i, f in avail_mass)
    system_methane_average = total_methane / total_mass
    
    # Only one alternative-capacity pair can be selected
    m.addConstr(gp.quicksum(y[a, c] for a in range(len(alternative_configs)) for c in capacity_levels) <= 1, "OneAlt")
    
    # Plant is active if any alternative-capacity pair is selected
    is_active = m.addVar(vtype=GRB.BINARY, name="is_active")
    m.addConstr(is_active == gp.quicksum(y[a, c] for a in range(len(alternative_configs)) for c in capacity_levels), "ActiveLink")
    m.addConstr(Omega <= max(capacity_levels)/1e6 * is_active, "OmegaUpper")
    m.addConstr(Omega >= params["Q_MIN"] * is_active, "OmegaLower")
    
    # Link Omega to selected capacity
    m.addConstr(Omega == gp.quicksum(y[a, c] * (c / 1e6) for a in range(len(alternative_configs)) for c in capacity_levels), "OmegaCap")
    
    # EEG volume limit calculation
    eeg_volume_limit_bg = (params["auction_bg_limit"] * params["FLH_max"] /
                          (params["alphaHV"] * system_methane_average)) / 1e6
    eeg_volume_limit_bm = (params["auction_bm_limit"] * params["FLH_max"] /
                          (params["alphaHV"] * system_methane_average)) / 1e6
    
    # EEG limits for biogas and biomethane
    for a, alt in enumerate(alternative_configs):
        for c in capacity_levels:
            if alt["category"] == "FlexEEG_biogas":
                m.addGenConstrIndicator(
                    y[a, c], True,
                    Omega <= eeg_volume_limit_bg - cumulative_eeg_bg,
                    name=f"EEG_limit_bg_{j}_{a}_{c}"
                )
            elif alt["category"] == "FlexEEG_biomethane":
                m.addGenConstrIndicator(
                    y[a, c], True,
                    Omega <= eeg_volume_limit_bm - cumulative_eeg_bm,
                    name=f"EEG_limit_bm_{j}_{a}_{c}"
                )
    
    # MANURE USAGE CONSTRAINT (PER PLANT)
    manure_types = [f for f in feedstock_types if is_manure(f)]
    plant_manure = gp.quicksum(x[i,f] for i in supply_nodes for f in manure_types)
    plant_total_feed = gp.quicksum(x[i,f] for i in supply_nodes for f in feedstock_types)
    m.addConstr(plant_manure <= params["manure_percent_limit"] * plant_total_feed, 
               "manure_limit_per_plant")
    
    # CAPEX CALCULATIONS
    capex_terms = []
    for a, alt in enumerate(alternative_configs):
        for c in capacity_levels:
            if alt["category"] in ["EEG_CHP_small"]:
                capex_val = (params["EEG_small_m3"] * Capex_params["capex_coeff"] * 
                            (params["EEG_small_m3"]) ** Capex_params["capex_exp"]) / 1e6
            elif alt["category"] in ["EEG_CHP_large"]:
                capex_val = (params["EEG_large_m3"] * Capex_params["capex_coeff"] * 
                            (params["EEG_large_m3"]) ** Capex_params["capex_exp"]) / 1e6
            else:
                capex_val = (c * Capex_params["capex_coeff"] * 
                            c ** Capex_params["capex_exp"]) / 1e6
            
            if alt["category"] in ["Upgrading", "FlexEEG_biomethane"]:
                upg_capex = ((c / params["FLH_max"]) * Upg_params["capex_coeff"] * 
                            (c / params["FLH_max"]) ** Upg_params["capex_exp"]) / 1e6
                capex_val += upg_capex + 1  # Grid connection cost
            capex_terms.append(y[a, c] * capex_val)
    
    total_capex = gp.quicksum(capex_terms)
    
    # OPEX CALCULATIONS
    opex_terms = []
    for a, alt in enumerate(alternative_configs):
        for c in capacity_levels:
            opex_val = Opex_params["opex_coeff"] * c ** Opex_params["opex_exp"] / 1e6
            if alt["category"] in ["Upgrading", "FlexEEG_biomethane"]:
                opex_val += params["variable_upg_cost"] * N_CH4
            opex_terms.append(y[a, c] * opex_val)
    
    total_opex = gp.quicksum(opex_terms)
    
    # CONSTRAINTS
    m.addConstr(Omega == gp.quicksum(
        x[i,f] * feed_yield[f]['biogas_m3_per_ton'] 
        for i in supply_nodes for f in feedstock_types
    ), "Omega_def")
    
    m.addConstr(N_CH4 == gp.quicksum(
        x[i,f] * feed_yield[f]['biogas_m3_per_ton'] * feed_yield[f]['ch4_content']
        for i in supply_nodes for f in feedstock_types
    ), "N_CH4_def")
    
    total_feed = gp.quicksum(x[i,f] for i in supply_nodes for f in feedstock_types)
    total_cn = gp.quicksum(x[i,f] * feed_yield[f]['CN'] for i in supply_nodes for f in feedstock_types)
    m.addConstr(total_cn >= params["CN_min"] * total_feed, "CN_min")
    m.addConstr(total_cn <= params["CN_max"] * total_feed, "CN_max")
    
    for i in supply_nodes:
        for f in feedstock_types:
            available = avail_mass.get((i, f), 0)
            m.addConstr(x[i,f] <= available / 1e6, f"supply_{i}_{f}")
    
    # Economics
    gas_price_m3 = params["gas_price_mwh"] * (params["alphaHV"] / 1000)
    co2_price = params["co2_price_ton"] / 556.2
    
    # EEG BONUS CALCULATION
    threshold_m3 = (100 * params["FLH_max"]) / (params["chp_elec_eff"] * system_methane_average * params["alphaHV"]) / 1e6
    bonus = 0
    for a, alt in enumerate(alternative_configs):
        for c in capacity_levels:
            if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"] and (c / 1e6) > threshold_m3:
                bonus += params["bonus_rate"] * ((c / 1e6) - threshold_m3) * y[a, c] / 1e6
    
    # Feedstock totals
    total_feed_j = gp.quicksum(x[i,f] for (i,f) in avail_mass)
    manure_feed_j = gp.quicksum(x[i,f] for (i,f) in avail_mass if is_manure(f))
    clover_feed_j = gp.quicksum(x[i,f] for (i,f) in avail_mass if is_clover(f))
    
    avg_discount = sum(0.99**t for t in range(1, params['years']+1)) / params['years']
    
    # Revenue calculations for each alternative
    rev_alternatives = []
    for a, alt in enumerate(alternative_configs):
        for c in capacity_levels:
            rev_a = m.addVar(name=f"rev_alt_{a}_{c}")
            if alt["category"] in ["EEG_CHP_small", "EEG_CHP_large"]:
                m.addGenConstrIndicator(
                    y[a, c], True,
                    Omega <= alt["max_cap_m3_year"] / 1e6,
                    name=f"MaxCap_{j}_{a}_{c}"
                )
                if alt["feed_constraint"] == 1:
                    m.addConstr(manure_feed_j >= 0.80 * total_feed_j * y[a, c],
                               name=f"manure80_{j}_{a}_{c}")
                else:  # feed_constraint == 2
                    m.addConstr(manure_feed_j >= 0.70 * total_feed_j * y[a, c],
                               name=f"manure70_{j}_{a}_{c}")
                    m.addConstr(clover_feed_j >= 0.10 * total_feed_j * y[a, c],
                               name=f"clover10_{j}_{a}_{c}")
                
                effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                E_elec = N_CH4 * (params["chp_elec_eff"] * params["alphaHV"] / 1000)
                rev_val = E_elec * effective_EEG + \
                         N_CH4 * (params["chp_heat_eff"] * params["heat_price"] * (params["alphaHV"]/1000))
                m.addGenConstrIndicator(y[a, c], True, rev_a == rev_val, name=f"rev_on_{j}_{a}_{c}")
                m.addGenConstrIndicator(y[a, c], False, rev_a == 0, name=f"rev_off_{j}_{a}_{c}")
            
            elif alt["category"] == "FlexEEG_biogas":
                effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                U_elec = N_CH4 * params["chp_elec_eff"] * params["alphaHV"] / 1000
                EEG_cap = U_elec * alt["prod_cap_factor"]
                EEG_rev = EEG_cap * effective_EEG
                spot_rev = (U_elec - EEG_cap) * params["electricity_spot_price"]
                heat_rev = N_CH4 * params["chp_heat_eff"] * params["alphaHV"] / 1000 * params["heat_price"]
                revenue_alt = EEG_rev + spot_rev + heat_rev
                m.addGenConstrIndicator(y[a, c], True, rev_a == revenue_alt, name=f"rev_alt_{a}_{c}_on")
                m.addGenConstrIndicator(y[a, c], False, rev_a == 0, name=f"rev_alt_{a}_{c}_off")
            
            elif alt["category"] == "Upgrading":
                revenue_alt = (
                    N_CH4 * gas_price_m3 + 
                    (Omega - N_CH4) * co2_price + 
                    gp.quicksum(x[i,f] * premium[f] for i in supply_nodes for f in feedstock_types))
                m.addGenConstrIndicator(y[a, c], True, rev_a == revenue_alt, name=f"rev_alt_{a}_{c}_on")
                m.addGenConstrIndicator(y[a, c], False, rev_a == 0, name=f"rev_alt_{a}_{c}_off")
            
            elif alt["category"] == "CHP_nonEEG":
                U_elec = N_CH4 * params["chp_elec_eff"] * params["alphaHV"] / 1000
                spot_rev = U_elec * alt["rev_price"]["spot"]
                heat_rev = N_CH4 * params["chp_heat_eff"] * params["alphaHV"] / 1000 * alt["rev_price"]["heat"]
                revenue_alt = spot_rev + heat_rev
                m.addGenConstrIndicator(y[a, c], True, rev_a == revenue_alt, name=f"rev_alt_{a}_{c}_on")
                m.addGenConstrIndicator(y[a, c], False, rev_a == 0, name=f"rev_alt_{a}_{c}_off")
            
            elif alt["category"] == "FlexEEG_biomethane":
                effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                U_elec = N_CH4 * params["chp_elec_eff"] * params["alphaHV"] / 1000
                EEG_cap = U_elec * alt["prod_cap_factor"]
                EEG_rev = EEG_cap * effective_EEG
                spot_rev = (U_elec - EEG_cap) * params["electricity_spot_price"]
                heat_rev = N_CH4 * params["chp_heat_eff"] * params["alphaHV"] / 1000 * params["heat_price"]
                revenue_alt = EEG_rev + spot_rev + heat_rev
                m.addGenConstrIndicator(y[a, c], True, rev_a == revenue_alt, name=f"rev_alt_{a}_{c}_on")
                m.addGenConstrIndicator(y[a, c], False, rev_a == 0, name=f"rev_alt_{a}_{c}_off")
            
            rev_alternatives.append(rev_a)
    
    total_revenue = gp.quicksum(rev_alternatives) + bonus
    
    # FEEDSTOCK COST CALCULATIONS
    feed_cost = gp.quicksum(
        x[i,f] * feed_yield[f]['price']
        for i in supply_nodes 
        for f in feedstock_types
    )
    
    # LOADING + TRANSPORT COST
    transport_cost = gp.quicksum(
        x[i,f] * 1e6 * (
            (feed_yield[f]['loading'] / feed_yield[f]['capacity_load']) + 
            distances.get((i, j), 0) * feed_yield[f]['cost_ton_km']
        ) / 1e6
        for i in supply_nodes
        for f in feedstock_types
    )
    
    # DIGESTATE COST
    digestate_cost = gp.quicksum(
        x[i,f] * 1e6 * (feed_yield[f]['digestate_frac']) * (
            (params["loading_cost_dig"] / params["capacity_dig"]) + 
            distances.get((i,j), 0) * params["cost_ton_km_dig"]
        ) / 1e6
        for i in supply_nodes
        for f in feedstock_types
    )
    
    # TOTAL FEEDSTOCK-RELATED COSTS
    total_feedstock_cost = feed_cost + transport_cost + digestate_cost
    
    npv = -total_capex + gp.quicksum(
        (total_revenue - total_opex - total_feedstock_cost) / (1 + params["r"])**t 
        for t in range(1, params["years"]+1))
    
    m.setObjective(npv, GRB.MAXIMIZE)
    
    return m, x, y, Omega, N_CH4, total_capex, total_opex, total_revenue, total_feedstock_cost, feed_cost, transport_cost, digestate_cost

# 4) GREEDY HEURISTIC
def greedy_heuristic():
    MIN_NPV = 0
    MIN_IRR = 0.06
    feedstock_df, plant_df, distance_df, yields_df = load_data()
    params = initialize_parameters()
    alternative_configs = get_alternative_configs(params)
    
    cumulative_eeg_bm = 0.0
    cumulative_eeg_bg = 0.0
    cumulative_manure = 0.0
    cumulative_feed = 0.0
    
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
        }
        for _, row in yields_df.iterrows()
    }
    
    distances = {
        (row['Feedstock_LAU'], row['Location']): row['Distance_km']
        for _, row in distance_df.iterrows()
    }
    
    avail_mass = {
        (row['GISCO_ID'], row['substrat_ENG']): row['nutz_pot_tFM']
        for _, row in feedstock_df.iterrows()
    }
    
    Capex_params = {'capex_coeff': 150.12, 'capex_exp': -0.311}
    Opex_params = {'opex_coeff': 2.1209, 'opex_exp': 0.8359}
    Upg_params = {'capex_coeff': 47777, 'capex_exp': -0.421, 'variable_upg_cost': 0.05}
    
    premium = {
        f: max(0, (params["alpha_GHG_comp"] - feed_yield[f]['GHG_intensity']))
           * feed_yield[f]['biogas_m3_per_ton']
           * feed_yield[f]['ch4_content']
           * params["alphaHV"] * 3.6
           * params["GHG_certificate_price"] / 1e6
        for f in feed_yield
    }
    
    plant_locs = plant_df['Location'].tolist()
    selected_plants = []
    results = []
    dist_ik = distances
    
    while len(selected_plants) < len(plant_locs):
        best_npv = -np.inf
        best_plant = None
        best_result = None
        
        for j in plant_locs:
            if j in selected_plants:
                continue
            
            supply_nodes = feedstock_df['GISCO_ID'].unique().tolist()
            feedstock_types = list(feed_yield.keys())
            
            try:
                (m, x, y, Omega, N_CH4,
                 total_capex, total_opex, total_revenue,
                 total_feedstock_cost, feed_cost,
                 transport_cost, digestate_cost) = build_single_plant_model(
                    j,
                    avail_mass=avail_mass,
                    supply_nodes=supply_nodes,
                    feedstock_types=feedstock_types,
                    feed_yield=feed_yield,
                    params=params,
                    Capex_params=Capex_params,
                    Opex_params=Opex_params,
                    Upg_params=Upg_params,
                    premium=premium,
                    distances=distances,
                    cumulative_eeg_bg=cumulative_eeg_bg,
                    cumulative_eeg_bm=cumulative_eeg_bm,
                    manure_used=cumulative_manure,
                    total_feed_used=cumulative_feed
                )
                
                m.Params.NumericFocus = 3
                m.Params.ScaleFlag = 2
                m.Params.Presolve = 2
                m.optimize()
                
                if m.status == GRB.OPTIMAL:
                    npv = m.objVal
                    if npv <= MIN_NPV:
                        continue
                    
                    annual_net = (
                        total_revenue.getValue()
                        - total_opex.getValue()
                        - total_feedstock_cost.getValue()
                    )
                    capex_val = total_capex.getValue()
                    if capex_val > 1e-9:
                        cash_flows = [-capex_val] + [annual_net] * params['years']
                        irr = nf.irr(cash_flows)
                    else:
                        irr = -np.inf
                    
                    if npv > best_npv:
                        best_npv = npv
                        best_plant = j
                        
                        # Record which alternative and capacity were chosen
                        chosen_alt = None
                        chosen_cap = None
                        for a, alt in enumerate(alternative_configs):
                            for c in capacity_levels:
                                if y[a, c].X > 0.5:
                                    chosen_alt = alt["name"]
                                    chosen_cap = c
                                    break
                            if chosen_alt:
                                break
                        
                        best_result = {
                            'model': m,
                            'x': x, 'y': y,
                            'Omega': Omega, 'N_CH4': N_CH4,
                            'total_capex': total_capex,
                            'total_opex': total_opex,
                            'feed+trans': total_feedstock_cost,
                            'used_feedstock': {
                                (i,f): x[i,f].X * 1e6
                                for i,f in x.keys() if x[i,f].X > 1e-6
                            },
                            'selected_alt': chosen_alt,
                            'selected_cap': chosen_cap,
                            'annual_net': annual_net,
                            'irr': irr,
                            'npv': npv
                        }
            
            except Exception as e:
                print(f"Error solving for {j}: {e}")
                continue
        
        if best_plant is None:
            print("No further plants with positive NPV – stopping.")
            break
        
        alt_name = best_result['selected_alt']
        if alt_name == "FlexEEG_biogas":
            cumulative_eeg_bg += best_result['Omega'].X
        elif alt_name == "FlexEEG_biomethane_tech1":
            cumulative_eeg_bm += best_result['Omega'].X
        
        selected_plants.append(best_plant)
        for (i,f), used in best_result['used_feedstock'].items():
            avail_mass[(i,f)] = max(avail_mass.get((i,f),0) - used, 0)
        
        results.append({
            'plant': best_plant,
            'npv': best_result['npv'],
            'irr': best_result['irr'],
            'Methane' : best_result['N_CH4'].X,
            'annual_net': best_result['annual_net'],
            'capacity': best_result['selected_cap'],  # Use selected capacity
            'config': alt_name,
            'capex': best_result['total_capex'].getValue(),
            'opex': best_result['total_opex'].getValue(),
            'feed+trans': best_result['feed+trans'].getValue(),
            'used_feedstock': best_result['used_feedstock'],
            'coordinates': (
                plant_df.loc[plant_df['Location']==best_plant, 'Longitude'].iloc[0],
                plant_df.loc[plant_df['Location']==best_plant, 'Latitude'].iloc[0]
            )
        })
        
        print(
            f"Selected {best_plant} ({alt_name}): "
            f"NPV {best_result['npv']:.2f} M€, "
            f"IRR {best_result['irr']:.1%}, "
            f"annual net {best_result['annual_net']:.2f} M€, "
            f"capex {best_result['total_capex'].getValue():.2f} M€, "
            f"capacity {best_result['selected_cap']:,.0f} m³"
        )
    
    return results, dist_ik

# 5) OUTPUT GENERATION
def generate_outputs(results, dist_ik, output_dir):
    financials = []
    for res in results:
        financials.append({
            'PlantLocation': res['plant'],
            'Longitude': res['coordinates'][0],
            'Latitude': res['coordinates'][1],
            'Alternative': res['config'],
            'Capacity': res['capacity'],  # Already in m³
            'NPV_EUR': res['npv'],
            'CAPEX_EUR': res['capex'],
            'OPEX_EUR': res['opex'],
            'Methane' : res['Methane'],
            'Feed_Trans_Cost': res['feed+trans']
        })
    
    pd.DataFrame(financials).to_csv(os.path.join(output_dir, "Financials_20_greedy.csv"), index=False)
    
    flows = []
    for res in results:
        for (i, f), qty in res['used_feedstock'].items():
            flows.append({
                'PlantLocation': res['plant'],
                'SupplyNode': i,
                'Feedstock': f,
                'FlowTons': qty,
                'Distance_km': dist_ik.get((i, res['plant']), 0)
            })
    
    pd.DataFrame(flows).to_csv(os.path.join(output_dir, "Flows_20_greedy.csv"), index=False)

if __name__ == '__main__':
    output_dir = os.path.join("C:/Clone/Master/results/large_scale_cont/10_greedy_with_alternatives/greedy_100/")
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    results, dist_ik = greedy_heuristic()
    
    if results:
        print(f"\nTotal NPV: €{sum(r['npv'] for r in results):,.0f}")
        generate_outputs(results, dist_ik, output_dir)
    
    print(f"Execution time: {time.time()-start_time:.1f}s")
    