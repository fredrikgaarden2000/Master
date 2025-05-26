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
        plant_df = safe_load_csv(f"{BASE_DIR}equally_spaced_locations_100.csv")
        distance_df = safe_load_csv(f"{BASE_DIR}Distance_Matrix_100.csv")
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
        "CN_min": 15.0,  # Changed from original
        "CN_max": 35.0,  # Changed from original
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
        "variable_upg_cost": 0.05,
        "alpha_GHG_comp": 94.0,
        "GHG_certificate_price": 50,
        "Q_MAX": 80,
        "Q_MIN": 0.01,
        "cap_biogas": 0.45,
        "cap_biomethane": 0.10,  # Added from alternatives
        "bonus_rate": 100,
        "loading_cost_dig": 27,
        "capacity_dig": 37,
        "cost_ton_km_dig": 0.104,
        "auction_bg_limit": 225000,
        "auction_bm_limit": 125000,
        "EEG_small_m3" : 255870*(8000/8760),
        "EEG_large_m3" : 511740*(8000/8760),
        "manure_percent_limit": 1,
        "boiler_eff": 0.9  # Added from alternatives
    }

# Define alternative configurations
def get_alternative_configs(params):
    return [
        {"name": "FlexEEG_biogas", "category": "FlexEEG_biogas", "prod_cap_factor": params["cap_biogas"], 
         "max_cap_m3_year": None, "upg_cost_coeff": 0, "upg_cost_exp": 0, 
         "rev_price": {"EEG": params["EEG_skip_chp_price"]}, "EEG_flag": True, 
         "GHG_eligible": False, "feed_constraint": None, "capex_coeff": 150.12, 
         "capex_exp": -0.311, "capex_type": "standard", "opex_coeff": 2.1209, 
         "opex_exp": 0.8359, "opex_type": "standard"},
        
        {"name": "Upgrading_tech1", "category": "Upgrading", "prod_cap_factor": 1.0, 
         "max_cap_m3_year": None, "upg_cost_coeff": 47777, "upg_cost_exp": -0.421, 
         "rev_price": {"gas": params["gas_price_mwh"] * (params["alphaHV"] / 1000), 
                      "co2": params["co2_price_ton"] / 556.2}, 
         "EEG_flag": False, "GHG_eligible": True, "feed_constraint": None, 
         "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard", 
         "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
        
        {"name": "nonEEG_CHP", "category": "CHP_nonEEG", "prod_cap_factor": 1.0, 
         "max_cap_m3_year": None, "upg_cost_coeff": 0, "upg_cost_exp": 0, 
         "rev_price": {"spot": params["electricity_spot_price"], 
                      "heat": params["heat_price"]}, 
         "EEG_flag": False, "GHG_eligible": False, "feed_constraint": None, 
         "capex_coeff": 150.12, "capex_exp": -0.311, "capex_type": "standard", 
         "opex_coeff": 2.1209, "opex_exp": 0.8359, "opex_type": "standard"},
        
        {"name": "FlexEEG_biomethane_tech1", "category": "FlexEEG_biomethane", 
         "prod_cap_factor": params["cap_biomethane"], "max_cap_m3_year": None, 
         "upg_cost_coeff": 47777, "upg_cost_exp": -0.421, 
         "rev_price": {"EEG": params["EEG_skip_upg_price"]}, "EEG_flag": True, 
         "GHG_eligible": False, "feed_constraint": None, "capex_coeff": 150.12, 
         "capex_exp": -0.311, "capex_type": "standard", "opex_coeff": 2.1209, 
         "opex_exp": 0.8359, "opex_type": "standard"},

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

# Helper functions for feedstock classification
def is_manure(ftype):
    return 'man' in ftype.lower() or 'slu' in ftype.lower()

def is_clover(ftype):
    return 'clover' in ftype.lower()

def is_maize_cereal(ftype):
    return 'maize' in ftype.lower() or 'cereal' in ftype.lower()

# 4) PLANT MODEL BUILDER - UPDATED WITH ALTERNATIVES
def build_single_plant_model(j, avail_mass, supply_nodes, feedstock_types, feed_yield, 
                            params, Capex_params, Opex_params, Upg_params, premium, distances,
                            cumulative_eeg_bg=0,cumulative_eeg_bm=0, manure_used=0, total_feed_used=0):
    m = gp.Model(f"Plant_{j}")
    m.setParam('OutputFlag', 0)
    
    # Get alternative configurations
    alternative_configs = get_alternative_configs(params)
    
    # Variables
    x = m.addVars(supply_nodes, feedstock_types, lb=0, name="x")
    y = m.addVars(range(len(alternative_configs)), vtype=GRB.BINARY, name="y")  # One for each alternative
    Omega = m.addVar(lb=params["Q_MIN"], ub=params["Q_MAX"], name="Omega")
    N_CH4 = m.addVar(lb=0, ub=params["Q_MAX"], name="N_CH4")
    total_methane = sum(avail_mass[i, f] * feed_yield[f]['ch4_content'] for i, f in avail_mass)
    total_mass = sum(avail_mass[i, f] for i, f in avail_mass)
    system_methane_average = total_methane / total_mass
    # Only one alternative can be selected
    m.addConstr(gp.quicksum(y[a] for a in range(len(alternative_configs))) <= 1, "OneAlt")
    
    # Plant is active if any alternative is selected
    is_active = m.addVar(vtype=GRB.BINARY, name="is_active")
    m.addConstr(is_active == gp.quicksum(y[a] for a in range(len(alternative_configs))), "ActiveLink")
    m.addConstr(Omega <= params["Q_MAX"] * is_active, "OmegaUpper")
    m.addConstr(Omega >= params["Q_MIN"] * is_active, "OmegaLower")

    # EEG volume limit calculation
    eeg_volume_limit_bg = (params["auction_bg_limit"] * params["FLH_max"] /
                       (params["alphaHV"] * system_methane_average)) / 1e6
    
        # EEG volume limit calculation
    eeg_volume_limit_bm = (params["auction_bm_limit"] * params["FLH_max"] /
                       (params["alphaHV"] * system_methane_average)) / 1e6
    # only apply the biogas‐EEG limit to the biogas alt
    for a, alt in enumerate(alternative_configs):
        if alt["category"] == "FlexEEG_biogas":
            m.addGenConstrIndicator(
                y[a], True,
                Omega <= eeg_volume_limit_bg - cumulative_eeg_bg,
                name=f"EEG_limit_bg_{j}_{a}"
            )

    # only apply the biomethane‐EEG limit to the biomethane alt
    for a, alt in enumerate(alternative_configs):
        if alt["category"] == "FlexEEG_biomethane":
            m.addGenConstrIndicator(
                y[a], True,
                Omega <= eeg_volume_limit_bm - cumulative_eeg_bm,
                name=f"EEG_limit_bm_{j}_{a}"
            )

    # MANURE USAGE CONSTRAINT (PER PLANT)
    manure_types = [f for f in feedstock_types if is_manure(f)]
    plant_manure = gp.quicksum(x[i,f] for i in supply_nodes for f in manure_types)
    plant_total_feed = gp.quicksum(x[i,f] for i in supply_nodes for f in feedstock_types)
    m.addConstr(plant_manure <= params["manure_percent_limit"] * plant_total_feed, 
               "manure_limit_per_plant")
    # CAPEX CALCULATIONS --------------------------------------------------------
    BREAKS = np.linspace(params["Q_MIN"], params["Q_MAX"], 11)

    # --- 1) piecewise base CHP cost ---
    base_capex_vals = [
        ((b * 1e6) * Capex_params["capex_coeff"] * (b * 1e6) ** Capex_params["capex_exp"]) / 1e6
        for b in BREAKS
    ]
    base_hat = m.addVar(name="base_hat")
    m.addGenConstrPWL(Omega, base_hat, BREAKS.tolist(), base_capex_vals)

    # --- 2) piecewise upgrading cost ---
    upg_capex_vals = [
        ((b * 1e6 / params["FLH_max"]) * Upg_params["capex_coeff"] *
         (b * 1e6 / params["FLH_max"]) ** Upg_params["capex_exp"]) / 1e6
        for b in BREAKS
    ]
    upg_hat = m.addVar(name="upg_hat")
    m.addGenConstrPWL(Omega, upg_hat, BREAKS.tolist(), upg_capex_vals)

    # --- 3) linking var for upgrading alternative ---
    upg_eff = m.addVar(lb=0, name="upg_eff")
    upgrading_idxs = [
        idx for idx, alt in enumerate(alternative_configs)
        if alt["category"] == "Upgrading"
    ]
    if upgrading_idxs:
        upg_sel = m.addVar(vtype=GRB.BINARY, name="upg_sel")
        # upg_sel == 1 if any y[a] for upgrading alternatives is 1
        m.addConstr(upg_sel == gp.quicksum(y[a] for a in upgrading_idxs), "LinkUpg")
        lin_on  = upg_eff - upg_hat
        lin_off = upg_eff
        m.addGenConstrIndicator(upg_sel, 1, lin_on,  GRB.EQUAL, 0, name="UpgOn")
        m.addGenConstrIndicator(upg_sel, 0, lin_off, GRB.EQUAL, 0, name="UpgOff")
    else:
        m.addConstr(upg_eff == 0, "NoUpg")

    # 2) assemble total_capex by alternative
    # ------------------------------------------------------------------------
    capex_terms = []
    for a, alt in enumerate(alternative_configs):
        if alt["category"] == "EEG_CHP_small":
            capex_terms.append(y[a] * (params["EEG_small_m3"] * Capex_params["capex_coeff"] * (params["EEG_small_m3"]) ** Capex_params["capex_exp"]) / 1e6)
        elif alt["category"] == "EEG_CHP_large":
            capex_terms.append(y[a] * (params["EEG_large_m3"] * Capex_params["capex_coeff"] * (params["EEG_large_m3"]) ** Capex_params["capex_exp"]) / 1e6)
        # add upgrading capex only on the upgrading alt
        elif alt["category"] in ["Upgrading", "FlexEEG_biomethane"]:
            # include *both* base CHP capex and upgrading capex
            capex_terms.append(y[a] * base_hat)
            capex_terms.append(y[a] * (upg_eff + 1)) #1 million grid connection
        else:
            # for FlexEEG_biogas, FlexEEG_biomethane, CHP_nonEEG, etc.
            capex_terms.append(y[a] * base_hat)

    total_capex = gp.quicksum(capex_terms)

    # OPEX CALCULATIONS ---------------------------------------------------------
    base_opex_vals = [
        Opex_params["opex_coeff"] * (b * 1e6) ** Opex_params["opex_exp"] / 1e6
        for b in BREAKS
    ]
    opex_biogas = m.addVar(name="opex_biogas")
    m.addGenConstrPWL(Omega, opex_biogas, BREAKS.tolist(), base_opex_vals)
    
    # Upgrading OPEX - only applies if upgrading is selected
    upg_opex = m.addVar(name="upg_opex")
    if upgrading_idxs:
        m.addGenConstrIndicator(upg_sel, True, upg_opex == params["variable_upg_cost"] * N_CH4, name="upg_opex_on")
        m.addGenConstrIndicator(upg_sel, False, upg_opex == 0, name="upg_opex_off")
    else:
        m.addConstr(upg_opex == 0, "NoUpgOpex")
    
    total_opex = opex_biogas + upg_opex
    
    # CONSTRAINTS ---------------------------------------------------------------
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
    excess = m.addVar(lb=0, name="excess")
    diff = m.addVar(name="diff")
    m.addConstr(diff == Omega - threshold_m3, "bonus_diff")
    m.addGenConstrMax(excess, [diff, 0], name="bonus_excess")
    bonus = params["bonus_rate"] * excess / 1e6
   # Feedstock totals
    # 1) with a dict‐comprehension# 2) with an explicit loop
    total_feed_j   = gp.quicksum(x[i,f] 
                                for (i,f) in avail_mass)

    # total manure feed into j
    manure_feed_j  = gp.quicksum(x[i,f] 
                                for (i,f) in avail_mass
                                if is_manure(f))

    # total clover feed into j
    clover_feed_j  = gp.quicksum(x[i,f] 
                                for (i,f) in avail_mass
                                if is_clover(f))

    avg_discount = sum(0.99**t for t in range(1, params['years']+1)) / params['years']
    # Revenue calculations for each alternative
    rev_alternatives = []
    for a, alt in enumerate(alternative_configs):
        rev_a = m.addVar(name=f"rev_alt_{a}")
        if alt["category"] in ["EEG_CHP_small", "EEG_CHP_large"]:
            m.addGenConstrIndicator(
                y[a], True,
                Omega <= alt["max_cap_m3_year"] / 1e6,
                name=f"MaxCap_{j}_{a}"
            )
            # 2) manures/clover mix
            # total_feed  = sum(x[i,f] for all i,f)
            # manure_feed = sum(x[i,f] for manure‐types)
            # clover_feed = sum(x[i,f] for clover‐types)
            if alt["feed_constraint"] == 1:
                # ≥80% manure if selected
                m.addConstr(manure_feed_j >= 0.80 * total_feed_j * y[a],
                            name=f"manure80_{j}_{a}")
            else:  # feed_constraint == 2
                m.addConstr(manure_feed_j >= 0.70 * total_feed_j * y[a],
                            name=f"manure70_{j}_{a}")
                m.addConstr(clover_feed_j  >= 0.10 * total_feed_j * y[a],
                            name=f"clover10_{j}_{a}")
                    # 3) revenue
            effective_EEG = alt["rev_price"]["EEG"] * avg_discount
            E_elec = N_CH4 * (params["chp_elec_eff"] * params["alphaHV"] / 1000)
            rev_val = E_elec * effective_EEG \
                    + N_CH4 * (params["chp_heat_eff"] * params["heat_price"] * (params["alphaHV"]/1000))
            m.addGenConstrIndicator(y[a], True, rev_a == rev_val, name=f"rev_on_{j}_{a}")
            m.addGenConstrIndicator(y[a], False, rev_a == 0, name=f"rev_off_{j}_{a}")

        elif alt["category"] == "FlexEEG_biogas":
            # CAPACITY-BASED EEG REVENUE
            effective_EEG = alt["rev_price"]["EEG"] * avg_discount
            U_elec = N_CH4 * params["chp_elec_eff"] * params["alphaHV"] / 1000
            EEG_cap = U_elec * alt["prod_cap_factor"]
            
            EEG_rev = EEG_cap * effective_EEG
            spot_rev = (U_elec - EEG_cap) * params["electricity_spot_price"]
            heat_rev = N_CH4 * params["chp_heat_eff"] * params["alphaHV"] / 1000 * params["heat_price"]
            
            revenue_alt = EEG_rev + spot_rev + heat_rev + bonus
            m.addGenConstrIndicator(y[a], True, rev_a == revenue_alt, name=f"rev_alt_{a}_on")
            m.addGenConstrIndicator(y[a], False, rev_a == 0, name=f"rev_alt_{a}_off")
            
        elif alt["category"] == "Upgrading":
            revenue_alt = (
                N_CH4 * gas_price_m3 
                + (Omega - N_CH4) * co2_price 
                + gp.quicksum(x[i,f] * premium[f] for i in supply_nodes for f in feedstock_types))
            m.addGenConstrIndicator(y[a], True, rev_a == revenue_alt, name=f"rev_alt_{a}_on")
            m.addGenConstrIndicator(y[a], False, rev_a == 0, name=f"rev_alt_{a}_off")
            
        elif alt["category"] == "CHP_nonEEG":
            U_elec = N_CH4 * params["chp_elec_eff"] * params["alphaHV"] / 1000
            spot_rev = U_elec * alt["rev_price"]["spot"]
            heat_rev = N_CH4 * params["chp_heat_eff"] * params["alphaHV"] / 1000 * alt["rev_price"]["heat"]
            
            revenue_alt = spot_rev + heat_rev
            m.addGenConstrIndicator(y[a], True, rev_a == revenue_alt, name=f"rev_alt_{a}_on")
            m.addGenConstrIndicator(y[a], False, rev_a == 0, name=f"rev_alt_{a}_off")
            
        elif alt["category"] == "FlexEEG_biomethane":
            # Similar to biogas but with different cap factor
            effective_EEG = alt["rev_price"]["EEG"] * avg_discount
            U_elec = N_CH4 * params["chp_elec_eff"] * params["alphaHV"] / 1000
            EEG_cap = U_elec * alt["prod_cap_factor"]
            
            EEG_rev = EEG_cap * effective_EEG
            spot_rev = (U_elec - EEG_cap) * params["electricity_spot_price"]
            heat_rev = N_CH4 * params["chp_heat_eff"] * params["alphaHV"] / 1000 * params["heat_price"]
            
            revenue_alt = EEG_rev + spot_rev + heat_rev + bonus
            m.addGenConstrIndicator(y[a], True, rev_a == revenue_alt, name=f"rev_alt_{a}_on")
            m.addGenConstrIndicator(y[a], False, rev_a == 0, name=f"rev_alt_{a}_off")
            
        rev_alternatives.append(rev_a)
    
    total_revenue = gp.quicksum(rev_alternatives)

    # FEEDSTOCK COST CALCULATIONS ==============================================
    feed_cost = gp.quicksum(
        x[i,f] * feed_yield[f]['price']
        for i in supply_nodes 
        for f in feedstock_types
    )

    # LOADING + TRANSPORT COST -------------------------------------------------
    transport_cost = gp.quicksum(
        x[i,f] * 1e6 * (
            (feed_yield[f]['loading'] / feed_yield[f]['capacity_load']) + 
            distances.get((i, j), 0) * feed_yield[f]['cost_ton_km']
        ) / 1e6
        for i in supply_nodes
        for f in feedstock_types
    )

    # DIGESTATE COST -----------------------------------------------------------
    digestate_cost = gp.quicksum(
        x[i,f]* 1e6 *(feed_yield[f]['digestate_frac']) * (
            (params["loading_cost_dig"] / params["capacity_dig"]) + 
            distances.get((i,j), 0) * params["cost_ton_km_dig"]
        ) /1e6
        for i in supply_nodes
        for f in feedstock_types
    )

    # TOTAL FEEDSTOCK-RELATED COSTS =============================================
    total_feedstock_cost = feed_cost + transport_cost + digestate_cost

    npv = -total_capex + gp.quicksum(
        (total_revenue - total_opex - total_feedstock_cost) / (1 + params["r"])**t 
        for t in range(1, params["years"]+1))
    
    m.setObjective(npv, GRB.MAXIMIZE)
    
    return m, x, y, Omega, N_CH4, total_capex, total_opex, total_revenue, total_feedstock_cost, feed_cost, transport_cost, digestate_cost, upg_hat

def greedy_heuristic():
    MIN_IRR = 0.06   # 10% minimum
    feedstock_df, plant_df, distance_df, yields_df = load_data()
    params = initialize_parameters()
    alternative_configs = get_alternative_configs(params)

    cumulative_eeg_bm = 0.0
    cumulative_eeg_bg = 0.0
    cumulative_manure = 0.0
    cumulative_feed   = 0.0

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
    Opex_params  = {'opex_coeff': 2.1209,  'opex_exp': 0.8359}
    Upg_params   = {'capex_coeff': 47777,  'capex_exp': -0.421, 'variable_upg_cost': 0.05}

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
        best_irr    = -np.inf
        best_plant  = None
        best_result = None

        for j in plant_locs:
            if j in selected_plants:
                continue

            supply_nodes    = feedstock_df['GISCO_ID'].unique().tolist()
            feedstock_types = list(feed_yield.keys())

            try:
                (m, x, y, Omega, N_CH4,
                 total_capex, total_opex, total_revenue,
                 total_feedstock_cost, feed_cost,
                 transport_cost, digestate_cost, upg_hat) = build_single_plant_model(
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
                m.Params.ScaleFlag    = 2
                m.Params.Presolve      = 2
                m.optimize()

                if m.status == GRB.OPTIMAL:
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

                    if irr < MIN_IRR:
                        continue   # skip anything below 10%

                    if irr > best_irr:
                        best_irr    = irr
                        best_plant  = j

                        # record which alt was chosen
                        chosen = None
                        for idx, alt in enumerate(alternative_configs):
                            if y[idx].X > 0.5:
                                chosen = alt["name"]
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
                            'selected_alt': chosen,
                            'annual_net': annual_net,
                            'irr': irr,
                            'npv' : m.objVal
                        }

            except Exception as e:
                print(f"Error solving for {j}: {e}")
                continue
        # after scanning all j …
        if best_irr < MIN_IRR or best_result is None:
            print(f"No further plants achieve at least {MIN_IRR:.0%} IRR – stopping.")
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
            'npv' : best_result['npv'],
            'irr': best_result['irr'],
            'annual_net': best_result['annual_net'],
            'capacity': best_result['Omega'].X,
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
            f"IRR {best_result['irr']:.1%}, "
            f"annual net {best_result['annual_net']:.2f} M€, "
            f"capex {best_result['total_capex'].getValue():.2f} M€, "
            f"capacity {best_result['Omega'].X*1e6:,.0f} m³"
        )

    return results, dist_ik


# 6) OUTPUT GENERATION - UPDATED
def generate_outputs(results, dist_ik, output_dir):
    # Financials
    financials = []
    for res in results:
        financials.append({
            'PlantID': res['plant'],
            'Longitude': res['coordinates'][0],
            'Latitude': res['coordinates'][1],
            'Configuration': res['config'],
            'TotalCapacity_m3': res['capacity'] * 1e6,
            'NPV_EUR': res['npv'],
            'IRR' : res['irr'],
            'CAPEX_EUR': res['capex'],
            'OPEX_EUR': res['opex'],
            'Feed_Trans_Cost': res['feed+trans']
        })
    
    pd.DataFrame(financials).to_csv(os.path.join(output_dir, "Financials.csv"), index=False)
    
    # Flows
    flows = []
    for res in results:
        for (i, f), qty in res['used_feedstock'].items():
            flows.append({
                'PlantID': res['plant'],
                'SupplyCluster': i,
                'Feedstock': f,
                'Quantity_t': qty,
                'Distance_km': dist_ik.get((i, res['plant']), 0)
            })
    
    pd.DataFrame(flows).to_csv(os.path.join(output_dir, "Flows.csv"), index=False)

if __name__ == '__main__':
    output_dir = os.path.join("C:/Clone/Master/results/large_scale_cont/10_greedy_with_alternatives")
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    results, dist_ik = greedy_heuristic()
    
    if results:
        print(f"\nTotal NPV: €{sum(r['npv'] for r in results):,.0f}")
        generate_outputs(results, dist_ik, output_dir)
    
    print(f"Execution time: {time.time()-start_time:.1f}s")

    # build DataFrame from results
    df = pd.DataFrame(results)

    # chart colors by configuration
    colors = {
        "FlexEEG_biomethane": "#66c2a5",
        "Upgrading": "#fc8d62",
        "FlexEEG_biogas": "#8da0cb",
        "NonEEG_CHP": "#e78ac3"
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, row in df.iterrows():
        width = row["capacity"]        # capacity in Mm³/yr
        irr_pct = row["irr"] * 100     # convert to %
        ax.barh(
            y=irr_pct,
            width=width,
            left=0,                     # all bars start at zero
            height=0.8,
            color=colors.get(row["config"], "#a6d854"),  # default color if not in dict
            edgecolor="black"
        )
        ax.text(
            width + 0.01 * df["capacity"].max(),  # offset label slightly right of bar
            irr_pct,
            row["plant"],
            ha="left",
            va="center",
            fontsize=8,
            color="black"
        )

    ax.set_xlabel("Capacity Ω [Mm³ biogas / year]")
    ax.set_ylabel("Internal Rate of Return [%]")
    ax.set_title("Greedy solution – plant IRR vs. capacity")
    ax.set_ylim(0, df["irr"].max() * 100 * 1.1)

    # Generate legend based on unique configurations in the DataFrame
    unique_configs = df['config'].unique()
    legend_handles = [
        Patch(
            facecolor=colors.get(config, "#a6d854"), 
            edgecolor='black', 
            label=config
        ) for config in unique_configs
    ]

    ax.legend(
        handles=legend_handles,
        title="Technology",
        bbox_to_anchor=(1.04, 1),
        loc="upper left"
    )
    plt.tight_layout()
    plt.show()