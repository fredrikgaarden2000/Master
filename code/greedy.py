import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
import time

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
        plant_df = safe_load_csv(f"{BASE_DIR}equally_spaced_locations_100_copy.csv")
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
        "CN_min": 20.0,
        "CN_max": 30.0,
        "heat_price": 20,
        "chp_elec_eff": 0.4,
        "chp_heat_eff": 0.4,
        "electricity_spot_price": 60,
        "EEG_skip_chp_price": 194.3,
        "r": 0.042,
        "years": 25,
        "gas_price_mwh": 30,
        "co2_price_ton": 20,
        "variable_upg_cost": 0.05,
        "alpha_GHG_comp": 94.0,
        "GHG_certificate_price": 50,
        "Q_MAX": 60,
        "Q_MIN": 5,
        "cap_biogas" : 0.45,
        "bonus_rate" : 100,
        "loading_cost_dig": 27,      # €/ton digestate loading cost
        "capacity_dig": 37,          # ton/truck digestate capacity
        "cost_ton_km_dig": 0.104,      # €/ton/km digestate transport
        "auction_chp_limit": 225000,  # kW
        "auction_bm_limit": 125000,    # kW
        "manure_percent_limit": 1    # 50% max manure usage
    }

# 4) PLANT MODEL BUILDER
def build_single_plant_model(j, avail_mass, supply_nodes, feedstock_types, feed_yield, 
                            params, Capex_params, Upg_params, Opex_params, premium, distances,
                            cumulative_eeg=0, manure_used=0, total_feed_used=0):
    m = gp.Model(f"Plant_{j}")
    m.setParam('OutputFlag', 0)
        # Variables
    x = m.addVars(supply_nodes, feedstock_types, lb=0, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    Omega = m.addVar(lb=params["Q_MIN"], ub=params["Q_MAX"], name="Omega")
    N_CH4 = m.addVar(lb=0, ub=params["Q_MAX"], name="N_CH4")
    total_methane = sum(avail_mass[i, f] * feed_yield[f]['ch4_content'] for i, f in avail_mass)
    total_mass = sum(avail_mass[i, f] for i, f in avail_mass)
    system_methane_average = total_methane / total_mass

    eeg_volume_limit = (params["auction_chp_limit"] 
                        * params["FLH_max"] /
                       (params["alphaHV"] * system_methane_average)) / 1e6  # Convert MW to Mm³
    
    # Only apply to biogas plants (y=0)
    m.addGenConstrIndicator(y, False, Omega <= eeg_volume_limit - cumulative_eeg,
                           name=f"EEG_limit_{j}")
    
     # CAPEX CALCULATIONS --------------------------------------------------------
    BREAKS = np.linspace(params["Q_MIN"], params["Q_MAX"], 11)
    
    # MANURE USAGE CONSTRAINT (PER PLANT) --------------------------------------
    manure_types = [f for f in feedstock_types if 'man' in f.lower() or 'slu' in f.lower()]
    
    # Total manure used at this plant
    plant_manure = gp.quicksum(x[i,f] for i in supply_nodes for f in manure_types)
    
    # Total feedstock at this plant
    plant_total_feed = gp.quicksum(x[i,f] for i in supply_nodes for f in feedstock_types)
    
    # Add constraint: manure <= X% of total feedstock
    m.addConstr(plant_manure <= params["manure_percent_limit"] * plant_total_feed, 
               "manure_limit_per_plant")

    # Base biogas plant CAPEX (CHP)
    base_capex_vals = [
        ((b * 1e6) * Capex_params["capex_coeff"] * (b * 1e6) ** Capex_params["capex_exp"]) / 1e6
        for b in BREAKS
    ]
    base_hat = m.addVar(name="base_hat")
    m.addGenConstrPWL(Omega, base_hat, BREAKS.tolist(), base_capex_vals)
    
    # Upgrading CAPEX (biomethane)
    upg_capex_vals = [
        ((b * 1e6 / params["FLH_max"]) * Upg_params["capex_coeff"] * 
        (b * 1e6 / params["FLH_max"]) ** Upg_params["capex_exp"]) / 1e6 
        for b in BREAKS
    ]
    upg_hat = m.addVar(name="upg_hat")
    upg_eff = m.addVar(lb=0, name="upg_eff")
    m.addGenConstrPWL(Omega, upg_hat, BREAKS.tolist(), upg_capex_vals)
    
    # CAPEX linking constraints
    m.addGenConstrIndicator(y, True, upg_eff == upg_hat, name="upg_cap_on")
    m.addGenConstrIndicator(y, False, upg_eff == 0, name="upg_cap_off")
    total_capex = base_hat + upg_eff
    
    # OPEX CALCULATIONS ---------------------------------------------------------
    base_opex_vals = [
        Opex_params["opex_coeff"] * (b * 1e6) ** Opex_params["opex_exp"] / 1e6
        for b in BREAKS
    ]
    opex_biogas = m.addVar(name="opex_biogas")
    m.addGenConstrPWL(Omega, opex_biogas, BREAKS.tolist(), base_opex_vals)
    
    # Replace the existing upg_opex line with:
    upg_opex = m.addVar(name="upg_opex")
    m.addGenConstrIndicator(y, True, upg_opex == Upg_params["variable_upg_cost"] * N_CH4, name="upg_opex_on")
    m.addGenConstrIndicator(y, False, upg_opex == 0, name="upg_opex_off")

    total_opex = opex_biogas + upg_opex  # Keep this line
    
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
            # Get availability (0 if not listed)
            available = avail_mass.get((i, f), 0) 
            # Always add constraint, even if availability is 0
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

    # CAPACITY-BASED EEG REVENUE
    U_elec = N_CH4 * params["chp_elec_eff"] * params["alphaHV"] / 1000
    EEG_cap = U_elec * params["cap_biogas"]
    
    EEG_rev = EEG_cap * params["EEG_skip_chp_price"]
    spot_rev = (U_elec - EEG_cap) * params["electricity_spot_price"]
    heat_rev = N_CH4 * params["chp_heat_eff"] * params["alphaHV"] / 1000 * params["heat_price"]
    
    revenue_biogas = EEG_rev + spot_rev + heat_rev + bonus
                   
    revenue_upg = (
    N_CH4 * gas_price_m3 
    + (Omega - N_CH4) * co2_price 
    + gp.quicksum(x[i,f] * premium[f] for i in supply_nodes for f in feedstock_types))
    
    # FEEDSTOCK COST CALCULATIONS ==============================================
    feed_cost = gp.quicksum(
        x[i,f] * feed_yield[f]['price']  # Base feedstock cost
        for i in supply_nodes 
        for f in feedstock_types
    )

    # LOADING + TRANSPORT COST -------------------------------------------------
    transport_cost = gp.quicksum(
        x[i,f] * 1e6 * (  # x is in million tons, convert to tons
            (feed_yield[f]['loading'] / feed_yield[f]['capacity_load']) + 
            distances.get((i, j), 0) * feed_yield[f]['cost_ton_km']
        ) / 1e6  # Convert back to million €
        for i in supply_nodes
        for f in feedstock_types
    )

    # DIGESTATE COST -----------------------------------------------------------
    digestate_cost = gp.quicksum(
        x[i,f]* 1e6 *(feed_yield[f]['digestate_frac'])  * (
            (params["loading_cost_dig"] / params["capacity_dig"]) + 
            distances.get((i,j), 0) * params["cost_ton_km_dig"]
        ) /1e6
        for i in supply_nodes
        for f in feedstock_types
    )

    # TOTAL FEEDSTOCK-RELATED COSTS =============================================
    total_feedstock_cost = feed_cost + transport_cost + digestate_cost

    total_revenue = (1 - y) * revenue_biogas + y * revenue_upg
    
    npv = -total_capex + gp.quicksum(
        (total_revenue - total_opex - total_feedstock_cost) / (1 + params["r"])**t 
        for t in range(1, params["years"]+1))
    
    m.setObjective(npv, GRB.MAXIMIZE)
    
    return m, x, y, Omega, N_CH4, total_capex, total_opex, total_revenue, revenue_biogas, revenue_upg, total_feedstock_cost,feed_cost, transport_cost, digestate_cost, upg_eff

# 5) GREEDY HEURISTIC IMPLEMENTATION
def greedy_heuristic():
    feedstock_df, plant_df, distance_df, yields_df = load_data()
    params = initialize_parameters()

    cumulative_eeg = 0
    cumulative_manure = 0
    cumulative_feed = 0

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

    distances = {(row['Feedstock_LAU'], row['Location']): row['Distance_km'] 
                for _, row in distance_df.iterrows()}
    
    avail_mass = {(row['GISCO_ID'], row['substrat_ENG']): row['nutz_pot_tFM'] 
                 for _, row in feedstock_df.iterrows()}
    
    Capex_params = {'capex_coeff': 150.12, 'capex_exp': -0.311}
    Opex_params = {'opex_coeff': 2.1209, 'opex_exp': 0.8359}
    Upg_params = {'capex_coeff': 47777, 'capex_exp': -0.421, "variable_upg_cost" : 0.05}
    
    premium = {
        f: max(0, (params["alpha_GHG_comp"] - feed_yield[f]['GHG_intensity'])) 
        * feed_yield[f]['biogas_m3_per_ton']  # Include biogas yield
        * feed_yield[f]['ch4_content']        # Include methane content
        * params["alphaHV"] * 3.6             # Energy conversion
        * params["GHG_certificate_price"] 
        / 1e6                                 # Convert grams to tons
        for f in feed_yield.keys()
    }

    plant_locs = plant_df['Location'].tolist()
    selected_plants = []
    results = []
    dist_ik = {(row['Feedstock_LAU'], row['Location']): row['Distance_km'] for _, row in distance_df.iterrows()}
    
    while len(selected_plants) < len(plant_locs):
        best_npv = -np.inf
        best_plant = None
        best_result = None
        
        for j in plant_locs:
            if j in selected_plants:
                continue
            supply_nodes      = feedstock_df['GISCO_ID'].unique().tolist()
            feedstock_types   = list(feed_yield.keys())
            try:
                m, x, y, Omega, N_CH4, total_capex, total_opex, total_revenue, \
                rev_biogas, rev_upg, total_feedstock_cost, feed_cost, \
                transport_cost, digestate_cost, upg_eff = build_single_plant_model(
                        j                       ,
                        avail_mass              ,
                        supply_nodes=supply_nodes,
                        feedstock_types=feedstock_types,
                        feed_yield=feed_yield   ,
                        params=params           ,
                        Capex_params=Capex_params,
                        Upg_params=Upg_params   ,
                        Opex_params=Opex_params ,
                        premium=premium         ,
                        distances=distances     ,
                        cumulative_eeg=cumulative_eeg,
                        manure_used=cumulative_manure,
                        total_feed_used=cumulative_feed)

                m.Params.NumericFocus = 3
                m.Params.ScaleFlag = 2
                m.Params.Presolve = 2
                m.optimize()
                
                if m.status == GRB.OPTIMAL and m.objVal > best_npv:
                    best_npv = m.objVal
                    best_plant = j
                    best_result = {
                        'model': m,
                        'x': x,
                        'y': y,
                        'Omega': Omega,
                        'N_CH4': N_CH4,
                        'total_capex': total_capex,
                        'total_opex': total_opex,
                        'feed+trans' : total_feedstock_cost,
                        'used_feedstock': {(i,f): x[i,f].X*1e6 for i,f in x.keys() if x[i,f].X > 1e-6}
                    }
                    print(f"\n--- DEBUG: Plant {j} ---")
                    print(f"Configuration: {'Upgrading' if y.X > 0.5 else 'Biogas'}")
                    print(f"Omega (Total Biogas): {Omega.X:.2f} Mm³/yr")
                    print(f"N_CH4 (Methane): {N_CH4.X:.2f} Mm³/yr")
                    print(f"CAPEX: {total_capex.getValue():.2f} M€")
                    print(f"CAPEX: {upg_eff.X:.2f} M€")
                    print(f"OPEX: {total_opex.getValue():.2f} M€/yr")
                    print(f"Revenue: {total_revenue.getValue():.2f} M€/yr")
                    print(f"Feed + Trans Cost: {total_feedstock_cost.getValue():.2f} M€/yr")
                    print(f"Feed: {feed_cost.getValue():.2f} M€/yr")
                    print(f"Trans Cost: {transport_cost.getValue():.2f} M€/yr")
                    print(f"Digestate Cost: {digestate_cost.getValue():.2f} M€/yr")
                    print(f"NPV: {m.objVal:.2f} M€")
                        
            except Exception as e:
                print(f"Error solving for {j}: {str(e)}")
                continue
                
        if not best_result or best_npv <= 0:
            print("No profitable plants remaining")
            break
            
        if best_result['y'].X < 0.5:  # Biogas plant
            cumulative_eeg += Omega.X

                # Update tracking FIRST
        selected_plants.append(best_plant)  # Critical fix: Mark plant as selected
            
        # Update feedstock
        for (i, f), used in best_result['used_feedstock'].items():
            if (i, f) in avail_mass:
                avail_mass[(i, f)] = max(avail_mass[(i, f)] - used, 0)
        
        # Store results
        results.append({
            'plant': best_plant,
            'npv': best_npv,
            'capacity': best_result['Omega'].X,
            'config': "Upgrading" if best_result['y'].X > 0.5 else "Biogas",
            'capex': best_result['total_capex'].getValue(),
            'opex': best_result['total_opex'].getValue(),
            'feed+trans' : best_result['feed+trans'].getValue(),
            'used_feedstock': best_result['used_feedstock'],
            'coordinates': (
                plant_df[plant_df['Location'] == best_plant]['Longitude'].values[0],
                plant_df[plant_df['Location'] == best_plant]['Latitude'].values[0]
            )
        })
        
        print(f"Selected {best_plant}: NPV €{best_npv:,.0f}, " +
              f"Capacity {best_result['Omega'].X*1e6:,.0f}m³")
    
    return results, dist_ik

# 6) OUTPUT GENERATION
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
            'CAPEX_EUR': res['capex'],
            'OPEX_EUR': res['opex'],
            'Feed_Trans': res['feed+trans']
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
    output_dir = os.path.join("C:/Clone/Master/results/large_scale_cont/10_greedy")
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    results, dist_ik = greedy_heuristic()
    
    if results:
        print(f"\nTotal NPV: €{sum(r['npv'] for r in results):,.0f}")
        generate_outputs(results, dist_ik, output_dir)
    
    print(f"Execution time: {time.time()-start_time:.1f}s")