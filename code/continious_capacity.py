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

output_dir = os.path.join(BASE_DIR, "results/large_scale_cont/35")
os.makedirs(output_dir, exist_ok=True)

feedstock_df = pd.read_csv(f"{BASE_DIR}aggregated_bavaria_supply_nodes.csv")
plant_df = pd.read_csv(f"{BASE_DIR}equally_spaced_locations_35.csv")
distance_df = pd.read_csv(f"{BASE_DIR}Distance_Matrix_35.csv")
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
]

premium = {f: max(0, (alpha_GHG_comp - feed_yield[f]['GHG_intensity'])) * (alphaHV * 3.6) * GHG_certificate_price / 1e6 for f in feedstock_types}
threshold_m3 = (100 * FLH_max) / (chp_elec_eff * system_methane_average * alphaHV) / 1e6
FLH_min_limit = 1000
Q_MAX = 80_000_000 / 1e6  # Maximum capacity in Mm³/yr
Q_MIN = 20_000_000 / 1e6  # Minimum capacity in Mm³/yr
M_large = Q_MAX * 1.01
avg_discount = sum(0.99**t for t in range(1, years+1)) / years
M_j = {j: sum(avail_mass[i, f] for i, f in avail_mass) / 1e6 for j in plant_locs}

# Define PWL breakpoints and values
BREAKS = np.linspace(0.01, Q_MAX, 11)
alt_base = alternative_configs[0]
alt_upg = alternative_configs[1]
base_capex_vals = [((b * 1e6) * alt_base["capex_coeff"] * (b * 1e6) ** alt_base["capex_exp"]) / 1e6 for b in BREAKS]
base_opex_vals = [alt_base["opex_coeff"] * (b * 1e6) ** alt_base["opex_exp"] / 1e6 for b in BREAKS]
upg_capex_vals = [((b * 1e6/ FLH_max) * alt_upg["upg_cost_coeff"] * ((b * 1e6 / FLH_max) ** alt_upg["upg_cost_exp"])) for b in BREAKS]

# 4) CONSTRAINT FUNCTIONS
def add_supply_constraints(m, avail_mass, x, plant_locs):
    m.addConstrs(
        (gp.quicksum(x[i, f, j] for j in plant_locs) <= amt / 1e6 for (i, f), amt in avail_mass.items()),
        name="Supply"
    )

def add_cn_constraints(m, x, avail_mass, plant_locs, feed_yield, cn_min=20.0, cn_max=30.0):
    m.addConstrs(
        (gp.quicksum(x[i, f, j] * feed_yield[f]['CN'] for (i, f) in avail_mass) >= 
         cn_min * gp.quicksum(x[i, f, j] for (i, f) in avail_mass) for j in plant_locs),
        name="CN_min"
    )
    m.addConstrs(
        (gp.quicksum(x[i, f, j] * feed_yield[f]['CN'] for (i, f) in avail_mass) <= 
         cn_max * gp.quicksum(x[i, f, j] for (i, f) in avail_mass) for j in plant_locs),
        name="CN_max"
    )

def add_ghg_constraints(m, x, avail_mass, plant_locs, feed_yield, alpha_ghg_lim):
    for j in plant_locs:
        total_feed_j = gp.quicksum(x[i, f, j] for i, f in avail_mass)
        total_GHG_j = gp.quicksum(x[i, f, j] * feed_yield[f]['GHG_intensity'] for i, f in avail_mass)
        m.addConstr(total_GHG_j <= alpha_ghg_lim * total_feed_j, name=f"GHG_average_{j}")

def add_auction_constraints(m, Cap, y, plant_locs, alternative_configs):
    total_EEG_capacity = gp.quicksum(Cap[j] * y[j, a] for j in plant_locs for a, alt in enumerate(alternative_configs) if alt["EEG_flag"])
    m.addConstr(total_EEG_capacity <= auction_chp_limit, name="EEG_Auction_Limit")

'''
def add_flh_constraints(m, Omega, Cap, plant_locs, N_CH4):
    m.addConstrs(
        (Omega[j] <= (FLH_max / 8760.0) * Cap[j] for j in plant_locs),
        name="FLH_limit"
    )
    m.addConstrs(
        (N_CH4[j] <= (FLH_max / 8760.0) * Omega[j] for j in plant_locs),
        name="FLH_limit_NCH4"
    )
'''
config = {
    "name": "Baseline",
    "eeg_enabled": False,
    "supply_enabled": True,
    "digestate_enabled": False,
    "digestate_return_frac": 0.99,
    "cn_enabled": True,
    "maize_enabled": False,
    "ghg_enabled": False,
    "auction_enabled": True,
    "flh_enabled": True
}


def build_model(config):
    m = gp.Model("ShadowPlant_Biogas_Model")
    m.setParam("NodefileStart", 50)  # Start offloading node data to disk after 40 GB

    # Global parameters for continuous capacity
    Q_MAX = 80  # [Mm³ biogas per year]
    Q_MIN = 20

    # AFTER  (0 first, then 10 equal steps up to Q_MAX)
    BREAKS = np.linspace(0, Q_MAX, 5)

    # Decision variables
    Omega = m.addVars(plant_locs, lb=0, ub=Q_MAX, vtype=GRB.CONTINUOUS, name="Omega")
    N_CH4 = m.addVars(plant_locs, lb=0, ub=Q_MAX, vtype=GRB.CONTINUOUS, name="N_CH4")
    m_up = m.addVars(plant_locs, feedstock_types, lb=0, ub=Q_MAX, vtype=GRB.CONTINUOUS, name="m_up")
    x = m.addVars(
        supply_nodes, feedstock_types, plant_locs,
        lb=0,
        ub={(i, f, j): avail_mass.get((i, f), 0) / 1e6 for i in supply_nodes for f in feedstock_types for j in plant_locs},
        vtype=GRB.CONTINUOUS,
        name="x"
    )
    y = m.addVars(plant_locs, range(len(alternative_configs)), vtype=GRB.BINARY, name="y")

    # Precompute zero flows
    zero_triplets = [(i, f, j) for i in supply_nodes for f in feedstock_types if (i, f) not in avail_mass for j in plant_locs]
    m.addConstrs((x[i, f, j] == 0 for (i, f, j) in zero_triplets), name="ZeroFlow")

    # Logical constraints
    m.addConstrs(
        (gp.quicksum(y[j, a] for a in range(len(alternative_configs))) <= 1 for j in plant_locs),
        name="OneAlt"
    )
    for j in plant_locs:
        is_active = m.addVar(vtype=GRB.BINARY, name=f"is_active_{j}")
        m.addConstr(is_active == gp.quicksum(y[j, a] for a in range(len(alternative_configs))), name=f"ActiveLink_{j}")
        m.addConstr(Omega[j] <= Q_MAX * is_active, name=f"OmegaUpper_{j}")
        m.addConstr(Omega[j] >= Q_MIN * is_active, name=f"OmegaLower_{j}")

    # Upgrading selection
    UpgSel = {j: m.addVar(vtype=GRB.BINARY, name=f"UpgSel_{j}") for j in plant_locs}
    for j in plant_locs:
        m.addConstr(
            UpgSel[j] == gp.quicksum(y[j, a] for a, alt in enumerate(alternative_configs) if alt["category"] == "Upgrading"),
            name=f"LinkUpg_{j}"
        )

    
    for j in plant_locs:
        for f in feedstock_types:
            prod_f = gp.quicksum(
                x[i,f,j] *
                feed_yield[f]['biogas_m3_per_ton'] *
                feed_yield[f]['ch4_content']
                for i in supply_nodes)

            # upgrading ON  →  m_up equals the methane you really get
            m.addGenConstrIndicator(
                UpgSel[j], True,
                m_up[j,f] == prod_f,
                name=f"UpgProd_on_{j}_{f}")

            # upgrading OFF →  m_up = 0
            m.addGenConstrIndicator(
                UpgSel[j], False,
                m_up[j,f] == 0,
                name=f"UpgProd_off_{j}_{f}")


    def safe_pow(base: float, exp: float) -> float:
        return 0.0 if base == 0 else base ** exp

    # PWL for CAPEX
    Capex_site = m.addVars(plant_locs, name="Capex_site")
    base_hat   = m.addVars(plant_locs, name="baseCap_hat") 
    α, β = alt_base["capex_coeff"], alt_base["capex_exp"]
    base_capex_vals = [(b*1e6) * α * safe_pow(b*1e6, β) / 1e6 for b in BREAKS]

    upg_hat   = m.addVars(plant_locs, name="upgCap_hat")       # PWL helper
    upg_eff   = m.addVars(plant_locs, lb=0, name="upgCap_eff")
    cu, eu = alt_upg["upg_cost_coeff"], alt_upg["upg_cost_exp"]
    upg_capex_vals = [(b*1e6/FLH_max) * cu * safe_pow(b*1e6/FLH_max, eu) / 1e6 for b in BREAKS]
    for j in plant_locs:
        m.addGenConstrPWL(Omega[j], base_hat[j], list(BREAKS), base_capex_vals,name=f"PWL_baseCap_{j}")
            # raw PWL curve
        m.addGenConstrPWL(Omega[j], upg_hat[j], list(BREAKS), upg_capex_vals,
                        name=f"PWL_upgCap_{j}")

        # indicator: upg_hat counts only when UpgSel[j]==1
        m.addGenConstrIndicator(UpgSel[j], True,
                                upg_eff[j] == upg_hat[j],
                                name=f"UpgCap_on_{j}")
        m.addGenConstrIndicator(UpgSel[j], False,
                                upg_eff[j] == 0,
                                name=f"UpgCap_off_{j}")

        # total CAPEX for the site = base + (optional) upgrading
        m.addConstr(Capex_site[j] == base_hat[j] + upg_eff[j],
                    name=f"Capex_total_{j}")

    # PWL for OPEX
    Opex_site = m.addVars(plant_locs, name="Opex_site")
    κ, γ = alt_base["opex_coeff"],  alt_base["opex_exp"]
    base_opex_vals = [κ * safe_pow(b*1e6, γ) / 1e6 for b in BREAKS]
    for j in plant_locs:
        op_hat = m.addVar(name=f"opHat_{j}")
        m.addGenConstrPWL(Omega[j], op_hat, list(BREAKS), base_opex_vals, name=f"PWL_OPEX_{j}")
        m.addConstr(Opex_site[j] == op_hat, name=f"Opex_link_{j}")

    # Revenue and cost variables
    Rev_loc = m.addVars(plant_locs, lb=0, vtype=GRB.CONTINUOUS, name="Rev_loc")
    Cost_loc = m.addVars(plant_locs, lb=0, vtype=GRB.CONTINUOUS, name="Cost_loc")
    Rev_alt_selected = m.addVars(plant_locs, range(len(alternative_configs)), lb=0, vtype=GRB.CONTINUOUS, name="Rev_alt_selected")
    Cost_alt_selected = m.addVars(plant_locs, range(len(alternative_configs)), lb=0, vtype=GRB.CONTINUOUS, name="Cost_alt_selected")

    # Define Omega and N_CH4 from feedstock
    for j in plant_locs:
        m.addConstr(Omega[j] == gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] for i, f in avail_mass), name=f"Omega_Feed_{j}")
        m.addConstr(N_CH4[j] == gp.quicksum(x[i, f, j] * feed_yield[f]['biogas_m3_per_ton'] * feed_yield[f]['ch4_content'] for i, f in avail_mass), name=f"N_CH4_Feed_{j}")

    # Feedstock totals
    total_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass) for j in plant_locs}
    manure_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if is_manure(f)) for j in plant_locs}
    clover_feed = {j: gp.quicksum(x[i, f, j] for i, f in avail_mass if is_clover(f)) for j in plant_locs}

    BIG_M = max(premium.values()) * Q_MAX * 0.7      # safe upper bound €/plant

    GHG_rev = {j: m.addVar(lb=0, name=f"GHGrev_{j}") for j in plant_locs}

    for j in plant_locs:
        # actual premium earned (already correct units)
        m.addConstr(
            GHG_rev[j] ==
            gp.quicksum(premium[f] * m_up[j,f] for f in feedstock_types),
            name=f"DefGHGrev_{j}")

        # disable it when UpgSel[j] = 0             (linear ‚big-M‘ link)
        m.addConstr(
            GHG_rev[j] <=  BIG_M * UpgSel[j],
            name=f"GHGrev_logic_{j}")

    
    BONUS_RATE = (
        100 * system_methane_average * chp_elec_eff * alphaHV / FLH_max
    )

    # Revenue and cost per alternative
    for j in plant_locs:
        excess_j = m.addVar(lb=0, name=f"excess_{j}")
        diff_j   = m.addVar(name=f"diff_{j}")
        m.addConstr(diff_j == Omega[j] - threshold_m3, name=f"diffConstr_{j}")
        m.addGenConstrMax(excess_j, [diff_j, 0], name=f"excessConstr_{j}")
        bonus_expr = BONUS_RATE * excess_j

        for a, alt in enumerate(alternative_configs):
            rev_val = gp.LinExpr(0)
            cost_val = gp.LinExpr(0)
            if not alt["EEG_flag"]:
                if alt["category"] == "Upgrading":
                    rev_val = N_CH4[j] * alt["rev_price"]["gas"] + (Omega[j] - N_CH4[j]) * alt["rev_price"]["co2"]
                    cost_val = variable_upg_cost * N_CH4[j]
            else:
                effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                if alt["category"] == "FlexEEG_biogas":
                    cap_fraction = Cap_biogas
                    U_elec = N_CH4[j] * chp_elec_eff * alphaHV / 1000.0
                    cap_production_elec = cap_fraction * U_elec
                    E_actual = N_CH4[j] * (chp_elec_eff * alphaHV / 1000.0)
                    EEG_rev = cap_production_elec * effective_EEG
                    spot_rev = (E_actual - cap_production_elec) * electricity_spot_price
                    heat_rev = heat_price * (N_CH4[j] * chp_heat_eff * alphaHV / 1000.0)
                    rev_val = EEG_rev + spot_rev + heat_rev + bonus_expr

            m.addGenConstrIndicator(y[j, a], True, Rev_alt_selected[j, a] == rev_val, name=f"Rev_on_{j}_{a}")
            m.addGenConstrIndicator(y[j, a], False, Rev_alt_selected[j, a] == 0, name=f"Rev_off_{j}_{a}")
            m.addGenConstrIndicator(y[j, a], True, Cost_alt_selected[j, a] == cost_val, name=f"Cost_on_{j}_{a}")
            m.addGenConstrIndicator(y[j, a], False, Cost_alt_selected[j, a] == 0, name=f"Cost_off_{j}_{a}")

    # Link local revenue and cost
    for j in plant_locs:
        m.addConstr(Rev_loc[j] == gp.quicksum(Rev_alt_selected[j, a] for a in range(len(alternative_configs))), name=f"Rev_link_{j}")
        m.addConstr(Cost_loc[j] == Opex_site[j] + gp.quicksum(Cost_alt_selected[j, a] for a in range(len(alternative_configs))), name=f"Cost_link_{j}")

    # Constraints
    add_supply_constraints(m, avail_mass, x, plant_locs)
    if config["cn_enabled"]:
        add_cn_constraints(m, x, avail_mass, plant_locs, feed_yield, CN_min, CN_max)
    '''
    if config["flh_enabled"]:
        m.addConstrs((N_CH4[j] <= (FLH_max / 8760.0) * Omega[j] for j in plant_locs), name="FLH_limit_NCH4")
    '''
    # Feedstock cost calculations (unchanged)
    FeedstockCost = gp.LinExpr()
    FeedstockCostPerPlant = {j: gp.LinExpr() for j in plant_locs}
    BaseFeedstockCost = {j: gp.LinExpr() for j in plant_locs}
    LoadingCost = {j: gp.LinExpr() for j in plant_locs}
    TransportCost = {j: gp.LinExpr() for j in plant_locs}
    DigestateCost = {j: gp.LinExpr() for j in plant_locs}

    flows = [(i, f, j) for (i, f) in avail_mass for j in plant_locs]
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

    # Objective
    TotalRev = gp.quicksum(Rev_loc[j] for j in plant_locs)
    TotalCost = FeedstockCost + gp.quicksum(Cost_loc[j] for j in plant_locs)
    TotalCapex = gp.quicksum(Capex_site[j] for j in plant_locs)
    GHGRevenue = gp.quicksum(GHG_rev[j] for j in plant_locs)
    NPV_expr = -TotalCapex
    for t in range(1, years + 1):
        discount_factor = 1 / (1 + r) ** t
        NPV_expr += discount_factor * (TotalRev + GHGRevenue - TotalCost)
    penalty = 1e-3 * gp.quicksum(y[j, a] for j in plant_locs for a in range(len(alternative_configs)))
    NPV_expr -= penalty

    m.setObjective(NPV_expr, GRB.MAXIMIZE)

    return m, Omega, N_CH4, x, y, m_up,UpgSel ,Rev_loc, Cost_loc, Capex_site, TotalRev, TotalCost, FeedstockCost, GHG_rev, TotalCapex, Rev_alt_selected, Cost_alt_selected, FeedstockCostPerPlant, BaseFeedstockCost, LoadingCost, TransportCost, DigestateCost, bonus_expr

if __name__ == '__main__':
    print("Running full model...")
    m, Omega, N_CH4, x, y, m_up, UpgSel,Rev_loc, Cost_loc, Capex_site, TotalRev, TotalCost, FeedstockCost, GHG_rev, TotalCapex, Rev_alt_selected, Cost_alt_selected, FeedstockCostPerPlant, BaseFeedstockCost, LoadingCost, TransportCost, DigestateCost, bonus_expr = build_model(config)
    m.update()
    # –– Warm‐start if a solution exists
    '''
    warmstart_path = os.path.join(output_dir, "warmstart.sol")
    if os.path.isfile(warmstart_path):
        print(f"Loading warm‐start from {warmstart_path}")
        m.read(warmstart_path)
    '''
    print(f"  Quadratic constraints: {m.NumQConstrs}")
    print(f"  Quadratic objective terms (non-zeros): {m.NumQNZs}")

    opt_start_time = time.time()
    m.optimize()
    opt_end_time = time.time()

    if m.status == GRB.OPTIMAL:
        print(f"\nFull model: NPV = {m.objVal:,.2f} €, Solve time = {m.Runtime:.2f} s, MIP Gap = {m.MIPGap:.4f}")
        for j in plant_locs:
            if Omega[j].X > 1e-6:
                print(f"Plant {j}: Capacity = {Omega[j].X * 1e6:,.0f} m³/yr, Omega = {Omega[j].X * 1e6:,.0f} m³/yr, N_CH4 = {N_CH4[j].X * 1e6:,.0f} m³/yr, CH4 Fraction = {N_CH4[j].X / Omega[j].X:.3f}")
                selected_alt = next((a for a in range(len(alternative_configs)) if y[j, a].X > 0.5), None)
                if selected_alt is not None:
                    alt_name = alternative_configs[selected_alt]["name"]
                    print(f"  Selected Alternative: {alt_name}")

    if m.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        m.computeIIS()
        m.write("model.iis")
        print("IIS written to model.iis")
    else:
        print(f"Model status: {m.status}")
    '''
    for j in plant_locs:
        print(j, UpgSel[j].X,
                [y[j,a].X for a,_ in enumerate(alternative_configs)])
    
    for j in plant_locs:
        if UpgSel[j].X > 0.5:
            tot_up = sum(m_up[j,f].X for f in feedstock_types)
            print(f"{j:8s}  m_up = {tot_up:10.3f}  Mm³")
    for f in feedstock_types:
        print(f"{f:20s}  premium = {premium[f]:8.4f}  €/Mm³")

    for j in plant_locs:
        ghg_expr = sum(premium[f] * m_up[j,f].X for f in feedstock_types)
        print(f"{j:8s}  model GHG_rev = {GHG_rev[j].X:8.2f} €   manual = {ghg_expr:8.2f} €")

    
    # Feedstock cost breakdown
    for j in plant_locs:
        if Omega[j].X > 1e-6:
            print(f"--- Plant {j} Feedstock Cost Breakdown ---")
            print(f"  Base feedstock: €{BaseFeedstockCost[j].getValue():,.2f}")
            print(f"  Loading:        €{LoadingCost[j].getValue():,.2f}")
            print(f"  Transport:      €{TransportCost[j].getValue():,.2f}")
            print(f"  Digestate:      €{DigestateCost[j].getValue():,.2f}")
            print(f"  TOTAL:          €{FeedstockCostPerPlant[j].getValue():,.2f}")

    # Revenue debug for a specific alternative (e.g., EEG_CHP_small1)
    for j in plant_locs:
        if Omega[j].X > 1e-6:
            selected_alt = next((a for a in range(len(alternative_configs)) if y[j, a].X > 0.5), None)
            if selected_alt is not None and alternative_configs[selected_alt]["name"] == "EEG_CHP_small1":
                alt = alternative_configs[selected_alt]
                N_val = N_CH4[j].X * 1e6
                omega_val = Omega[j].X * 1e6
                effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                ch4_energy = N_val * (alphaHV / 1000)
                electricity_revenue = ch4_energy * chp_elec_eff * effective_EEG
                heat_revenue = ch4_energy * chp_heat_eff * heat_price
                total_revenue = electricity_revenue + heat_revenue
                print(f"\nPlant {j} selected EEG_CHP_small1:")
                print(f"  Y[{j}, {selected_alt}] = {y[j, selected_alt].X:.6f}")
                print(f"  N_CH4 = {N_val:.6f} m³")
                print(f"  Omega = {omega_val:.6f} m³")
                print(f"  CH4 Energy Input = {ch4_energy:.6f} MWh")
                print(f"  Electricity Revenue = €{electricity_revenue:.6f}")
                print(f"  Heat Revenue = €{heat_revenue:.6f}")
                print(f"  Total Calculated Revenue = €{total_revenue:.6f}")
                print(f"  Model Revenue = €{Rev_loc[j].X:.6f}")

    # Feedstock usage
    for f in feedstock_types:
        used = sum(x[i, f, j].X for i in supply_nodes for j in plant_locs)
        avail = sum(avail_mass.get((i, f), 0) for i in supply_nodes) / 1e6
        print(f"{f:20s}  {used:8.1f} / {avail:8.1f}  ({100 * used / avail:5.1f}%)")
    '''
    # Inflow data
    inflow_rows = []
    for j in plant_locs:
        for i in supply_nodes:
            for f in feedstock_types:
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
    in_flow_df.to_csv(os.path.join(output_dir, "Output_in_flow.csv"), index=False)

    # Check for No_build inconsistencies
    for j in plant_locs:
        no_build_selected = False
        selected_alt = next((a for a in range(len(alternative_configs)) if y[j, a].X > 0.5), None)
        if selected_alt is not None and alternative_configs[selected_alt]["name"] == "No_build":
            no_build_selected = True
            print(f"Plant {j} selected No_build with Y[{j}, {selected_alt}] = {y[j, selected_alt].X:.6f}")
        if no_build_selected and Omega[j].X > 1e-6:
            print(f"Warning: Plant {j} has Omega = {Omega[j].X * 1e6:,.0f} but selected No_build")

    # Financial metrics
    plant_npvs = {}
    plant_annual_cf = {}
    plant_irr = {}
    for j in plant_locs:
        if Omega[j].X > 1e-6:
            rev_j = Rev_loc[j].X
            cost_j = Cost_loc[j].X + FeedstockCostPerPlant[j].getValue()
            ghg_j = GHG_rev[j].X
            capex_j = Capex_site[j].X

            # Annual net cash flow
            annual_net = rev_j - cost_j + ghg_j
            plant_annual_cf[j] = annual_net

            # NPV
            discounted_operating = sum((1 / (1 + r) ** t) * annual_net for t in range(1, years + 1))
            plant_npvs[j] = -capex_j + discounted_operating

            # IRR
            cf_series = [-capex_j] + [annual_net] * years
            plant_irr[j] = nf.irr(cf_series) if len(cf_series) > 1 else 0

    # Financial output
    merged_rows = []
    for j in plant_locs:
        if Omega[j].X > 1e-6:
            selected_alt = next((a for a in range(len(alternative_configs)) if y[j, a].X > 0.5), None)
            if selected_alt is not None:
                alt = alternative_configs[selected_alt]
                alt_name = alt["name"]
                feed_cost_j = FeedstockCostPerPlant[j].getValue()
                row_data = {
                    "PlantLocation": j,
                    "Alternative": alt_name,
                    "Capacity": Omega[j].X * 1e6,
                    "Plant_NPV": plant_npvs.get(j, 0),
                    "Plant_IRR": plant_irr.get(j, 0),
                    "Omega": Omega[j].X * 1e6,
                    "N_CH4": N_CH4[j].X * 1e6,
                    "CO2_Production": (Omega[j].X - N_CH4[j].X) * 1e6,
                    "Revenue": Rev_loc[j].X,
                    "Cost": Cost_loc[j].X,
                    "Feed_Trans_Cost": feed_cost_j,
                    "Capex": Capex_site[j].X,
                    "GHG": GHG_rev[j].X,
                    "FLH": (Omega[j].X / Omega[j].X) * 8760 if Omega[j].X > 0 else 0,
                    "PlantLatitude": plant_coords.get(j, (None, None))[1],
                    "PlantLongitude": plant_coords.get(j, (None, None))[0]
                }
                if alt["category"] in ["FlexEEG_biogas", "FlexEEG_biomethane"]:
                    effective_EEG = alt["rev_price"]["EEG"] * avg_discount
                    E_actual = N_CH4[j].X * (chp_elec_eff * alphaHV / 1000.0)
                    cap_fraction = Cap_biogas if alt["category"] == "FlexEEG_biogas" else Cap_biomethane
                    EEG_rev = cap_fraction * E_actual * effective_EEG if cap_fraction else 0
                    spot_rev = (E_actual - (cap_fraction * E_actual if cap_fraction else 0)) * electricity_spot_price
                    heat_rev = heat_price * (N_CH4[j].X * chp_heat_eff * alphaHV / 1000.0)
                    row_data.update({
                        "EEG_Revenue": EEG_rev,
                        "Spot_Revenue": spot_rev,
                        "Heat_Revenue": heat_rev,
                        "Bonus": bonus_expr.getValue()
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
    fin_df.to_csv(os.path.join(output_dir, "Output_financials.csv"), index=False)

    # Warm-start solution
    warmstart_path = os.path.join(output_dir, "warmstart.sol")
    m.write(warmstart_path)
    print(f"Warm-start solution written to: {warmstart_path}")

    for v in m.getVars():
        v.start = v.X

    # Execution time logging
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    opt_time = opt_end_time - opt_start_time
    with open(f'{BASE_DIR}/Solutions/aggregated/execution_times.txt', 'a') as f:
        f.write(f"Total script execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)\n")
        f.write(f"Optimization time: {opt_time:.2f} seconds ({opt_time/60:.2f} minutes)\n")

