#!/usr/bin/env python3
# scenario_analysis_all_alternatives_corrected.py
# ------------------------------------------------------------
# Multi-technology break-even analysis for biogas pathways with corrected K sensitivities.
# ------------------------------------------------------------

import os
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import pandas as pd
import math

# Load & Fit Transport+Feedstock Cost
BASE_DIR = "/home/fredrgaa/Master/"
if not os.path.exists(BASE_DIR):
    BASE_DIR = "C:/Clone/Master/"
output_dir = os.path.join(BASE_DIR, "results/large_scale_cont")
os.makedirs(output_dir, exist_ok=True)

fin = pd.read_csv(os.path.join(BASE_DIR, "results/large_scale_cont/10_greedy_with_alternatives/Financials.csv"))
fin["Capacity_Mm3"] = fin["Capacity"] / 1e6
fin["FeedTrans_M€"] = fin["Feed_Trans_Cost"]

a, b = np.polyfit(fin["Capacity_Mm3"], fin["FeedTrans_M€"], 1)
fin["unit_cost"] = fin["FeedTrans_M€"] / fin["Capacity_Mm3"]
coef_min, coef_max = np.percentile(fin["unit_cost"], [10, 90])

print(f"feed_cost_coef    = {a:.4f}  M€/Mm³ → €/Nm³ = {a*1e6/1e6:.4f}")
print(f"feed_cost_const   = {b*1e6:,.0f} €/yr")
print(f"feed_cost_range   = [{coef_min:.4f}, {coef_max:.4f}] M€/Mm³")

# Techno-Economic Parameters
P = {
    "FLH_max": 8000,
    "alphaHV": 9.97,
    "r": 0.042,
    "years": 25,
    "gas_price_mwh": 30,
    "co2_price_ton": 20,
    "GHG_certificate_price": 50,
    "var_upg_cost": 0.05,
    "alpha_GHG_ref": 94.0,
    "feed_cost_coef": a,
    "feed_cost_const": b * 1e6,
    "feed_cost_coef_range": [coef_min, coef_max],
    "digestate_frac": 0.9,
    "digestate_unit_cost": (27 / 37 + 0.104 * 20),
    "Q_MIN": 5,
    "Q_MAX": 60,
    "chp_elec_eff": 0.4,
    "chp_heat_eff": 0.4,
    "boiler_eff": 0.9,
    "eeg_bg_price": 194.3,
    "eeg_bm_price": 210.4,
    "cap_biogas": 0.45,
    "cap_biomethane": 0.1,
    "elec_spot_price": 60,
    "heat_price": 20,
    "bonus_rate": 100,  
}

AVG_CH4_CONTENT = 0.588
AVG_BIOGAS_YIELD = 67
AVG_GHG = -78

def npv_parts(cap, p, tech="Upgrading", feed_cost_coef=None):
    """Return (init, annual_cashflow, K_dict) with technology-specific sensitivities."""
    fc = feed_cost_coef if feed_cost_coef is not None else p["feed_cost_coef"]
    Q_bio = cap * 1e6  # m³/yr
    Q_ch4 = Q_bio * AVG_CH4_CONTENT  # m³/yr

    # Common Costs
    capex_bio = Q_bio * 150.12 * (Q_bio ** -0.311) / 1e6 if tech != "Boiler" else 0  # M€
    opex_bio = 2.1209 * (Q_bio ** 0.8359) / 1e6 if tech != "Boiler" else 0  # M€/yr
    trans = (fc * Q_bio + p["feed_cost_const"]) / 1e6  # M€/yr
    avg_discount = sum(0.99 ** t for t in range(1, p['years'] + 1)) / p['years']
    
    K_dict = {}

    if tech == "Upgrading":
        capex_upg = (Q_bio / p["FLH_max"]) * 47777 * ((Q_bio / p["FLH_max"]) ** -0.421) / 1e6 + 1
        opex_upg = p["var_upg_cost"] * Q_ch4 / 1e6
        gas_rev = Q_ch4 * (p["gas_price_mwh"] * p["alphaHV"] / 1000) / 1e6
        co2_rev = (Q_bio - Q_ch4) * (p["co2_price_ton"] / 556.2) / 1e6
        ghg_rev = (p["alpha_GHG_ref"] - AVG_GHG) * p["alphaHV"] * 3.6 * Q_ch4 * p["GHG_certificate_price"] / 1e12
        init = -(capex_bio + capex_upg)
        opex = opex_bio + opex_upg + trans
        ann = gas_rev + co2_rev + ghg_rev - opex
        K_dict = {
            "co2_price_ton": (Q_bio - Q_ch4) / 556.2 / 1e6,
            "GHG_certificate_price": (p["alpha_GHG_ref"] - AVG_GHG) * p["alphaHV"] * 3.6 * Q_ch4 / 1e12,
            "gas_price_mwh": Q_ch4 * (p["alphaHV"] / 1000) / 1e6
        }
    elif tech == "FlexEEG_biogas":
        eeg_rev = avg_discount * p["eeg_bg_price"] * p["cap_biogas"] * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        spot_rev = p["elec_spot_price"] * Q_ch4 * (1 - p["cap_biogas"]) * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        heat_rev = p["heat_price"] * Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        bonus_rev = Q_ch4 * p["bonus_rate"] * p["chp_heat_eff"] * p["alphaHV"]/ p["FLH_max"] / 1e6
        init = -capex_bio
        opex = opex_bio + trans
        ann = eeg_rev + spot_rev + heat_rev + bonus_rev - opex
        K_dict = {
            "eeg_bg_price": avg_discount * p["cap_biogas"] * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6,
            "elec_spot_price": Q_ch4 * (1 - p["cap_biogas"]) * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6,
            "heat_price": Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6,
            "bonus_rate":      Q_ch4  * p["chp_heat_eff"] * p["alphaHV"]/ p["FLH_max"] / 1e6       
        }

    elif tech == "FlexEEG_biomethane":
        capex_upg = (Q_bio / p["FLH_max"]) * 47777 * ((Q_bio / p["FLH_max"]) ** -0.421) / 1e6 + 1
        opex_upg = p["var_upg_cost"] * Q_ch4 / 1e6
        eeg_rev = avg_discount * p["eeg_bm_price"] * p["cap_biomethane"] * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        spot_rev = p["elec_spot_price"] * Q_ch4 * (1 - p["cap_biomethane"]) * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        heat_rev = p["heat_price"] * Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        bonus_rev = Q_ch4 * p["bonus_rate"] * p["chp_heat_eff"] * p["alphaHV"]/ p["FLH_max"] / 1e6        
        init = -(capex_bio + capex_upg)
        opex = opex_bio + opex_upg + trans
        ann = eeg_rev + spot_rev + heat_rev + bonus_rev - opex
        K_dict = {
            "eeg_bm_price": avg_discount * p["cap_biomethane"] * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6,
            "elec_spot_price": Q_ch4 * (1 - p["cap_biomethane"]) * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6,
            "heat_price": Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6,
                        "bonus_rate":      Q_ch4  * p["chp_heat_eff"] * p["alphaHV"]/ p["FLH_max"] / 1e6      
        }
    elif tech == "NonEEG_CHP":
        spot_rev = p["elec_spot_price"] * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        heat_rev = p["heat_price"] * Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        init = -capex_bio
        opex = opex_bio + trans
        ann = spot_rev + heat_rev - opex
        K_dict = {
            "elec_spot_price": Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6,
            "heat_price": Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        }
    elif tech == "Boiler":
        MW = Q_ch4 * p["boiler_eff"] * p["alphaHV"] / (p["FLH_max"] * 1000)
        capex_upg = 110000 * MW / 1e6
        heat_rev = p["heat_price"] * Q_ch4 * p["boiler_eff"] * p["alphaHV"] / 1000 / 1e6
        fixed_opex = 3000 * MW
        variable_opex = 0.5 * Q_ch4 * p["alphaHV"] * p["boiler_eff"] / 1000
        opex = (fixed_opex + variable_opex) / 1e6 + trans
        init = -capex_upg
        ann = heat_rev - opex
        K_dict = {
            "heat_price": Q_ch4 * p["boiler_eff"] * p["alphaHV"] / 1000 / 1e6
        }
    else:
        raise ValueError(f"Unknown technology: {tech}")

    return init, ann, K_dict

def break_even_price(cap, p, key, tech="Upgrading", feed_cost_coef=None):
    """Calculate break-even price using the correct sensitivity from K_dict."""
    init, ann, K_dict = npv_parts(cap, p, tech, feed_cost_coef)
    pv = (1 - (1 + p["r"]) ** (-p["years"])) / p["r"]
    K = K_dict.get(key, 0)
    cur = p.get(key, 0)
    npv0 = init + pv * (ann - K * cur)
    if npv0 >= 0 or K <= 0:
        return 0.0
    return -npv0 / (pv * K)

# Define Technologies
alternative_configs = [
    {"name": "Upgrading", "metrics": ["co2_price_ton", "GHG_certificate_price", "gas_price_mwh"],
     "labels": ["CO₂ price [€/t]", "GHG quota [€/t]", "Gas [€/MWh]"]},
    {"name": "FlexEEG_biogas", "metrics": ["eeg_bg_price", "elec_spot_price", "heat_price", "bonus_rate"],
     "labels": ["EEG tariff [€/MWh]", "Spot elec [€/MWh]", "Heat [€/MWh]", "Flexibility Bonus [€/kW]"]},
    {"name": "FlexEEG_biomethane", "metrics": ["eeg_bm_price", "elec_spot_price", "heat_price", "bonus_rate"],
     "labels": ["EEG tariff [€/MWh]", "Spot elec [€/MWh]", "Heat [€/MWh]", "Flexibility Bonus [€/kW]"]},
    {"name": "NonEEG_CHP", "metrics": ["elec_spot_price", "heat_price"],
     "labels": ["Elec price [€/MWh]", "Heat price [€/MWh]"]},
    {"name": "Boiler", "metrics": ["heat_price"], "labels": ["Heat price [€/MWh]"]},
]

# Debug Prints
debug_caps = [P["Q_MIN"], 0.5 * (P["Q_MIN"] + P["Q_MAX"]), P["Q_MAX"]]

print("\n----- DEBUG BREAK-EVEN PRICES -----")
for cap in debug_caps:
    print(f"\nCapacity = {cap:.1f} Mm³/yr")
    for alt in alternative_configs:
        print(f"  Technology: {alt['name']}")
        for key, label in zip(alt["metrics"], alt["labels"]):
            base = break_even_price(cap, P, key, alt["name"])
            low = break_even_price(cap, P, key, alt["name"], feed_cost_coef=P["feed_cost_coef_range"][0])
            high = break_even_price(cap, P, key, alt["name"], feed_cost_coef=P["feed_cost_coef_range"][1])
            print(f"    {label:20s} → base = {base:7.2f},  low = {low:7.2f},  high = {high:7.2f}")
    print("-" * 50)
print("----- END DEBUG -----\n")

# Detailed Breakdown
print("\n----- DETAILED COST & REVENUE BREAKDOWN -----")
for cap in debug_caps:
    print(f"\nCapacity = {cap:.1f} Mm³/yr")
    Q_bio = cap * 1e6
    Q_ch4 = Q_bio * AVG_CH4_CONTENT
    for alt in alternative_configs:
        tech = alt["name"]
        print(f"  Technology: {tech}")
        init, ann, K_dict = npv_parts(cap, P, tech)
        
        capex_bio = Q_bio * 150.12 * (Q_bio ** -0.311) / 1e6 if tech != "Boiler" else 0
        opex_bio = 2.1209 * (Q_bio ** 0.8359) / 1e6 if tech != "Boiler" else 0
        trans = (P["feed_cost_coef"] * Q_bio + P["feed_cost_const"]) / 1e6
        avg_discount = sum(0.99 ** t for t in range(1, P['years'] + 1)) / P['years']
        
        if tech == "Upgrading":
            capex_upg = (Q_bio / P["FLH_max"]) * 47777 * ((Q_bio / P["FLH_max"]) ** -0.421) / 1e6 + 1
            opex_upg = P["var_upg_cost"] * Q_ch4 / 1e6
            gas_rev = Q_ch4 * (P["gas_price_mwh"] * P["alphaHV"] / 1000) / 1e6
            co2_rev = (Q_bio - Q_ch4) * (P["co2_price_ton"] / 556.2) / 1e6
            ghg_rev = (P["alpha_GHG_ref"] - AVG_GHG) * P["alphaHV"] * 3.6 * Q_ch4 * P["GHG_certificate_price"] / 1e12
            eeg_rev = spot_rev = heat_rev = bonus_rev = 0
        elif tech == "FlexEEG_biogas":
            capex_upg = opex_upg = 0
            eeg_rev = avg_discount * P["eeg_bg_price"] * P["cap_biogas"] * Q_ch4 * P["chp_elec_eff"] * P["alphaHV"] / 1000 / 1e6
            spot_rev = P["elec_spot_price"] * Q_ch4 * (1 - P["cap_biogas"]) * P["chp_elec_eff"] * P["alphaHV"] / 1000 / 1e6
            heat_rev = P["heat_price"] * Q_ch4 * P["chp_heat_eff"] * P["alphaHV"] / 1000 / 1e6
            bonus_rev = Q_ch4 *P["bonus_rate"] * P["chp_heat_eff"] * P["alphaHV"]/ P["FLH_max"] / 1e6  
            gas_rev = co2_rev = ghg_rev = 0
        elif tech == "FlexEEG_biomethane":
            capex_upg = (Q_bio / P["FLH_max"]) * 47777 * ((Q_bio / P["FLH_max"]) ** -0.421) / 1e6 + 1
            opex_upg = P["var_upg_cost"] * Q_ch4 / 1e6
            eeg_rev = avg_discount * P["eeg_bm_price"] * P["cap_biomethane"] * Q_ch4 * P["chp_elec_eff"] * P["alphaHV"] / 1000 / 1e6
            spot_rev = P["elec_spot_price"] * Q_ch4 * (1 - P["cap_biomethane"]) * P["chp_elec_eff"] * P["alphaHV"] / 1000 / 1e6
            heat_rev = P["heat_price"] * Q_ch4 * P["chp_heat_eff"] * P["alphaHV"] / 1000 / 1e6
            bonus_rev = Q_ch4 *P["bonus_rate"] * P["chp_heat_eff"] * P["alphaHV"]/ P["FLH_max"] / 1e6
            gas_rev = co2_rev = ghg_rev = 0
        elif tech == "NonEEG_CHP":
            capex_upg = opex_upg = 0
            spot_rev = P["elec_spot_price"] * Q_ch4 * P["chp_elec_eff"] * P["alphaHV"] / 1000 / 1e6
            heat_rev = P["heat_price"] * Q_ch4 * P["chp_heat_eff"] * P["alphaHV"] / 1000 / 1e6
            eeg_rev = gas_rev = co2_rev = ghg_rev = bonus_rev = 0
        elif tech == "Boiler":
            MW = Q_ch4 * P["boiler_eff"] * P["alphaHV"] / (P["FLH_max"] * 1000)
            capex_upg = 110000 * MW / 1e6
            fixed_opex = 3000 * MW
            variable_opex = 0.5 * Q_ch4 * P["alphaHV"] * P["boiler_eff"] / 1000
            opex_upg = (fixed_opex + variable_opex) / 1e6
            heat_rev = P["heat_price"] * Q_ch4 * P["boiler_eff"] * P["alphaHV"] / 1000 / 1e6
            eeg_rev = spot_rev = gas_rev = co2_rev = ghg_rev = bonus_rev = 0

        print(f"    CAPEX (M€):")
        if capex_bio > 0:
            print(f"      Biogas:      {capex_bio:10.2f}")
        if capex_upg > 0:
            print(f"      Upgrading:   {capex_upg:10.2f}")
        print(f"      Total:       {-init:10.2f}")
        print(f"    OPEX (M€/yr):")
        if opex_bio > 0:
            print(f"      Biogas:      {opex_bio:10.2f}")
        if opex_upg > 0:
            print(f"      Upgrading:   {opex_upg:10.2f}")
        print(f"      Transport:   {trans:10.2f}")
        print(f"      Total:       {opex_bio + opex_upg + trans:10.2f}")
        print(f"    Revenue (M€/yr):")
        if gas_rev > 0:
            print(f"      Gas:         {gas_rev:10.2f}")
        if co2_rev > 0:
            print(f"      CO2:         {co2_rev:10.2f}")
        if ghg_rev > 0:
            print(f"      GHG:         {ghg_rev:10.2f}")
        if eeg_rev > 0:
            print(f"      EEG:         {eeg_rev:10.2f}")
        if spot_rev > 0:
            print(f"      Spot Elec:   {spot_rev:10.2f}")
        if heat_rev > 0:
            print(f"      Heat:        {heat_rev:10.2f}")
        if bonus_rev > 0:
            print(f"      Bonus:       {bonus_rev:10.2f}")
        print(f"      Total:       {gas_rev + co2_rev + ghg_rev + eeg_rev + spot_rev + heat_rev + bonus_rev:10.2f}")
        print(f"    Net Annual (M€/yr): {ann:10.2f}")
    print("-" * 50)
print("----- END DETAILED BREAKDOWN -----\n")

# Run & Plot
if __name__ == "__main__":
    caps = np.linspace(P["Q_MIN"], P["Q_MAX"], 200)

    # ── build a grid that is only as large as required ───────────────────
    n_alt   = len(alternative_configs)            # = 5
    n_cols  = 2                                   # 2 columns look nice
    n_rows  = math.ceil(n_alt / n_cols)           # → 3 rows
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(12, 10), constrained_layout=True
    )
    axes = axes.flatten()

    # ── plot each alternative ────────────────────────────────────────────
    for ax, alt in zip(axes, alternative_configs):
        for key, label in zip(alt["metrics"], alt["labels"]):
            base_curve = [break_even_price(c, P, key, alt["name"]) for c in caps]
            low_curve  = [break_even_price(c, P, key, alt["name"],
                                           feed_cost_coef=P["feed_cost_coef_range"][0]) for c in caps]
            high_curve = [break_even_price(c, P, key, alt["name"],
                                           feed_cost_coef=P["feed_cost_coef_range"][1]) for c in caps]

            ax.plot(caps, base_curve, label=label, lw=2)
            ax.fill_between(caps, low_curve, high_curve, alpha=0.2)

        ax.set_title(alt["name"])
        ax.set_xlabel("Capacity [Mm³ / yr]")
        ax.set_ylabel("Break-even price")
        ax.legend()
        ax.grid(alpha=0.3)

    # ── remove any unused axes (there will be exactly one) ───────────────
    for ax in axes[n_alt:]:
        fig.delaxes(ax)

    plt.savefig(os.path.join(output_dir, "break_even_all_alts.png"))
    plt.show()
