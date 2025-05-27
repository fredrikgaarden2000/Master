#!/usr/bin/env python3
# scenario_analysis_all_alternatives.py
# ------------------------------------------------------------
# Multi‐technology break‐even analysis for biogas pathways.
# ------------------------------------------------------------

import os
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 1) LOAD & FIT TRANSPORT+FEEDSTOCK COST
# ─────────────────────────────────────────────────────────────
BASE_DIR = "/home/fredrgaa/Master/"
if not os.path.exists(BASE_DIR):
    BASE_DIR = "C:/Clone/Master/"
output_dir = os.path.join(BASE_DIR, "results/large_scale_cont")
os.makedirs(output_dir, exist_ok=True)

fin = pd.read_csv(
    os.path.join(BASE_DIR,
                 "results/large_scale_cont/10_greedy_with_alternatives/Financials_20_greedy.csv")
)
fin["Capacity_Mm3"] = fin["Capacity"] / 1e6
fin["FeedTrans_M€"] = fin["Feed_Trans_Cost"]

a, b = np.polyfit(fin["Capacity_Mm3"], fin["FeedTrans_M€"], 1)
fin["unit_cost"] = fin["FeedTrans_M€"] / fin["Capacity_Mm3"]
coef_min, coef_max = np.percentile(fin["unit_cost"], [10, 90])

print(f"feed_cost_coef    = {a:.4f}  M€/Mm³ → €/Nm³ = {a*1e6/1e6:.4f}")
print(f"feed_cost_const   = {b*1e6:,.0f} €/yr")
print(f"feed_cost_range   = [{coef_min:.4f}, {coef_max:.4f}] M€/Mm³")

# ─────────────────────────────────────────────────────────────
# 2) PARAMS & NPVs
# ─────────────────────────────────────────────────────────────
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
    "feed_cost_const": b*1e6,
    "feed_cost_coef_range": [coef_min, coef_max],
    "digestate_frac": 0.9,
    "digestate_unit_cost": (27/37 + 0.104*20),
    "Q_MIN": 5,
    "Q_MAX": 60,
    "chp_elec_eff": 0.4,
    "chp_heat_eff": 0.4,
    "eeg_bg_price": 194.3,
    "eeg_bm_price": 210.4,
    "cap_biogas": 0.45,
    "cap_biomethane": 0.1,
    "elec_spot_price": 60,
    "heat_price": 20,
    "bonus_rate": 0.0293118,  # €/Mm³
}

AVG_CH4_CONTENT = 0.588
AVG_BIOGAS_YIELD = 67
AVG_GHG = -78

def npv_parts(cap, p, tech="Upgrading", feed_cost_coef=None):
    """Return (init, annual_cashflow, K_GHG, K_gas, K_co2)."""
    fc = feed_cost_coef if feed_cost_coef is not None else p["feed_cost_coef"]
    Q_bio = cap * 1e6  # m³/yr
    Q_ch4 = Q_bio * AVG_CH4_CONTENT  # m³/yr

    # Common Costs
    capex_bio = Q_bio * 150.12 * (Q_bio ** -0.311) / 1e6  # M€
    opex_bio = 2.1209 * (Q_bio ** 0.8359) / 1e6  # M€/yr
    trans = (fc * Q_bio + p["feed_cost_const"]) / 1e6  # M€/yr

    # Technology-Specific
    if tech == "Upgrading":
        capex_upg = (Q_bio / p["FLH_max"]) * 47777 * ((Q_bio / p["FLH_max"]) ** -0.421) / 1e6 + 1
        opex_upg = p["var_upg_cost"] * Q_ch4 / 1e6
        gas_rev = Q_ch4 * (p["gas_price_mwh"] * p["alphaHV"] / 1000) / 1e6
        co2_rev = (Q_bio - Q_ch4) * (p["co2_price_ton"] / 556.2) / 1e6
        ghg_rev = (p["alpha_GHG_ref"] - AVG_GHG) * p["alphaHV"] * 3.6 * Q_ch4 * p["GHG_certificate_price"] / 1e12
        init = -(capex_bio + capex_upg)
        opex = opex_bio + opex_upg + trans
        ann = gas_rev + co2_rev + ghg_rev - opex
        K_ghg = (p["alpha_GHG_ref"]) * p["alphaHV"] * 3.6 * Q_ch4 / 1e12
        K_gas = Q_ch4 * (p["alphaHV"] / 1000) / 1e6
        K_co2 = (Q_bio - Q_ch4) / 556.2 / 1e6
    elif tech == "FlexEEG_biogas":

        opex_chp = 2.1209 * ((Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000) ** 0.8359) / 1e6
        eeg_rev = p["eeg_bg_price"] *p["cap_biogas"] * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        spot_rev = p["elec_spot_price"] * Q_ch4 *(1-p["cap_biogas"]) *p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        heat_rev =  p["heat_price"] * Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        bonus_rev = Q_bio * p["bonus_rate"] / 1e6  # M€/yr
        init = -(capex_bio)
        opex = opex_bio + opex_chp + trans
        ann = eeg_rev + spot_rev + heat_rev + bonus_rev - opex
        K_ghg = 0
        K_gas = Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        K_co2 = 0
    elif tech == "FlexEEG_biomethane":
        capex_upg = (Q_bio / p["FLH_max"]) * 47777 * ((Q_bio / p["FLH_max"]) ** -0.421) / 1e6 + 1

        opex_upg = p["var_upg_cost"] * Q_ch4 / 1e6
        opex_chp = 2.1209 * ((Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000) ** 0.8359) / 1e6
        eeg_rev = p["eeg_bm_price"] *p["cap_biomethane"] * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        spot_rev = p["elec_spot_price"] * Q_ch4 *(1-p["cap_biomethane"]) *p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        heat_rev =  p["heat_price"] * Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        bonus_rev = Q_bio * p["bonus_rate"] / 1e6
        init = -(capex_bio + capex_upg)
        opex = opex_bio + opex_upg + opex_chp + trans
        ann = eeg_rev + spot_rev + heat_rev + bonus_rev - opex
        K_ghg = 0
        K_gas = Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        K_co2 = 0
    elif tech == "NonEEG_CHP":
        opex_chp = 2.1209 * ((Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000) ** 0.8359) / 1e6
        spot_rev = p["elec_spot_price"] * Q_ch4 *p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        heat_rev =  p["heat_price"] * Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        init = -(capex_bio)
        opex = opex_bio + opex_chp + trans
        ann = spot_rev + heat_rev - opex
        K_ghg = 0
        K_gas = Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        K_co2 = 0
    else:
        raise ValueError(f"Unknown technology: {tech}")

    return init, ann, K_ghg, K_gas, K_co2

def break_even_price(cap, p, key, tech="Upgrading", feed_cost_coef=None):
    init, ann, Kghg, Kg, Kco2 = npv_parts(cap, p, tech, feed_cost_coef)
    pv = (1 - (1 + p["r"]) ** (-p["years"])) / p["r"]
    mapping = {
        "GHG_certificate_price": (Kghg, p["GHG_certificate_price"]),
        "gas_price_mwh": (Kg, p["gas_price_mwh"]),
        "co2_price_ton": (Kco2, p["co2_price_ton"]),
        "eeg_bg_price": (Kg, p["eeg_bg_price"]),
        "eeg_bm_price": (Kg, p["eeg_bm_price"]),
        "elec_spot_price": (Kg, p["elec_spot_price"]),
        "heat_price": (Kg, p["heat_price"]),
    }
    K, cur = mapping.get(key, (0, 0))
    npv0 = init + pv * (ann - K * cur)
    if npv0 >= 0 or K <= 0:
        return 0.0
    return -npv0 / (pv * K)

# ─────────────────────────────────────────────────────────────
# 3) DEFINE YOUR TECHNOLOGIES
# ─────────────────────────────────────────────────────────────
alternative_configs = [
    {
        "name": "Upgrading",
        "metrics": ["co2_price_ton", "GHG_certificate_price", "gas_price_mwh"],
        "labels": ["CO₂ price [€/t]", "GHG quota [€/t]", "Gas [€/MWh]"],
    },
    {
        "name": "FlexEEG_biogas",
        "metrics": ["eeg_bg_price", "elec_spot_price", "heat_price"],
        "labels": ["EEG tariff [€/MWh]", "Spot elec [€/MWh]", "Heat [€/MWh]"],
    },
    {
        "name": "FlexEEG_biomethane",
        "metrics": ["eeg_bm_price", "elec_spot_price", "heat_price"],
        "labels": ["EEG tariff [€/MWh]", "Spot elec [€/MWh]", "Heat [€/MWh]"],
    },
    {
        "name": "NonEEG_CHP",
        "metrics": ["elec_spot_price", "heat_price"],
        "labels": ["Elec price [€/MWh]", "Heat price [€/MWh]"],
    },
]

# ─────────────────────────────────────────────────────────────
# DEBUG PRINTS: show base / low / high for a few capacities
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# DEBUG: Detailed CAPEX, OPEX, Revenue Breakdown
# ─────────────────────────────────────────────────────────────
print("\n----- DETAILED COST & REVENUE BREAKDOWN -----")
for cap in debug_caps:
    print(f"\nCapacity = {cap:.1f} Mm³/yr")
    Q_bio = cap * 1e6
    Q_ch4 = Q_bio * AVG_CH4_CONTENT
    for alt in alternative_configs:
        tech = alt["name"]
        print(f"  Technology: {tech}")
        init, ann, K_ghg, K_gas, K_co2 = npv_parts(cap, P, tech)
        
        # Common Costs
        capex_bio = Q_bio * 150.12 * (Q_bio ** -0.311) / 1e6
        opex_bio = 2.1209 * (Q_bio ** 0.8359) / 1e6
        trans = (P["feed_cost_coef"] * Q_bio + P["feed_cost_const"]) / 1e6
        
        # Technology-Specific
        if tech == "Upgrading":
            capex_upg = (Q_bio / P["FLH_max"]) * 47777 * ((Q_bio / P["FLH_max"]) ** -0.421) / 1e6 + 1
            capex_chp = 0
            opex_upg = P["var_upg_cost"] * Q_ch4 / 1e6
            opex_chp = 0
            gas_rev = Q_ch4 * (P["gas_price_mwh"] * P["alphaHV"] / 1000) / 1e6
            co2_rev = (Q_bio - Q_ch4) * (P["co2_price_ton"] / 556.2) / 1e6
            ghg_rev = (P["alpha_GHG_ref"] - AVG_GHG) * P["alphaHV"] * 3.6 * Q_ch4 * P["GHG_certificate_price"] / 1e12
            eeg_rev = spot_rev = heat_rev = bonus_rev = 0
        elif tech == "FlexEEG_biogas":
            capex_upg = 0
            opex_upg = 0
            opex_chp = 2.1209 * ((Q_ch4 * P["chp_elec_eff"] * P["alphaHV"] / 1000) ** 0.8359) / 1e6
            eeg_rev = P["eeg_bg_price"] *P["cap_biogas"] * Q_ch4 * P["chp_elec_eff"] * P["alphaHV"] / 1000 / 1e6
            spot_rev = P["elec_spot_price"] * Q_ch4 *(1-P["cap_biogas"]) *P["chp_elec_eff"] * P["alphaHV"] / 1000 / 1e6
            heat_rev =  P["heat_price"] * Q_ch4 * P["chp_heat_eff"] * P["alphaHV"] / 1000 / 1e6
            bonus_rev = Q_bio * P["bonus_rate"] / 1e6
            gas_rev = co2_rev = ghg_rev = 0
        elif tech == "FlexEEG_biomethane":
            capex_upg = (Q_bio / P["FLH_max"]) * 47777 * ((Q_bio / P["FLH_max"]) ** -0.421) / 1e6 + 1
            opex_upg = P["var_upg_cost"] * Q_ch4 / 1e6
            opex_chp = 2.1209 * ((Q_ch4 * P["chp_elec_eff"] * P["alphaHV"] / 1000) ** 0.8359) / 1e6
            eeg_rev = P["eeg_bm_price"] *P["cap_biomethane"] * Q_ch4 * P["chp_elec_eff"] * P["alphaHV"] / 1000 / 1e6
            spot_rev = P["elec_spot_price"] * Q_ch4 *(1-P["cap_biomethane"]) *P["chp_elec_eff"] * P["alphaHV"] / 1000 / 1e6
            heat_rev =  P["heat_price"] * Q_ch4 * P["chp_heat_eff"] * P["alphaHV"] / 1000 / 1e6
            bonus_rev = Q_bio * P["bonus_rate"] / 1e6
            gas_rev = co2_rev = ghg_rev = 0
        elif tech == "NonEEG_CHP":
            capex_upg = 0
            opex_upg = 0
            opex_chp = 2.1209 * ((Q_ch4 * P["chp_elec_eff"] * P["alphaHV"] / 1000) ** 0.8359) / 1e6
            spot_rev = P["elec_spot_price"] * Q_ch4 *P["chp_elec_eff"] * P["alphaHV"] / 1000 / 1e6
            heat_rev =  P["heat_price"] * Q_ch4 * P["chp_heat_eff"] * P["alphaHV"] / 1000 / 1e6
            eeg_rev = gas_rev = co2_rev = ghg_rev = bonus_rev = 0

        print(f"    CAPEX (M€):")
        print(f"      Biogas:      {capex_bio:10.2f}")
        if capex_upg > 0:
            print(f"      Upgrading:   {capex_upg:10.2f}")
        print(f"      Total:       {-init:10.2f}")
        print(f"    OPEX (M€/yr):")
        print(f"      Biogas:      {opex_bio:10.2f}")
        if opex_upg > 0:
            print(f"      Upgrading:   {opex_upg:10.2f}")
        if opex_chp > 0:
            print(f"      CHP:         {opex_chp:10.2f}")
        print(f"      Transport:   {trans:10.2f}")
        print(f"      Total:       {opex_bio + opex_upg + opex_chp + trans:10.2f}")
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

# ─────────────────────────────────────────────────────────────
# 4) RUN & PLOT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    caps = np.linspace(P["Q_MIN"], P["Q_MAX"], 200)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10), constrained_layout=True)
    axes = axes.flatten()

    for ax, alt in zip(axes, alternative_configs):
        for key, label in zip(alt["metrics"], alt["labels"]):
            base_curve = [break_even_price(c, P, key, alt["name"]) for c in caps]
            low_curve = [break_even_price(c, P, key, alt["name"], feed_cost_coef=P["feed_cost_coef_range"][0]) for c in caps]
            high_curve = [break_even_price(c, P, key, alt["name"], feed_cost_coef=P["feed_cost_coef_range"][1]) for c in caps]
            ax.plot(caps, base_curve, label=label, lw=2)
            ax.fill_between(caps, low_curve, high_curve, alpha=0.2)
        ax.set_title(alt["name"])
        ax.set_xlabel("Capacity [Mm³/yr]")
        ax.set_ylabel("Break-even price")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.savefig(os.path.join(output_dir, "break_even_all_alts.png"))
    plt.show()