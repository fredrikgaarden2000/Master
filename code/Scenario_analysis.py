#!/usr/bin/env python3
# scenario_analysis_upgrading_with_updated_costs.py
# ------------------------------------------------------------
# Scenario analysis for biomethane-upgrading alternative:
# Break-even prices for GHG quota, gas, and biogenic CO2
# vs. plant capacity, with uncertainty boundaries.
# Updated with transport and feedstock costs from graph.
# ------------------------------------------------------------

import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 1. PARAMETERS – consistent with original models, updated costs
# ─────────────────────────────────────────────────────────────
P = {
    # techno-economic
    "FLH_max"      : 8000,        # full-load hours, h/yr
    "alphaHV"      : 9.97,        # kWh per Nm³ CH₄ (HHV)
    "r"            : 0.042,       # discount rate
    "years"        : 25,          # project life

    # market prices (defaults for non-varying parameters)
    "gas_price_mwh": 30,          # €/MWh
    "co2_price_ton": 20,          # €/t CO₂
    "GHG_certificate_price": 50,  # €/t CO₂-eq
    "var_upg_cost" : 0.05,        # €/Nm³ CH₄
    "alpha_GHG_ref": 94.0,        # baseline gCO₂e/MJ for certificates

    # simplified logistics (updated from graph)
    "feed_cost_coef" : 0.1,       # €/Nm³ biogas (from 1E-07 M€/Nm³ = 0.1 €/Nm³)
    "feed_cost_const": 251200,    # €/yr (from 0.2512 M€/yr)

    # digestate handling
    "digestate_frac"     : 0.9,   # kg digestate per kg feed
    "digestate_unit_cost": (27/37 + 0.104*20),  # €/t digestate

    # sizing range
    "Q_MIN" : 5,               # Mm³ biogas/yr
    "Q_MAX" : 80,                # Mm³ biogas/yr

    # uncertainty ranges
    "feed_cost_coef_range" : [0.05, 0.15],  # ±50% of 0.1
    "digestate_unit_cost_range" : [2.81*0.7, 2.81*1.3],  # ±30%
}

# constants
AVG_GHG_INTENSITY = 30           # gCO₂e/MJ typical feedstock
AVG_CH4_CONTENT   = 0.60         # CH₄ share in raw biogas
AVG_BIOGAS_YIELD  = 100          # Nm³ biogas / t feedstock

# ─────────────────────────────────────────────────────────────
# 2. CASH-FLOW COMPONENTS
# ─────────────────────────────────────────────────────────────
def npv_parts(cap_m3MM: float, p: dict, feed_cost_coef: float = None, 
              digestate_unit_cost: float = None, var_upg_cost: float = None):
    """
    cap_m3MM : capacity, Mm³ biogas per year
    returns   : (initial_M€, annual_no_var_M€/yr, K_GHG, K_gas, K_co2)
    where K_x are coefficients for each variable price (€/t or €/MWh)
    """
    # Use provided values or defaults
    feed_cost = feed_cost_coef if feed_cost_coef is not None else p["feed_cost_coef"]
    digestate_cost = digestate_unit_cost if digestate_unit_cost is not None else p["digestate_unit_cost"]
    upg_cost = var_upg_cost if var_upg_cost is not None else p["var_upg_cost"]

    Q_biogas = cap_m3MM * 1e6                # Nm³ biogas/yr
    Q_CH4    = Q_biogas * AVG_CH4_CONTENT    # Nm³ CH₄ / yr

    # CAPEX
    capex_bio = Q_biogas * 150.12 * Q_biogas**-0.311 / 1e6
    capex_upg = (Q_biogas / p["FLH_max"]) * 47777 * (Q_biogas / p["FLH_max"])**-0.421 / 1e6 + 1
    initial = -(capex_bio + capex_upg)

    # OPEX
    opex_bio = 2.1209 * Q_biogas**0.8359 / 1e6
    opex_upg = upg_cost * Q_CH4 / 1e6
    trans_feed = (feed_cost * Q_biogas + p["feed_cost_const"]) / 1e6
    digestate = (Q_biogas / AVG_BIOGAS_YIELD) * p["digestate_frac"] * digestate_cost / 1e6
    total_opex = opex_bio + opex_upg + trans_feed + digestate

    # Revenues (excluding variable parameter)
    gas_rev = Q_CH4 * (p["gas_price_mwh"] * p["alphaHV"] / 1000) / 1e6
    co2_rev = (Q_biogas - Q_CH4) * (p["co2_price_ton"] / 556.2) / 1e6
    ghg_rev = (p["alpha_GHG_ref"] - AVG_GHG_INTENSITY) * p["alphaHV"] * 3.6 * Q_CH4 * p["GHG_certificate_price"] / 1e12
    annual_no_var = gas_rev + co2_rev + ghg_rev - total_opex

    # Coefficients for each variable price
    K_GHG = (p["alpha_GHG_ref"] - AVG_GHG_INTENSITY) * p["alphaHV"] * 3.6 * Q_CH4 / 1e12  # M€/yr per €/t CO₂
    K_gas = Q_CH4 * (p["alphaHV"] / 1000) / 1e6  # M€/yr per €/MWh
    K_co2 = (Q_biogas - Q_CH4) / 556.2 / 1e6  # M€/yr per €/t CO₂

    return initial, annual_no_var, K_GHG, K_gas, K_co2

# ─────────────────────────────────────────────────────────────
# 3. BREAK-EVEN PRICE FUNCTIONS
# ─────────────────────────────────────────────────────────────
def break_even_ghg_price(cap_m3MM: float, p: dict, feed_cost_coef: float = None, 
                         digestate_unit_cost: float = None, var_upg_cost: float = None) -> float:
    """
    Return the €/t CO₂-eq certificate price that yields NPV = 0.
    """
    init, annual, K_GHG, _, _ = npv_parts(cap_m3MM, p, feed_cost_coef, digestate_unit_cost, var_upg_cost)
    a = (1 - (1 + p["r"])**(-p["years"])) / p["r"]  # PV factor
    npv0 = init + a * (annual - (p["alpha_GHG_ref"] - AVG_GHG_INTENSITY) * p["alphaHV"] * 3.6 * (cap_m3MM * 1e6 * AVG_CH4_CONTENT) * p["GHG_certificate_price"] / 1e12)
    if npv0 >= 0:
        return 0.0
    if K_GHG <= 0:
        return np.nan
    return -npv0 / (a * K_GHG)

def break_even_gas_price(cap_m3MM: float, p: dict, feed_cost_coef: float = None, 
                         digestate_unit_cost: float = None, var_upg_cost: float = None) -> float:
    """
    Return the €/MWh gas price that yields NPV = 0.
    """
    init, annual, _, K_gas, _ = npv_parts(cap_m3MM, p, feed_cost_coef, digestate_unit_cost, var_upg_cost)
    a = (1 - (1 + p["r"])**(-p["years"])) / p["r"]
    npv0 = init + a * (annual - (cap_m3MM * 1e6 * AVG_CH4_CONTENT) * (p["alphaHV"] / 1000) * p["gas_price_mwh"] / 1e6)
    if npv0 >= 0:
        return 0.0
    if K_gas <= 0:
        return np.nan
    return -npv0 / (a * K_gas)

def break_even_co2_price(cap_m3MM: float, p: dict, feed_cost_coef: float = None, 
                         digestate_unit_cost: float = None, var_upg_cost: float = None) -> float:
    """
    Return the €/t CO₂ price that yields NPV = 0.
    """
    init, annual, _, _, K_co2 = npv_parts(cap_m3MM, p, feed_cost_coef, digestate_unit_cost, var_upg_cost)
    a = (1 - (1 + p["r"])**(-p["years"])) / p["r"]
    npv0 = init + a * (annual - (cap_m3MM * 1e6 * (1 - AVG_CH4_CONTENT)) * (p["co2_price_ton"] / 556.2) / 1e6)
    if npv0 >= 0:
        return 0.0
    if K_co2 <= 0:
        return np.nan
    return -npv0 / (a * K_co2)

# ─────────────────────────────────────────────────────────────
# 4. RUN & PLOT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    caps = np.linspace(P["Q_MIN"], P["Q_MAX"], 200)  # Mm³ / yr

    # Default prices
    ghg_prices = [break_even_ghg_price(c, P) for c in caps]
    gas_prices = [break_even_gas_price(c, P) for c in caps]
    co2_prices = [break_even_co2_price(c, P) for c in caps]

    # Uncertainty bounds
    lower_params = {
        "feed_cost_coef": P["feed_cost_coef_range"][0],
        "digestate_unit_cost": P["digestate_unit_cost_range"][0]
    }
    upper_params = {
        "feed_cost_coef": P["feed_cost_coef_range"][1],
        "digestate_unit_cost": P["digestate_unit_cost_range"][1]
    }

    ghg_prices_lower = [break_even_ghg_price(c, P, **lower_params) for c in caps]
    ghg_prices_upper = [break_even_ghg_price(c, P, **upper_params) for c in caps]
    gas_prices_lower = [break_even_gas_price(c, P, **lower_params) for c in caps]
    gas_prices_upper = [break_even_gas_price(c, P, **upper_params) for c in caps]
    co2_prices_lower = [break_even_co2_price(c, P, **lower_params) for c in caps]
    co2_prices_upper = [break_even_co2_price(c, P, **upper_params) for c in caps]

    # --- Plot setup ---
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 3, 3, 3])

    # --- Left: Alternative title ---
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.text(0.5, 0.5, "Upgrading", fontsize=14, ha='center', va='center', rotation=90)
    ax_title.set_axis_off()

    # --- Right: Three subplots ---
    # GHG Price
    ax_ghg = fig.add_subplot(gs[0, 1])
    ax_ghg.plot(caps, ghg_prices, lw=2, color='blue', label='Base')
    ax_ghg.fill_between(caps, ghg_prices_lower, ghg_prices_upper, color='blue', alpha=0.2, label='Uncertainty')
    ax_ghg.set_xlabel("Capacity [Mm³ biogas yr⁻¹]")
    ax_ghg.set_ylabel("Break-even GHG price [€/t CO₂-eq]")
    ax_ghg.set_title("GHG Quota Price")
    ax_ghg.grid(alpha=0.3)
    ax_ghg.legend()

    # Gas Price
    ax_gas = fig.add_subplot(gs[0, 2])
    ax_gas.plot(caps, gas_prices, lw=2, color='green', label='Base')
    ax_gas.fill_between(caps, gas_prices_lower, gas_prices_upper, color='green', alpha=0.2, label='Uncertainty')
    ax_gas.set_xlabel("Capacity [Mm³ biogas yr⁻¹]")
    ax_gas.set_ylabel("Break-even Gas price [€/MWh]")
    ax_gas.set_title("Gas Price")
    ax_gas.grid(alpha=0.3)
    ax_gas.legend()

    # CO2 Price
    ax_co2 = fig.add_subplot(gs[0, 3])
    ax_co2.plot(caps, co2_prices, lw=2, color='red', label='Base')
    ax_co2.fill_between(caps, co2_prices_lower, co2_prices_upper, color='red', alpha=0.2, label='Uncertainty')
    ax_co2.set_xlabel("Capacity [Mm³ biogas yr⁻¹]")
    ax_co2.set_ylabel("Break-even CO₂ price [€/t CO₂]")
    ax_co2.set_title("Biogenic CO₂ Price")
    ax_co2.grid(alpha=0.3)
    ax_co2.legend()

    plt.tight_layout()
    plt.savefig('scenario_analysis_upgrading_with_updated_costs.png')
    plt.show()

    # --- Save table ---
    df = pd.DataFrame({
        "capacity_Mm3": caps,
        "break_even_ghg_price_EUR_per_t": ghg_prices,
        "break_even_ghg_price_lower": ghg_prices_lower,
        "break_even_ghg_price_upper": ghg_prices_upper,
        "break_even_gas_price_EUR_per_MWh": gas_prices,
        "break_even_gas_price_lower": gas_prices_lower,
        "break_even_gas_price_upper": gas_prices_upper,
        "break_even_co2_price_EUR_per_t": co2_prices,
        "break_even_co2_price_lower": co2_prices_lower,
        "break_even_co2_price_upper": co2_prices_upper
    })
    df.to_csv("scenario_analysis_upgrading_with_updated_costs.csv", index=False)
    print("Saved table → scenario_analysis_upgrading_with_updated_costs.csv")