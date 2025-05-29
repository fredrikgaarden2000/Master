import numpy as np
import matplotlib.pyplot as plt

# Define parameters
P = {
    "FLH_max": 8000,
    "alphaHV": 9.97,
    "r": 0.042,
    "years": 25,
    "gas_price_mwh": 30,
    "co2_price_ton": 20,
    "GHG_certificate_price": 70,
    "var_upg_cost": 0.2,
    "alpha_GHG_ref": 94.0,
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
    "eeg_price_small": 220,
    "eeg_price_large": 190,
    "EEG_small_m3": 255870 * (8000 / 8760),
    "EEG_large_m3": 511740 * (8000 / 8760),
    "AVG_CH4_CONTENT": 0.588,
    "AVG_BIOGAS_YIELD": 67,
    "AVG_GHG": -78,
    "capex_coeff": 150.12,
    "capex_exp": -0.311,
    "opex_coeff": 2.1209,
    "opex_exp": 0.8359,
    "upg_cost_coeff": 47777,
    "upg_cost_exp": -0.421,
}

# Feedstock cost coefficients for sensitivity analysis
feed_cost_coef_range = [0.0202, 0.062]  # €/m³ (low, high)

def npv_parts(cap, p, tech, feed_cost_coef=None):
    """Calculate NPV components for EEG_CHP_Small and EEG_CHP_Large."""
    Q_bio = cap * 1e6  # m³/yr
    Q_ch4 = Q_bio * p["AVG_CH4_CONTENT"]  # m³/yr

    # Validate capacity ranges
    if tech == "EEG_CHP_Small" and not (0.1 <= cap <= p["EEG_small_m3"] / 1e6):
        raise ValueError(f"Capacity {cap} Mm³ out of range for EEG_CHP_Small")
    elif tech == "EEG_CHP_Large" and not (0.1 <= cap <= p["EEG_large_m3"] / 1e6):
        raise ValueError(f"Capacity {cap} Mm³ out of range for EEG_CHP_Large")

    # Feedstock and transport cost
    fc = feed_cost_coef if feed_cost_coef is not None else 0.0428  # €/m³ (midpoint)
    trans = (fc * Q_bio) / 1e6  # M€/yr

    # CAPEX and OPEX
    capex_bio = Q_bio * p["capex_coeff"] * (Q_bio ** p["capex_exp"]) / 1e6  # M€
    opex_bio = p["opex_coeff"] * (Q_bio ** p["opex_exp"]) / 1e6  # M€/yr

    # Average discount factor
    avg_discount = sum(0.99 ** t for t in range(1, p["years"] + 1)) / p["years"]

    # Revenue components
    if tech == "EEG_CHP_Small":
        eeg_rev = avg_discount * p["eeg_price_small"] * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        heat_rev = p["heat_price"] * Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        K_dict = {
            "eeg_price_small": avg_discount * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6,
            "heat_price": Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        }
    elif tech == "EEG_CHP_Large":
        eeg_rev = avg_discount * p["eeg_price_large"] * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6
        heat_rev = p["heat_price"] * Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        K_dict = {
            "eeg_price_large": avg_discount * Q_ch4 * p["chp_elec_eff"] * p["alphaHV"] / 1000 / 1e6,
            "heat_price": Q_ch4 * p["chp_heat_eff"] * p["alphaHV"] / 1000 / 1e6
        }
    else:
        raise ValueError(f"Technology {tech} not supported")

    init = -capex_bio
    opex = opex_bio + trans
    ann = eeg_rev + heat_rev - opex
    return init, ann, K_dict

def break_even_price(cap, p, key, tech, feed_cost_coef=None):
    """Calculate break-even price for a given metric."""
    init, ann, K_dict = npv_parts(cap, p, tech, feed_cost_coef)
    pv = (1 - (1 + p["r"]) ** (-p["years"])) / p["r"]
    K = K_dict.get(key, 0)
    cur = p.get(key, 0)
    npv0 = init + pv * (ann - K * cur)
    return -npv0 / (pv * K) if npv0 < 0 and K > 0 else 0.0

# Define capacity ranges
cap_small = np.linspace(0.1, P["EEG_small_m3"] / 1e6, 50)  # Up to ~0.2336 Mm³
cap_large = np.linspace(0.1, P["EEG_large_m3"] / 1e6, 50)  # Up to ~0.4672 Mm³

# Calculate break-even prices for EEG_CHP_Small
base_curves_small = {
    "eeg_price_small": [break_even_price(c, P, "eeg_price_small", "EEG_CHP_Small") for c in cap_small],
    "heat_price": [break_even_price(c, P, "heat_price", "EEG_CHP_Small") for c in cap_small]
}
low_curves_small = {
    "eeg_price_small": [break_even_price(c, P, "eeg_price_small", "EEG_CHP_Small", feed_cost_coef_range[0]) for c in cap_small],
    "heat_price": [break_even_price(c, P, "heat_price", "EEG_CHP_Small", feed_cost_coef_range[0]) for c in cap_small]
}
high_curves_small = {
    "eeg_price_small": [break_even_price(c, P, "eeg_price_small", "EEG_CHP_Small", feed_cost_coef_range[1]) for c in cap_small],
    "heat_price": [break_even_price(c, P, "heat_price", "EEG_CHP_Small", feed_cost_coef_range[1]) for c in cap_small]
}

# Calculate break-even prices for EEG_CHP_Large
base_curves_large = {
    "eeg_price_large": [break_even_price(c, P, "eeg_price_large", "EEG_CHP_Large") for c in cap_large],
    "heat_price": [break_even_price(c, P, "heat_price", "EEG_CHP_Large") for c in cap_large]
}
low_curves_large = {
    "eeg_price_large": [break_even_price(c, P, "eeg_price_large", "EEG_CHP_Large", feed_cost_coef_range[0]) for c in cap_large],
    "heat_price": [break_even_price(c, P, "heat_price", "EEG_CHP_Large", feed_cost_coef_range[0]) for c in cap_large]
}
high_curves_large = {
    "eeg_price_large": [break_even_price(c, P, "eeg_price_large", "EEG_CHP_Large", feed_cost_coef_range[1]) for c in cap_large],
    "heat_price": [break_even_price(c, P, "heat_price", "EEG_CHP_Large", feed_cost_coef_range[1]) for c in cap_large]
}

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# EEG_CHP_Small subplot
ax1.plot(cap_small, base_curves_small["eeg_price_small"], color="green", label="EEG Small Tariff")
ax1.fill_between(cap_small, low_curves_small["eeg_price_small"], high_curves_small["eeg_price_small"], color="green", alpha=0.2)
ax1.plot(cap_small, base_curves_small["heat_price"], color="red", label="Heat Price")
ax1.fill_between(cap_small, low_curves_small["heat_price"], high_curves_small["heat_price"], color="red", alpha=0.2)
ax1.set_title("EEG_CHP_Small Break-even Prices")
ax1.set_xlabel("Capacity [Mm³/yr]")
ax1.set_ylabel("Break-even Price [€/MWh]")
current_eeg = P["eeg_price_small"]
current_heat = P["heat_price"]
ax1.axhline(current_eeg,
            color="green",
            linestyle="--",
            linewidth=1,
            label=f"Current EEG tariff ({current_eeg} €/MWh)")
ax1.axhline(current_heat,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Current Heat price ({current_heat} €/MWh)")
ax1.legend()
ax1.grid(alpha=0.3)

# EEG_CHP_Large subplot
ax2.plot(cap_large, base_curves_large["eeg_price_large"], color="green", label="EEG Large Tariff")
ax2.fill_between(cap_large, low_curves_large["eeg_price_large"], high_curves_large["eeg_price_large"], color="green", alpha=0.2)
ax2.plot(cap_large, base_curves_large["heat_price"], color="red", label="Heat Price")
ax2.fill_between(cap_large, low_curves_large["heat_price"], high_curves_large["heat_price"], color="red", alpha=0.2)
ax2.set_title("EEG_CHP_Large Break-even Prices")
ax2.set_xlabel("Capacity [Mm³/yr]")
ax2.set_ylabel("Break-even Price [€/MWh]")
current_eeg = P["eeg_price_large"]
current_heat = P["heat_price"]
ax2.axhline(current_eeg,
            color="green",
            linestyle="--",
            linewidth=1,
            label=f"Current EEG tariff ({current_eeg} €/MWh)")
ax2.axhline(current_heat,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Current Heat price ({current_heat} €/MWh)")
ax2.legend()
ax2.grid(alpha=0.3)

# Add a title for the entire figure
fig.suptitle("Break-even Prices for EEG_CHP Technologies", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot
plt.savefig("break_even_eeg_chp.png")

# Print capacity ranges for verification
print(f"EEG_CHP_Small capacity range: {cap_small[0]:.4f} to {cap_small[-1]:.4f} Mm³")
print(f"EEG_CHP_Large capacity range: {cap_large[0]:.4f} to {cap_large[-1]:.4f} Mm³")