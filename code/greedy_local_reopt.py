# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

def plot_greedy_solution(csv_path, save_path=None):
    """Plot greedy solution results from Financials.csv"""
    # Load data with error checking
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find results file at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Convert capacity to Mm³/year
    df["Capacity_Mm3"] = df["TotalCapacity_m3"] / 1e6
    
    # Chart colors by configuration
    colors = {
        "FlexEEG_biomethane": "#66c2a5",
        "Upgrading": "#fc8d62",
        "FlexEEG_biogas": "#8da0cb",
        "NonEEG_CHP": "#e78ac3"
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each plant
    for idx, row in df.iterrows():
        width = row["Capacity_Mm3"]
        irr_pct = row["IRR"] * 100  # Convert IRR from fraction to %
        
        ax.barh(
            y=irr_pct,
            width=width,
            left=0,
            height=0.8,
            color=colors.get(row["Configuration"], "#a6d854"),
            edgecolor="black"
        )
        
        ax.text(
            width + 0.01 * df["Capacity_Mm3"].max(),
            irr_pct,
            row["PlantID"],
            ha="left",
            va="center",
            fontsize=8,
            color="black"
        )

    # Axis labels and formatting
    ax.set_xlabel("Capacity Ω [Mm³ biogas / year]")
    ax.set_ylabel("Internal Rate of Return [%]")
    ax.set_title("Greedy solution – plant IRR vs. capacity")
    ax.set_ylim(0, df["IRR"].max()*100*1.1)

    # Dynamic legend based on present configurations
    unique_configs = df["Configuration"].unique()
    legend_handles = [
        Patch(
            facecolor=colors.get(config, "#a6d854"),
            edgecolor="black",
            label=config
        ) for config in unique_configs
    ]

    ax.legend(
        handles=legend_handles,
        title="Technology",
        bbox_to_anchor=(1.04, 1),
        loc="upper left"
    )
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.tight_layout()
        plt.show()


def plot_capacity_by_irr(csv_path, save_path=None):
    """Plot capacities ordered by descending IRR."""
    # Load data
    df = pd.read_csv(csv_path)
    df["Capacity_Mm3"] = df["TotalCapacity_m3"] / 1e6
    # Ensure IRR column is fraction, convert if necessary
    if df["IRR"].max() > 1:
        df["IRR"] = df["IRR"] / 100.0

    # Sort by IRR descending
    df = df.sort_values("IRR", ascending=False).reset_index(drop=True)

    # Chart colors by configuration
    colors = {
        "FlexEEG_biomethane": "#66c2a5",
        "Upgrading":          "#fc8d62",
        "FlexEEG_biogas":     "#8da0cb",
        "NonEEG_CHP":         "#e78ac3"
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars: x is plant rank, y is capacity
    x = range(len(df))
    bar_colors = [colors.get(cfg, "#a6d854") for cfg in df["Configuration"]]
    ax.bar(
        x,
        df["Capacity_Mm3"],
        color=bar_colors,
        edgecolor="black"
    )

    # Annotate with PlantID above each bar
    for xi, plant in zip(x, df["PlantID"]):
        ax.text(
            xi,
            df.loc[df["PlantID"] == plant, "Capacity_Mm3"].values[0] + 0.02 * df["Capacity_Mm3"].max(),
            plant,
            ha="center",
            va="bottom",
            fontsize=8
        )

    # Labels and title
    ax.set_xlabel("Build Order (sorted by IRR)")
    ax.set_ylabel("Capacity Ω [Mm³ biogas / year]")
    ax.set_title("Capacities of Plants Ordered by IRR")

    # X-ticks as blank or use small tick marks
    ax.set_xticks(x)
    ax.set_xticklabels([])

    # Legend
    unique_configs = df["Configuration"].unique()
    legend_handles = [
        Patch(facecolor=colors.get(cfg, "#a6d854"), edgecolor="black", label=cfg)
        for cfg in unique_configs
    ]
    ax.legend(handles=legend_handles, title="Technology", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    results_dir = "C:/Clone/Master/results/large_scale_cont/10_greedy_with_alternatives"
    plot_greedy_solution(
        csv_path=os.path.join(results_dir, "Financials.csv")

        #save_path=os.path.join(results_dir, "capacity_vs_irr.png")
    )
    plot_capacity_by_irr(os.path.join(results_dir, "Financials.csv"))