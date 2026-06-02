import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

def set_pub_style():
    """Sets a clean, publication-ready style using matplotlib and seaborn."""
    # Increased font scale for better readability relative to graph size
    sns.set_context("paper", font_scale=1.8) 
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "grid.alpha": 0.3
    })

def main():
    parser = argparse.ArgumentParser(description="Visualize model training ranges and potential extrapolation issues.")
    parser.add_argument("-i", "--input", required=True, help="Path to validation results.")
    parser.add_argument("-o", "--output", default="training_range.png", help="Output plot filename.")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    set_pub_style()
    # 4x3 aspect ratio
    plt.figure(figsize=(6.5, 4.5))

    # Add core region rectangle
    from matplotlib.patches import Rectangle
    # Core region: phi [0.5, 1.8] for any blend
    core_rect = Rectangle((0.5, 0.0), 1.3, 1.0, linewidth=0, edgecolor='green', facecolor='green', alpha=0.3, label="Core Robust Region", zorder=0)
    plt.gca().add_patch(core_rect)

    # 1. Plot all data points to show coverage
    if "SL_rel_dev_pct" in df.columns:
        error_abs = np.abs(df["SL_rel_dev_pct"])
        # Increased symbol size to 40 for publication
        sc = plt.scatter(df["phi"], df["Xf_H2"], c=error_abs, cmap="viridis", 
                         alpha=0.6, s=40, label="Validation Data", vmin=0, vmax=25, edgecolors='none')
        # Temporarily set alpha to 1.0 so the colorbar is drawn fully opaque
        sc.set_alpha(1.0)
        cbar = plt.colorbar(sc, label=r"|Relative Error| [%]")
        # Restore alpha for the plot points
        sc.set_alpha(0.6)
    else:
        plt.scatter(df["phi"], df["Xf_H2"], alpha=0.2, s=40, color='gray', label="Validation Data", edgecolors='none')

    # 2. Identify anchors and Grouped EGR levels
    anchors = [0.0, 0.4, 0.8, 0.95, 1.0]
    groups = [
        {"yegrs": [0.0, 0.1], "label": r"Train-Lim $Y_{egr} \leq 0.1$", "ls": "-", "color": "black"},
        {"yegrs": [0.2, 0.3], "label": r"Train-Lim $Y_{egr} \in [0.2, 0.3]$", "ls": "--", "color": "red"}
    ]

    for g in groups:
        phi_mins = []
        phi_maxs = []
        valid_anchors = []
        
        ref_y = g["yegrs"][0]
        for a in anchors:
            sub = df[(df['Yegr'] == ref_y) & (np.abs(df['Xf_H2'] - a) < 1e-4)]
            if not sub.empty:
                phi_mins.append(sub['phi'].min())
                phi_maxs.append(sub['phi'].max())
                valid_anchors.append(a)
        
        if valid_anchors:
            # Plot min range line
            plt.plot(phi_mins, valid_anchors, color=g["color"], linestyle=g["ls"], linewidth=1.0, label=g["label"])
            # Plot max range line (no duplicate label)
            plt.plot(phi_maxs, valid_anchors, color=g["color"], linestyle=g["ls"], linewidth=1.0)

    plt.xlabel(r"$\phi$ / -")
    plt.ylabel(r"$X_{\mathrm{f,H}_2}$ / mol/mol")
    # No title for cleaner publication look
    
    plt.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    plt.xlim(0, 4.2)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
