
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import argparse
import sys
import os

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
        "legend.fontsize": 14,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "grid.alpha": 0.3
    })

def get_label(col):
    """Helper to parse column names into LaTeX formatted labels."""
    mapping = {
        "phi": r"$\phi$ / -",
        "Yegr": r"$Y_{ed}$ / -",
        "Xf_H2": r"$X_{\mathrm{f,H}_2}$ / mol/mol",
        "Tu_K": r"$T_u$ / K",
        "p_bar": r"$p$ / bar",
        "SL_cm_s": r"Simulated $S_{\mathrm{L}}$ / cm/s",
        "SL_model_cm_s": r"Modeled $S_{\mathrm{L}}$ / cm/s",
        "Tb_K": r"$T_\mathrm{b}$ / K",
        "Ti_K": r"$T_\mathrm{i}$ / K"
    }
    return mapping.get(col, col)

def plot_parity(df, output_path=None, log_scale=False):
    """Generates a binned parity plot: Modeled vs. Validated SL, color by density."""
    has_truth = "SL_cm_s" in df.columns
    if not has_truth and "SL_uncert_abs_cm_s" not in df.columns:
        print("Error: DataFrame missing both 'SL_cm_s' (experimental) and 'SL_uncert_abs_cm_s' (estimated uncertainty).")
        return
    if "SL_model_cm_s" not in df.columns:
        print("Error: DataFrame missing 'SL_model_cm_s' (model results).")
        return

    # 4x3 aspect ratio (6 x 4.5 inches)
    plt.figure(figsize=(6, 4.5))
    
    y = df["SL_model_cm_s"].values
    if has_truth:
        x = df["SL_cm_s"].values
    else:
        # Uncertainty Parity mode: Use model results for both axes, 
        # but "blur" the y-axis with the mapped uncertainty to show expected spread
        print("Note: Experimental data missing. Generating Expected Parity Plot using mapped uncertainties.")
        x = y.copy()
        # We simulate a "parity spread" using the estimated absolute uncertainty
        # To show the density of the uncertainty band, we create two points per case
        # representing the +/- 1-sigma bounds.
        sigma = df["SL_uncert_abs_cm_s"].values
        x = np.concatenate([x, x])
        y = np.concatenate([y + sigma, y - sigma])

    # Filter out non-positive values for log scale
    if log_scale:
        mask = (x > 0) & (y > 0)
        x, y = x[mask], y[mask]
        plt.xscale("log")
        plt.yscale("log")

    # Adaptive gridsize: adjusted to produce larger bins (lower resolution)
    # Logic: map log10(N) to gridsize [50, 150]
    n_pts = len(x)
    gs = int(np.clip(30 * np.log10(n_pts) - 15, 80, 85))
    
    # Using standard Purples colormap
    if log_scale:
        hb = plt.hexbin(x, y, gridsize=gs, cmap="Purples", mincnt=1, bins='log', xscale='log', yscale='log', zorder=1)
        plt.xlim(left=1.0)
        plt.ylim(bottom=1.0)
    else:
        hb = plt.hexbin(x, y, gridsize=gs, cmap="Purples", mincnt=1, bins='log', zorder=1)
    
    plt.colorbar(hb, label="Log10(Point Density)")

    # 45-degree line
    min_val = 1.0 if log_scale else 0.0
    max_val = max(x.max(), y.max())
    line_range = np.array([min_val, max_val])
    
    plt.plot(line_range, line_range, 'k--', linewidth=1, alpha=0.8, label="Parity", zorder=2)
    
    # +/- 10% bounds
    plt.plot(line_range, line_range*1.1, 'k:', linewidth=1, alpha=0.5, label=r"$\pm 10\%$", zorder=2)
    plt.plot(line_range, line_range*0.9, 'k:', linewidth=1, alpha=0.5, zorder=2)
    
    # +/- 25% bounds
    plt.plot(line_range, line_range*1.25, 'k:', linewidth=1, alpha=0.3, label=r"$\pm 25\%$", zorder=2)
    plt.plot(line_range, line_range*0.75, 'k:', linewidth=1, alpha=0.3, zorder=2)

    plt.xlabel(get_label("SL_cm_s"))
    plt.ylabel(get_label("SL_model_cm_s"))
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.2, which="both" if log_scale else "major")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"{'Log-log ' if log_scale else ''}Binned parity plot saved to {output_path}")
    else:
        plt.show()

def plot_error_heatmap(df, x_col="phi", y_col="Tu_K", output_path=None, use_uncert=False):
    """Generates a smooth heatmap using grid interpolation for non-distorted field representation."""
    from scipy.interpolate import griddata

    # Logic to determine what column to plot
    if use_uncert:
        val_col = "SL_uncert_rel_pct"
        label = "Est. Uncertainty / %"
        cmap = "Purples"
    else:
        val_col = "SL_rel_dev_pct"
        label = "Mean Abs. Relative Error / %"
        cmap = "Reds"

    # Check data availability
    if val_col not in df.columns:
        if not use_uncert:
            if "SL_cm_s" in df.columns and "SL_model_cm_s" in df.columns:
                df[val_col] = np.abs((df["SL_model_cm_s"] - df["SL_cm_s"]) / np.maximum(df["SL_cm_s"], 1e-12) * 100.0)
            elif "SL_uncert_rel_pct" in df.columns:
                val_col = "SL_uncert_rel_pct"
                label = "Est. Uncertainty / %"
                cmap = "Purples"
            else:
                print(f"Error: Column '{val_col}' or 'SL_cm_s' not found.", file=sys.stderr)
                return
        else:
            if val_col not in df.columns:
                print(f"Error: Column '{val_col}' not found for uncertainty heatmap.", file=sys.stderr)
                return

    # Heatmaps usually benefit from a slightly wider aspect ratio
    plt.figure(figsize=(6.5, 3.0))
    
    # Aggregate data to handle duplicates in the 2D projection
    df_plot = df.groupby([x_col, y_col])[val_col].mean().reset_index()

    if len(df_plot) > 50:
        # Create a regular grid for smooth, non-stretched interpolation
        # Using 300x300 grid for high resolution
        xi = np.linspace(df_plot[x_col].min(), df_plot[x_col].max(), 300)
        yi = np.linspace(df_plot[y_col].min(), df_plot[y_col].max(), 300)
        XI, YI = np.meshgrid(xi, yi)
        
        # Interpolate scattered data onto the grid
        ZI = griddata((df_plot[x_col], df_plot[y_col]), df_plot[val_col], (XI, YI), method='linear')
        
        # Plot using imshow for a perfectly rectangular, non-distorted grid
        im = plt.imshow(ZI, extent=(xi.min(), xi.max(), yi.min(), yi.max()), 
                        origin='lower', aspect='auto', cmap=cmap)
        plt.colorbar(im, label=label)
        
        # Overlay subtle contour lines
        plt.contour(XI, YI, ZI, levels=15, colors='k', linewidths=0.3, alpha=0.2)
    else:
        sc = plt.scatter(df_plot[x_col], df_plot[y_col], c=df_plot[val_col], 
                         cmap=cmap, s=50, alpha=0.8, edgecolors='w', linewidth=0.5)
        plt.colorbar(sc, label=label)

    plt.xlabel(get_label(x_col))
    plt.ylabel(get_label(y_col))
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Heatmap saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready accuracy plots for the SL model.")
    parser.add_argument("-i", "--input", required=True, help="Path to the validation results CSV (val_results.csv).")
    parser.add_argument("--parity", help="Filename for the parity plot (e.g., parity.pdf).")
    parser.add_argument("--heatmap", help="Filename for the error heatmap (e.g., heatmap.pdf).")
    parser.add_argument("--uncert-heatmap", help="Filename for the uncertainty heatmap (e.g., uncert_map.pdf).")
    parser.add_argument("--x-axis", default="phi", help="X-axis variable for heatmap (default: phi).")
    parser.add_argument("--y-axis", default="Tu_K", help="Y-axis variable for heatmap (default: Tu_K).")
    parser.add_argument("--filter", help="Simple filter string in pandas query format (e.g., 'p_bar == 10 and Yegr < 0.1').")
    parser.add_argument("--robust", action="store_true", help="Restrict to robust conditions: 4-point blending or anchor points.")
    parser.add_argument("--log", action="store_true", help="Use log-log scale for parity plot.")
    parser.add_argument("--show-types", action="store_true", help="Print unique blend types and counts, then exit.")
    
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    if args.show_types:
        if "blend_type" in df.columns:
            print("Unique blend types in dataset:")
            print(df["blend_type"].value_counts())
        else:
            print("Column 'blend_type' not found.")
        sys.exit(0)

    if args.robust:
        print("Restricting to robust conditions (4-point blending or anchor points)...")
        # Define anchor points (with slight tolerance)
        anchors = [0.0, 0.4, 0.8, 0.9, 1.0] # Included 0.9
        is_anchor = df["Xf_H2"].apply(lambda x: any(np.abs(x - a) < 1e-4 for a in anchors))
        is_4point = df["blend_type"].isin(["4-point", "4-point (adaptive)"])
        # Also include explicit 'is_robust' if present
        if "is_robust" in df.columns:
            is_rob = df["is_robust"] == True
        else:
            is_rob = is_4point # fallback
            
        df = df[is_anchor | is_rob]
        print(f"Remaining points: {len(df)}")

    if args.filter:
        print(f"Applying filter: {args.filter}")
        df = df.query(args.filter)
        if df.empty:
            print("Warning: Filter resulted in an empty dataset.")
            sys.exit(0)

    set_pub_style()

    if args.parity:
        plot_parity(df, args.parity, log_scale=args.log)
    
    if args.heatmap:
        plot_error_heatmap(df, args.x_axis, args.y_axis, args.heatmap, use_uncert=False)
        
    if args.uncert_heatmap:
        plot_error_heatmap(df, args.x_axis, args.y_axis, args.uncert_heatmap, use_uncert=True)

    if not any([args.parity, args.heatmap, args.uncert_heatmap]):
        print("No plot requested. Use --parity, --heatmap, or --uncert-heatmap.")

if __name__ == "__main__":
    main()
