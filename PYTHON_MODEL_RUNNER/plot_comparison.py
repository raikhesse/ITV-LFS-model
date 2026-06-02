import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os

def set_pub_style():
    """Sets a clean, publication-ready style using matplotlib and seaborn."""
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
        "Tb_K": r"Simulated $T_{\mathrm{b}}$ / K",
        "Tb_model_K": r"Modeled $T_{\mathrm{b}}$ / K",
        "Ti_K": r"Simulated $T_{\mathrm{i}}$ / K",
        "Ti_model_K": r"Modeled $T_{\mathrm{i}}$ / K"
    }
    return mapping.get(col, col)

def main():
    parser = argparse.ArgumentParser(description="Generate trend comparison plots (Model vs Simulation/Exp). Fully flexible: use --x to plot against any variable (phi, Yegr, p_bar, etc.) and --hue to group.")
    parser.add_argument("-i", "--input", required=True, help="Path to results CSV.")
    parser.add_argument("-o", "--output", required=True, help="Output plot filename (e.g. compare_Tb.pdf).")
    parser.add_argument("--x", default="phi", help="X-axis variable (default: phi).")
    parser.add_argument("--y-sim", help="Simulated/Experimental Y variable (symbols).")
    parser.add_argument("--y-mod", required=True, help="Modeled Y variable (lines).")
    parser.add_argument("--hue", default="Xf_H2", help="Grouping variable (default: Xf_H2).")
    parser.add_argument("--filter", help="Pandas query filter (e.g. 'Tu_K == 300 and p_bar == 1').")
    parser.add_argument("--robust", action="store_true", help="Restrict to robust conditions: 4-point blending or anchor points.")
    parser.add_argument("--xlim", nargs=2, type=float, help="X-axis limits: min max.")
    parser.add_argument("--ylim", nargs=2, type=float, help="Y-axis limits: min max.")
    
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    if args.filter:
        print(f"Applying filter: {args.filter}")
        df = df.query(args.filter)
        if df.empty:
            print("Error: Filter resulted in an empty dataset.")
            sys.exit(1)

    if args.robust:
        print("Restricting to robust conditions (4-point blending or anchor points)...")
        anchors = [0.0, 0.4, 0.8, 0.95, 1.0]
        is_anchor = df["Xf_H2"].apply(lambda x: any(np.abs(x - a) < 1e-4 for a in anchors))
        is_rob_blend = df["blend_type"].isin(["4-point", "4-point (adaptive)"])
        if "is_robust" in df.columns:
            is_rob = df["is_robust"] == True
        else:
            is_rob = is_rob_blend
        df = df[is_anchor | is_rob]
        print(f"Remaining points: {len(df)}")
        if df.empty:
            print("Warning: Robust filtering resulted in an empty dataset.")
            sys.exit(0)

    set_pub_style()
    plt.figure(figsize=(6.5, 4.5)) # 4x3 ratio

    # Unique values of the grouping variable
    hue_vals = sorted(df[args.hue].unique())
    colors = sns.color_palette("viridis", len(hue_vals))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'P', '*', 'X', 'd']

    sim_handles = []
    mod_handles = []
    lbl_handles = []

    for i, val in enumerate(hue_vals):
        sub = df[df[args.hue] == val].sort_values(by=args.x)
        
        m_color = colors[i % len(colors)]
        m_marker = markers[i % len(markers)]

        # Determine line style: solid for anchors, dashed for interpolated
        line_style = '-'
        if args.hue == "Xf_H2":
            anchors = [0.0, 0.4, 0.8, 0.95, 1.0]
            if not any(np.abs(val - a) < 1e-4 for a in anchors):
                line_style = '--'

        # Plot Modeled data as Lines
        l_h, = plt.plot(sub[args.x], sub[args.y_mod], linestyle=line_style, color=m_color, linewidth=1.0, label="_nolegend_")
        mod_handles.append(l_h)

        # Plot Simulated/Experimental data as Symbols
        if args.y_sim and args.y_sim in sub.columns:
            s_h, = plt.plot(sub[args.x], sub[args.y_sim], m_marker, color=m_color, markersize=5, 
                            markerfacecolor='none', markeredgewidth=1.0, alpha=0.8, label="_nolegend_")
            sim_handles.append(s_h)
        else:
            sim_handles.append(plt.Line2D([0], [0], color='none'))

        # Placeholder for label handles
        lbl_handles.append(plt.Line2D([0], [0], color='none'))

    plt.xlabel(get_label(args.x))
    plt.ylabel(get_label(args.y_mod).replace("Modeled ", "")) # Use neutral label
    # --- Tabulated Legend Construction ---
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.text import Text

    class TextHandler(HandlerBase):
        def __init__(self, text, bold=False):
            self.text = text
            self.bold = bold
            super().__init__()
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            tx = Text(width/2., height/2., self.text, fontsize=fontsize,
                      fontweight='bold' if self.bold else 'normal',
                      ha="center", va="center", transform=trans)
            return [tx]

    N = len(hue_vals)
    h_sim_head = plt.Line2D([0], [0], color='none')
    h_mod_head = plt.Line2D([0], [0], color='none')
    h_lbl_head = plt.Line2D([0], [0], color='none')

    # Map headers
    h_map = {
        h_sim_head: TextHandler("Sim."),
        h_mod_head: TextHandler("Model"),
        h_lbl_head: TextHandler(get_label(args.hue).split(" / ")[0])
    }

    # Map labels in Col 3
    val_handles = [plt.Line2D([0], [0], color='none') for _ in range(N)]
    for i, v in enumerate(hue_vals):
        label_text = f"{v:.2f}" if args.hue=="Xf_H2" else str(v)
        h_map[val_handles[i]] = TextHandler(label_text)

    # Column-major ordering for Matplotlib legend (ncol=3)
    final_handles = [h_sim_head] + sim_handles + [h_mod_head] + mod_handles + [h_lbl_head] + val_handles
    final_labels = [""] * len(final_handles)

    plt.legend(final_handles, final_labels, ncol=3, loc='best', frameon=True, framealpha=0.8,
               handler_map=h_map, handlelength=2.5, columnspacing=0.2, handletextpad=0.0,
               borderpad=0.4, labelspacing=0.4)

    if args.xlim:
        plt.xlim(args.xlim)
    if args.ylim:
        plt.ylim(args.ylim)

    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(args.output)
    print(f"Comparison plot saved to {args.output}")

if __name__ == "__main__":
    main()
