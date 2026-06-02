# run_model.py
import argparse
import json
import sys
import time
import os
import numpy as np
import pandas as pd

from sl_ti_model import sl_ti_model, prepare_model_opts
from thermo import fuel_mass_fractions_from_molar_x, load_exhaust_surrogate, compute_rho_u
from blending import parse_xf_from_filename, chen_blend_massflux, interpolate_property_piecewise

try:
    from uncertainty import add_uncertainty_to_df
    HAS_UNCERTAINTY = True
except ImportError:
    HAS_UNCERTAINTY = False

def load_config(params_path):
    """Load parameter JSON and return (par, aux, model_opts). Supports old and new formats."""
    with open(params_path, "r") as f:
        config = json.load(f)
    if "Info" in config:
        par = config["par"]
        info = config.get("Info", {})
        return par, info.get("aux", {}), prepare_model_opts(info.get("modelOpts", {}))
    par = config["par"]
    aux = config.get("aux", {}).get("par", config.get("aux", {}))
    ref = config.get("ref", {})
    opts = prepare_model_opts({"T_ref": ref.get("T_ref", 298.0), "p_ref": ref.get("p_ref", 1.0)})
    return par, aux, opts

def calculate_deviations(df):
    """Calculate absolute and relative deviations for validation."""
    if "SL_cm_s" in df.columns:
        df["SL_abs_dev"] = df["SL_model_cm_s"] - df["SL_cm_s"]
        df["SL_rel_dev_pct"] = np.divide(df["SL_abs_dev"], df["SL_cm_s"], 
                                         out=np.full(len(df), np.nan), where=np.abs(df["SL_cm_s"]) > 1e-12) * 100.0
    if "Tb_K" in df.columns: df["Tb_abs_dev"] = df["Tb_model_K"] - df["Tb_K"]
    ti_col = "Ti_K" if "Ti_K" in df.columns else "T0" if "T0" in df.columns else None
    if ti_col: df["Ti_abs_dev"] = df["Ti_model_K"] - df[ti_col]
    return df

def apply_robustness_filter(df, z_all):
    """Marks points outside the high-confidence domain and preserves SL values."""
    phi = df["phi"].values
    xf = df["Xf_H2"].values
    # Core region: phi [0.5, 1.8]
    is_robust = (phi >= 0.5 - 1e-6) & (phi <= 1.8 + 1e-6)
    # Anchor-specific ranges
    for z_a in z_all:
        match = np.abs(xf - z_a) < 1e-4
        if np.abs(z_a - 1.0) < 1e-4: rng = (phi >= 0.2 - 1e-6) & (phi <= 4.0 + 1e-6)
        elif z_a < 0.5: rng = (phi >= 0.6 - 1e-6) & (phi <= 1.6 + 1e-6)
        else: rng = (phi >= 0.5 - 1e-6) & (phi <= 2.0 + 1e-6)
        is_robust |= (match & rng)
    df["is_robust"] = is_robust
    df.loc[~is_robust, "blend_type"] = "Outside robust domain"
    return df

def run_csv_single(args, compare=False):
    par, aux, opts = load_config(args.params[0])
    df = pd.concat([pd.read_csv(p) for p in args.input]) if isinstance(args.input, list) else pd.read_csv(args.input)
    X = df[["Tu_K", "p_bar", "Yegr", "phi"]].to_numpy(dtype=float)
    t0 = time.perf_counter()
    SL, Ti, Tb, _ = sl_ti_model(par, X, aux, opts)
    print(f"Single model: {X.shape[0]} cases, {time.perf_counter()-t0:.6f} s", file=sys.stderr)
    df["Tb_model_K"], df["Ti_model_K"], df["SL_model_cm_s"] = Tb, Ti, SL
    if compare: df = calculate_deviations(df)
    if args.estimate_uncert and HAS_UNCERTAINTY:
        ed_table = load_exhaust_surrogate(args.ed_table) if args.ed_table else None
        df = add_uncertainty_to_df(df, pd.read_csv(args.val_data), [par], [aux], [opts], np.array([parse_xf_from_filename(args.params[0])]), ed_table)
    df.to_csv(args.output, index=False) if args.output else df.to_csv(sys.stdout, index=False)

def run_csv_blend(args, compare=False):
    ed_table = load_exhaust_surrogate(args.ed_table)
    anchors_cfg = sorted([{"p": p, "xf": parse_xf_from_filename(p)} for p in args.params], key=lambda x: x["xf"])
    pars, auxs, opts_l, z_all = [load_config(a["p"])[0] for a in anchors_cfg], [load_config(a["p"])[1] for a in anchors_cfg], [load_config(a["p"])[2] for a in anchors_cfg], np.array([a["xf"] for a in anchors_cfg])
    
    df = pd.concat([pd.read_csv(p) for p in args.input]) if isinstance(args.input, list) else pd.read_csv(args.input)
    Xf_H2 = df[args.xf_h2_col or ("Xblend" if "Xblend" in df.columns else None)].to_numpy(dtype=float) if (args.xf_h2_col or "Xblend" in df.columns) else np.full(len(df), float(args.xf_h2))
    X = df[["Tu_K", "p_bar", "Yegr", "phi"]].to_numpy(dtype=float)
    Tu, pbar, Ye, phi = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    t0 = time.perf_counter()
    m_all = np.column_stack([compute_rho_u(Tu, pbar, Ye, phi, z, ed_table) * sl_ti_model(pars[i], X, auxs[i], opts_l[i])[0] * 0.01 for i, z in enumerate(z_all)])
    Yc_all = np.array([fuel_mass_fractions_from_molar_x(z)[0] for z in z_all])
    Yh_all = np.array([fuel_mass_fractions_from_molar_x(z)[1] for z in z_all])
    Yc_t, Yh_t = fuel_mass_fractions_from_molar_x(Xf_H2)

    m_b, c_v, b_t = chen_blend_massflux(m_all, z_all, Yc_all, Yh_all, Xf_H2, Yc_t, Yh_t)
    SL_b = (m_b / np.maximum(compute_rho_u(Tu, pbar, Ye, phi, Xf_H2, ed_table), 1e-12)) * 100.0
    Tb_b = interpolate_property_piecewise(z_all, [sl_ti_model(pars[i], X, auxs[i], opts_l[i])[2] for i in range(len(z_all))], Xf_H2)
    Ti_b = interpolate_property_piecewise(z_all, [sl_ti_model(pars[i], X, auxs[i], opts_l[i])[1] for i in range(len(z_all))], Xf_H2)
    
    print(f"Blend model: {X.shape[0]} cases, {time.perf_counter()-t0:.6f} s", file=sys.stderr)
    df["Tb_model_K"], df["Ti_model_K"], df["SL_model_cm_s"], df["blend_type"], df["Xf_H2"], df["c_blend"] = Tb_b, Ti_b, SL_b, b_t, Xf_H2, c_v
    df = apply_robustness_filter(df, z_all)
    if compare: df = calculate_deviations(df)
    if args.estimate_uncert and HAS_UNCERTAINTY: 
        df = add_uncertainty_to_df(df, pd.read_csv(args.val_data), pars, auxs, opts_l, z_all, ed_table)
    df.to_csv(args.output, index=False) if args.output else df.to_csv(sys.stdout, index=False)

def run_interactive_mode(args):
    par, aux, opts = load_config(args.params[0])
    xf = parse_xf_from_filename(args.params[0])
    print(f"Interactive model (Anchor: Xf_H2 = {xf:.2f}). Enter: Tu_K p_bar phi Yegr")
    while True:
        try:
            line = input("> ").strip()
            if not line: break
            parts = list(map(float, line.split()))
            if len(parts) != 4: continue
            SL, Ti, Tb, _ = sl_ti_model(par, np.array([parts]), aux, opts)
            phi = parts[3]
            rng = (phi>=0.2) if xf>0.9 else (phi>=0.6 and phi<=1.6) if xf<0.5 else (phi>=0.5 and phi<=2.0)
            warn = "" if rng else " [WARNING: Outside robust domain]"
            print(f"Tb={Tb[0]:.6g} K, Ti={Ti[0]:.6g} K, SL={SL[0]:.6g} cm/s{warn}\n")
        except (EOFError, KeyboardInterrupt): break
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="LFS SL–Ti model orchestrator.")
    parser.add_argument("--params", "-p", nargs="+", required=True)
    parser.add_argument("--mode", "-m", choices=["csv", "interactive", "val"], required=True)
    parser.add_argument("--input", "-i", nargs="+")
    parser.add_argument("--output", "-o")
    parser.add_argument("--blend", action="store_true")
    parser.add_argument("--xf-h2-col")
    parser.add_argument("--xf-h2", type=float)
    parser.add_argument("--ed-table")
    parser.add_argument("--estimate-uncert", action="store_true")
    parser.add_argument("--val-data")
    args = parser.parse_args()
    if args.mode in ["csv", "val"]:
        if args.blend: run_csv_blend(args, compare=(args.mode == "val"))
        else: run_csv_single(args, compare=(args.mode == "val"))
    else: run_interactive_mode(args)

if __name__ == "__main__": main()
