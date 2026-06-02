import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.interpolate import NearestNDInterpolator

# We need access to the model to compute errors on val_df if not present
from sl_ti_model import sl_ti_model
from blending import chen_blend_massflux, interpolate_property_piecewise
from thermo import fuel_mass_fractions_from_molar_x, compute_rho_u

def compute_val_results(val_df, pars, auxs, opts_l, z_all, ed_table):
    """
    Computes model results for validation data to establish the error map.
    """
    X = val_df[["Tu_K", "p_bar", "Yegr", "phi"]].to_numpy(dtype=float)
    Xf_H2 = val_df["Xf_H2"].to_numpy(dtype=float)
    Tu, pbar, Ye, phi = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    if len(pars) == 1:
        # Single model
        SL, Ti, Tb, _ = sl_ti_model(pars[0], X, auxs[0], opts_l[0])
    else:
        # Blend model
        m_all = np.column_stack([
            compute_rho_u(Tu, pbar, Ye, phi, z, ed_table) * 
            sl_ti_model(pars[i], X, auxs[i], opts_l[i])[0] * 0.01 
            for i, z in enumerate(z_all)
        ])
        Yc_all = np.array([fuel_mass_fractions_from_molar_x(z)[0] for z in z_all])
        Yh_all = np.array([fuel_mass_fractions_from_molar_x(z)[1] for z in z_all])
        Yc_t, Yh_t = fuel_mass_fractions_from_molar_x(Xf_H2)

        m_b, _, _ = chen_blend_massflux(m_all, z_all, Yc_all, Yh_all, Xf_H2, Yc_t, Yh_t)
        SL = (m_b / np.maximum(compute_rho_u(Tu, pbar, Ye, phi, Xf_H2, ed_table), 1e-12)) * 100.0
        
        # Interpolate Tb and Ti for each case
        Tb = interpolate_property_piecewise(z_all, [sl_ti_model(pars[i], X, auxs[i], opts_l[i])[2] for i in range(len(z_all))], Xf_H2)
        Ti = interpolate_property_piecewise(z_all, [sl_ti_model(pars[i], X, auxs[i], opts_l[i])[1] for i in range(len(z_all))], Xf_H2)
    
    val_df["SL_model_cm_s"] = SL
    val_df["Tb_model_K"] = Tb
    val_df["Ti_model_K"] = Ti
    return val_df

class UncertaintyEstimator:
    """
    Estimates model uncertainty based on validation data errors.
    Uses nearest-neighbor interpolation in normalized parameter space.
    """
    def __init__(self, val_df, input_cols=['Tu_K', 'p_bar', 'Yegr', 'phi', 'Xf_H2']):
        self.input_cols = input_cols
        
        # Ensure we have the required columns
        if 'SL_model_cm_s' not in val_df.columns:
            raise KeyError("val_df missing 'SL_model_cm_s'. Model results must be computed first.")
        if 'SL_cm_s' not in val_df.columns:
            raise KeyError("val_df missing 'SL_cm_s'. Experimental data required.")

        # Prepare data for OOD and interpolation
        if 'SL_rel_dev_pct' not in val_df.columns:
            # Use absolute relative deviation for uncertainty map
            val_df['SL_rel_dev_pct'] = np.abs(np.divide(
                val_df['SL_model_cm_s'] - val_df['SL_cm_s'],
                np.maximum(val_df['SL_cm_s'], 1e-12),
                out=np.zeros(len(val_df)),
                where=val_df['SL_cm_s'] > 1e-12
            ) * 100.0)
        else:
            # Ensure it is absolute
            val_df['SL_rel_dev_pct'] = np.abs(val_df['SL_rel_dev_pct'])
        
        # We also want absolute deviation in cm/s as a fallback for low speeds
        val_df['SL_abs_dev_cm_s'] = np.abs(val_df['SL_model_cm_s'] - val_df['SL_cm_s'])
        
        self.data = val_df[input_cols].values
        self.errors_rel = val_df['SL_rel_dev_pct'].values
        self.errors_abs = val_df['SL_abs_dev_cm_s'].values
        
        # Normalization factors (min-max)
        self.min_vals = self.data.min(axis=0)
        self.max_vals = self.data.max(axis=0)
        self.ranges = np.maximum(self.max_vals - self.min_vals, 1e-12)
        
        self.norm_data = (self.data - self.min_vals) / self.ranges
        
        # Interpolators for both relative and absolute error
        self.interp_rel = NearestNDInterpolator(self.norm_data, self.errors_rel)
        self.interp_abs = NearestNDInterpolator(self.norm_data, self.errors_abs)
        
        # KDTree for OOD (nearest distance)
        self.tree = KDTree(self.norm_data)

    def estimate(self, X):
        """
        X: np.ndarray [N, 5] -> [Tu_K, p_bar, Yegr, phi, Xf_H2]
        Returns: (uncertainty_pct, uncertainty_abs, is_ood)
        """
        X = np.asarray(X)
        X_norm = (X - self.min_vals) / self.ranges
        
        # Interpolate errors
        uncert_rel = self.interp_rel(X_norm)
        uncert_abs = self.interp_abs(X_norm)
        
        # Bounding box check
        in_bbox = np.all((X[:, :4] >= self.min_vals[:4] - 1e-3) & 
                         (X[:, :4] <= self.max_vals[:4] + 1e-3), axis=1)
        
        # Density check (4D)
        tree_4d = KDTree(self.norm_data[:, :4])
        dist_4d, _ = tree_4d.query(X_norm[:, :4], k=1)
        is_near = dist_4d < 0.25 
        
        # Special check for Yegr > 0.3
        is_yegr_high = X[:, 2] > 0.3 + 1e-6
        
        is_ood = ~(in_bbox & is_near) | is_yegr_high
        
        return uncert_rel, uncert_abs, is_ood

def add_uncertainty_to_df(df, val_df, pars, auxs, opts_l, z_all, ed_table, input_cols=['Tu_K', 'p_bar', 'Yegr', 'phi', 'Xf_H2']):
    print("Building uncertainty estimator from validation data...", file=sys.stderr, flush=True)
    
    if "SL_model_cm_s" not in val_df.columns:
        print("Calculating model results for validation data...", file=sys.stderr)
        val_df = compute_val_results(val_df, pars, auxs, opts_l, z_all, ed_table)

    # Filter out extremely low speeds to avoid noise in relative error
    val_clean = val_df[val_df['SL_cm_s'] > 0.01].copy()
    if len(val_clean) < 100:
        val_clean = val_df.copy()
        
    estimator = UncertaintyEstimator(val_clean, input_cols)
    
    X = df[input_cols].values
    uncert_rel, uncert_abs, is_ood = estimator.estimate(X)
    
    df['SL_uncert_rel_pct'] = uncert_rel
    df['SL_uncert_abs_cm_s'] = uncert_abs
    df['out_of_domain'] = is_ood
    return df
