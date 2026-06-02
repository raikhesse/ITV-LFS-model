import numpy as np
import os
import re

def parse_xf_from_filename(path):
    """
    Extract fuel-side molar H2 fraction Xf_H2 from a filename of the form
    ..._<A>_<B>bld_... where A and B are integers.
    """
    fname = os.path.basename(path)
    m = re.search(r'(\d+)_(\d+)bld', fname)
    if not m:
        raise ValueError(
            f"Could not parse Xf_H2 from filename '{fname}'. "
            "Expected pattern like '*_0_80bld_*'."
        )
    whole = int(m.group(1))
    frac_str = m.group(2)
    frac = int(frac_str) / (10 ** len(frac_str))
    return float(whole + frac)


def chen_blend_massflux(m_all, z_all, Yc_all, Yh_all, z_target, Yc_target, Yh_target, eps=1e-12):
    """
    Adaptive Chen-type blending law in mass-flux space for binary CH4/H2.
    Handles rows where some anchors are non-flammable (SL=0).
    Assumes z_all is sorted in ascending order.
    """
    N, M = m_all.shape
    m_blend = np.zeros(N)
    c_vals = np.ones(N)
    blend_type = np.full(N, "Non-flammable", dtype=object)

    def _compute_c(FL, FH, FS, wLS, wHS):
        num_c = wHS * (FH - FS)
        den_c = wLS * (FS - FL)
        c = np.divide(num_c, den_c, out=np.ones_like(num_c), where=np.abs(den_c) >= eps)
        return np.maximum(c, eps)

    # 1. Row-wise flammable mask
    is_flam = m_all > 1e-10
    powers = 2 ** np.arange(M)
    masks = (is_flam * powers).sum(axis=1)

    unique_masks = np.unique(masks)
    for mask in unique_masks:
        row_idx = np.where(masks == mask)[0]
        
        # Check if any row is exactly an anchor (precedence to "Anchor" label)
        z_t_mask = z_target[row_idx]
        for i in range(M):
            match = np.abs(z_t_mask - z_all[i]) < 1e-6
            if np.any(match):
                match_rows = row_idx[match]
                m_blend[match_rows] = m_all[match_rows, i]
                blend_type[match_rows] = "Anchor"
                row_idx = row_idx[~match]
                z_t_mask = z_target[row_idx]
        
        if len(row_idx) == 0 or mask == 0:
            continue

        I = [i for i in range(M) if (mask & (2 ** i))]
        n_flam = len(I)

        idx_L_flam, idx_H_flam = I[0], I[-1]
        S_idx = I[1:-1]

        # Range check: within span of anchors
        z_t = z_target[row_idx]
        in_range = (z_t >= z_all[0] - 1e-6) & (z_t <= z_all[-1] + 1e-6)
        blend_type[row_idx[~in_range]] = "Outside anchor range"
        
        rows = row_idx[in_range]
        if len(rows) == 0:
            continue

        z_val = z_target[rows]
        is_between_flam = (z_val >= z_all[idx_L_flam] - 1e-6) & (z_val <= z_all[idx_H_flam] + 1e-6)
        
        # --- Case A: Within flammable bracket ---
        rows_in = rows[is_between_flam]
        if len(rows_in) > 0:
            idx_L, idx_H = idx_L_flam, idx_H_flam
            m_L, m_H = m_all[rows_in, idx_L], m_all[rows_in, idx_H]
            F_L, F_H = m_L**2, m_H**2
            Yc_L, Yc_H = Yc_all[idx_L], Yc_all[idx_H]
            w_L = (Yc_target[rows_in] - Yc_H) / (Yc_L - Yc_H)
            w_H = 1.0 - w_L
            
            if n_flam < 2:
                m_blend[rows_in] = 0.0 
                blend_type[rows_in] = "Outside flammable range"
            elif n_flam == 2:
                c = np.ones(len(rows_in), dtype=float)
                m_blend[rows_in], c_vals[rows_in], bt = _apply_chen(F_L, F_H, w_L, w_H, c, "2-point (linear)")
                blend_type[rows_in] = bt
            elif n_flam == 3:
                idx_S = S_idx[0]
                w_LS = (Yc_all[idx_S] - Yc_H) / (Yc_L - Yc_H)
                c = _compute_c(F_L, F_H, m_all[rows_in, idx_S]**2, w_LS, 1.0 - w_LS)
                m_blend[rows_in], c_vals[rows_in], bt = _apply_chen(F_L, F_H, w_L, w_H, c, "3-point")
                blend_type[rows_in] = bt
            elif n_flam == 4:
                # Optimized vectorized path for standard 4-point blending
                idx_S1, idx_S2 = S_idx[0], S_idx[1]
                w_LS1 = (Yc_all[idx_S1] - Yc_H) / (Yc_L - Yc_H)
                c1 = _compute_c(F_L, F_H, m_all[rows_in, idx_S1]**2, w_LS1, 1.0 - w_LS1)
                w_LS2 = (Yc_all[idx_S2] - Yc_H) / (Yc_L - Yc_H)
                c2 = _compute_c(F_L, F_H, m_all[rows_in, idx_S2]**2, w_LS2, 1.0 - w_LS2)
                b = (c2 - c1) / (z_all[idx_S2] - z_all[idx_S1])
                a = c1 - b * z_all[idx_S1]
                c = a + b * z_target[rows_in]
                m_blend[rows_in], c_vals[rows_in], bt = _apply_chen(F_L, F_H, w_L, w_H, c, "4-point")
                blend_type[rows_in] = bt
            else: # n_flam > 4, Adaptive selection required
                z_in = z_target[rows_in]
                Yc_in = Yc_target[rows_in]
                for i, k in enumerate(rows_in):
                    zt = z_in[i]
                    # Select the 4 closest flammable anchors
                    best_I = sorted(I, key=lambda idx: abs(z_all[idx] - zt))[:4]
                    best_I = sorted(best_I) # restore ascending order
                    
                    idx_L, idx_H = best_I[0], best_I[3]
                    idx_S1, idx_S2 = best_I[1], best_I[2]
                    
                    FL, FH = m_all[k, idx_L]**2, m_all[k, idx_H]**2
                    YcL, YcH = Yc_all[idx_L], Yc_all[idx_H]
                    wl = (Yc_in[i] - YcH) / (YcL - YcH)
                    wh = 1.0 - wl
                    
                    # Calculate c at S1
                    w_LS1 = (Yc_all[idx_S1] - YcH) / (YcL - YcH)
                    c1 = _compute_c(FL, FH, m_all[k, idx_S1]**2, w_LS1, 1.0 - w_LS1)
                    
                    # Calculate c at S2
                    w_LS2 = (Yc_all[idx_S2] - YcH) / (YcL - YcH)
                    c2 = _compute_c(FL, FH, m_all[k, idx_S2]**2, w_LS2, 1.0 - w_LS2)
                    
                    # Interpolate c to target zt
                    slope = (c2 - c1) / (z_all[idx_S2] - z_all[idx_S1])
                    intercept = c1 - slope * z_all[idx_S1]
                    c_target = intercept + slope * zt
                    
                    # Apply standard Chen rule using the dynamic bounds
                    mb, cv, bt_arr = _apply_chen(FL, FH, wl, wh, c_target, "4-point (adaptive)")
                    m_blend[k] = mb
                    c_vals[k] = cv
                    blend_type[k] = bt_arr.item()

        # --- Case B: Outside flammable bracket, but within anchor span ---
        rows_out = rows[~is_between_flam]
        if len(rows_out) > 0:
            for r in rows_out:
                zt = z_target[r]
                ih = np.searchsorted(z_all, zt, side='right')
                ih = np.clip(ih, 1, M - 1)
                il = ih - 1
                if (il in I) or (ih in I):
                    F_L, F_H = m_all[r, il]**2, m_all[r, ih]**2
                    Yc_L, Yc_H = Yc_all[il], Yc_all[ih]
                    wl = (Yc_target[r] - Yc_H) / (Yc_L - Yc_H)
                    wh = 1.0 - wl
                    m_blend[r], c_vals[r], _ = _apply_chen(F_L, F_H, wl, wh, 1.0, "")
                    blend_type[r] = "2-point (linear)"
                else:
                    m_blend[r] = 0.0
                    blend_type[r] = "Outside flammable range"

    return m_blend, c_vals, blend_type

def _apply_chen(F_L, F_H, w_L, w_H, c, label, eps=1e-12):
    c = np.maximum(c, eps)
    denom_F = c * w_L + w_H
    F_blend = (c * w_L * F_L + w_H * F_H) / np.maximum(denom_F, eps)
    m_blend = np.sqrt(np.maximum(F_blend, 0.0))
    return m_blend, c, np.full_like(m_blend, label, dtype=object)


def interpolate_property_piecewise(z_anchors, q_anchors_list, z_target):
    zA = np.asarray(z_anchors, dtype=float)
    M = zA.size
    if M < 2:
        raise ValueError("Need at least two anchors for interpolation.")
    qA = np.vstack([np.asarray(q, dtype=float) for q in q_anchors_list])
    N = qA.shape[1]
    order = np.argsort(zA)
    zA, qA = zA[order], qA[order, :]
    z_t = np.asarray(z_target, dtype=float)
    idx_high = np.searchsorted(zA, z_t, side="right")
    idx_high = np.clip(idx_high, 1, M - 1)
    idx_low = idx_high - 1
    zL, zH = zA[idx_low], zA[idx_high]
    rows = np.arange(N)
    qL, qH = qA[idx_low, rows], qA[idx_high, rows]
    denom = zH - zL
    alpha = np.divide(z_t - zL, denom, out=np.zeros_like(z_t), where=np.abs(denom) >= 1e-12)
    return (1.0 - alpha) * qL + alpha * qH
