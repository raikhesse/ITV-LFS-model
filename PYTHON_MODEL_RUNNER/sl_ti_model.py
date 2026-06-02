# sl_t0_model.py
import numpy as np


def prepare_model_opts(model_opts=None):
    """
    Fill model options with defaults where missing.

    Keys used here:
    - T_ref, p_ref : reference T and p from JSON["ref"]
    - w_rl         : width of tilt gate in phin
    """
    defaults = {
        "T_ref": 1250.0,
        "p_ref": 10.0,
        "w_rl": 0.5,
    }
    opts = defaults.copy()
    if model_opts is not None:
        opts.update(model_opts)
    return opts


def _get(par, key, *aliases, default=None, required=True):
    """
    Robust accessor for parameter dictionaries.

    Looks for 'key' and then any 'aliases' in 'par'.
    If not found and required=True and default is None, raises KeyError.
    Otherwise returns 'default'.
    """
    if key in par:
        return par[key]
    for a in aliases:
        if a in par:
            return par[a]
    if required and default is None:
        raise KeyError(f"Required parameter '{key}' not found in par.")
    return default


def sl_ti_model(par, X, aux, model_opts=None):
    """
    Python translation of SL_Ti_model.m (global SL–Ti–Tb model).

    Parameters
    ----------
    par : dict
        Parameter dictionary from JSON["par"].
        Supports both the MATLAB names (S_0, Theta_a, ...) and your
        earlier Python names where obvious.

    X : np.ndarray, shape (N, 4) or (N, >=5)
        Columns: [T_K, p_bar, Yegr, phi, (optional code, ...)].
        T_K  : unburned gas temperature [K]
        p_bar: pressure [bar]
        Yegr : external dilution mass fraction
        phi  : fuel–air equivalence ratio
        code : optional; ignored here (always return SL, Ti, Tb separately).

    aux : dict
        For the new JSON format, pass JSON["aux"]["par"], which must contain:
          q_T, A_0, A_T, alpha_0, tau_Y, xi_Y, m_r, phi_Tref, phi_Sref.
        For the old format, the function tries to fall back to keys
        like phi_peak, A0, AT, betaT, gammaT, xiT, qT, mr, phi0.

    model_opts : dict or None
        Should at least contain T_ref, p_ref. For the new JSON format,
        these are taken from JSON["ref"] in load_config().

    Returns
    -------
    SL : np.ndarray, shape (N,)
        Laminar flame speed [cm/s] (same unit as S_0).
    Ti : np.ndarray, shape (N,)
        Inner-layer temperature T_i [K].
    Tb : np.ndarray, shape (N,)
        Adiabatic flame temperature [K] (full φ-dependence).
    extras : dict
        Additional diagnostic fields.
    """
    opts = prepare_model_opts(model_opts)

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] < 4:
        raise ValueError("X must have at least 4 columns: [T_K, p_bar, Yegr, phi].")

    Tu = X[:, 0]  # [K]
    p = X[:, 1]   # [bar]
    Y = X[:, 2]   # Yegr [-]
    phi = X[:, 3]

    # ------------------------------------------------------------------
    # Model options
    # ------------------------------------------------------------------
    T_ref = float(opts["T_ref"])
    p_ref = float(opts["p_ref"])
    w_rl = float(opts["w_rl"])

    # ------------------------------------------------------------------
    # Aux parameters (Tb sub-model)
    # ------------------------------------------------------------------
    def _aux(key, *aliases, default=None, required=True):
        return _get(aux, key, *aliases, default=default, required=required)

    # Try new-style names first; fall back to old ones if present
    phi_Sref = _aux("phi_Sref")
    phi_Tref = _aux("phi_Tref")
    q_T = _aux("q_T")
    A_0 = _aux("A_0")
    A_T = _aux("A_T")
    alpha_0 = _aux("alpha_0")  # in old code this played a similar role
    tau_Y = _aux("tau_Y", default=0.0, required=False)
    xi_Y = _aux("xi_Y")
    m_r = _aux("m_r")

    # ------------------------------------------------------------------
    # Unpack parameters (JSON → MATLAB notation)
    # ------------------------------------------------------------------
    # Core kinetic / cut parameters
    S_0 = _get(par, "S_0")
    Theta_a = _get(par, "Theta_a")
    n_p = _get(par, "n_p")
    n_p2 = _get(par, "n_p2")
    r_th = _get(par, "r_th")
    Theta_i = _get(par, "Theta_i")

    B = _get(par, "B")
    a_T = _get(par, "a_T")
    delta = _get(par, "delta")
    n = _get(par, "n")  # φ-independent collapse exponent

    q_Y = _get(par, "q_Y", default=1.0, required=False)

    aY = _get(par, "a_Y")
    eta = _get(par, "eta")
    pc_r = _get(par, "pc_r")
    lambda_ = _get(par, "lambda", default=1.0, required=False)
    n_r = _get(par, "n_r")

    # Shape dependency coefficients
    beta_L0 = _get(par, "beta_L0", "betaL0")
    beta_Lp = _get(par, "beta_Lp", "betaLp")
    beta_LT = _get(par, "beta_LT", "betaLT")
    beta_LY = _get(par, "beta_LY", "betaLY")

    beta_R0 = _get(par, "beta_R0", "betaR0")
    beta_Rp = _get(par, "beta_Rp", "betaRp")
    beta_RT = _get(par, "beta_RT", "betaRT")
    beta_RY = _get(par, "beta_RY", "betaRY")

    # Peak location coefficients (zeta_* in example JSON)
    c_0 = _get(par, "c_0", "zeta_0", "c0", required=False, default=0.0)
    c_P = _get(par, "c_P", "zeta_p", "cP", required=False, default=0.0)
    c_T = _get(par, "c_T", "zeta_T", "cT", required=False, default=0.0)
    c_Y = _get(par, "c_Y", "zeta_Y", "cY", required=False, default=0.0)

    gamma_0 = _get(par, "gamma_0", "gamma0")
    gamma_P = _get(par, "gamma_P", "gamma_p", "gammaP")
    gamma_T = _get(par, "gamma_T", "gammaT")
    gamma_Y = _get(par, "gamma_Y", "gammaY")

    kappa_t0 = _get(par, "kappa_t0", "kappaT0")
    kappa_tp = _get(par, "kappa_tp", "kappaTp")
    kappa_tT = _get(par, "kappa_tT", "kappaTT")
    kappa_tY = _get(par, "kappa_tY", "kappaTY")

    # ------------------------------------------------------------------
    # Normalized inputs
    # ------------------------------------------------------------------
    Tn = Tu / T_ref
    pn = np.log(p / p_ref)

    # ------------------------------------------------------------------
    # Peak location model (phiS_hat) and normalized equivalence ratio phin
    # ------------------------------------------------------------------
    phiS_hat = phi_Sref * np.exp(c_0 + c_P * pn + c_T * Tn + c_Y * Y)
    phin = np.log(np.maximum(phi / np.maximum(phiS_hat, 1e-12), 1e-12))

    # ------------------------------------------------------------------
    # Effective pressure and chain-termination saturation
    # ------------------------------------------------------------------
    peff = p * (1.0 + eta * Y) ** q_Y
    pc = pc_r * p_ref
    s = peff / (peff + pc)           # saturation fraction in [0,1]
    g = 1.0 + lambda_ * s            # preheat amplification

    # ------------------------------------------------------------------
    # Adiabatic flame temperature Tb(φ) sub-model
    # ------------------------------------------------------------------
    psi_star = phi_Sref / np.maximum(phi_Tref, 1e-12)
    psi = phi / np.maximum(phi_Tref, 1e-12)

    beta_S = (1.0 + m_r) * q_T / 2.0
    A_Tu = A_0 + A_T * Tu

    # Tb at reference peak (phi_Sref) -> dT_star
    u_star = beta_S * np.log(np.maximum(psi_star, 1e-12))
    abs_u_star = np.abs(u_star)
    log_cosh_u_star = abs_u_star + np.log1p(np.exp(-2.0 * abs_u_star)) - np.log(2.0)
    log_ShapeT_star = (1.0 - m_r) / 2.0 * np.log(np.maximum(psi_star, 1e-12)) - (1.0 / q_T) * log_cosh_u_star
    ShapeT_star = np.exp(log_ShapeT_star)

    denom_star = np.maximum(1.0 + alpha_0 * ShapeT_star + tau_Y * Y, 1e-12)
    dT_star = (A_Tu * (1.0 - xi_Y * Y)) * ShapeT_star / denom_star

    # Tb(φ): full φ-dependence -> dT
    u = beta_S * np.log(np.maximum(psi, 1e-12))
    abs_u = np.abs(u)
    log_cosh_u = abs_u + np.log1p(np.exp(-2.0 * abs_u)) - np.log(2.0)
    log_ShapeT = (1.0 - m_r) / 2.0 * np.log(np.maximum(psi, 1e-12)) - (1.0 / q_T) * log_cosh_u
    ShapeT = np.exp(log_ShapeT)

    denom = np.maximum(1.0 + alpha_0 * ShapeT + tau_Y * Y, 1e-12)
    dT = (A_Tu * (1.0 - xi_Y * Y)) * ShapeT / denom

    Tb = Tu + dT

    # ------------------------------------------------------------------
    # Ti at phi(SL_max) i.e. at the same operating point
    # ------------------------------------------------------------------
    D = np.log(np.maximum(s, 1e-12)) - B + a_T * g * Tn
    # Use np.divide to handle D=0 cases robustly without RuntimeWarning
    T_kin = np.divide(-Theta_i, D, out=np.zeros_like(D), where=np.abs(D) >= 1e-12)
    Ti = T_kin - aY * Y

    # ------------------------------------------------------------------
    # Pressure amplitude A_p
    # ------------------------------------------------------------------
    A_p = np.exp(-n_p * pn - n_p2 * pn ** 2)

    # ------------------------------------------------------------------
    # Shape in φ (lean/rich asymmetry, tilt, wing bending)
    # ------------------------------------------------------------------
    f1 = np.ones_like(pn)
    f = np.column_stack((f1, pn, Tn, Y))

    # mL_raw, mR_raw, softplus to keep them positive-ish
    mL_raw = (beta_L0 * f[:, 0] +
              beta_Lp * f[:, 1] +
              beta_LT * f[:, 2] +
              beta_LY * f[:, 3])
    mR_raw = (beta_R0 * f[:, 0] +
              beta_Rp * f[:, 1] +
              beta_RT * f[:, 2] +
              beta_RY * f[:, 3])

    mL = np.log1p(np.exp(mL_raw))
    mR = np.log1p(np.exp(mR_raw))

    # Peak curvature qS_raw, softplus to avoid negative or huge magnitudes
    qS_raw = (gamma_0 * f[:, 0] +
              gamma_P * f[:, 1] +
              gamma_T * f[:, 2] +
              gamma_Y * f[:, 3])
    qS = np.log1p(np.exp(qS_raw))

    # Tilt parameter k_tilt >= 0
    k_raw = (kappa_t0 * f[:, 0] +
             kappa_tp * f[:, 1] +
             kappa_tT * f[:, 2] +
             kappa_tY * f[:, 3])
    k_tilt = np.log1p(np.exp(k_raw))

    # Base asymmetric cosh envelope
    beta_env = 0.5 * (mL + mR) * qS
    u_env = beta_env * phin
    abs_u_env = np.abs(u_env)
    logcosh_env = abs_u_env + np.log1p(np.exp(-2.0 * abs_u_env)) - np.log(2.0)
    ShapeBase = np.exp(0.5 * (mL - mR) * phin - (1.0 / qS) * logcosh_env)

    # Tilt gate (odd, zero value and slope at peak)
    x = phin / w_rl
    sech_x = 1.0 / np.cosh(x)
    h_tilt = np.tanh(x) * (1.0 - sech_x)
    M_tilt = np.exp(k_tilt * h_tilt)

    ShapeS = ShapeBase * M_tilt

    # ------------------------------------------------------------------
    # Collapse factors
    # ------------------------------------------------------------------
    # Global (φ-independent), based on dT_star
    FG_star = np.maximum(Tu + dT_star - delta * np.maximum(Ti, 0.0), 0.0) / np.maximum(dT_star, 1e-6)
    C_cut_star = FG_star ** n

    # φ-dependent collapse correction using dT (full Tb(φ))
    FG = np.maximum(Tu + dT - delta * np.maximum(Ti, 0.0), 0.0) / np.maximum(dT, 1e-6)
    Rphi = FG / np.maximum(FG_star, 1e-6)
    Rphi = np.maximum(Rphi, 0.0)
    C_phi = Rphi ** n_r

    # ------------------------------------------------------------------
    # Kinetics and corrections
    # ------------------------------------------------------------------
    Ti_chem = np.maximum(Ti, 1.0)
    A_chem = S_0 * A_p * np.exp(-Theta_a * (1.0 / Ti_chem - 1.0 / T_ref))
    Theta_th = (Tu / Ti_chem) ** r_th

    SL = A_chem * ShapeS * Theta_th * C_cut_star * C_phi

    extras = {
        "Ti": Ti,
        "Tb": Tb,
        "Tb_star": Tu + dT_star,
        "dT": dT,
        "dT_star": dT_star,
        "s": s,
        "D": D,
        "T_kin": T_kin,
        "A_p": A_p,
        "ShapeS": ShapeS,
    }
    return SL, Ti, Tb, extras