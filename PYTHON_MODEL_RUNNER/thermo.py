import numpy as np
import json

def fuel_mass_fractions_from_molar_x(x_h2, M_H2=2.016, M_CH4=16.04):
    """
    Convert fuel-side molar fraction X_f,H2 to mass fractions of CH4 and H2
    in the fuel stream (Y_f,CH4, Y_f,H2), such that Y_f,CH4 + Y_f,H2 = 1.
    """
    x_h2 = np.asarray(x_h2, dtype=float)
    x_h2 = np.clip(x_h2, 0.0, 1.0)
    x_ch4 = 1.0 - x_h2

    denom = x_h2 * M_H2 + x_ch4 * M_CH4
    Y_H2 = x_h2 * M_H2 / denom
    Y_CH4 = x_ch4 * M_CH4 / denom
    return Y_CH4, Y_H2


class ExhaustSurrogate:
    """
    Tabulated surrogate for external dilution as function of Xf_H2.

    Expected JSON structure (minimal):
    {
      "Xf_H2": [0.0, 0.1, ..., 1.0],
      "W_ed":  [27.53, ..., 24.68]
    }
    All W_ed are in g/mol.
    """

    def __init__(self, data):
        self.Xf_H2 = np.array(data["Xf_H2"], dtype=float)
        self.W_ed = np.array(data["W_ed"], dtype=float)
        if self.Xf_H2.shape != self.W_ed.shape:
            raise ValueError("Xf_H2 and W_ed must have the same length in EGR table.")

        # sort by Xf_H2 to guarantee monotonic interpolation
        order = np.argsort(self.Xf_H2)
        self.Xf_H2 = self.Xf_H2[order]
        self.W_ed = self.W_ed[order]

    def W_ed_from_Xf(self, Xf_H2):
        X = np.asarray(Xf_H2, dtype=float)
        X_clipped = np.clip(X, self.Xf_H2.min(), self.Xf_H2.max())
        return np.interp(X_clipped, self.Xf_H2, self.W_ed)


def load_exhaust_surrogate(path):
    with open(path, "r") as f:
        data = json.load(f)
    return ExhaustSurrogate(data)


def compute_rho_u(Tu, p_bar, Yegr, phi, Xf_H2, ed_table):
    """
    Compute unburned density rho_u [kg/m^3] for given Tu, p_bar, Yegr, phi,
    and fuel molar H2 fraction Xf_H2, using the exhaust-surrogate table.
    """
    Ru = 8.314462618  # J/(mol·K)

    # ---- constants (g/mol) ----
    M_CH4 = 16.04
    M_H2 = 2.016
    M_O2 = 31.999
    M_N2 = 28.014

    X_O2_air = 0.2094
    X_N2_air = 1.0 - X_O2_air
    M_air = X_O2_air * M_O2 + X_N2_air * M_N2

    Tu = np.asarray(Tu, dtype=float)
    p_b = np.asarray(p_bar, dtype=float)
    Ye = np.asarray(Yegr, dtype=float)
    phi = np.asarray(phi, dtype=float)
    Xf = np.asarray(Xf_H2, dtype=float)

    # fuel mixture molar mass
    M_f = (1.0 - Xf) * M_CH4 + Xf * M_H2  # g/mol

    # stoichiometric and actual O2 requirement per 1 kmol of fuel mixture
    nu_O2_stoich = 2.0 - 1.5 * Xf
    phi_safe = np.maximum(phi, 1e-6)
    nu_O2 = nu_O2_stoich / phi_safe

    # moles of air per 1 kmol of fuel mixture
    n_air = nu_O2 / X_O2_air

    m_fuel = M_f
    m_air = n_air * M_air
    m_fresh = m_fuel + m_air

    Y_f_total = (1.0 - Ye) * (m_fuel / m_fresh)
    Y_air_total = (1.0 - Ye) * (m_air / m_fresh)

    # EGR molar mass from table
    W_ed = ed_table.W_ed_from_Xf(Xf)  # g/mol

    inv_M_mix = (Y_f_total / M_f +
                 Y_air_total / M_air +
                 Ye / W_ed)
    M_mix = 1.0 / np.maximum(inv_M_mix, 1e-12)  # g/mol

    return (p_b * 1e5) * (M_mix / 1000.0) / (Ru * Tu)  # kg/m^3
