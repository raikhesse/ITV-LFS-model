#!/usr/bin/env python3
"""
Generate a CSV with all combinations of Yegr, T_K, p_bar, phi, and Xbl.

Edit the *_values lists below to define your grid.
"""

import itertools
import csv
import numpy as np

# ---- EDIT THESE LISTS TO DEFINE YOUR GRID ------------------------

# Example: calculate SL for blends at atmospheric conditions without 
Yegr_values = np.linspace(0.0, 0.3, num=4)       # external dilution 
T_K_values  = np.linspace(600.0, 1000.0, num=5)   # temperature in K
p_bar_values = np.linspace(1.0, 150.0, num=16)    # pressure in bar
phi_values  = np.linspace(0.4, 2.0, num=33)       # equivalence ratio
Xbl_values = np.linspace(0.0, 1.0, num=21)        # blending ratio


# Output file name
output_file = "../OUTPUT/cases.csv"

# ---- NO CHANGES NEEDED BELOW THIS LINE ---------------------------

def main():
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tu_K", "p_bar", "Yegr", "phi", "Xf_H2"])

        for Yegr, T_K, p_bar, phi, Xbl in itertools.product(
            Yegr_values, T_K_values, p_bar_values, phi_values, Xbl_values
        ):
            writer.writerow([T_K, p_bar, Yegr, phi, Xbl])

    n = (len(Yegr_values) *
         len(T_K_values) *
         len(p_bar_values) *
         len(phi_values) * 
         len(Xbl_values))
    print(f"Wrote {n} rows to '{output_file}'.")

if __name__ == "__main__":
    main()