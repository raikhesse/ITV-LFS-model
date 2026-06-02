# ITV-LFS-model
[![License: CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code_License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**ITV-LFS-model** 

Additional supplementary material to the submission of "Physics-guided laminar flame speed correlation for methane–hydrogen–air mixtures with varying dilution."

Authors: Raik Hesse, Christian Schwenzer, Roman Glaznev, Florence Cameron, Heinz Pitsch, Joachim Beeckmann

Affiliation: Institute for Combustion Technology, RWTH Aachen University,
Templergraben 64, 52056 Aachen, Germany

Corresponding author: Raik Hesse 

---

## License

This repository uses a **Dual License** structure:
- **Data & Models (`MODEL_INPUT/`, `VAL_DATA/`)**: Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).
- **Software Code (`PYTHON_MODEL_RUNNER/`)**: Licensed under the [MIT License](PYTHON_MODEL_RUNNER/LICENSE).

---

## How to Cite

If you use this code in your research, please cite our preprint (currently under review at the *International Journal of Hydrogen Energy*):

```bibtex
@misc{Hesse2026_hydrogen_preprint,
      title={Physics-guided laminar flame speed correlation for methane-hydrogen-air mixtures with varying dilution}, 
      author={Raik Hesse and Christian Schwenzer and Roman Glaznev and Florence Cameron and Heinz Pitsch and Joachim Beeckmann},
      year={2026},
      eprint={2603.26568},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn},
      note={submitted to Int. J. Hydrogen Energy}
}
```
*(Note: This citation will be updated with the final DOI once the paper is published.)*

---

## 1. Repository Structure

1. **[`MODEL_INPUT`](MODEL_INPUT/)**  
   Contains model parameter sets (`FitPar_*.json`) and the EGR surrogate composition (`exhaust_surrogate.json`).

2. **[`OUTPUT`](OUTPUT/)**  
   Recommended directory for storing generated CSV results and visualization plots (PDF/PNG).

3. **[`PYTHON_MODEL_RUNNER`](PYTHON_MODEL_RUNNER/)**  
   The core Python implementation, including the model orchestrator, physics modules, and plotting utilities.

4. **[`VAL_DATA`](VAL_DATA/)**  
   Experimental validation data used for accuracy assessment and uncertainty mapping. The outer bounds of the validation dataset are:
   - **$T_u$**: 298.0 K to 1098.6 K
   - **$p$**: 1.013 bar to 150.0 bar
   - **$\phi$**: 0.25 to 4.0
   - **$Y_{ed}$**: 0.0 to 0.3
   - **$X_{\mathrm{f,H}_2}$**: 0.0 to 1.0

---

## 2. Typical Run Order

You can run the entire workflow (grid generation, model evaluation on validation data, grid prediction, and plotting) using the provided automated script:

```bash
./run_all.sh
```

Alternatively, follow the manual steps below. All Python commands should be executed from within the `PYTHON_MODEL_RUNNER` directory to ensure proper module discovery.

### 2.1. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
pip install numpy pandas scipy matplotlib seaborn
```

### 2.2. Generating Input Conditions
Use `generate_grid.py` to create a CSV file containing the operating conditions (Tu, p, phi, Yegr) you wish to simulate. By default, the output is saved to the `OUTPUT` directory:
```bash
cd PYTHON_MODEL_RUNNER
python generate_grid.py
```
*Creates `../OUTPUT/cases.csv`.*

### 2.3. Running the Model (LFS Prediction)
The `run_model.py` script is the primary interface. It supports single-anchor calculations or multi-anchor blending.

#### Example: Model evaluation on full validation dataset
```bash
python run_model.py -m val --blend \
    -p "../MODEL_INPUT/FitPar_Present study_0_00bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_40bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_80bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_95bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_1_00bld_CH4_H2_global.json" \
    -i "../VAL_DATA/input_data_full.csv" \
    --xf-h2-col Xf_H2 \
    --ed-table "../MODEL_INPUT/exhaust_surrogate.json" \
    -o "../OUTPUT/results_full.csv"
```

#### Example: Grid prediction with Uncertainty Mapping
```bash
python run_model.py -m val --blend \
    -p "../MODEL_INPUT/FitPar_Present study_0_00bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_40bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_80bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_95bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_1_00bld_CH4_H2_global.json" \
    -i "../OUTPUT/cases.csv" \
    --xf-h2-col Xf_H2 \
    --ed-table "../MODEL_INPUT/exhaust_surrogate.json" \
    --estimate-uncert \
    --val-data "../VAL_DATA/input_data_full.csv" \
    -o "../OUTPUT/results_cases.csv"
```

---

## 3. Visualization and Accuracy Assessment

The `plot_accuracy.py` utility generates plots to assess model accuracy and uncertainty.

### 3.1. Heatmaps (Smooth Grid Interpolation)
Visualize **Estimated Uncertainty [%]** or **Relative Error** across the operating space:
```bash
# EGR Fraction vs Equivalence Ratio
python plot_accuracy.py -i "../OUTPUT/results_cases.csv" --heatmap "../OUTPUT/map_phi_Yegr.pdf" --x-axis phi --y-axis Yegr

# Hydrogen Fraction vs Equivalence Ratio
python plot_accuracy.py -i "../OUTPUT/results_cases.csv" --heatmap "../OUTPUT/map_phi_XfH2.pdf" --x-axis phi --y-axis Xf_H2
```

### 3.2. Parity Plots (Binned Density)
Compare model predictions against validation data or visualize expected spread for grid data:
```bash
# Log-Log Parity plot for full validation dataset
python plot_accuracy.py -i "../OUTPUT/results_full.csv" --parity "../OUTPUT/parity_full.pdf" --log

# Expected Parity (Uncertainty-aware) for grid results
python plot_accuracy.py -i "../OUTPUT/results_cases.csv" --parity "../OUTPUT/parity_cases.pdf" --log

# Expected Parity (Uncertainty-aware) for grid results showing only the robust region
python plot_accuracy.py -i "../OUTPUT/results_cases.csv" --parity "../OUTPUT/parity_robust_cases.pdf" --log --robust
```

### 3.3. Trend Comparisons
The `plot_comparison.py` utility generates 2D functional trend comparisons (e.g., $S_L$ or $T_b$ vs. $\phi$). It visually distinguishes direct anchor evaluations (solid lines) from interpolated blends (dashed lines).

```bash
# Flame temperature vs phi, grouped by Hydrogen fraction, filtering for the robust regime
python plot_comparison.py \
    -i "../OUTPUT/results_full.csv" \
    -o "../OUTPUT/compare_Tb_phi_298K_1bar.pdf" \
    --x phi --xlim 0.5 1.8 \
    --y-mod Tb_model_K \
    --y-sim Tb_K \
    --hue Xf_H2 \
    --filter "Tu_K == 298.0 and p_bar == 1.01325 and Yegr == 0.0" \
    --robust
```

---

## 4. Model Outputs
The runner computes the following for each point:
- `SL_model_cm_s`: Laminar flame speed $S_L$ [cm/s]
- `Tb_model_K`: Adiabatic flame temperature $T_b$ [K]
- `Ti_model_K`: Inner-layer temperature $T_i$ [K]
- `SL_uncert_rel_pct`: Estimated relative uncertainty [%]
- `SL_uncert_abs_cm_s`: Estimated absolute uncertainty [cm/s]
- `out_of_domain`: Boolean flag for conditions far from validation data.
- `is_robust`: Boolean flag indicating if the point is within the high-confidence domain.

---

## 5. Python Modules Overview

The `PYTHON_MODEL_RUNNER/` directory contains several modular scripts designed to separate physics logic from data orchestration and visualization:

*   **`run_model.py`**: The main execution orchestrator. Parses command-line arguments, loads JSON parameter sets, applies the adaptive blending logic across batch CSV inputs, filters for robust domains, and triggers uncertainty estimations.
*   **`sl_ti_model.py`**: Contains the core physics formulation (`sl_ti_model()`). It implements the global asymptotic laminar flame speed correlation and the associated inner-layer temperature ($T_i$) equations based on the provided parameter set.
*   **`blending.py`**: Handles all interpolation logic between distinct fuel blends. Contains `chen_blend_massflux()` for the adaptive multi-anchor (e.g., 4-point) Chen-type blending of $S_L$, and `interpolate_property_piecewise()` for the linear interpolation of thermodynamic properties like $T_b$ and $T_i$.
*   **`thermo.py`**: Manages thermodynamic calculations, specifically computing unburned mixture densities (`compute_rho_u()`) and converting molar fractions to mass fractions using the provided EGR surrogate data (`exhaust_surrogate.json`).
*   **`uncertainty.py`**: Implements the `UncertaintyEstimator` class. It uses nearest-neighbor interpolation (`NearestNDInterpolator`) in a normalized 5D space to map validation errors to new predictive grids, generating "confidence bands" and detecting out-of-domain (OOD) queries.
*   **`generate_grid.py`**: A utility script to generate a multi-dimensional Cartesian grid (CSV) of operating conditions ($T_u, p, \phi, Y_{ed}, X_{\mathrm{f,H}_2}$) for large-scale model predictions.
*   **`plot_accuracy.py`**: A visualization tool for generating publication-ready parity plots (binned log-density) and smooth heatmaps (`tricontourf`/`griddata` interpolation) of model errors and uncertainties.
*   **`plot_comparison.py`**: A visualization tool for generating 2D functional trend comparisons (e.g., $S_L$ or $T_b$ vs. $\phi$), grouping data by fuel blend with distinct markers and line styles (solid for anchors, dashed for blends).
*   **`plot_training_range.py`**: Generates a scatter plot mapping the availability of validation data and the model's core robust region across the $\phi$ vs $X_{\mathrm{f,H}_2}$ domain.
