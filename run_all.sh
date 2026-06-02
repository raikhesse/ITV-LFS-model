#!/bin/bash

# Exit on any error
set -e

echo "=== RWTH LFS Model Runner: Full Workflow ==="

# 1. Setup Environment (assumes venv is already created as per README)
# If .env exists, we use it. Otherwise we use the system python.
if [ -d "PYTHON_MODEL_RUNNER/.env" ]; then
    PYTHON_CMD="./.env/bin/python"
    echo "Using virtual environment: PYTHON_MODEL_RUNNER/.env"
else
    PYTHON_CMD="python"
    echo "Using system python (make sure dependencies are installed: numpy, pandas, scipy, matplotlib, seaborn)"
fi

# 0. Move into the runner directory
cd PYTHON_MODEL_RUNNER

# 1. Generate Input Grid
echo "--- Step 1: Generating Input Conditions ---"
$PYTHON_CMD generate_grid.py

# 2. Run the Model for the validation data
echo "--- Step 2.1: Running LFS Model for the full input data ---"
$PYTHON_CMD run_model.py -m val --blend \
    -p "../MODEL_INPUT/FitPar_Present study_0_00bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_40bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_80bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_95bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_1_00bld_CH4_H2_global.json" \
    -i "../VAL_DATA/input_data_full.csv" \
    --xf-h2-col Xf_H2 \
    --ed-table "../MODEL_INPUT/exhaust_surrogate.json" \
    -o "../OUTPUT/results_full.csv"

echo "--- Step 2.2: Running LFS Model for the anchor points ---"
$PYTHON_CMD run_model.py -m val --blend \
    -p "../MODEL_INPUT/FitPar_Present study_0_00bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_40bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_80bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_0_95bld_CH4_H2_global.json" \
       "../MODEL_INPUT/FitPar_Present study_1_00bld_CH4_H2_global.json" \
    -i "../VAL_DATA/input_data_anchor.csv" \
    --xf-h2-col Xf_H2 \
    --ed-table "../MODEL_INPUT/exhaust_surrogate.json" \
    -o "../OUTPUT/results_anchor.csv"

# 3. Run the Model for the grid (Blending + Uncertainty Estimation)
echo "--- Step 3: Running LFS Model (4-Point Blending + Uncertainty) ---"
$PYTHON_CMD run_model.py -m val --blend \
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

# 4. Generate Visualization Plots
echo "--- Step 4: Generating Plots ---"

# Parity Plot (Log-Log Density)
echo "Generating parity plots..."
$PYTHON_CMD plot_accuracy.py -i "../OUTPUT/results_full.csv" \
    --parity "../OUTPUT/parity_robust_full.pdf" --log --robust

$PYTHON_CMD plot_accuracy.py -i "../OUTPUT/results_anchor.csv" \
    --parity "../OUTPUT/parity_anchor.pdf" --log

$PYTHON_CMD plot_accuracy.py -i "../OUTPUT/results_cases.csv" \
    --parity "../OUTPUT/parity_robust_cases.pdf" --log --robust  

# Heatmap: EGR vs Phi
echo "Generating EGR heatmap..."
$PYTHON_CMD plot_accuracy.py -i "../OUTPUT/results_cases.csv" \
    --heatmap "../OUTPUT/map_phi_Yegr.pdf" --x-axis phi --y-axis Yegr

# Heatmap: Hydrogen vs Phi
echo "Generating Hydrogen heatmap..."
$PYTHON_CMD plot_accuracy.py -i "../OUTPUT/results_cases.csv" \
    --heatmap "../OUTPUT/map_phi_XfH2.pdf" --x-axis phi --y-axis Xf_H2

# Heatmap: temperature vs Phi
echo "Generating Temperature heatmap..."
$PYTHON_CMD plot_accuracy.py -i "../OUTPUT/results_cases.csv" \
    --heatmap "../OUTPUT/map_phi_Tu_K.pdf" --x-axis phi --y-axis Tu_K

# Heatmap: pressure vs Phi
echo "Generating Pressure heatmap..."
$PYTHON_CMD plot_accuracy.py -i "../OUTPUT/results_cases.csv" \
    --heatmap "../OUTPUT/map_phi_p_bar.pdf" --x-axis phi --y-axis p_bar

# Training Range Plot
echo "Generating training range plot..."
$PYTHON_CMD plot_training_range.py -i "../OUTPUT/results_full.csv" \
    -o "../OUTPUT/training_range.png"

# Plotting SL comparison for robust data range
$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_SL_phi_298K_1bar_EGR0_0.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod SL_model_cm_s \
        --y-sim SL_cm_s \
        --hue Xf_H2 \
        --filter "Tu_K == 298 and p_bar == 1.01325 and Yegr == 0.0" --robust

$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_SL_phi_298K_1bar_EGR0_3.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod SL_model_cm_s \
        --y-sim SL_cm_s \
        --hue Xf_H2 \
        --filter "Tu_K == 298 and p_bar == 1.01325 and Yegr == 0.3" --robust

$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_SL_phi_687K_20bar_EGR0_0.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod SL_model_cm_s \
        --y-sim SL_cm_s \
        --hue Xf_H2 \
        --filter "Tu_K == 687 and p_bar == 20.0 and Yegr == 0.0" --robust

$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_SL_phi_687K_20bar_EGR0_3.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod SL_model_cm_s \
        --y-sim SL_cm_s \
        --hue Xf_H2 \
        --filter "Tu_K == 687 and p_bar == 20.0 and Yegr == 0.3" --robust

$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_SL_phi_971K_80bar_EGR0_0.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod SL_model_cm_s \
        --y-sim SL_cm_s \
        --hue Xf_H2 \
        --filter "Tu_K == 971.4 and p_bar == 80.0 and Yegr == 0.0" --robust

$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_SL_phi_971K_80bar_EGR0_3.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod SL_model_cm_s \
        --y-sim SL_cm_s \
        --hue Xf_H2 \
        --filter "Tu_K == 971.4 and p_bar == 80.0 and Yegr == 0.3" --robust

# Plotting Tb comparison for robust data range
$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_Tb_phi_298K_1bar_EGR0_0.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod Tb_model_K \
        --y-sim Tb_K \
        --hue Xf_H2 \
        --filter "Tu_K == 298 and p_bar == 1.01325 and Yegr == 0.0" --robust

$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_Tb_phi_298K_1bar_EGR0_3.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod Tb_model_K \
        --y-sim Tb_K \
        --hue Xf_H2 \
        --filter "Tu_K == 298 and p_bar == 1.01325 and Yegr == 0.3" --robust

$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_Tb_phi_687K_20bar_EGR0_0.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod Tb_model_K \
        --y-sim Tb_K \
        --hue Xf_H2 \
        --filter "Tu_K == 687 and p_bar == 20.0 and Yegr == 0.0" --robust

$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_Tb_phi_687K_20bar_EGR0_3.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod Tb_model_K \
        --y-sim Tb_K \
        --hue Xf_H2 \
        --filter "Tu_K == 687 and p_bar == 20.0 and Yegr == 0.3" --robust

$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_Tb_phi_971K_80bar_EGR0_0.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod Tb_model_K \
        --y-sim Tb_K \
        --hue Xf_H2 \
        --filter "Tu_K == 971.4 and p_bar == 80.0 and Yegr == 0.0" --robust

$PYTHON_CMD plot_comparison.py \
        -i "../OUTPUT/results_full.csv" \
        -o "../OUTPUT/compare_Tb_phi_971K_80bar_EGR0_3.pdf" \
        --x phi --xlim 0.25 4.0\
        --y-mod Tb_model_K \
        --y-sim Tb_K \
        --hue Xf_H2 \
        --filter "Tu_K == 971.4 and p_bar == 80.0 and Yegr == 0.3" --robust

echo "=== Workflow Complete. Results are in the 'OUTPUT' directory. ==="
