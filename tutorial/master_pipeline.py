import m3Drop as M3Drop
import os
import pickle
import time
import pandas as pd
import matplotlib

# Prevent pop-ups for batch processing
matplotlib.use('Agg')

# ==========================================
#        MASTER CONFIGURATION
# ==========================================
DATASET_BASENAME = "test_data"

# --- TOGGLES (A, B, C) ---
RUN_FEATURE_SELECTION = True
RUN_DIAGNOSTICS       = True  # Set to True to generate Model Comparison & Plots
RUN_NORMALIZATION     = False # Set to True to generate Pearson Residuals (Heavy IO)

# --- FILE PATHS ---
# Input
RAW_DATA_FILE = f"{DATASET_BASENAME}.h5ad"

# Core Intermediates (Shared)
CLEANED_DATA_FILE = f"{DATASET_BASENAME}_cleaned.h5ad"
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
FIT_OUTPUT_FILE   = f"{DATASET_BASENAME}_fit.pkl"

# Branch A: Feature Selection Outputs
HIGH_VAR_OUTPUT_CSV      = f"{DATASET_BASENAME}_4A_high_variance_genes.csv"
COMBINED_DROP_OUTPUT_CSV = f"{DATASET_BASENAME}_4A_combined_dropout_genes.csv"
VOLCANO_PLOT_FILE        = f"{DATASET_BASENAME}_4A_volcano_plot.png"

# Branch B: Diagnostics Outputs
DISP_VS_MEAN_PLOT_FILE = f"{DATASET_BASENAME}_4B_disp_vs_mean.png"
COMPARISON_PLOT_FILE   = f"{DATASET_BASENAME}_4B_NBumiCompareModels.png"

# Branch C: Normalization Outputs
PEARSON_FULL_OUTPUT_FILE   = f"{DATASET_BASENAME}_4C_pearson_residuals.h5ad"
PEARSON_APPROX_OUTPUT_FILE = f"{DATASET_BASENAME}_4C_pearson_residuals_approx.h5ad"


# ==========================================
#        MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"############################################################")
    print(f"   M3DROP+ MASTER PIPELINE: {DATASET_BASENAME}")
    print(f"############################################################\n")

    # ---------------------------------------------------------
    # STAGE 1: DATA CLEANING (Universal)
    # ---------------------------------------------------------
    print(">>> STAGE 1: DATA CLEANING")
    if not os.path.exists(CLEANED_DATA_FILE):
        M3Drop.ConvertDataSparseGPU(
            input_filename=RAW_DATA_FILE,
            output_filename=CLEANED_DATA_FILE
        )
    else:
        print(f"STATUS: Found existing file '{CLEANED_DATA_FILE}'. Skipping.")
    print("------------------------------------------------------------\n")

    # ---------------------------------------------------------
    # STAGE 2: STATISTICS (Universal)
    # ---------------------------------------------------------
    print(">>> STAGE 2: STATISTICS CALCULATION")
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = M3Drop.hidden_calc_valsGPU(filename=CLEANED_DATA_FILE)
        with open(STATS_OUTPUT_FILE, 'wb') as f:
            pickle.dump(stats, f)
        print(f"STATUS: Statistics saved to '{STATS_OUTPUT_FILE}'.")
    else:
        print(f"STATUS: Loading statistics from '{STATS_OUTPUT_FILE}'...")
        with open(STATS_OUTPUT_FILE, 'rb') as f:
            stats = pickle.load(f)
    print("------------------------------------------------------------\n")

    # ---------------------------------------------------------
    # STAGE 3: MODEL FITTING (Universal)
    # ---------------------------------------------------------
    print(">>> STAGE 3: MODEL FITTING")
    if not os.path.exists(FIT_OUTPUT_FILE):
        fit_results = M3Drop.NBumiFitModelGPU(
            cleaned_filename=CLEANED_DATA_FILE,
            stats=stats
        )
        with open(FIT_OUTPUT_FILE, 'wb') as f:
            pickle.dump(fit_results, f)
        print(f"STATUS: Fit results saved to '{FIT_OUTPUT_FILE}'.")
    else:
        print(f"STATUS: Loading fit results from '{FIT_OUTPUT_FILE}'...")
        with open(FIT_OUTPUT_FILE, 'rb') as f:
            fit_results = pickle.load(f)
    print("------------------------------------------------------------\n")

    # ---------------------------------------------------------
    # STAGE 4A: FEATURE SELECTION
    # ---------------------------------------------------------
    if RUN_FEATURE_SELECTION:
        print(">>> STAGE 4A: FEATURE SELECTION")
        
        # Method 1: High Variance
        if not os.path.exists(HIGH_VAR_OUTPUT_CSV):
            print("   Running High Variance Selection...")
            high_var_genes = M3Drop.NBumiFeatureSelectionHighVarGPU(fit=fit_results)
            high_var_genes.to_csv(HIGH_VAR_OUTPUT_CSV, index=False)
        else:
            print(f"   Skipping High Variance (Output exists: {HIGH_VAR_OUTPUT_CSV})")

        # Method 2: Combined Dropout
        if not os.path.exists(COMBINED_DROP_OUTPUT_CSV):
            print("   Running Combined Dropout Selection...")
            combined_drop_genes = M3Drop.NBumiFeatureSelectionCombinedDropGPU(
                fit=fit_results,
                cleaned_filename=CLEANED_DATA_FILE
            )
            combined_drop_genes.to_csv(COMBINED_DROP_OUTPUT_CSV, index=False)
            
            # Volcano Plot
            if not os.path.exists(VOLCANO_PLOT_FILE):
                M3Drop.NBumiCombinedDropVolcanoGPU(
                    results_df=combined_drop_genes,
                    plot_filename=VOLCANO_PLOT_FILE
                )
        else:
            print(f"   Skipping Combined Dropout (Output exists: {COMBINED_DROP_OUTPUT_CSV})")
        print("------------------------------------------------------------\n")

    # ---------------------------------------------------------
    # STAGE 4B: DIAGNOSTICS
    # ---------------------------------------------------------
    if RUN_DIAGNOSTICS:
        print(">>> STAGE 4B: DIAGNOSTICS")
        
        # 1. Dispersion vs Mean Plot
        if not os.path.exists(DISP_VS_MEAN_PLOT_FILE):
            M3Drop.NBumiPlotDispVsMeanGPU(
                fit=fit_results,
                plot_filename=DISP_VS_MEAN_PLOT_FILE
            )
        
        # 2. Model Comparison (Optimized In-Memory)
        if not os.path.exists(COMPARISON_PLOT_FILE):
            print("   Running Model Comparison (Optimized)...")
            M3Drop.NBumiCompareModelsGPU(
                raw_filename=RAW_DATA_FILE,
                cleaned_filename=CLEANED_DATA_FILE,
                stats=stats,
                fit_adjust=fit_results,
                plot_filename=COMPARISON_PLOT_FILE
            )
        else:
            print(f"   Skipping Comparison (Plot exists: {COMPARISON_PLOT_FILE})")
        print("------------------------------------------------------------\n")

    # ---------------------------------------------------------
    # STAGE 4C: NORMALIZATION
    # ---------------------------------------------------------
    if RUN_NORMALIZATION:
        print(">>> STAGE 4C: NORMALIZATION")
        
        # 1. Full Pearson Residuals
        if not os.path.exists(PEARSON_FULL_OUTPUT_FILE):
            print("   Generating Full Pearson Residuals...")
            M3Drop.NBumiPearsonResidualsGPU(
                cleaned_filename=CLEANED_DATA_FILE,
                fit_filename=FIT_OUTPUT_FILE,
                output_filename=PEARSON_FULL_OUTPUT_FILE
            )
        
        # 2. Approx Pearson Residuals
        if not os.path.exists(PEARSON_APPROX_OUTPUT_FILE):
            print("   Generating Approximate Pearson Residuals...")
            M3Drop.NBumiPearsonResidualsApproxGPU(
                cleaned_filename=CLEANED_DATA_FILE,
                stats_filename=STATS_OUTPUT_FILE,
                output_filename=PEARSON_APPROX_OUTPUT_FILE
            )
        print("------------------------------------------------------------\n")

    total_time = time.time() - pipeline_start_time
    print(f"############################################################")
    print(f" PIPELINE COMPLETE: {total_time/60:.2f} minutes")
    print(f"############################################################")
