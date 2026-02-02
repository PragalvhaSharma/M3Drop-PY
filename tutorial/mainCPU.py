import os
import pickle
import time
import pandas as pd
import matplotlib
import sys

# Prevent pop-ups for batch processing (headless mode)
matplotlib.use('Agg')

# ==========================================
#        IMPORTS & SETUP (STANDALONE)
# ==========================================
try:
    import m3Drop as M3Drop
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import the 'm3Drop' package.\nDetails: {e}")
    sys.exit(1)

# ==========================================
#        MASTER CONFIGURATION
# ==========================================
DATASET_BASENAME = "test_data2"
CONTROL_MODE = "auto" 
MANUAL_TARGET = 3000

# --- TOGGLES ---
RUN_FEATURE_SELECTION = True
RUN_DIAGNOSTICS       = True
RUN_NORMALIZATION     = True

# --- FILE PATHS ---
RAW_DATA_FILE    = f"{DATASET_BASENAME}.h5ad"
MASK_OUTPUT_FILE = f"{DATASET_BASENAME}_mask.pkl" 
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
FIT_OUTPUT_FILE   = f"{DATASET_BASENAME}_fit.pkl"

HIGH_VAR_OUTPUT_CSV      = f"{DATASET_BASENAME}_4A_high_variance_genes.csv"
COMBINED_DROP_OUTPUT_CSV = f"{DATASET_BASENAME}_4A_combined_dropout_genes.csv"
VOLCANO_PLOT_FILE        = f"{DATASET_BASENAME}_4A_volcano_plot.png"

DISP_VS_MEAN_PLOT_FILE = f"{DATASET_BASENAME}_4B_disp_vs_mean.png"
COMPARISON_PLOT_FILE   = f"{DATASET_BASENAME}_4B_NBumiCompareModels.png"

PEARSON_FULL_OUTPUT_FILE   = f"{DATASET_BASENAME}_4C_pearson_residuals.h5ad"
PEARSON_APPROX_OUTPUT_FILE = f"{DATASET_BASENAME}_4C_pearson_residuals_approx.h5ad"

# [NEW] Visualization Outputs
PLOT_SUMMARY_FILE = f"{DATASET_BASENAME}_4C_Summary_Diagnostics.png"
PLOT_DETAIL_FILE  = f"{DATASET_BASENAME}_4C_Residual_Shrinkage.png"


# ==========================================
#        MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"############################################################")
    print(f"   M3DROP mainCPU.py: {DATASET_BASENAME}")
    print(f"   Mode: {CONTROL_MODE.upper()}")
    print(f"############################################################\n")

    # STAGE 1: MASK
    print(">>> STAGE 1: MASK GENERATION")
    if not os.path.exists(MASK_OUTPUT_FILE):
        M3Drop.ConvertDataSparseCPU(
            input_filename=RAW_DATA_FILE,
            output_mask_filename=MASK_OUTPUT_FILE,
            mode=CONTROL_MODE,
            manual_target=MANUAL_TARGET
        )
    else:
        print(f"STATUS: Found existing mask. Skipping.")
    print("------------------------------------------------------------\n")

    # STAGE 2: STATS
    print(">>> STAGE 2: STATISTICS CALCULATION")
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = M3Drop.hidden_calc_valsCPU(
            filename=RAW_DATA_FILE,     
            mask_filename=MASK_OUTPUT_FILE, 
            mode=CONTROL_MODE,
            manual_target=MANUAL_TARGET
        )
        with open(STATS_OUTPUT_FILE, 'wb') as f: pickle.dump(stats, f)
    else:
        print(f"STATUS: Loading statistics...")
        with open(STATS_OUTPUT_FILE, 'rb') as f: stats = pickle.load(f)
    print("------------------------------------------------------------\n")

    # STAGE 3: FITTING
    print(">>> STAGE 3: MODEL FITTING")
    if not os.path.exists(FIT_OUTPUT_FILE):
        fit_results = M3Drop.NBumiFitModelCPU(
            raw_filename=RAW_DATA_FILE,     
            mask_filename=MASK_OUTPUT_FILE,
            stats=stats,
            mode=CONTROL_MODE,
            manual_target=MANUAL_TARGET
        )
        with open(FIT_OUTPUT_FILE, 'wb') as f: pickle.dump(fit_results, f)
    else:
        print(f"STATUS: Loading fit results...")
        with open(FIT_OUTPUT_FILE, 'rb') as f: fit_results = pickle.load(f)
    print("------------------------------------------------------------\n")

    # STAGE 4A: FEATURE SELECTION
    if RUN_FEATURE_SELECTION:
        print(">>> STAGE 4A: FEATURE SELECTION")
        if not os.path.exists(HIGH_VAR_OUTPUT_CSV):
            M3Drop.NBumiFeatureSelectionHighVarCPU(fit=fit_results).to_csv(HIGH_VAR_OUTPUT_CSV, index=False)

        if not os.path.exists(COMBINED_DROP_OUTPUT_CSV):
            cdf = M3Drop.NBumiFeatureSelectionCombinedDropCPU(
                fit=fit_results,
                raw_filename=RAW_DATA_FILE,
                mode=CONTROL_MODE,
                manual_target=MANUAL_TARGET
            )
            cdf.to_csv(COMBINED_DROP_OUTPUT_CSV, index=False)
            if not os.path.exists(VOLCANO_PLOT_FILE):
                M3Drop.NBumiCombinedDropVolcanoCPU(results_df=cdf, plot_filename=VOLCANO_PLOT_FILE)
        print("------------------------------------------------------------\n")

    # STAGE 4B: DIAGNOSTICS
    if RUN_DIAGNOSTICS:
        print(">>> STAGE 4B: DIAGNOSTICS")
        if not os.path.exists(DISP_VS_MEAN_PLOT_FILE):
            M3Drop.NBumiPlotDispVsMeanCPU(fit=fit_results, plot_filename=DISP_VS_MEAN_PLOT_FILE)
        
        if not os.path.exists(COMPARISON_PLOT_FILE):
            M3Drop.NBumiCompareModelsCPU(
                raw_filename=RAW_DATA_FILE,    
                mask_filename=MASK_OUTPUT_FILE, 
                stats=stats,
                fit_adjust=fit_results,
                plot_filename=COMPARISON_PLOT_FILE,
                mode=CONTROL_MODE,
                manual_target=MANUAL_TARGET
            )
        print("------------------------------------------------------------\n")

    # STAGE 4C: NORMALIZATION (Updated for Sidecar Viz)
    if RUN_NORMALIZATION:
        print(">>> STAGE 4C: NORMALIZATION")
        
        # Check if PLOTS exist (force run if plots are missing, even if h5ad exists)
        if (not os.path.exists(PEARSON_FULL_OUTPUT_FILE) or 
            not os.path.exists(PEARSON_APPROX_OUTPUT_FILE) or
            not os.path.exists(PLOT_SUMMARY_FILE) or
            not os.path.exists(PLOT_DETAIL_FILE)):
            
            M3Drop.NBumiPearsonResidualsCombinedCPU(
                raw_filename=RAW_DATA_FILE,
                mask_filename=MASK_OUTPUT_FILE,
                fit_filename=FIT_OUTPUT_FILE,
                stats_filename=STATS_OUTPUT_FILE,
                output_filename_full=PEARSON_FULL_OUTPUT_FILE,
                output_filename_approx=PEARSON_APPROX_OUTPUT_FILE,
                # [NEW] Pass the plot filenames
                plot_summary_filename=PLOT_SUMMARY_FILE,
                plot_detail_filename=PLOT_DETAIL_FILE,
                mode=CONTROL_MODE,
                manual_target=MANUAL_TARGET
            )
        else:
            print("   Skipping Normalization (All Outputs & Plots exist)")
            
        print("------------------------------------------------------------\n")

    total_time = time.time() - pipeline_start_time
    print(f"PIPELINE COMPLETE: {total_time/60:.2f} minutes")
