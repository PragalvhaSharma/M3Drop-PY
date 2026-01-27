import os
import pickle
import time
import pandas as pd
import matplotlib
import sys

# Prevent pop-ups for batch processing
matplotlib.use('Agg')

# ==========================================
#        PACKAGE IMPORTS
# ==========================================
try:
    # Importing submodules from the M3Drop package
    from M3Drop import CoreGPU
    from M3Drop import DiagnosticsGPU
    from M3Drop import NormalizationGPU
    
    # Note: We do not need to import ControlDevice directly here, 
    # as it is handled internally by the Core/Diagnostics modules.
    
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import M3Drop package.")
    print(f"Ensure the 'M3Drop' folder is in the same directory or installed in your Python environment.")
    print(f"Details: {e}")
    sys.exit(1)
    
# ==========================================
#        MASTER CONFIGURATION
# ==========================================
DATASET_BASENAME = "test_data2"

# Modes: "auto" (L3 Cache Optimized) | "manual" (User Defined)
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


# ==========================================
#        MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"############################################################")
    print(f"   M3DROP+ MASTER PIPELINE: {DATASET_BASENAME}")
    print(f"   Mode: {CONTROL_MODE.upper()}")
    if CONTROL_MODE == "manual":
        print(f"   Target: {MANUAL_TARGET}")
    print(f"############################################################\n")

    # ---------------------------------------------------------
    # STAGE 1: MASK GENERATION
    # ---------------------------------------------------------
    print(">>> STAGE 1: MASK GENERATION")
    if not os.path.exists(MASK_OUTPUT_FILE):
        CoreGPU.ConvertDataSparseGPU(
            input_filename=RAW_DATA_FILE,
            output_mask_filename=MASK_OUTPUT_FILE,
            mode=CONTROL_MODE,
            manual_target=MANUAL_TARGET
        )
    else:
        print(f"STATUS: Found existing mask '{MASK_OUTPUT_FILE}'. Skipping.")
    print("------------------------------------------------------------\n")

    # ---------------------------------------------------------
    # STAGE 2: STATISTICS (Virtual)
    # ---------------------------------------------------------
    print(">>> STAGE 2: STATISTICS CALCULATION")
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = CoreGPU.hidden_calc_valsGPU(
            filename=RAW_DATA_FILE,     # Reads Raw
            mask_filename=MASK_OUTPUT_FILE, # Applies Mask
            mode=CONTROL_MODE,
            manual_target=MANUAL_TARGET
        )
        with open(STATS_OUTPUT_FILE, 'wb') as f:
            pickle.dump(stats, f)
        print(f"STATUS: Statistics saved to '{STATS_OUTPUT_FILE}'.")
    else:
        print(f"STATUS: Loading statistics from '{STATS_OUTPUT_FILE}'...")
        with open(STATS_OUTPUT_FILE, 'rb') as f:
            stats = pickle.load(f)
    print("------------------------------------------------------------\n")

    # ---------------------------------------------------------
    # STAGE 3: MODEL FITTING (Virtual)
    # ---------------------------------------------------------
    print(">>> STAGE 3: MODEL FITTING")
    if not os.path.exists(FIT_OUTPUT_FILE):
        fit_results = CoreGPU.NBumiFitModelGPU(
            raw_filename=RAW_DATA_FILE,     # Reads Raw
            mask_filename=MASK_OUTPUT_FILE, # Applies Mask
            stats=stats,
            mode=CONTROL_MODE,
            manual_target=MANUAL_TARGET
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
        stage4a_start = time.time()
        
        if not os.path.exists(HIGH_VAR_OUTPUT_CSV):
            high_var_genes = CoreGPU.NBumiFeatureSelectionHighVarGPU(fit=fit_results)
            high_var_genes.to_csv(HIGH_VAR_OUTPUT_CSV, index=False)
        else:
            print(f"   Skipping High Variance (Output exists)")

        if not os.path.exists(COMBINED_DROP_OUTPUT_CSV):
            combined_drop_genes = CoreGPU.NBumiFeatureSelectionCombinedDropGPU(
                fit=fit_results,
                raw_filename=RAW_DATA_FILE,
                mode=CONTROL_MODE,
                manual_target=MANUAL_TARGET
            )
            combined_drop_genes.to_csv(COMBINED_DROP_OUTPUT_CSV, index=False)
            
            if not os.path.exists(VOLCANO_PLOT_FILE):
                CoreGPU.NBumiCombinedDropVolcanoGPU(
                    results_df=combined_drop_genes,
                    plot_filename=VOLCANO_PLOT_FILE
                )
        else:
            print(f"   Skipping Combined Dropout (Output exists)")
            
        print(f"Stage 4A Complete. Total Time: {time.time() - stage4a_start:.2f} seconds.")
        print("------------------------------------------------------------\n")

    # ---------------------------------------------------------
    # STAGE 4B: DIAGNOSTICS
    # ---------------------------------------------------------
    if RUN_DIAGNOSTICS:
        print(">>> STAGE 4B: DIAGNOSTICS")
        
        if not os.path.exists(DISP_VS_MEAN_PLOT_FILE):
            DiagnosticsGPU.NBumiPlotDispVsMeanGPU(
                 fit=fit_results,
                 plot_filename=DISP_VS_MEAN_PLOT_FILE
            )
        
        if not os.path.exists(COMPARISON_PLOT_FILE):
            DiagnosticsGPU.NBumiCompareModelsGPU(
                raw_filename=RAW_DATA_FILE,     # Reads Raw
                mask_filename=MASK_OUTPUT_FILE, # Applies Mask
                stats=stats,
                fit_adjust=fit_results,
                plot_filename=COMPARISON_PLOT_FILE,
                mode=CONTROL_MODE,
                manual_target=MANUAL_TARGET
            )
        else:
            print(f"   Skipping Comparison (Plot exists)")
        print("------------------------------------------------------------\n")

    # ---------------------------------------------------------
    # STAGE 4C: NORMALIZATION (Optimized: Combined)
    # ---------------------------------------------------------
    if RUN_NORMALIZATION:
        print(">>> STAGE 4C: NORMALIZATION")
        stage4c_start = time.time()
        
        if not os.path.exists(PEARSON_FULL_OUTPUT_FILE) or not os.path.exists(PEARSON_APPROX_OUTPUT_FILE):
            NormalizationGPU.NBumiPearsonResidualsCombinedGPU(
                raw_filename=RAW_DATA_FILE,
                mask_filename=MASK_OUTPUT_FILE,
                fit_filename=FIT_OUTPUT_FILE,
                stats_filename=STATS_OUTPUT_FILE,
                output_filename_full=PEARSON_FULL_OUTPUT_FILE,
                output_filename_approx=PEARSON_APPROX_OUTPUT_FILE,
                mode=CONTROL_MODE,
                manual_target=MANUAL_TARGET
            )
        else:
            print("   Skipping Normalization (Outputs exist)")
            
        print(f"Stage 4C Complete. Total Time: {time.time() - stage4c_start:.2f} seconds.")
        print("------------------------------------------------------------\n")

    total_time = time.time() - pipeline_start_time
    print(f"############################################################")
    print(f" PIPELINE COMPLETE: {total_time/60:.2f} minutes")
    print(f"############################################################")
