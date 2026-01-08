import m3Drop as M3Drop
import os
import pickle
import time
import matplotlib

# --- PREVENT PLOT POP-UPS ---
# Use a non-interactive backend that only saves to files
matplotlib.use('Agg') 

# --- 1. DEFINE FILE PATHS AND PARAMETERS ---
# !! CHANGE THIS LINE TO SWITCH DATASETS !!
DATASET_BASENAME = "Human_Heart"


# --- Input File ---
RAW_DATA_FILE = f"{DATASET_BASENAME}.h5ad"

# --- Intermediate Files ---
CLEANED_DATA_FILE = f"{DATASET_BASENAME}_cleaned.h5ad"
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
ADJUSTED_FIT_OUTPUT_FILE = f"{DATASET_BASENAME}_adjusted_fit.pkl"

# --- Final Output ---
DISP_VS_MEAN_PLOT_FILE = f"{DATASET_BASENAME}_disp_vs_mean.png"
COMPARISON_PLOT_FILE = f"{DATASET_BASENAME}_NBumiCompareModels.png"

# --- 2. MAIN DIAGNOSTIC PIPELINE SCRIPT ---
if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"--- Initializing M3Drop+ Diagnostic Pipeline for {RAW_DATA_FILE} ---\n")

    # STAGE 1: Data Cleaning
    print("--- PIPELINE STAGE 1: DATA CLEANING ---")
    if not os.path.exists(CLEANED_DATA_FILE):
        M3Drop.ConvertDataSparseGPU(
            input_filename=RAW_DATA_FILE,
            output_filename=CLEANED_DATA_FILE
        )
    else:
        print(f"STATUS: Found existing file '{CLEANED_DATA_FILE}'. Skipping.\n")

    # STAGE 2: Statistics Calculation
    print("--- PIPELINE STAGE 2: STATISTICS CALCULATION ---")
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = M3Drop.hidden_calc_valsGPU(
            filename=CLEANED_DATA_FILE
        )
        print(f"STATUS: Saving statistics to '{STATS_OUTPUT_FILE}'...")
        with open(STATS_OUTPUT_FILE, 'wb') as f:
            pickle.dump(stats, f)
        print("STATUS: COMPLETE\n")
    else:
        print(f"STATUS: Loading existing statistics from '{STATS_OUTPUT_FILE}'...")
        with open(STATS_OUTPUT_FILE, 'rb') as f:
            stats = pickle.load(f)
        print("STATUS: COMPLETE\n")

    # STAGE 3: Adjusted Model Fitting
    print("--- PIPELINE STAGE 3: ADJUSTED MODEL FITTING ---")
    if not os.path.exists(ADJUSTED_FIT_OUTPUT_FILE):
        fit_adjust = M3Drop.NBumiFitModelGPU(
            cleaned_filename=CLEANED_DATA_FILE,
            stats=stats
        )
        print(f"STATUS: Saving adjusted fit to '{ADJUSTED_FIT_OUTPUT_FILE}'...")
        with open(ADJUSTED_FIT_OUTPUT_FILE, 'wb') as f:
            pickle.dump(fit_adjust, f)
        print("STATUS: COMPLETE\n")
    else:
        print(f"STATUS: Loading existing adjusted fit from '{ADJUSTED_FIT_OUTPUT_FILE}'...")
        with open(ADJUSTED_FIT_OUTPUT_FILE, 'rb') as f:
            fit_adjust = pickle.load(f)
        print("STATUS: COMPLETE\n")

    # STAGE 4: DISPERSION VS. MEAN PLOT
    print("--- PIPELINE STAGE 4: DISPERSION VS. MEAN PLOT ---")
    M3Drop.NBumiPlotDispVsMeanGPU(
        fit=fit_adjust,
        plot_filename=DISP_VS_MEAN_PLOT_FILE
    )

    # STAGE 5: Run Full Model Comparison
    print("--- PIPELINE STAGE 5: MODEL COMPARISON ---")
    M3Drop.NBumiCompareModelsGPU(
        raw_filename=RAW_DATA_FILE,
        cleaned_filename=CLEANED_DATA_FILE,
        stats=stats,
        fit_adjust=fit_adjust,
        plot_filename=COMPARISON_PLOT_FILE
    )

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"--- Diagnostic Pipeline Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes.")
