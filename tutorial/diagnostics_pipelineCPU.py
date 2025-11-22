import m3Drop as M3Drop
import os
import pickle
import time

# --- ADD THESE TWO LINES TO PREVENT PLOT POP-UPS ---
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend that only saves to files


# --- 1. DEFINE FILE PATHS AND PARAMETERS ---
# !! CHANGE THIS LINE TO SWITCH DATASETS !!
DATASET_BASENAME = "Human_Heart"
ROW_CHUNK = 2000


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

    # STAGE 1: Data Cleaning (from core.py)
    print("--- PIPELINE STAGE 1: DATA CLEANING ---")
    if not os.path.exists(CLEANED_DATA_FILE):
        M3Drop.ConvertDataSparseCPU(
            input_filename=RAW_DATA_FILE,
            output_filename=CLEANED_DATA_FILE,
            row_chunk_size=ROW_CHUNK
        )
    else:
        print(f"STATUS: Found existing file '{CLEANED_DATA_FILE}'. Skipping.\n")

    # STAGE 2: Statistics Calculation (from core.py)
    print("--- PIPELINE STAGE 2: STATISTICS CALCULATION ---")
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = M3Drop.hidden_calc_valsCPU(
            filename=CLEANED_DATA_FILE,
            chunk_size=ROW_CHUNK
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

    # STAGE 3: Adjusted Model Fitting (from core.py)
    print("--- PIPELINE STAGE 3: ADJUSTED MODEL FITTING ---")
    if not os.path.exists(ADJUSTED_FIT_OUTPUT_FILE):
        fit_adjust = M3Drop.NBumiFitModelCPU(
            cleaned_filename=CLEANED_DATA_FILE,
            stats=stats,
            chunk_size=ROW_CHUNK
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
    # By setting the backend at the top, we no longer need suppress_plot=True.
    # The plot will not pop up, but it will be saved if a filename is given.
    M3Drop.NBumiPlotDispVsMeanCPU(
        fit=fit_adjust,
        plot_filename=DISP_VS_MEAN_PLOT_FILE
    )

    # STAGE 5: Run Full Model Comparison (from diagnostics.py)
    print("--- PIPELINE STAGE 5: MODEL COMPARISON ---")
    M3Drop.NBumiCompareModelsCPU(
        raw_filename=RAW_DATA_FILE,
        cleaned_filename=CLEANED_DATA_FILE,
        stats=stats,
        fit_adjust=fit_adjust,
        plot_filename=COMPARISON_PLOT_FILE,
        chunk_size=ROW_CHUNK
    )

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"--- Diagnostic Pipeline Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes.")