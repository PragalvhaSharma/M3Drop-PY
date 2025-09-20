import os
import sys
import pickle
import time

# Ensure local package is importable when running this script directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- ADD THESE TWO LINES TO PREVENT PLOT POP-UPS ---
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend that only saves to files

# Import functions from the local package
from m3Drop.NB_UMI import (
    ConvertDataSparse,
    hidden_calc_vals,
    NBumiFitModel,
    NBumiCompareModels,
    NBumiPlotDispVsMean,
)

# --- 1. DEFINE FILE PATHS AND PARAMETERS ---
# Resolve dataset paths relative to repository data directory
REPO_ROOT = PROJECT_ROOT
DATA_DIR = os.path.join(REPO_ROOT, "data")
# Default dataset available in data/
DATASET_BASENAME = "GSM8267529_G-P28_raw_matrix"
RAW_DATA_FILE = os.path.join(DATA_DIR, f"{DATASET_BASENAME}.h5ad")

# --- Intermediate Files ---
CLEANED_DATA_FILE = f"{DATASET_BASENAME}_cleaned.h5ad"
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
ADJUSTED_FIT_OUTPUT_FILE = f"{DATASET_BASENAME}_adjusted_fit.pkl"

# --- Final Output ---
DISP_VS_MEAN_PLOT_FILE = f"{DATASET_BASENAME}_disp_vs_mean.png"
COMPARISON_PLOT_FILE = f"{DATASET_BASENAME}_NBumiCompareModels.png"

# --- Processing Parameters ---
ROW_CHUNK = 2000

# --- 2. MAIN DIAGNOSTIC PIPELINE SCRIPT ---
if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"--- Initializing M3Drop+ Diagnostic Pipeline for {RAW_DATA_FILE} ---\n")

    # STAGE 1: Data Cleaning
    print("--- PIPELINE STAGE 1: DATA CLEANING ---")
    if not os.path.exists(CLEANED_DATA_FILE):
        ConvertDataSparse(
            input_filename=RAW_DATA_FILE,
            output_filename=CLEANED_DATA_FILE,
            row_chunk_size=ROW_CHUNK,
        )
    else:
        print(f"STATUS: Found existing file '{CLEANED_DATA_FILE}'. Skipping.\n")

    # STAGE 2: Statistics Calculation
    print("--- PIPELINE STAGE 2: STATISTICS CALCULATION ---")
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = hidden_calc_vals(
            filename_or_counts=CLEANED_DATA_FILE,
            chunk_size=ROW_CHUNK,
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
        fit_adjust = NBumiFitModel(
            CLEANED_DATA_FILE,
            stats,
            chunk_size=ROW_CHUNK,
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
    NBumiPlotDispVsMean(
        fit=fit_adjust,
        suppress_plot=True,
        plot_filename=DISP_VS_MEAN_PLOT_FILE,
    )

    # STAGE 5: Run Full Model Comparison
    print("--- PIPELINE STAGE 5: MODEL COMPARISON ---")
    NBumiCompareModels(
        RAW_DATA_FILE,
        CLEANED_DATA_FILE,
        stats,
        fit_adjust,
        plot_filename=COMPARISON_PLOT_FILE,
        chunk_size=ROW_CHUNK,
        suppress_plot=True,
    )

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"--- Diagnostic Pipeline Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes.")



