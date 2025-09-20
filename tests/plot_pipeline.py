import os
import sys
import pickle
import time
import pandas as pd

# Ensure local package is importable when running this script directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Use a non-interactive backend to prevent plot pop-ups
import matplotlib
matplotlib.use('Agg')

# Import all necessary functions from the core and diagnostics libraries
from m3Drop.NB_UMI import (
    ConvertDataSparse,
    hidden_calc_vals,
    NBumiFitModel,
    NBumiFeatureSelectionCombinedDrop,
    NBumiCombinedDropVolcano,
    NBumiCompareModels,
    NBumiPlotDispVsMean,
)

# --- 1. DEFINE FILE PATHS AND PARAMETERS ---
# Resolve dataset paths relative to repository data directory
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT))
DATA_DIR = os.path.join(REPO_ROOT, "data")
DATASET_BASENAME = "GSM8267529_G-P28_raw_matrix"  # default dataset in data/
RAW_DATA_FILE = os.path.join(DATA_DIR, f"{DATASET_BASENAME}.h5ad")

# --- Intermediate Files (to be deleted after execution) ---
CLEANED_DATA_FILE = f"{DATASET_BASENAME}_cleaned.h5ad"
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
ADJUSTED_FIT_OUTPUT_FILE = f"{DATASET_BASENAME}_adjusted_fit.pkl"

# --- Final Plot Outputs ---
DISP_VS_MEAN_PLOT_FILE = f"{DATASET_BASENAME}_disp_vs_mean_final.png"
COMPARISON_PLOT_FILE = f"{DATASET_BASENAME}_model_comparison_final.png"
VOLCANO_PLOT_FILE = f"{DATASET_BASENAME}_volcano_plot_final.png"

# --- Processing Parameters ---
ROW_CHUNK = 2000

# --- 2. MAIN MASTER PIPELINE SCRIPT ---
if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"--- Initializing M3Drop+ Master Plotting Pipeline for {RAW_DATA_FILE} ---\n")

    try:
        # STAGE 1: Data Cleaning
        print("--- PIPELINE STAGE 1: DATA CLEANING ---")
        ConvertDataSparse(
            input_filename=RAW_DATA_FILE,
            output_filename=CLEANED_DATA_FILE,
            row_chunk_size=ROW_CHUNK
        )

        # STAGE 2: Statistics Calculation
        print("--- PIPELINE STAGE 2: STATISTICS CALCULATION ---")
        stats = hidden_calc_vals(
            filename=CLEANED_DATA_FILE,
            chunk_size=ROW_CHUNK
        )
        # Save stats temporarily for NBumiCompareModels
        with open(STATS_OUTPUT_FILE, 'wb') as f:
            pickle.dump(stats, f)

        # STAGE 3: Adjusted Model Fitting
        print("--- PIPELINE STAGE 3: ADJUSTED MODEL FITTING ---")
        fit_adjust = NBumiFitModel(
            cleaned_filename=CLEANED_DATA_FILE,
            stats=stats,
            chunk_size=ROW_CHUNK
        )
        # Save fit temporarily for NBumiCompareModels
        with open(ADJUSTED_FIT_OUTPUT_FILE, 'wb') as f:
            pickle.dump(fit_adjust, f)

        # STAGE 4: Generate Dispersion vs. Mean Plot
        print("--- PIPELINE STAGE 4: DISPERSION VS. MEAN PLOT ---")
        NBumiPlotDispVsMean(
            fit=fit_adjust,
            suppress_plot=True, # Ensure no pop-up
            plot_filename=DISP_VS_MEAN_PLOT_FILE
        )

        # STAGE 5: Generate Model Comparison Plot
        print("--- PIPELINE STAGE 5: MODEL COMPARISON PLOT ---")
        NBumiCompareModels(
            raw_filename=RAW_DATA_FILE,
            cleaned_filename=CLEANED_DATA_FILE,
            stats=stats,
            fit_adjust=fit_adjust,
            suppress_plot=True, # Ensure no pop-up
            plot_filename=COMPARISON_PLOT_FILE,
            chunk_size=ROW_CHUNK
        )

        # STAGE 6: Feature Selection for Volcano Plot
        print("--- PIPELINE STAGE 6: FEATURE SELECTION FOR VOLCANO PLOT ---")
        combined_drop_genes = NBumiFeatureSelectionCombinedDrop(
            fit=fit_adjust,
            cleaned_filename=CLEANED_DATA_FILE,
            chunk_size=ROW_CHUNK
        )

        # STAGE 7: Generate Volcano Plot
        print("--- PIPELINE STAGE 7: VOLCANO PLOT ---")
        NBumiCombinedDropVolcano(
            results_df=combined_drop_genes,
            suppress_plot=True, # Ensure no pop-up
            plot_filename=VOLCANO_PLOT_FILE
        )

    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}")
    
    finally:
        # STAGE 8: Cleanup of Intermediate Files
        print("\n--- PIPELINE STAGE 8: CLEANUP ---")
        files_to_delete = [
            CLEANED_DATA_FILE,
            STATS_OUTPUT_FILE,
            ADJUSTED_FIT_OUTPUT_FILE
        ]
        for f in files_to_delete:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"Successfully deleted intermediate file: {f}")
                except OSError as e:
                    print(f"Error deleting file {f}: {e}")
            else:
                print(f"Cleanup skipped: file not found {f}")

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"\n--- Master Plotting Pipeline Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes.")