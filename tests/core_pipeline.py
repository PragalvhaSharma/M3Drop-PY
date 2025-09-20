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

# --- ADD THESE TWO LINES TO PREVENT PLOT POP-UPS ---
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend that only saves to files

# Import required functions from the package modules
from m3Drop.NB_UMI import (
    ConvertDataSparse,
    hidden_calc_vals,
    NBumiFitModel,
    NBumiFeatureSelectionHighVar,
    NBumiFeatureSelectionCombinedDrop,
    NBumiCombinedDropVolcano,
)

# --- 1. DEFINE FILE PATHS AND PARAMETERS ---
# Resolve dataset paths relative to repository data directory
REPO_ROOT = PROJECT_ROOT
DATA_DIR = os.path.join(REPO_ROOT, "data")
DATASET_BASENAME = "GSM8267529_G-P28_raw_matrix"  # default dataset in data/
RAW_DATA_FILE = os.path.join(DATA_DIR, f"{DATASET_BASENAME}.h5ad")

# Derive dataset basename from the input file (without extension)
import os
DATASET_BASENAME = os.path.splitext(os.path.basename(RAW_DATA_FILE))[0]

# Intermediate and final output files are now generated automatically
CLEANED_DATA_FILE = f"{DATASET_BASENAME}_cleaned.h5ad"
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
FIT_OUTPUT_FILE = f"{DATASET_BASENAME}_fit.pkl"
HIGH_VAR_OUTPUT_CSV = f"{DATASET_BASENAME}_high_variance_genes.csv"
COMBINED_DROP_OUTPUT_CSV = f"{DATASET_BASENAME}_combined_dropout_genes.csv"
VOLCANO_PLOT_FILE = f"{DATASET_BASENAME}_volcano_plot.png"

# Processing parameters
ROW_CHUNK = 5000

# --- 2. MAIN PIPELINE SCRIPT ---
if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"--- Initializing M3Drop+ Pipeline for {RAW_DATA_FILE} ---\n")

    # --- STAGE 1: Data Cleaning ---
    print("--- PIPELINE STAGE 1: DATA CLEANING ---")
    if not os.path.exists(CLEANED_DATA_FILE):
        ConvertDataSparse(
            input_filename=RAW_DATA_FILE,
            output_filename=CLEANED_DATA_FILE,
            row_chunk_size=ROW_CHUNK
        )
    else:
        print(f"STATUS: Found existing file '{CLEANED_DATA_FILE}'. Skipping.\n")

    # --- STAGE 2: Core Statistics Calculation ---
    print("--- PIPELINE STAGE 2: STATISTICS CALCULATION ---")
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = hidden_calc_vals(
            filename_or_counts=CLEANED_DATA_FILE,
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

    # --- STAGE 3: Model Fitting ---
    print("--- PIPELINE STAGE 3: MODEL FITTING ---")
    if not os.path.exists(FIT_OUTPUT_FILE):
        fit_results = NBumiFitModel(
            CLEANED_DATA_FILE,
            stats,
            chunk_size=ROW_CHUNK
        )
        print(f"STATUS: Saving fit results to '{FIT_OUTPUT_FILE}'...")
        with open(FIT_OUTPUT_FILE, 'wb') as f:
            pickle.dump(fit_results, f)
        print("STATUS: COMPLETE\n")
    else:
        print(f"STATUS: Loading existing fit results from '{FIT_OUTPUT_FILE}'...")
        with open(FIT_OUTPUT_FILE, 'rb') as f:
            fit_results = pickle.load(f)
        print("STATUS: COMPLETE\n")

    # --- STAGE 4: Feature Selection ---
    print("--- PIPELINE STAGE 4: FEATURE SELECTION ---")
    # Method 1: High Variance Genes
    print("\n--- Method 1: High Variance ---")
    if not os.path.exists(HIGH_VAR_OUTPUT_CSV):
        high_var_genes = NBumiFeatureSelectionHighVar(fit=fit_results)
        print(f"STATUS: Saving high variance genes to '{HIGH_VAR_OUTPUT_CSV}'...")
        high_var_genes.to_csv(HIGH_VAR_OUTPUT_CSV, index=False)
        print("STATUS: COMPLETE\n")
    else:
        print(f"STATUS: Found existing file '{HIGH_VAR_OUTPUT_CSV}'. Skipping.\n")

    # Method 2: Combined Dropout
    print("--- Method 2: Combined Dropout ---")
    if not os.path.exists(COMBINED_DROP_OUTPUT_CSV):
        combined_drop_genes = NBumiFeatureSelectionCombinedDrop(
            fit=fit_results,
            cleaned_filename=CLEANED_DATA_FILE
        )
        print(f"STATUS: Saving combined dropout genes to '{COMBINED_DROP_OUTPUT_CSV}'...")
        combined_drop_genes.to_csv(COMBINED_DROP_OUTPUT_CSV, index=False)
        print("STATUS: COMPLETE\n")
    else:
        print(f"STATUS: Found existing file '{COMBINED_DROP_OUTPUT_CSV}'. Loading...")
        combined_drop_genes = pd.read_csv(COMBINED_DROP_OUTPUT_CSV)
        print("STATUS: COMPLETE\n")

    # --- STAGE 5: VISUALIZATION ---
    print("--- PIPELINE STAGE 5: VISUALIZATION ---")
    if not os.path.exists(VOLCANO_PLOT_FILE):
        # The plot will be saved to the filename but will not pop up
        NBumiCombinedDropVolcano(
            results_df=combined_drop_genes,
            plot_filename=VOLCANO_PLOT_FILE
        )
    else:
        print(f"STATUS: Found existing plot '{VOLCANO_PLOT_FILE}'. Skipping.\n")


    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"--- M3Drop+ Pipeline Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes.")