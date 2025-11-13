import os
import pickle
import time

# Import functions from your library files
from coreGPU import ConvertDataSparseGPU, hidden_calc_valsGPU, NBumiFitModelGPU
from normalizationGPU import NBumiPearsonResidualsGPU, NBumiPearsonResidualsApproxGPU

# --- 1. DEFINE FILE PATHS AND PARAMETERS ---
# !! CHANGE THIS LINE TO SWITCH DATASETS !!
DATASET_BASENAME = "healthy_liver"

# --- Input Files ---
RAW_DATA_FILE = f"{DATASET_BASENAME}.h5ad"

# --- Intermediate Files (Prerequisites) ---
CLEANED_DATA_FILE = f"{DATASET_BASENAME}_cleaned.h5ad"
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
FIT_OUTPUT_FILE = f"{DATASET_BASENAME}_fit.pkl"

# --- Final Output Files ---
PEARSON_FULL_OUTPUT_FILE = f"{DATASET_BASENAME}_pearson_residuals.h5ad"
PEARSON_APPROX_OUTPUT_FILE = f"{DATASET_BASENAME}_pearson_residuals_approx.h5ad"

# --- Processing Parameters ---
CHUNK_SIZE = 5000

# --- 2. MAIN NORMALIZATION PIPELINE SCRIPT ---
if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"--- Initializing Full Normalization Pipeline for {RAW_DATA_FILE} ---\n")

    # STAGE 1: Data Cleaning (from core.py)
    print("--- PIPELINE STAGE 1: DATA CLEANING ---")
    if not os.path.exists(CLEANED_DATA_FILE):
        ConvertDataSparseGPU(
            input_filename=RAW_DATA_FILE,
            output_filename=CLEANED_DATA_FILE,
            row_chunk_size=CHUNK_SIZE
        )
    else:
        print(f"STATUS: Found existing file '{CLEANED_DATA_FILE}'. Skipping.\n")

    # STAGE 2: Statistics Calculation (from core.py)
    print("--- PIPELINE STAGE 2: STATISTICS CALCULATION ---")
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = hidden_calc_valsGPU(
            filename=CLEANED_DATA_FILE,
            chunk_size=CHUNK_SIZE
        )
        print(f"STATUS: Saving statistics to '{STATS_OUTPUT_FILE}'...")
        with open(STATS_OUTPUT_FILE, 'wb') as f:
            pickle.dump(stats, f)
        print("STATUS: COMPLETE\n")
    else:
        print(f"STATUS: Found existing statistics file '{STATS_OUTPUT_FILE}'. Skipping calculation.\n")

    # STAGE 3: Model Fitting (from core.py)
    print("--- PIPELINE STAGE 3: MODEL FITTING ---")
    if not os.path.exists(FIT_OUTPUT_FILE):
        # Load stats object, as it's needed for this step
        print(f"STATUS: Loading statistics from '{STATS_OUTPUT_FILE}'...")
        with open(STATS_OUTPUT_FILE, 'rb') as f:
            stats = pickle.load(f)
        print("STATUS: COMPLETE")
        
        fit_results = NBumiFitModelGPU(
            cleaned_filename=CLEANED_DATA_FILE,
            stats=stats,
            chunk_size=CHUNK_SIZE
        )
        print(f"STATUS: Saving fit results to '{FIT_OUTPUT_FILE}'...")
        with open(FIT_OUTPUT_FILE, 'wb') as f:
            pickle.dump(fit_results, f)
        print("STATUS: COMPLETE\n")
    else:
        print(f"STATUS: Found existing fit file '{FIT_OUTPUT_FILE}'. Skipping.\n")

    # STAGE 4: Pearson Residuals Normalization
    print("--- PIPELINE STAGE 4: PEARSON RESIDUALS NORMALIZATION ---")
    
    # Method 1: Full, accurate method
    print("\n--- Method 1: Full Pearson Residuals ---")
    if not os.path.exists(PEARSON_FULL_OUTPUT_FILE):
        NBumiPearsonResidualsGPU(
            cleaned_filename=CLEANED_DATA_FILE,
            fit_filename=FIT_OUTPUT_FILE,
            output_filename=PEARSON_FULL_OUTPUT_FILE,
            chunk_size=CHUNK_SIZE
        )
    else:
        print(f"STATUS: Found existing file '{PEARSON_FULL_OUTPUT_FILE}'. Skipping.\n")

    # Method 2: Approximate, faster method
    print("--- Method 2: Approximate Pearson Residuals ---")
    if not os.path.exists(PEARSON_APPROX_OUTPUT_FILE):
        NBumiPearsonResidualsApproxGPU(
            cleaned_filename=CLEANED_DATA_FILE,
            stats_filename=STATS_OUTPUT_FILE,
            output_filename=PEARSON_APPROX_OUTPUT_FILE,
            chunk_size=CHUNK_SIZE
        )
    else:
        print(f"STATUS: Found existing file '{PEARSON_APPROX_OUTPUT_FILE}'. Skipping.\n")


    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"--- Normalization Pipeline Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes.")
