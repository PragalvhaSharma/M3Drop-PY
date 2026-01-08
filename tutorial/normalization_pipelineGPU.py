import m3Drop as M3Drop
import os
import pickle
import time

# --- 1. DEFINE FILE PATHS AND PARAMETERS ---
# !! CHANGE THIS LINE TO SWITCH DATASETS !!
DATASET_BASENAME = "Human_Heart"


# --- Input Files ---
RAW_DATA_FILE = f"{DATASET_BASENAME}.h5ad"

# --- Intermediate Files (Prerequisites) ---
CLEANED_DATA_FILE = f"{DATASET_BASENAME}_cleaned.h5ad"
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
FIT_OUTPUT_FILE = f"{DATASET_BASENAME}_fit.pkl"

# --- Final Output Files ---
PEARSON_FULL_OUTPUT_FILE = f"{DATASET_BASENAME}_pearson_residuals.h5ad"
PEARSON_APPROX_OUTPUT_FILE = f"{DATASET_BASENAME}_pearson_residuals_approx.h5ad"


# --- 2. MAIN NORMALIZATION PIPELINE SCRIPT ---
if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"--- Initializing Full Normalization Pipeline for {RAW_DATA_FILE} ---\n")

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
        print(f"STATUS: Found existing statistics file '{STATS_OUTPUT_FILE}'. Skipping calculation.\n")

    # STAGE 3: Model Fitting
    print("--- PIPELINE STAGE 3: MODEL FITTING ---")
    if not os.path.exists(FIT_OUTPUT_FILE):
        # Load stats object
        print(f"STATUS: Loading statistics from '{STATS_OUTPUT_FILE}'...")
        with open(STATS_OUTPUT_FILE, 'rb') as f:
            stats = pickle.load(f)
        print("STATUS: COMPLETE")
        
        fit_results = M3Drop.NBumiFitModelGPU(
            cleaned_filename=CLEANED_DATA_FILE,
            stats=stats
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
        M3Drop.NBumiPearsonResidualsGPU(
            cleaned_filename=CLEANED_DATA_FILE,
            fit_filename=FIT_OUTPUT_FILE,
            output_filename=PEARSON_FULL_OUTPUT_FILE
        )
    else:
        print(f"STATUS: Found existing file '{PEARSON_FULL_OUTPUT_FILE}'. Skipping.\n")

    # Method 2: Approximate, faster method
    print("--- Method 2: Approximate Pearson Residuals ---")
    if not os.path.exists(PEARSON_APPROX_OUTPUT_FILE):
        M3Drop.NBumiPearsonResidualsApproxGPU(
            cleaned_filename=CLEANED_DATA_FILE,
            stats_filename=STATS_OUTPUT_FILE,
            output_filename=PEARSON_APPROX_OUTPUT_FILE
        )
    else:
        print(f"STATUS: Found existing file '{PEARSON_APPROX_OUTPUT_FILE}'. Skipping.\n")


    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"--- Normalization Pipeline Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes.")
