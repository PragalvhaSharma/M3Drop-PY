import os
import pickle
import time

from coreCPU import ConvertDataSparseCPU, hidden_calc_valsCPU, NBumiFitModelCPU
from normalizationCPU import NBumiPearsonResidualsCPU, NBumiPearsonResidualsApproxCPU

DATASET_BASENAME = "healthy_liver"

RAW_DATA_FILE = f"{DATASET_BASENAME}.h5ad"
CLEANED_DATA_FILE = f"{DATASET_BASENAME}_cleaned.h5ad"
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
FIT_OUTPUT_FILE = f"{DATASET_BASENAME}_fit.pkl"

PEARSON_FULL_OUTPUT_FILE = f"{DATASET_BASENAME}_pearson_residuals_cpu.h5ad"
PEARSON_APPROX_OUTPUT_FILE = f"{DATASET_BASENAME}_pearson_residuals_approx_cpu.h5ad"

CHUNK_SIZE = 5000

if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"--- Initializing CPU Normalization Pipeline for {RAW_DATA_FILE} ---\n")

    print("--- PIPELINE STAGE 1: DATA CLEANING ---")
    if not os.path.exists(CLEANED_DATA_FILE):
        ConvertDataSparseCPU(
            input_filename=RAW_DATA_FILE,
            output_filename=CLEANED_DATA_FILE,
            row_chunk_size=CHUNK_SIZE
        )
    else:
        print(f"STATUS: Found existing file '{CLEANED_DATA_FILE}'. Skipping.\n")

    print("--- PIPELINE STAGE 2: STATISTICS CALCULATION ---")
    stats = None
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = hidden_calc_valsCPU(
            filename=CLEANED_DATA_FILE,
            chunk_size=CHUNK_SIZE
        )
        print(f"STATUS: Saving statistics to '{STATS_OUTPUT_FILE}'...")
        with open(STATS_OUTPUT_FILE, 'wb') as f:
            pickle.dump(stats, f)
        print("STATUS: COMPLETE\n")
    else:
        print(f"STATUS: Found existing statistics file '{STATS_OUTPUT_FILE}'. Loading...")

    if stats is None:
        with open(STATS_OUTPUT_FILE, 'rb') as f:
            stats = pickle.load(f)
        print("STATUS: COMPLETE\n")

    print("--- PIPELINE STAGE 3: MODEL FITTING ---")
    fit_results = None
    if not os.path.exists(FIT_OUTPUT_FILE):
        fit_results = NBumiFitModelCPU(
            cleaned_filename=CLEANED_DATA_FILE,
            stats=stats,
            chunk_size=CHUNK_SIZE
        )
        print(f"STATUS: Saving fit results to '{FIT_OUTPUT_FILE}'...")
        with open(FIT_OUTPUT_FILE, 'wb') as f:
            pickle.dump(fit_results, f)
        print("STATUS: COMPLETE\n")
    else:
        print(f"STATUS: Found existing fit file '{FIT_OUTPUT_FILE}'. Loading...")

    if fit_results is None:
        with open(FIT_OUTPUT_FILE, 'rb') as f:
            fit_results = pickle.load(f)
        print("STATUS: COMPLETE\n")

    print("--- PIPELINE STAGE 4: PEARSON RESIDUALS NORMALIZATION ---")
    print("\n--- Method 1: Full Pearson Residuals (CPU) ---")
    if not os.path.exists(PEARSON_FULL_OUTPUT_FILE):
        NBumiPearsonResidualsCPU(
            cleaned_filename=CLEANED_DATA_FILE,
            fit_filename=FIT_OUTPUT_FILE,
            output_filename=PEARSON_FULL_OUTPUT_FILE,
            chunk_size=CHUNK_SIZE
        )
    else:
        print(f"STATUS: Found existing file '{PEARSON_FULL_OUTPUT_FILE}'. Skipping.\n")

    print("--- Method 2: Approximate Pearson Residuals (CPU) ---")
    if not os.path.exists(PEARSON_APPROX_OUTPUT_FILE):
        NBumiPearsonResidualsApproxCPU(
            cleaned_filename=CLEANED_DATA_FILE,
            stats_filename=STATS_OUTPUT_FILE,
            output_filename=PEARSON_APPROX_OUTPUT_FILE,
            chunk_size=CHUNK_SIZE
        )
    else:
        print(f"STATUS: Found existing file '{PEARSON_APPROX_OUTPUT_FILE}'. Skipping.\n")

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"--- Normalization CPU Pipeline Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes.")
