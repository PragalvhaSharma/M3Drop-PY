import os
import pickle
import time

import matplotlib
matplotlib.use('Agg')

from coreCPU import (
    ConvertDataSparseCPU,
    hidden_calc_valsCPU,
    NBumiFitModelCPU
)
from diagnosticsCPU import NBumiCompareModelsCPU, NBumiPlotDispVsMeanCPU

RAW_DATA_FILE = "/Users/pragalvhasharma/Downloads/PragGOToDocuments/CompSci/myProjects/M3Drop/M3Drop-PY/m3Drop/Human_Heart.h5ad"
DATASET_BASENAME = "Human_Heart"
CLEANED_DATA_FILE = f"{DATASET_BASENAME}_cleaned.h5ad"
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
ADJUSTED_FIT_OUTPUT_FILE = f"{DATASET_BASENAME}_adjusted_fit.pkl"

DISP_VS_MEAN_PLOT_FILE = f"{DATASET_BASENAME}_disp_vs_mean_cpu.png"
COMPARISON_PLOT_FILE = f"{DATASET_BASENAME}_NBumiCompareModels_cpu.png"

ROW_CHUNK = 2000

if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"--- Initializing CPU Diagnostic Pipeline for {RAW_DATA_FILE} ---\n")

    print("--- PIPELINE STAGE 1: DATA CLEANING ---")
    if not os.path.exists(CLEANED_DATA_FILE):
        ConvertDataSparseCPU(
            input_filename=RAW_DATA_FILE,
            output_filename=CLEANED_DATA_FILE,
            row_chunk_size=ROW_CHUNK
        )
    else:
        print(f"STATUS: Found existing file '{CLEANED_DATA_FILE}'. Skipping.\n")

    print("--- PIPELINE STAGE 2: STATISTICS CALCULATION ---")
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = hidden_calc_valsCPU(
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

    print("--- PIPELINE STAGE 3: ADJUSTED MODEL FITTING ---")
    if not os.path.exists(ADJUSTED_FIT_OUTPUT_FILE):
        fit_adjust = NBumiFitModelCPU(
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

    print("--- PIPELINE STAGE 4: DISPERSION VS. MEAN PLOT ---")
    NBumiPlotDispVsMeanCPU(
        fit=fit_adjust,
        plot_filename=DISP_VS_MEAN_PLOT_FILE
    )

    print("--- PIPELINE STAGE 5: MODEL COMPARISON ---")
    NBumiCompareModelsCPU(
        raw_filename=RAW_DATA_FILE,
        cleaned_filename=CLEANED_DATA_FILE,
        stats=stats,
        fit_adjust=fit_adjust,
        plot_filename=COMPARISON_PLOT_FILE,
        chunk_size=ROW_CHUNK
    )

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"--- Diagnostic CPU Pipeline Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes.")
