import m3Drop as M3Drop
import os
import pickle
import time
import pandas as pd


# !! CHANGE THIS LINE TO SWITCH DATASETS !!
DATASET_BASENAME = "Human_Heart"
ROW_CHUNK = 5000


#########################################################################
# Input file
RAW_DATA_FILE = f"{DATASET_BASENAME}.h5ad"

# Intermediate and final output files are now generated automatically
CLEANED_DATA_FILE = f"{DATASET_BASENAME}_cleaned.h5ad"
STATS_OUTPUT_FILE = f"{DATASET_BASENAME}_stats.pkl"
FIT_OUTPUT_FILE = f"{DATASET_BASENAME}_fit.pkl"
HIGH_VAR_OUTPUT_CSV = f"{DATASET_BASENAME}_high_variance_genes.csv"
COMBINED_DROP_OUTPUT_CSV = f"{DATASET_BASENAME}_combined_dropout_genes.csv"
VOLCANO_PLOT_FILE = f"{DATASET_BASENAME}_volcano_plot.png"

# --- 2. MAIN PIPELINE SCRIPT ---
if __name__ == "__main__":
    pipeline_start_time = time.time()
    print(f"--- Initializing M3Drop+ Pipeline for {RAW_DATA_FILE} ---\n")

    # --- STAGE 1: Data Cleaning ---
    print("--- PIPELINE STAGE 1: DATA CLEANING ---")
    if not os.path.exists(CLEANED_DATA_FILE):
        M3Drop.ConvertDataSparseGPU(
            input_filename=RAW_DATA_FILE,
            output_filename=CLEANED_DATA_FILE,
            row_chunk_size=ROW_CHUNK
        )
    else:
        print(f"STATUS: Found existing file '{CLEANED_DATA_FILE}'. Skipping.\n")

    # --- STAGE 2: Core Statistics Calculation ---
    print("--- PIPELINE STAGE 2: STATISTICS CALCULATION ---")
    if not os.path.exists(STATS_OUTPUT_FILE):
        stats = M3Drop.hidden_calc_valsGPU(
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

    # --- STAGE 3: Model Fitting ---
    print("--- PIPELINE STAGE 3: MODEL FITTING ---")
    if not os.path.exists(FIT_OUTPUT_FILE):
        fit_results = M3Drop.NBumiFitModelGPU(
            cleaned_filename = CLEANED_DATA_FILE,
            stats = stats,
            #CLEANED_DATA_FILE,
            #stats,
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
        high_var_genes = M3Drop.NBumiFeatureSelectionHighVarGPU(fit=fit_results)
        print(f"STATUS: Saving high variance genes to '{HIGH_VAR_OUTPUT_CSV}'...")
        high_var_genes.to_csv(HIGH_VAR_OUTPUT_CSV, index=False)
        print("STATUS: COMPLETE\n")
    else:
        print(f"STATUS: Found existing file '{HIGH_VAR_OUTPUT_CSV}'. Skipping.\n")

    # Method 2: Combined Dropout
    print("--- Method 2: Combined Dropout ---")
    if not os.path.exists(COMBINED_DROP_OUTPUT_CSV):
        combined_drop_genes = M3Drop.NBumiFeatureSelectionCombinedDropGPU(
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
        M3Drop.NBumiCombinedDropVolcanoGPU(
            results_df=combined_drop_genes,
            plot_filename=VOLCANO_PLOT_FILE
        )
    else:
        print(f"STATUS: Found existing plot '{VOLCANO_PLOT_FILE}'. Skipping.\n")


    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"--- M3Drop+ Pipeline Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes.")
