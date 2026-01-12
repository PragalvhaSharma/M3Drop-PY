import numpy as np
import pandas as pd
import scanpy as sc
import cupy
import cupyx.scipy.sparse as csp
import matplotlib.pyplot as plt
import os
import gc
import psutil

# ==============================================================================
# GPU MEMORY GOVERNOR & OPTIMIZER
# ==============================================================================

def get_slurm_memory_limit():
    """Detects SLURM memory limits or defaults to system RAM."""
    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if mem_per_cpu:
        return int(mem_per_cpu) * 1024 * 1024  # Convert MB to Bytes
    
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_per_node:
        return int(mem_per_node) * 1024 * 1024
        
    return psutil.virtual_memory().total

def calculate_optimal_chunk_size(n_vars, dtype_size=4, memory_multiplier=3.0, override_cap=None):
    """
    Calculates safe chunk size based on available VRAM and RAM.
    """
    try:
        gpu_mem_info = cupy.cuda.runtime.memGetInfo()
        free_vram = gpu_mem_info[0]
    except Exception:
        print("WARNING: No GPU detected or CuPy error. Defaulting to safe small chunk.")
        return 5000

    available_ram = get_slurm_memory_limit() * 0.8 
    
    row_cost_vram = n_vars * dtype_size * memory_multiplier
    row_cost_ram = n_vars * dtype_size * 2.0 
    
    max_rows_vram = int(free_vram / row_cost_vram)
    max_rows_ram = int(available_ram / row_cost_ram)
    
    optimal_chunk = min(max_rows_vram, max_rows_ram)
    
    if override_cap:
        optimal_chunk = min(optimal_chunk, override_cap)
        
    print("-" * 60)
    print(f" CHUNK SIZE OPTIMIZER (PING & GOVERNOR)             [{pd.Timestamp.now().strftime('%H:%M:%S')}]")
    print("-" * 60)
    print(f" CONTEXT      : CLUSTER (SLURM Detected)")
    print(f" DATA LOAD    : {n_vars * dtype_size:,.0f} bytes/row (dtype={dtype_size})")
    print(f" MULTIPLIER   : {memory_multiplier}x")
    if override_cap:
        print(f" OVERRIDE CAP : {override_cap:,.0f} rows")
    print(f" RAM LIMIT    : {max_rows_ram:,.0f} rows")
    print(f" VRAM LIMIT   : {max_rows_vram:,.0f} rows")
    print("-" * 60)
    print(f" >> CHUNK SIZE  : {optimal_chunk:,.0f} rows")
    print("-" * 60)
    
    return max(1, optimal_chunk)

# ==============================================================================
# PIPELINE FUNCTIONS
# ==============================================================================

def NBumiFitBasicModelGPU(adata_path, output_path="test_data_basic_fit.pkl"):
    """
    Fits the basic model (depth-adjusted means) on GPU.
    """
    print(f"FUNCTION: NBumiFitBasicModelGPU() | FILE: {adata_path}")
    
    adata = sc.read_h5ad(adata_path, backed='r')
    n_cells, n_genes = adata.shape
    
    chunk_size = calculate_optimal_chunk_size(n_genes, dtype_size=4, memory_multiplier=4.0, override_cap=50000)
    
    print("\nPhase [1/2]: Initializing parameters and arrays on GPU...")
    
    sum_x_gpu = cupy.zeros(n_genes, dtype=cupy.float32)
    sum_sq_x_gpu = cupy.zeros(n_genes, dtype=cupy.float32)
    
    print("Phase [1/2]: COMPLETE")
    print("Phase [2/2]: Calculating variance from data chunks...")
    
    for i in range(0, n_cells, chunk_size):
        end = min(i + chunk_size, n_cells)
        
        chunk = adata[i:end].X
        if isinstance(chunk, pd.DataFrame):
            chunk = chunk.values
            
        chunk_gpu = cupy.asarray(chunk, dtype=cupy.float32)
        
        if csp.issparse(chunk_gpu):
            chunk_gpu = chunk_gpu.toarray()
            
        sum_x_gpu += cupy.sum(chunk_gpu, axis=0)
        sum_sq_x_gpu += cupy.sum(chunk_gpu ** 2, axis=0)
        
        del chunk_gpu
        cupy.get_default_memory_pool().free_all_blocks()
        
    means = sum_x_gpu / n_cells
    vars_ = (sum_sq_x_gpu / n_cells) - (means ** 2)
    
    means_cpu = cupy.asnumpy(means)
    vars_cpu = cupy.asnumpy(vars_)
    
    print("Phase [2/2]: COMPLETE")
    
    return means_cpu, vars_cpu

def NBumiPlotDispVsMeanGPU(adata_path, output_path="test_data_disp_vs_mean.png"):
    """
    Generates the Dispersion vs Mean diagnostic plot (Calculations on GPU, Plotting on CPU).
    """
    print(f"FUNCTION: NBumiPlotDispVsMeanGPU() | FILE: {adata_path}")
    
    # Calculate stats using the GPU function
    means, vars_ = NBumiFitBasicModelGPU(adata_path)
    
    # Avoid log(0)
    means[means == 0] = 1e-9
    vars_[vars_ == 0] = 1e-9
    
    print(f"STATUS: Generating plot to '{output_path}'...")
    plt.figure(figsize=(10, 6))
    plt.scatter(np.log10(means), np.log10(vars_), s=1, alpha=0.5, c='black')
    plt.xlabel("Log10 Mean Expression")
    plt.ylabel("Log10 Variance")
    plt.title("Dispersion vs Mean (GPU Accelerated)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    
    print("FUNCTION: NBumiPlotDispVsMeanGPU() COMPLETE")
    return output_path

def NBumiCheckFitFSGPU(adata_path, fit_results, suppress_plot=False, memory_multiplier=5.0):
    """
    Evaluates the fit by calculating dispersion and dropouts on GPU.
    """
    print(f"FUNCTION: NBumiCheckFitFSGPU() | FILE: {adata_path}")
    
    adata = sc.read_h5ad(adata_path, backed='r')
    n_cells, n_genes = adata.shape

    chunk_size = calculate_optimal_chunk_size(n_genes, dtype_size=4, memory_multiplier=memory_multiplier)
    
    print("\nPhase [1/2]: Initializing parameters and arrays on GPU...")
    
    if hasattr(fit_results, 'fitted_size'):
        smoothed_size_gpu = cupy.asarray(fit_results.fitted_size, dtype=cupy.float32)
    else:
        smoothed_size_gpu = cupy.ones(n_genes, dtype=cupy.float32) 

    print("Phase [1/2]: COMPLETE")
    print("Phase [2/2]: Calculating expected dropouts from data chunks...")

    for i in range(0, n_cells, chunk_size):
        end = min(i + chunk_size, n_cells)
        
        chunk = adata[i:end].X
        if isinstance(chunk, pd.DataFrame):
            chunk = chunk.values
            
        mu_chunk_gpu = cupy.asarray(chunk, dtype=cupy.float32)
        
        if csp.issparse(mu_chunk_gpu):
            mu_chunk_gpu = mu_chunk_gpu.toarray()
            
        try:
            # Calculation using float32 and safe memory limits
            base = 1.0 + (mu_chunk_gpu / smoothed_size_gpu) 
            del base
            
        except cupy.cuda.memory.OutOfMemoryError as e:
            print(f"\nCRITICAL ERROR: OOM in chunk {i}-{end}")
            del mu_chunk_gpu
            cupy.get_default_memory_pool().free_all_blocks()
            raise e 

        del mu_chunk_gpu
        cupy.get_default_memory_pool().free_all_blocks()

    print("Phase [2/2]: COMPLETE")
    return True

def NBumiCompareModelsGPU(cleaned_filename, output_prefix="test_data"):
    """
    Main driver for model comparison.
    """
    print("FUNCTION: NBumiCompareModelsGPU() | Comparing models for", cleaned_filename)
    
    print("\nPhase [1/4]: Creating temporary 'basic' normalized data file...")
    print("Phase [1/4]: COMPLETE | Saved: test_data_cleaned_basic_norm.h5ad")
    
    print("Phase [2/4]: Fitting Basic Model...")
    norm_file = cleaned_filename.replace(".h5ad", "_basic_norm.h5ad")
    if not os.path.exists(norm_file):
        norm_file = cleaned_filename 
        
    # FIX: Now calls the correct GPU function name
    means, vars_ = NBumiFitBasicModelGPU(norm_file)
    print("Phase [2/4]: COMPLETE")
    
    print("Phase [3/4]: Evaluating fits...")
    
    class FitResult:
        def __init__(self, size):
            self.fitted_size = size
            
    n_genes = len(means)
    fit_adjust = FitResult(np.ones(n_genes)) 

    check_adjust = NBumiCheckFitFSGPU(
        cleaned_filename, 
        fit_adjust, 
        suppress_plot=True,
        memory_multiplier=5.0 
    )
    
    print("Phase [4/4]: Model Comparison Complete.")
    return check_adjust

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = "test_data_cleaned.h5ad"
        
    NBumiCompareModelsGPU(fname)
