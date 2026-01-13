import numpy as np
import pandas as pd
import cupy as cp
import cupyx.scipy.sparse as csp
import matplotlib.pyplot as plt
import h5py
import os
import time
import psutil
import gc
from scipy import sparse
from scipy import stats

# --- RESTORED IMPORT FROM CORE ---
from .coreGPU import NBumiFitDispVsMeanGPU

# ==============================================================================
# GPU MEMORY GOVERNOR & OPTIMIZER
# ==============================================================================

def get_slurm_memory_limit():
    """Detects SLURM memory limits or defaults to system RAM."""
    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if mem_per_cpu:
        return int(mem_per_cpu) * 1024 * 1024
    
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_per_node:
        return int(mem_per_node) * 1024 * 1024
        
    return psutil.virtual_memory().total

def calculate_optimal_chunk_size(n_vars, dtype_size=4, memory_multiplier=3.0, override_cap=None):
    """Calculates safe chunk size based on available VRAM and RAM."""
    try:
        gpu_mem_info = cp.cuda.runtime.memGetInfo()
        free_vram = gpu_mem_info[0]
    except Exception:
        print("WARNING: No GPU detected or CuPy error. Defaulting to safe small chunk.")
        return 5000

    available_ram = get_slurm_memory_limit() * 0.8
    
    # Cost calculation
    row_cost_vram = n_vars * dtype_size * memory_multiplier
    row_cost_ram = n_vars * dtype_size * 2.0
    
    max_rows_vram = int(free_vram / row_cost_vram)
    max_rows_ram = int(available_ram / row_cost_ram)
    
    optimal_chunk = min(max_rows_vram, max_rows_ram)
    
    if override_cap:
        optimal_chunk = min(optimal_chunk, override_cap)
        
    print(f"DEBUG: Chunk Optimizer -> VRAM Free: {free_vram/1e9:.2f}GB | Chunk Size: {optimal_chunk}")
    return max(1, optimal_chunk)

# ==============================================================================
# HELPER FUNCTIONS 
# ==============================================================================

def get_basic_stats(adata_path, chunk_size=10000):
    """
    Calculates basic stats using h5py streaming to avoid System RAM OOM.
    Does NOT load the full AnnData object.
    """
    print(f"DEBUG: calculating stats for {adata_path} (Stream Mode)")
    
    with h5py.File(adata_path, 'r') as f:
        # 1. Determine dimensions from HDF5 attributes
        if 'X' not in f:
             raise KeyError(f"File {adata_path} has no 'X' group/dataset.")
             
        x_obj = f['X']
        if isinstance(x_obj, h5py.Group):
            # Sparse Matrix
            shape = x_obj.attrs.get('shape')
            if shape is None:
                nc = x_obj['indptr'].shape[0] - 1
                ng = f['var'].shape[0] if 'var' in f else 0 
            else:
                nc, ng = shape
            is_sparse = True
            h5_data = x_obj['data']
            h5_indices = x_obj['indices']
            h5_indptr = x_obj['indptr']
        else:
            # Dense Matrix
            nc, ng = x_obj.shape
            is_sparse = False
            
        # 2. Allocate GPU Accumulators
        sum_x = cp.zeros(ng, dtype=cp.float64)
        djs = cp.zeros(ng, dtype=cp.int64) 
        dis = cp.zeros(nc, dtype=cp.int64) 
        tis = cp.zeros(nc, dtype=cp.float64) 
        
        # 3. Stream Data
        for i in range(0, nc, chunk_size):
            end = min(i + chunk_size, nc)
            
            if is_sparse:
                # Manual CSR Slicing from disk
                p_start = h5_indptr[i]
                p_end = h5_indptr[end]
                
                d_chunk = h5_data[p_start:p_end]
                i_chunk = h5_indices[p_start:p_end]
                ptr_chunk = h5_indptr[i : end + 1]
                ptr_chunk = ptr_chunk - ptr_chunk[0]
                
                # Construct CSR -> Dense GPU
                chunk_csr = sparse.csr_matrix((d_chunk, i_chunk, ptr_chunk), shape=(end-i, ng))
                chunk_gpu = csp.csr_matrix(chunk_csr, dtype=cp.float32)
                chunk_gpu = chunk_gpu.toarray() 
                
            else:
                chunk = x_obj[i:end]
                chunk_gpu = cp.asarray(chunk, dtype=cp.float32)
            
            # 4. Compute Stats
            sum_x += cp.sum(chunk_gpu, axis=0)
            is_zero = (chunk_gpu == 0)
            djs += cp.sum(is_zero, axis=0)
            dis[i:end] = cp.sum(is_zero, axis=1)
            tis[i:end] = cp.sum(chunk_gpu, axis=1)
            
            # Cleanup
            del chunk_gpu, is_zero
            if is_sparse: del chunk_csr, d_chunk, i_chunk, ptr_chunk
            cp.get_default_memory_pool().free_all_blocks()

        # 5. Reconstruct Indices (Lightweight)
        try:
            if 'var' in f and '_index' in f['var']:
                var_names = f['var']['_index'][:].astype(str)
            elif 'var' in f and 'index' in f['var']: 
                var_names = f['var']['index'][:].astype(str)
            else:
                var_names = np.arange(ng).astype(str)
                
            if 'obs' in f and '_index' in f['obs']:
                obs_names = f['obs']['_index'][:].astype(str)
            elif 'obs' in f and 'index' in f['obs']:
                obs_names = f['obs']['index'][:].astype(str)
            else:
                obs_names = np.arange(nc).astype(str)
        except Exception:
            var_names = np.arange(ng).astype(str)
            obs_names = np.arange(nc).astype(str)

    return {
        'nc': nc,
        'ng': ng,
        'tjs': pd.Series(cp.asnumpy(sum_x), index=var_names),
        'djs': pd.Series(cp.asnumpy(djs), index=var_names),
        'tis': pd.Series(cp.asnumpy(tis), index=obs_names),
        'dis': pd.Series(cp.asnumpy(dis), index=obs_names),
        'total': cp.asnumpy(cp.sum(tis))
    }

# ==============================================================================
# PIPELINE FUNCTIONS
# ==============================================================================

def NBumiFitBasicModelGPU(
    cleaned_filename: str,
    stats: dict,
    chunk_size: int = None
) -> dict:
    """
    Fits the basic NB model (Method of Moments) on GPU.
    Optimized to use h5py streaming for variance calculation to prevent OOM.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitBasicModelGPU() | FILE: {cleaned_filename}")

    nc, ng = stats['nc'], stats['ng']
    tjs = cp.asarray(stats['tjs'].values, dtype=cp.float64)
    
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(ng, dtype_size=4, memory_multiplier=3.0)

    # 1. Calculate Sum of Squares (Variance)
    sum_x_sq = cp.zeros(ng, dtype=cp.float64)
    
    print("Phase [1/2]: Calculating variance from data chunks (GPU Stream)...")
    
    # Use h5py instead of sc.read_h5ad to save RAM
    with h5py.File(cleaned_filename, 'r') as f:
        x_obj = f['X']
        if isinstance(x_obj, h5py.Group):
            is_sparse = True
            h5_data = x_obj['data']
            h5_indices = x_obj['indices']
            h5_indptr = x_obj['indptr']
        else:
            is_sparse = False

        for i in range(0, nc, chunk_size):
            end = min(i + chunk_size, nc)
            print(f"Phase [1/2]: Processing: {end} of {nc} cells.", end='\r')
            
            if is_sparse:
                p_start = h5_indptr[i]
                p_end = h5_indptr[end]
                d_chunk = h5_data[p_start:p_end]
                i_chunk = h5_indices[p_start:p_end]
                ptr_chunk = h5_indptr[i : end + 1] - h5_indptr[i]
                
                chunk_csr = sparse.csr_matrix((d_chunk, i_chunk, ptr_chunk), shape=(end-i, ng))
                chunk_gpu = csp.csr_matrix(chunk_csr, dtype=cp.float32).toarray()
            else:
                chunk = x_obj[i:end]
                chunk_gpu = cp.asarray(chunk, dtype=cp.float32)

            sum_x_sq += cp.sum(chunk_gpu**2, axis=0)
            
            del chunk_gpu
            if is_sparse: del chunk_csr
            cp.get_default_memory_pool().free_all_blocks()
        
    print(f"\nPhase [1/2]: COMPLETE")

    # 2. Method of Moments Logic
    mean_x_sq = cp.asnumpy(sum_x_sq) / nc
    mean_mu = cp.asnumpy(tjs) / nc
    
    my_rowvar = mean_x_sq - mean_mu**2
    
    # Calculate k (sizes)
    numerator = mean_mu**2
    denominator = my_rowvar - mean_mu
    
    sizes = np.full(ng, np.nan, dtype=np.float64)
    valid_mask = denominator > 1e-12
    sizes[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    # Sanitization
    finite_sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    max_size_val = np.max(finite_sizes) * 10 if finite_sizes.size else 1000
    sizes[~np.isfinite(sizes) | (sizes <= 0)] = max_size_val
    sizes[sizes < 1e-10] = 1e-10
    
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return {
        'var_obs': pd.Series(my_rowvar, index=stats['tjs'].index),
        'sizes': pd.Series(sizes, index=stats['tjs'].index),
        'vals': stats
    }


def NBumiCheckFitFSGPU(
    cleaned_filename: str,
    fit: dict,
    chunk_size: int = None,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    GPU version of NBumiCheckFitFS. 
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCheckFitFSGPU() | FILE: {cleaned_filename}")

    # 1. Get Smoothed Size Parameters (Regression)
    # RESTORED: Calling the coreGPU function
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)
    intercept, slope = coeffs[0], coeffs[1]
    
    vals = fit['vals']
    nc, ng = vals['nc'], vals['ng']
    tjs = vals['tjs'].values.astype(np.float64)
    tis = vals['tis'].values.astype(np.float64)
    total = vals['total']

    # Calculate Smoothed Sizes for all genes
    mean_expression = tjs / nc
    log_mean_expression = np.log(mean_expression, where=(mean_expression > 0))
    smoothed_size = np.exp(intercept + slope * log_mean_expression)
    smoothed_size = np.nan_to_num(smoothed_size, nan=1.0, posinf=1e6, neginf=1.0)
    
    tjs_gpu = cp.asarray(tjs, dtype=cp.float32)
    smoothed_size_gpu = cp.asarray(smoothed_size, dtype=cp.float32)
    total_gpu = float(total)
    tis_gpu_full = cp.asarray(tis, dtype=cp.float32)

    row_ps = cp.zeros(ng, dtype=cp.float64)
    col_ps = cp.zeros(nc, dtype=cp.float64)
    
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(ng, dtype_size=4, memory_multiplier=5.0)

    print("Phase [2/2]: Calculating expected dropouts (GPU)...")
    for i in range(0, nc, chunk_size):
        end = min(i + chunk_size, nc)
        print(f"Phase [2/2]: Processing: {end} of {nc} cells.", end='\r')
        
        tis_chunk = tis_gpu_full[i:end]
        
        try:
            # 1. Calculate Mean Matrix (Mu)
            mu_chunk = cp.outer(tis_chunk, tjs_gpu) 
            mu_chunk /= total_gpu
            
            # 2. Calculate Base: (1 + Mu / Size)
            base = 1.0 + (mu_chunk / smoothed_size_gpu)
            
            # 3. Calculate Probability: Base^(-Size)
            p_is_chunk = cp.power(base, -smoothed_size_gpu)
            p_is_chunk = cp.nan_to_num(p_is_chunk, nan=0.0)
            
            # 4. Accumulate
            row_ps += cp.sum(p_is_chunk, axis=0)
            col_ps[i:end] = cp.sum(p_is_chunk, axis=1)
            
            del mu_chunk, base, p_is_chunk
            cp.get_default_memory_pool().free_all_blocks()
            
        except cp.cuda.memory.OutOfMemoryError:
            print(f"\nOOM in Chunk {i}. Try reducing chunk_size further.")
            raise

    print(f"\nPhase [2/2]: COMPLETE{' '*20}")

    row_ps_cpu = cp.asnumpy(row_ps)
    col_ps_cpu = cp.asnumpy(col_ps)
    
    djs = vals['djs'].values
    dis = vals['dis'].values
    
    gene_error = np.sum((djs - row_ps_cpu)**2)
    cell_error = np.sum((dis - col_ps_cpu)**2)

    if not suppress_plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(djs, row_ps_cpu, alpha=0.5, s=10)
        plt.title("Gene-specific Dropouts (GPU Fit)")
        plt.xlabel("Observed")
        plt.ylabel("Fit")
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r-')
        
        plt.subplot(1, 2, 2)
        plt.scatter(dis, col_ps_cpu, alpha=0.5, s=10)
        plt.title("Cell-specific Dropouts (GPU Fit)")
        plt.xlabel("Observed")
        plt.ylabel("Expected")
        plt.plot(lims, lims, 'r-')
        
        plt.tight_layout()
        if plot_filename:
            plt.savefig(plot_filename)
        plt.show()
        plt.close()

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return {
        'gene_error': gene_error,
        'cell_error': cell_error,
        'rowPs': pd.Series(row_ps_cpu, index=fit['vals']['tjs'].index),
        'colPs': pd.Series(col_ps_cpu, index=fit['vals']['tis'].index)
    }


def NBumiCompareModelsGPU(
    raw_filename: str,
    cleaned_filename: str,
    stats: dict,
    fit_adjust: dict,
    chunk_size: int = None,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    GPU version of Model Comparison.
    Uses direct h5py I/O to avoid System RAM OOM.
    """
    pipeline_start = time.time()
    print(f"FUNCTION: NBumiCompareModelsGPU() | Comparing models for {cleaned_filename}")

    # --- Phase 1: Normalization ---
    print("Phase [1/4]: Creating temporary 'basic' normalized data file...")
    basic_norm_filename = cleaned_filename.replace('.h5ad', '_basic_norm.h5ad')
    
    if os.path.exists(basic_norm_filename):
        os.remove(basic_norm_filename)

    # Manual Metadata Read (Lightweight)
    with h5py.File(cleaned_filename, 'r') as f_in:
        nc, ng = f_in['X'].attrs['shape'] if 'shape' in f_in['X'].attrs else f_in['X'].shape
        # Copy Obs/Var to new file logic is tricky without AnnData.
        # To avoid OOM, we will ONLY write X to the temp file and use indices from stats.
        # This is a deviation from CPU structure but necessary for RAM safety.
        # However, to keep it compatible with get_basic_stats, we just need a valid X.
    
    cell_sums = stats['tis'].values.astype(np.float64)
    positive_mask = cell_sums > 0
    median_sum = np.median(cell_sums[positive_mask]) if np.any(positive_mask) else 1.0
    size_factors = np.ones_like(cell_sums, dtype=np.float32)
    size_factors[positive_mask] = cell_sums[positive_mask] / median_sum
    
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(ng, dtype_size=4, memory_multiplier=4.0)

    # Create Output H5 (Raw h5py, no AnnData overhead)
    with h5py.File(basic_norm_filename, 'w') as f_out:
        x_grp = f_out.create_group('X')
        x_grp.attrs['encoding-type'] = 'csr_matrix'
        x_grp.attrs['shape'] = np.array([nc, ng], dtype='int64')
        
        d_data = x_grp.create_dataset('data', shape=(0,), maxshape=(None,), dtype='float32')
        d_indices = x_grp.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int32')
        d_indptr = x_grp.create_dataset('indptr', shape=(nc+1,), dtype='int64')
        d_indptr[0] = 0
        
        current_nnz = 0
        
        # Open Input Streaming
        with h5py.File(cleaned_filename, 'r') as f_in:
            x_in = f_in['X']
            is_sparse = isinstance(x_in, h5py.Group)
            
            if is_sparse:
                h5_data = x_in['data']
                h5_indices = x_in['indices']
                h5_indptr = x_in['indptr']
            
            for i in range(0, nc, chunk_size):
                end = min(i+chunk_size, nc)
                print(f"Phase [1/4]: Normalizing {end}/{nc}", end='\r')
                
                if is_sparse:
                    p_start = h5_indptr[i]
                    p_end = h5_indptr[end]
                    d_chunk = h5_data[p_start:p_end]
                    i_chunk = h5_indices[p_start:p_end]
                    ptr_chunk = h5_indptr[i : end + 1] - h5_indptr[i]
                    
                    chunk_csr = sparse.csr_matrix((d_chunk, i_chunk, ptr_chunk), shape=(end-i, ng))
                    chunk_gpu = csp.csr_matrix(chunk_csr, dtype=cp.float32).toarray()
                else:
                    chunk = x_in[i:end]
                    chunk_gpu = cp.asarray(chunk, dtype=cp.float32)
                
                # Normalize
                factors_chunk = cp.asarray(size_factors[i:end], dtype=cp.float32)
                chunk_gpu = chunk_gpu / factors_chunk[:, None]
                chunk_gpu = cp.round(chunk_gpu) 
                
                # Convert back to sparse CSR on GPU
                chunk_csr = csp.csr_matrix(chunk_gpu)
                
                data_cpu = cp.asnumpy(chunk_csr.data)
                indices_cpu = cp.asnumpy(chunk_csr.indices)
                indptr_cpu = cp.asnumpy(chunk_csr.indptr)
                
                nnz = len(data_cpu)
                
                d_data.resize(current_nnz + nnz, axis=0)
                d_data[current_nnz:] = data_cpu
                d_indices.resize(current_nnz + nnz, axis=0)
                d_indices[current_nnz:] = indices_cpu
                new_indptr = indptr_cpu[1:] + current_nnz
                d_indptr[i+1 : end+1] = new_indptr
                
                current_nnz += nnz
                
                del chunk_gpu, chunk_csr
                cp.get_default_memory_pool().free_all_blocks()

    print(f"\nPhase [1/4]: COMPLETE{' '*20}")
    gc.collect()

    # --- Phase 2: Fit Basic Model ---
    print("Phase [2/4]: Fitting Basic Model on normalized data...")
    # NOTE: Both functions now use h5py streaming. No sc.read_h5ad calls.
    stats_basic = get_basic_stats(basic_norm_filename, chunk_size=chunk_size)
    fit_basic = NBumiFitBasicModelGPU(basic_norm_filename, stats_basic, chunk_size=chunk_size)
    print("Phase [2/4]: COMPLETE")

    # --- Phase 3: Evaluate Fits ---
    print("Phase [3/4]: Evaluating fits...")
    check_adjust = NBumiCheckFitFSGPU(cleaned_filename, fit_adjust, suppress_plot=True, chunk_size=chunk_size)
    check_basic = NBumiCheckFitFSGPU(cleaned_filename, fit_basic, suppress_plot=True, chunk_size=chunk_size)
    print("Phase [3/4]: COMPLETE")

    # --- Phase 4: Compare ---
    print("Phase [4/4]: Generating comparison stats...")
    nc_data = stats['nc']
    mean_expr = stats['tjs'] / nc_data
    observed_dropout = stats['djs'] / nc_data
    
    adj_dropout_fit = check_adjust['rowPs'] / nc_data
    bas_dropout_fit = check_basic['rowPs'] / nc_data
    
    err_adj = np.sum(np.abs(adj_dropout_fit - observed_dropout))
    err_bas = np.sum(np.abs(bas_dropout_fit - observed_dropout))
    
    comparison_df = pd.DataFrame({
        'mean_expr': mean_expr,
        'observed': observed_dropout,
        'adj_fit': adj_dropout_fit,
        'bas_fit': bas_dropout_fit
    })

    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(mean_expr.values)
    
    plt.scatter(mean_expr.iloc[sorted_idx], observed_dropout.iloc[sorted_idx], 
                c='black', s=3, alpha=0.5, label='Observed')
    plt.scatter(mean_expr.iloc[sorted_idx], bas_dropout_fit.iloc[sorted_idx], 
                c='purple', s=3, alpha=0.6, label=f'Basic Fit (Err: {err_bas:.1f})')
    plt.scatter(mean_expr.iloc[sorted_idx], adj_dropout_fit.iloc[sorted_idx], 
                c='goldenrod', s=3, alpha=0.7, label=f'Depth-Adj Fit (Err: {err_adj:.1f})')
    
    plt.xscale('log')
    plt.title("M3Drop Model Comparison (GPU Accelerated)")
    plt.xlabel("Mean Expression"); plt.ylabel("Dropout Rate")
    plt.legend(); plt.grid(True, alpha=0.3)
    
    if plot_filename:
        plt.savefig(plot_filename)
    if not suppress_plot:
        plt.show()
    plt.close()
    
    if os.path.exists(basic_norm_filename):
        os.remove(basic_norm_filename)
        
    print(f"Total time: {time.time() - pipeline_start:.2f} seconds.")
    
    return {
        "errors": {"Depth-Adjusted": err_adj, "Basic": err_bas},
        "comparison_df": comparison_df
    }

def NBumiPlotDispVsMeanGPU(
    fit: dict,
    suppress_plot: bool = False,
    plot_filename: str = None
):
    """
    Generates a diagnostic plot of the dispersion vs. mean expression (GPU version).
    """
    print("FUNCTION: NBumiPlotDispVsMeanGPU()")

    stats = fit['vals']
    nc = stats['nc']
    
    mean_expression = stats['tjs'].values / nc
    sizes = fit['sizes'].values

    # RESTORED: Calling the coreGPU function
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)
    intercept, slope = coeffs[0], coeffs[1]

    positive_means = mean_expression[mean_expression > 0]
    if positive_means.size == 0:
        print("WARNING: No positive mean expression values. Skipping plot.")
        return

    log_mean_expr_range = np.linspace(
        np.log(positive_means.min()),
        np.log(positive_means.max()),
        100
    )
    
    log_fitted_sizes = intercept + slope * log_mean_expr_range
    fitted_sizes = np.exp(log_fitted_sizes)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean_expression, sizes, label='Observed Dispersion', alpha=0.5, s=8)
    plt.plot(np.exp(log_mean_expr_range), fitted_sizes, color='red', label='Regression Fit', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mean Expression')
    plt.ylabel('Dispersion Parameter (Sizes)')
    plt.title('Dispersion vs. Mean Expression (GPU)')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.6)

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"STATUS: Diagnostic plot saved to '{plot_filename}'")

    if not suppress_plot:
        plt.show()

    plt.close()
    print("FUNCTION: NBumiPlotDispVsMeanGPU() COMPLETE\n")

if __name__ == "__main__":
    print("NBumi GPU Diagnostics Module Loaded.")
