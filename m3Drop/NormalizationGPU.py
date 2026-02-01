import pickle
import time
import sys
import numpy as np
import h5py
import anndata
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import cupy
    from cupy.sparse import csr_matrix as cp_csr_matrix
    import cupyx
    HAS_GPU = True
except ImportError:
    cupy = None
    HAS_GPU = False

# Package-compatible import
try:
    from .ControlDeviceGPU import ControlDevice
except ImportError:
    try:
        from ControlDeviceGPU import ControlDevice
    except ImportError:
        print("CRITICAL ERROR: 'ControlDeviceGPU.py' not found.")
        sys.exit(1)

# ==========================================
#        KERNELS
# ==========================================

pearson_residual_kernel = cupy.ElementwiseKernel(
    'float64 count, float64 tj, float64 ti, float64 theta, float64 total', 'float64 out',
    '''
    double mu = (tj * ti) / total;
    double denom_sq = mu + ( (mu * mu) / theta );
    double denom = sqrt(denom_sq);
    if (denom < 1e-12) { out = (count == 0.0) ? 0.0 : 0.0; } else { out = (count - mu) / denom; }
    ''',
    'pearson_residual_kernel'
)

pearson_approx_kernel = cupy.ElementwiseKernel(
    'float64 count, float64 tj, float64 ti, float64 total', 'float64 out',
    '''
    double mu = (tj * ti) / total;
    double denom = sqrt(mu);
    if (denom < 1e-12) { out = 0.0; } else { out = (count - mu) / denom; }
    ''',
    'pearson_approx_kernel'
)

def NBumiPearsonResidualsCombinedGPU(
    raw_filename: str, 
    mask_filename: str, 
    fit_filename: str, 
    stats_filename: str,
    output_filename_full: str,
    output_filename_approx: str,
    plot_summary_filename: str = None,
    plot_detail_filename: str = None,
    mode: str = "auto",
    manual_target: int = 3000
):
    """
    Calculates Full and Approximate residuals in a SINGLE PASS.
    Includes "Sidecar" Visualization logic (Streaming Stats + Subsampling).
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResidualsCombined() | FILE: {raw_filename}")

    # 1. Load Mask
    with open(mask_filename, 'rb') as f: mask_cpu = pickle.load(f)
    mask_gpu = cupy.asarray(mask_cpu)
    ng_filtered = int(cupy.sum(mask_gpu))

    # 2. Manual Init
    with h5py.File(raw_filename, 'r') as f: 
        indptr_cpu = f['X']['indptr'][:]
        total_rows = len(indptr_cpu) - 1
        
    device = ControlDevice(indptr=indptr_cpu, total_rows=total_rows, n_genes=ng_filtered, mode=mode, manual_target=manual_target)
    nc = device.total_rows

    print("Phase [1/2]: Initializing parameters...")
    # Load parameters
    with open(fit_filename, 'rb') as f: fit = pickle.load(f)
    
    # Common params
    total = fit['vals']['total']
    tjs_gpu = cupy.asarray(fit['vals']['tjs'].values, dtype=cupy.float64)
    tis_gpu = cupy.asarray(fit['vals']['tis'].values, dtype=cupy.float64)
    sizes_gpu = cupy.asarray(fit['sizes'].values, dtype=cupy.float64)

    # Setup Output Files
    adata_in = anndata.read_h5ad(raw_filename, backed='r')
    filtered_var = adata_in.var[mask_cpu]
    
    # Create skeletons
    adata_out_full = anndata.AnnData(obs=adata_in.obs, var=filtered_var)
    adata_out_full.write_h5ad(output_filename_full, compression=None)
    
    adata_out_approx = anndata.AnnData(obs=adata_in.obs, var=filtered_var)
    adata_out_approx.write_h5ad(output_filename_approx, compression=None)
    
    # --- VISUALIZATION SETUP (THE SIDECAR) ---
    # 1. Sampling Rate (Target 5 Million Max)
    TARGET_SAMPLES = 5_000_000
    total_points = nc * ng_filtered
    
    if total_points <= TARGET_SAMPLES:
        sampling_rate = 1.0 # Take everything
    else:
        sampling_rate = TARGET_SAMPLES / total_points
        
    print(f"   > Visualization Sampling Rate: {sampling_rate*100:.4f}% (Target: {TARGET_SAMPLES:,} points)")

    # 2. Accumulators for Plot 1 (Variance) - EXACT MATH
    acc_raw_sum = cupy.zeros(ng_filtered, dtype=cupy.float64)
    
    acc_approx_sum = cupy.zeros(ng_filtered, dtype=cupy.float64)
    acc_approx_sq  = cupy.zeros(ng_filtered, dtype=cupy.float64)
    
    acc_full_sum   = cupy.zeros(ng_filtered, dtype=cupy.float64)
    acc_full_sq    = cupy.zeros(ng_filtered, dtype=cupy.float64)

    # 3. Lists for Plots 2 & 3 (Scatter/KDE) - SAMPLED
    viz_approx_samples = []
    viz_full_samples = []
    # -----------------------------------------

    # Storage Chunk Calc
    storage_chunk_rows = int(1_000_000_000 / (ng_filtered * 8)) 
    if storage_chunk_rows > nc: storage_chunk_rows = nc
    if storage_chunk_rows < 1: storage_chunk_rows = 1
    
    # Open files
    with h5py.File(output_filename_full, 'a') as f_full, h5py.File(output_filename_approx, 'a') as f_approx:
        if 'X' in f_full: del f_full['X']
        if 'X' in f_approx: del f_approx['X']
        
        out_x_full = f_full.create_dataset('X', shape=(nc, ng_filtered), chunks=(storage_chunk_rows, ng_filtered), dtype='float64')
        out_x_approx = f_approx.create_dataset('X', shape=(nc, ng_filtered), chunks=(storage_chunk_rows, ng_filtered), dtype='float64')

        with h5py.File(raw_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']
            
            current_row = 0
            while current_row < nc:
                # [SAFE MODE RESTORED] Multiplier 3.0 is efficient because we use IN-PLACE ops below.
                end_row = device.get_next_chunk(current_row, mode='dense', overhead_multiplier=3.0)
                if end_row is None or end_row <= current_row: break

                chunk_size = end_row - current_row
                print(f"Phase [2/2]: Processing rows {end_row} of {nc} | Chunk: {chunk_size}", end='\r')

                start_idx, end_idx = h5_indptr[current_row], h5_indptr[end_row]
                
                # Load Raw
                data_gpu_raw = cupy.asarray(h5_data[start_idx:end_idx], dtype=cupy.float64)
                indices_gpu_raw = cupy.asarray(h5_indices[start_idx:end_idx])
                indptr_gpu_raw = cupy.asarray(h5_indptr[current_row:end_row+1] - h5_indptr[current_row])
                
                chunk_gpu = cp_csr_matrix((data_gpu_raw, indices_gpu_raw, indptr_gpu_raw), shape=(chunk_size, len(mask_cpu)))
                chunk_gpu = chunk_gpu[:, mask_gpu]
                chunk_gpu.data = cupy.ceil(chunk_gpu.data)

                # Dense Conversion
                counts_dense = chunk_gpu.todense()
                del chunk_gpu, data_gpu_raw, indices_gpu_raw, indptr_gpu_raw
                cupy.get_default_memory_pool().free_all_blocks()

                # --- VIZ ACCUMULATION 1: RAW MEAN ---
                # Add raw sums to accumulator (column-wise sum)
                acc_raw_sum += cupy.sum(counts_dense, axis=0)
                
                # --- VIZ SAMPLING: GENERATE INDICES ---
                chunk_total_items = chunk_size * ng_filtered
                n_samples_chunk = int(chunk_total_items * sampling_rate)
                
                if n_samples_chunk > 0:
                    # [SAFE] Use randint (with replacement) to avoid VRAM spike
                    sample_indices = cupy.random.randint(0, int(chunk_total_items), size=n_samples_chunk)
                else:
                    sample_indices = None

                # ============================================
                #   CALC 1: APPROX (Optimize Order of Ops)
                # ============================================
                approx_out = cupy.empty_like(counts_dense)
                pearson_approx_kernel(
                    counts_dense,
                    tjs_gpu,
                    tis_gpu[current_row:end_row][:, cupy.newaxis], 
                    total,
                    approx_out 
                )
                
                # 1. Accumulate Sum (First Moment)
                acc_approx_sum += cupy.sum(approx_out, axis=0)
                
                # 2. Sample (Before we destroy the data)
                if sample_indices is not None:
                    sampled_vals = approx_out.ravel().take(sample_indices)
                    viz_approx_samples.append(cupy.asnumpy(sampled_vals))

                # 3. Write to Disk (Save the clean residuals)
                out_x_approx[current_row:end_row, :] = approx_out.get()

                # 4. Square IN-PLACE (Destroying VRAM copy to create squares without allocation)
                approx_out *= approx_out 
                
                # 5. Accumulate Sum of Squares (Second Moment)
                acc_approx_sq += cupy.sum(approx_out, axis=0)
                
                del approx_out

                # ============================================
                #   CALC 2: FULL (Optimize Order of Ops)
                # ============================================
                pearson_residual_kernel(
                    counts_dense,
                    tjs_gpu,
                    tis_gpu[current_row:end_row][:, cupy.newaxis], 
                    sizes_gpu,
                    total,
                    counts_dense # Overwrite input with Residuals
                )
                
                # 1. Accumulate Sum
                acc_full_sum += cupy.sum(counts_dense, axis=0)
                
                # 2. Sample
                if sample_indices is not None:
                    sampled_vals = counts_dense.ravel().take(sample_indices)
                    viz_full_samples.append(cupy.asnumpy(sampled_vals))

                # 3. Write to Disk
                out_x_full[current_row:end_row, :] = counts_dense.get()

                # 4. Square IN-PLACE
                counts_dense *= counts_dense
                
                # 5. Accumulate Sum of Squares
                acc_full_sq += cupy.sum(counts_dense, axis=0)
                
                del counts_dense, sample_indices
                cupy.get_default_memory_pool().free_all_blocks()
                current_row = end_row
        
        print(f"\nPhase [2/2]: COMPLETE{' '*50}")

    # ==========================================
    #        VIZ GENERATION (POST-PROCESS)
    # ==========================================
    if plot_summary_filename and plot_detail_filename:
        print("Phase [Viz]: Generating Diagnostics...")
        
        # 1. Finalize Variance Stats (GPU -> CPU)
        raw_sum = cupy.asnumpy(acc_raw_sum)
        
        approx_sum = cupy.asnumpy(acc_approx_sum)
        approx_sq  = cupy.asnumpy(acc_approx_sq)
        
        full_sum   = cupy.asnumpy(acc_full_sum)
        full_sq    = cupy.asnumpy(acc_full_sq)
        
        # Calculate Variance: E[X^2] - (E[X])^2
        mean_raw = raw_sum / nc
        
        mean_approx = approx_sum / nc
        mean_sq_approx = approx_sq / nc
        var_approx = mean_sq_approx - (mean_approx**2)
        
        mean_full = full_sum / nc
        mean_sq_full = full_sq / nc
        var_full = mean_sq_full - (mean_full**2)

        # 2. Finalize Samples
        if viz_approx_samples:
            flat_approx = np.concatenate(viz_approx_samples)
            flat_full   = np.concatenate(viz_full_samples)
        else:
            flat_approx = np.array([])
            flat_full = np.array([])
            
        print(f"   > Samples Collected: {len(flat_approx):,} points")

        # --- FILE 1: SUMMARY (1080p) ---
        print(f"   > Saving Summary Plot: {plot_summary_filename}")
        fig1, ax1 = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Variance Stabilization
        ax = ax1[0]
        ax.scatter(mean_raw, var_approx, s=2, alpha=0.5, color='red', label='Approx (Poisson)')
        ax.scatter(mean_raw, var_full, s=2, alpha=0.5, color='blue', label='Full (NB Pearson)')
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("Variance Stabilization Check")
        ax.set_xlabel("Mean Raw Expression (log)")
        ax.set_ylabel("Variance of Residuals (log)")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5) 
        ax.text(0.5, -0.15, "Goal: Blue dots should form a flat line at y=1", 
                transform=ax.transAxes, ha='center', fontsize=9, 
                bbox=dict(facecolor='#f0f0f0', edgecolor='black', alpha=0.7))

        # Plot 3: Distribution (Histogram + KDE Overlay) - LOG SCALE FIXED
        ax = ax1[1]
        if len(flat_approx) > 100:
            mask_kde = (flat_approx > -10) & (flat_approx < 10)
            
            # Histograms (The Truth)
            bins = np.linspace(-5, 5, 100)
            ax.hist(flat_approx[mask_kde], bins=bins, color='red', alpha=0.2, density=True, label='_nolegend_')
            ax.hist(flat_full[mask_kde], bins=bins, color='blue', alpha=0.2, density=True, label='_nolegend_')

            # KDEs (The Trend)
            sns.kdeplot(flat_approx[mask_kde], fill=False, color='red', linewidth=2, label='Approx', ax=ax, warn_singular=False)
            sns.kdeplot(flat_full[mask_kde], fill=False, color='blue', linewidth=2, label='Full', ax=ax, warn_singular=False)

        ax.set_yscale('log') # <--- THE CRITICAL FIX FOR "SHIT GRAPH"
        ax.set_ylim(bottom=0.001) # Safety floor for log(0)
        ax.set_xlim(-5, 5)
        ax.set_title("Distribution of Residuals (Log Scale)")
        ax.set_xlabel("Residual Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(0.5, -0.15, "Goal: Blue curve should be tighter (narrower) than Red", 
                transform=ax.transAxes, ha='center', fontsize=9, 
                bbox=dict(facecolor='#f0f0f0', edgecolor='black', alpha=0.7))

        plt.tight_layout()
        plt.savefig(plot_summary_filename, dpi=120) 
        plt.close()

        # --- FILE 2: DETAIL (4K) ---
        print(f"   > Saving Detail Plot: {plot_detail_filename}")
        fig2, ax2 = plt.subplots(figsize=(20, 11))
        
        if len(flat_approx) > 0:
            ax2.scatter(flat_approx, flat_full, s=1, alpha=0.5, color='purple')
            
            lims = [
                np.min([ax2.get_xlim(), ax2.get_ylim()]),
                np.max([ax2.get_xlim(), ax2.get_ylim()]),
            ]
            ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        
        ax2.set_title("Residual Shrinkage (Sampled)")
        ax2.set_xlabel("Approx Residuals")
        ax2.set_ylabel("Full Residuals")
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, -0.1, "Goal: Points below diagonal = Dispersion Penalty Working", 
                transform=ax2.transAxes, ha='center', fontsize=12, 
                bbox=dict(facecolor='#f0f0f0', edgecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(plot_detail_filename, dpi=200)
        plt.close()

    if hasattr(adata_in, "file") and adata_in.file is not None: adata_in.file.close()
    print(f"Total time: {time.perf_counter() - start_time:.2f} seconds.\n")
