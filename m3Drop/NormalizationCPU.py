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
from scipy import sparse

try:
    from numba import jit, prange
except ImportError:
    print("CRITICAL ERROR: 'numba' not found. Please install it (pip install numba).")
    sys.exit(1)

# Strict Relative Import
from .ControlDeviceCPU import ControlDevice

# ==========================================
#        NUMBA KERNELS (CPU)
# ==========================================

@jit(nopython=True, parallel=True, fastmath=True)
def pearson_residual_kernel_cpu(counts, tj, ti, theta, total, out_matrix):
    rows = counts.shape[0]
    cols = counts.shape[1]
    for r in prange(rows):
        ti_val = ti[r]
        for c in range(cols):
            count_val = counts[r, c]
            mu = (tj[c] * ti_val) / total
            theta_val = theta[c]
            denom_sq = mu + ((mu * mu) / theta_val)
            denom = np.sqrt(denom_sq)
            if denom < 1e-12:
                out_matrix[r, c] = 0.0
            else:
                out_matrix[r, c] = (count_val - mu) / denom

@jit(nopython=True, parallel=True, fastmath=True)
def pearson_approx_kernel_cpu(counts, tj, ti, total, out_matrix):
    rows = counts.shape[0]
    cols = counts.shape[1]
    for r in prange(rows):
        ti_val = ti[r]
        for c in range(cols):
            count_val = counts[r, c]
            mu = (tj[c] * ti_val) / total
            denom = np.sqrt(mu)
            if denom < 1e-12:
                out_matrix[r, c] = 0.0
            else:
                out_matrix[r, c] = (count_val - mu) / denom

# ==========================================
#        NORMALIZATION FUNCTION
# ==========================================

def NBumiPearsonResidualsCombinedCPU(
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
    CPU-Optimized: Calculates Full and Approximate residuals in a SINGLE PASS.
    Includes "Sidecar" Visualization logic (Streaming Stats + Subsampling).
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResidualsCombinedCPU() | FILE: {raw_filename}")

    # 1. Load Mask
    with open(mask_filename, 'rb') as f: mask = pickle.load(f)
    ng_filtered = int(np.sum(mask))

    # 2. Init Device
    with h5py.File(raw_filename, 'r') as f: indptr_cpu = f['X']['indptr'][:]; total_rows = len(indptr_cpu) - 1
    device = ControlDevice(indptr=indptr_cpu, total_rows=total_rows, n_genes=ng_filtered, mode=mode, manual_target=manual_target)
    nc = device.total_rows

    print("Phase [1/2]: Initializing parameters...")
    with open(fit_filename, 'rb') as f: fit = pickle.load(f)
    
    total = fit['vals']['total']
    tjs = fit['vals']['tjs'].values.astype(np.float64)
    tis = fit['vals']['tis'].values.astype(np.float64)
    sizes = fit['sizes'].values.astype(np.float64)

    # Setup Output Files
    adata_in = anndata.read_h5ad(raw_filename, backed='r')
    filtered_var = adata_in.var[mask]
    
    adata_out_full = anndata.AnnData(obs=adata_in.obs, var=filtered_var)
    adata_out_full.write_h5ad(output_filename_full, compression=None)
    
    adata_out_approx = anndata.AnnData(obs=adata_in.obs, var=filtered_var)
    adata_out_approx.write_h5ad(output_filename_approx, compression=None)
    
    # --- VISUALIZATION SETUP (THE SIDECAR) ---
    # 1. Sampling Rate (Strict Cap to prevent CPU RAM explosion)
    TARGET_SAMPLES = 5_000_000
    total_points = nc * ng_filtered
    
    if total_points <= TARGET_SAMPLES:
        sampling_rate = 1.0 
    else:
        sampling_rate = TARGET_SAMPLES / total_points
        
    print(f"Phase [1/2]: Visualization Sampling Rate: {sampling_rate*100:.4f}% (Target: {TARGET_SAMPLES:,} points)")

    # 2. Accumulators (Numpy Arrays - Small memory footprint)
    acc_raw_sum    = np.zeros(ng_filtered, dtype=np.float64)
    acc_approx_sum = np.zeros(ng_filtered, dtype=np.float64)
    acc_approx_sq  = np.zeros(ng_filtered, dtype=np.float64)
    acc_full_sum   = np.zeros(ng_filtered, dtype=np.float64)
    acc_full_sq    = np.zeros(ng_filtered, dtype=np.float64)

    # 3. Lists for Plots (Sampled Only)
    viz_approx_samples = []
    viz_full_samples = []
    # -----------------------------------------

    storage_chunk_rows = int(1_000_000_000 / (ng_filtered * 8)) 
    if storage_chunk_rows > nc: storage_chunk_rows = nc
    if storage_chunk_rows < 1: storage_chunk_rows = 1
    
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
                end_row = device.get_next_chunk(current_row, mode='dense', overhead_multiplier=3.0) 
                if end_row is None or end_row <= current_row: break

                chunk_size = end_row - current_row
                print(f"Phase [2/2]: Processing rows {end_row} of {nc} | Chunk: {chunk_size}", end='\r')

                start_idx, end_idx = h5_indptr[current_row], h5_indptr[end_row]
                
                data = np.array(h5_data[start_idx:end_idx], dtype=np.float64)
                indices = np.array(h5_indices[start_idx:end_idx])
                indptr = np.array(h5_indptr[current_row:end_row+1] - h5_indptr[current_row])
                
                chunk_csr = sparse.csr_matrix((data, indices, indptr), shape=(chunk_size, len(mask)))
                chunk_csr = chunk_csr[:, mask]
                chunk_csr.data = np.ceil(chunk_csr.data)

                # Numba needs dense
                counts_dense = chunk_csr.toarray()

                # --- VIZ ACCUMULATION 1: RAW MEAN ---
                acc_raw_sum += np.sum(counts_dense, axis=0)

                # --- VIZ SAMPLING: GENERATE INDICES ---
                chunk_total_items = chunk_size * ng_filtered
                n_samples_chunk = int(chunk_total_items * sampling_rate)
                sample_indices = None
                
                if n_samples_chunk > 0:
                    sample_indices = np.random.randint(0, int(chunk_total_items), size=n_samples_chunk)
                
                # --- CALC 1: APPROX ---
                approx_out = np.empty_like(counts_dense)
                pearson_approx_kernel_cpu(
                    counts_dense,
                    tjs,
                    tis[current_row:end_row], 
                    total,
                    approx_out 
                )
                
                # Accumulate
                acc_approx_sum += np.sum(approx_out, axis=0)
                
                # Sample
                if sample_indices is not None:
                    # Ravel creates a view, take copies the data. Safe.
                    viz_approx_samples.append(np.take(approx_out.ravel(), sample_indices))

                # Write
                out_x_approx[current_row:end_row, :] = approx_out

                # Square (Explicit multiplication for safety)
                approx_out = approx_out * approx_out
                acc_approx_sq += np.sum(approx_out, axis=0)
                del approx_out

                # --- CALC 2: FULL (In-place on counts_dense) ---
                pearson_residual_kernel_cpu(
                    counts_dense,
                    tjs,
                    tis[current_row:end_row], 
                    sizes,
                    total,
                    counts_dense # Overwrite input
                )
                
                # Accumulate
                acc_full_sum += np.sum(counts_dense, axis=0)
                
                # Sample
                if sample_indices is not None:
                    viz_full_samples.append(np.take(counts_dense.ravel(), sample_indices))

                # Write
                out_x_full[current_row:end_row, :] = counts_dense
                
                # Square
                counts_dense = counts_dense * counts_dense
                acc_full_sq += np.sum(counts_dense, axis=0)
                
                current_row = end_row
        
        print(f"\nPhase [2/2]: COMPLETE{' '*50}")

    # ==========================================
    #        VIZ GENERATION (POST-PROCESS)
    # ==========================================
    if plot_summary_filename and plot_detail_filename:
        print("Phase [Viz]: Generating Diagnostics (CPU)...")
        
        # 1. Finalize Variance Stats
        mean_raw = acc_raw_sum / nc
        
        mean_approx = acc_approx_sum / nc
        mean_sq_approx = acc_approx_sq / nc
        var_approx = mean_sq_approx - (mean_approx**2)
        
        mean_full = acc_full_sum / nc
        mean_sq_full = acc_full_sq / nc
        var_full = mean_sq_full - (mean_full**2)

        # 2. Finalize Samples
        if viz_approx_samples:
            flat_approx = np.concatenate(viz_approx_samples)
            flat_full   = np.concatenate(viz_full_samples)
        else:
            flat_approx = np.array([])
            flat_full = np.array([])
            
        print(f"Phase [Viz]: Samples Collected... n = {len(flat_approx):,}")

        # --- FILE 1: SUMMARY (1080p) ---
        print(f"Saving Summary Plot to {plot_summary_filename}")
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

        # Plot 2: Distribution (Histogram + KDE Overlay)
        ax = ax1[1]
        if len(flat_approx) > 100:
            mask_kde = (flat_approx > -10) & (flat_approx < 10)
            bins = np.linspace(-5, 5, 100)
            ax.hist(flat_approx[mask_kde], bins=bins, color='red', alpha=0.2, density=True, label='_nolegend_')
            ax.hist(flat_full[mask_kde], bins=bins, color='blue', alpha=0.2, density=True, label='_nolegend_')

            sns.kdeplot(flat_approx[mask_kde], fill=False, color='red', linewidth=2, label='Approx', ax=ax, warn_singular=False)
            sns.kdeplot(flat_full[mask_kde], fill=False, color='blue', linewidth=2, label='Full', ax=ax, warn_singular=False)

        ax.set_yscale('log')
        ax.set_ylim(bottom=0.001)
        ax.set_xlim(-5, 5)
        ax.set_title("Distribution of Residuals (Log Scale)")
        ax.set_xlabel("Residual Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_summary_filename, dpi=120) 
        plt.close()

        # --- FILE 2: DETAIL (4K) ---
        print(f"Saving Detail plot to: {plot_detail_filename}")
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
        
        plt.tight_layout()
        plt.savefig(plot_detail_filename, dpi=200)
        plt.close()

    if hasattr(adata_in, "file") and adata_in.file is not None: adata_in.file.close()
    print(f"Total time: {time.perf_counter() - start_time:.2f} seconds.\n")
