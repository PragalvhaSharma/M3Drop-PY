
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import time
import os
import h5py
import anndata
import cupy
from cupy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import csr_matrix as sp_csr_matrix
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

#### Fitting #####

def ConvertDataSparse(
    input_filename: str,
    output_filename: str,
    row_chunk_size: int = 5000
):
    """
    Performs out-of-core data cleaning on a standard (cell, gene) sparse
    .h5ad file. It correctly identifies and removes genes with zero counts
    across all cells.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: ConvertDataSparse() | FILE: {input_filename}")

    with h5py.File(input_filename, 'r') as f_in:
        x_group_in = f_in['X']
        n_cells, n_genes = x_group_in.attrs['shape']

        print("Phase [1/2]: Identifying genes with non-zero counts...")
        genes_to_keep_mask = np.zeros(n_genes, dtype=bool)
        
        h5_indptr = x_group_in['indptr']
        h5_indices = x_group_in['indices']

        for i in range(0, n_cells, row_chunk_size):
            end_row = min(i + row_chunk_size, n_cells)
            print(f"Phase [1/2]: Processing: {end_row} of {n_cells} cells.", end='\r')

            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx:
                continue

            indices_slice = h5_indices[start_idx:end_idx]
            unique_in_chunk = np.unique(indices_slice)
            genes_to_keep_mask[unique_in_chunk] = True

        n_genes_to_keep = np.sum(genes_to_keep_mask)
        print(f"\nPhase [1/2]: COMPLETE | Result: {n_genes_to_keep} / {n_genes} genes retained.")

        print("Phase [2/2]: Rounding up decimals and saving filtered output to disk...")
        adata_meta = anndata.read_h5ad(input_filename, backed='r')
        filtered_var_df = adata_meta.var[genes_to_keep_mask]
        
        adata_out_template = anndata.AnnData(obs=adata_meta.obs, var=filtered_var_df, uns=adata_meta.uns)
        adata_out_template.write_h5ad(output_filename, compression="gzip")

        with h5py.File(output_filename, 'a') as f_out:
            if 'X' in f_out:
                del f_out['X']
            x_group_out = f_out.create_group('X')

            out_data = x_group_out.create_dataset('data', shape=(0,), maxshape=(None,), dtype='float32')
            out_indices = x_group_out.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int32')
            out_indptr = x_group_out.create_dataset('indptr', shape=(n_cells + 1,), dtype='int64')
            out_indptr[0] = 0
            current_nnz = 0

            h5_data = x_group_in['data']

            for i in range(0, n_cells, row_chunk_size):
                end_row = min(i + row_chunk_size, n_cells)
                print(f"Phase [2/2]: Processing: {end_row} of {n_cells} cells.", end='\r')

                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                chunk = sp_csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end_row-i, n_genes))
                filtered_chunk = chunk[:, genes_to_keep_mask]
                filtered_chunk.data = np.ceil(filtered_chunk.data).astype('float32')

                out_data.resize(current_nnz + filtered_chunk.nnz, axis=0)
                out_data[current_nnz:] = filtered_chunk.data

                out_indices.resize(current_nnz + filtered_chunk.nnz, axis=0)
                out_indices[current_nnz:] = filtered_chunk.indices

                new_indptr_list = filtered_chunk.indptr[1:].astype(np.int64) + current_nnz
                out_indptr[i + 1 : end_row + 1] = new_indptr_list
                
                current_nnz += filtered_chunk.nnz

            x_group_out.attrs['encoding-type'] = 'csr_matrix'
            x_group_out.attrs['encoding-version'] = '0.1.0'
            x_group_out.attrs['shape'] = np.array([n_cells, n_genes_to_keep], dtype='int64')
        print(f"\nPhase [2/2]: COMPLETE | Output: {output_filename} {' ' * 50}")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

def _hidden_calc_vals_counts(counts):
    """Original in-memory calculation of hidden values (genes x cells)."""
    if np.any(counts < 0):
        raise ValueError("Expression matrix contains negative values! Please provide raw UMI counts!")
    if not np.allclose(counts, np.round(counts)):
        raise ValueError("Error: Expression matrix is not integers! Please provide raw UMI counts.")
    if isinstance(counts, pd.DataFrame):
        if counts.index.empty:
            counts.index = [str(i) for i in range(counts.shape[0])]
    elif isinstance(counts, np.ndarray):
        counts = pd.DataFrame(counts, index=[str(i) for i in range(counts.shape[0])])
    if not sp.issparse(counts):
        counts_sparse = sp.csr_matrix(counts.values if isinstance(counts, pd.DataFrame) else counts)
    else:
        counts_sparse = counts
    tjs = np.array(counts_sparse.sum(axis=1)).flatten()
    no_detect = np.sum(tjs <= 0)
    if no_detect > 0:
        raise ValueError(f"Error: contains {no_detect} undetected genes.")
    tis = np.array(counts_sparse.sum(axis=0)).flatten()
    if np.any(tis <= 0):
        raise ValueError("Error: all cells must have at least one detected molecule.")
    djs = counts_sparse.shape[1] - np.array((counts_sparse > 0).sum(axis=1)).flatten()
    dis = counts_sparse.shape[0] - np.array((counts_sparse > 0).sum(axis=0)).flatten()
    nc = counts_sparse.shape[1]
    ng = counts_sparse.shape[0]
    total = np.sum(tis)
    return {
        'tis': tis,
        'tjs': tjs,
        'dis': dis,
        'djs': djs,
        'total': total,
        'nc': nc,
        'ng': ng
    }

def hidden_calc_vals(filename_or_counts, chunk_size: int = 5000):
    """
    Calculates key statistics from a large, sparse (cell, gene) .h5ad file
    using a memory-safe, GPU-accelerated, single-pass algorithm. If a matrix
    is provided instead of a filename, falls back to the original in-memory
    implementation (genes x cells).
    """
    if isinstance(filename_or_counts, str) and os.path.exists(filename_or_counts):
        filename = filename_or_counts
        start_time = time.perf_counter()
        print(f"FUNCTION: hidden_calc_vals() | FILE: {filename}")
        adata_meta = anndata.read_h5ad(filename, backed='r')
        print("Phase [1/3]: Finding nc and ng...")
        nc, ng = adata_meta.shape
        print(f"Phase [1/3]: COMPLETE")
        tis = np.zeros(nc, dtype='int64')
        cell_non_zeros = np.zeros(nc, dtype='int64')
        tjs_gpu = cupy.zeros(ng, dtype=cupy.float32)
        gene_non_zeros_gpu = cupy.zeros(ng, dtype=cupy.int32)
        print("Phase [2/3]: Calculating tis and tjs...")
        with h5py.File(filename, 'r') as f_in:
            x_group = f_in['X']
            h5_indptr = x_group['indptr']
            h5_data = x_group['data']
            h5_indices = x_group['indices']
            for i in range(0, nc, chunk_size):
                end_row = min(i + chunk_size, nc)
                print(f"Phase [2/3]: Processing: {end_row} of {nc} cells.", end='\r')
                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]
                data_gpu = cupy.asarray(data_slice.copy(), dtype=cupy.float32)
                indices_gpu = cupy.asarray(indices_slice.copy())
                indptr_gpu = cupy.asarray(indptr_slice.copy())
                chunk_gpu = cp_csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(end_row-i, ng))
                tis[i:end_row] = chunk_gpu.sum(axis=1).get().flatten()
                cell_non_zeros_chunk = cupy.diff(indptr_gpu)
                cell_non_zeros[i:end_row] = cell_non_zeros_chunk.get()
                cupy.add.at(tjs_gpu, indices_gpu, data_gpu)
                unique_indices_gpu, counts_gpu = cupy.unique(indices_gpu, return_counts=True)
                cupy.add.at(gene_non_zeros_gpu, unique_indices_gpu, counts_gpu)
        tjs = cupy.asnumpy(tjs_gpu)
        gene_non_zeros = cupy.asnumpy(gene_non_zeros_gpu)
        print(f"Phase [2/3]: COMPLETE{' ' * 50}")
        print("Phase [3/3]: Calculating dis, djs, and total...")
        dis = ng - cell_non_zeros
        djs = nc - gene_non_zeros
        total = tjs.sum()
        print("Phase [3/3]: COMPLETE")
        end_time = time.perf_counter()
        print(f"Total time: {end_time - start_time:.2f} seconds.\n")
        return {
            'tis': pd.Series(tis, index=adata_meta.obs.index),
            'tjs': pd.Series(tjs, index=adata_meta.var.index),
            'dis': pd.Series(dis, index=adata_meta.obs.index),
            'djs': pd.Series(djs, index=adata_meta.var.index),
            'total': total,
            'nc': nc,
            'ng': ng
        }
    else:
        return _hidden_calc_vals_counts(filename_or_counts)

def NBumiConvertToInteger(mat):
    """Convert matrix to integer format."""
    mat = np.ceil(np.asarray(mat)).astype(int)
    # Remove genes with zero total counts
    row_sums = np.sum(mat, axis=1)
    mat = mat[row_sums > 0, :]
    return mat

def _NBumiFitModel_counts(counts):
    """Original in-memory NB fit using counts (genes x cells)."""
    vals = _hidden_calc_vals_counts(counts)
    min_size = 1e-10
    my_rowvar = np.zeros(counts.shape[0])
    for i in range(counts.shape[0]):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        if isinstance(counts, pd.DataFrame):
            row_data = counts.iloc[i, :].values
        else:
            row_data = counts[i, :]
        my_rowvar[i] = np.var(row_data - mu_is)
    numerator = vals['tjs']**2 * (np.sum(vals['tis']**2) / vals['total']**2)
    denominator = (vals['nc'] - 1) * my_rowvar - vals['tjs']
    size = numerator / denominator
    max_size = 10 * np.max(size[size > 0])
    size[size < 0] = max_size
    size[size < min_size] = min_size
    return {
        'var_obs': my_rowvar,
        'sizes': size,
        'vals': vals
    }

def NBumiFitModel(*args, **kwargs):
    """
    GPU-accelerated NB fit for cleaned .h5ad data when called with
    (cleaned_filename: str, stats: dict, chunk_size: int = 5000).
    Falls back to original in-memory implementation when called with a
    single counts argument.
    """
    if len(args) == 1 and not kwargs:
        return _NBumiFitModel_counts(args[0])

    if len(args) >= 2 and isinstance(args[1], dict):
        cleaned_filename = args[0]
        stats = args[1]
        chunk_size = kwargs.get('chunk_size', 5000)
        start_time = time.perf_counter()
        print(f"FUNCTION: NBumiFitModel() | FILE: {cleaned_filename}")
        tjs = stats['tjs'].values if isinstance(stats['tjs'], pd.Series) else stats['tjs']
        tis = stats['tis'].values if isinstance(stats['tis'], pd.Series) else stats['tis']
        nc, ng = stats['nc'], stats['ng']
        total = stats['total']
        tjs_gpu = cupy.asarray(tjs, dtype=cupy.float64)
        tis_gpu = cupy.asarray(tis, dtype=cupy.float64)
        sum_x_sq_gpu = cupy.zeros(ng, dtype=cupy.float64)
        sum_2xmu_gpu = cupy.zeros(ng, dtype=cupy.float64)
        print("Phase [1/3]: Pre-calculating sum of squared expectations...")
        sum_tis_sq_gpu = cupy.sum(tis_gpu**2)
        sum_mu_sq_gpu = (tjs_gpu**2 / total**2) * sum_tis_sq_gpu
        print("Phase [1/3]: COMPLETE")
        print("Phase [2/3]: Calculating variance components from data chunks...")
        with h5py.File(cleaned_filename, 'r') as f_in:
            x_group = f_in['X']
            h5_indptr = x_group['indptr']
            h5_data = x_group['data']
            h5_indices = x_group['indices']
            for i in range(0, nc, chunk_size):
                end_row = min(i + chunk_size, nc)
                print(f"Phase [2/3]: Processing: {end_row} of {nc} cells.", end='\r')
                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                if start_idx == end_idx:
                    continue
                data_gpu = cupy.asarray(h5_data[start_idx:end_idx], dtype=cupy.float64)
                indices_gpu = cupy.asarray(h5_indices[start_idx:end_idx])
                indptr_gpu = cupy.asarray(h5_indptr[i:end_row+1] - h5_indptr[i])
                cupy.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
                nnz_in_chunk = indptr_gpu[-1].item()
                cell_boundary_markers = cupy.zeros(nnz_in_chunk, dtype=cupy.int32)
                if len(indptr_gpu) > 1:
                    cell_boundary_markers[indptr_gpu[:-1]] = 1
                cell_indices_chunk = cupy.cumsum(cell_boundary_markers, axis=0) - 1
                cell_indices_gpu = cell_indices_chunk + i
                tis_per_nz = tis_gpu[cell_indices_gpu]
                tjs_per_nz = tjs_gpu[indices_gpu]
                term_vals = 2 * data_gpu * tjs_per_nz * tis_per_nz / total
                cupy.add.at(sum_2xmu_gpu, indices_gpu, term_vals)
                del data_gpu, indices_gpu, indptr_gpu, cell_indices_gpu
                del tis_per_nz, tjs_per_nz, term_vals
                if i % (chunk_size * 10) == 0:
                    cupy.get_default_memory_pool().free_all_blocks()
        print(f"Phase [2/3]: COMPLETE {' ' * 50}")
        print("Phase [3/3]: Finalizing dispersion and variance calculations...")
        sum_sq_dev_gpu = sum_x_sq_gpu - sum_2xmu_gpu + sum_mu_sq_gpu
        var_obs_gpu = sum_sq_dev_gpu / (nc - 1)
        sizes_gpu = cupy.full(ng, 10000.0)
        numerator_gpu = (tjs_gpu**2 / total**2) * sum_tis_sq_gpu
        denominator_gpu = sum_sq_dev_gpu - tjs_gpu
        stable_mask = denominator_gpu > 1e-6
        sizes_gpu[stable_mask] = numerator_gpu[stable_mask] / denominator_gpu[stable_mask]
        sizes_gpu[sizes_gpu <= 0] = 10000.0
        var_obs_cpu = var_obs_gpu.get()
        sizes_cpu = sizes_gpu.get()
        print("Phase [3/3]: COMPLETE")
        end_time = time.perf_counter()
        print(f"Total time: {end_time - start_time:.2f} seconds.\n")
        return {
            'var_obs': pd.Series(var_obs_cpu, index=stats['tjs'].index if isinstance(stats['tjs'], pd.Series) else None),
            'sizes': pd.Series(sizes_cpu, index=stats['tjs'].index if isinstance(stats['tjs'], pd.Series) else None),
            'vals': stats
        }
    raise ValueError("NBumiFitModel called with unsupported arguments")

def NBumiFitBasicModel(counts):
    """Fit basic negative binomial model."""
    vals = hidden_calc_vals(counts)
    
    mus = vals['tjs'] / vals['nc']
    
    if isinstance(counts, pd.DataFrame):
        gm = counts.mean(axis=1).values
        v = ((counts.subtract(gm, axis=0))**2).sum(axis=1).values / (counts.shape[1] - 1)
    else:
        gm = np.mean(counts, axis=1)
        v = np.sum((counts - gm[:, np.newaxis])**2, axis=1) / (counts.shape[1] - 1)
    
    errs = v < mus
    v[errs] = mus[errs] + 1e-10
    
    size = mus**2 / (v - mus)
    max_size = np.max(mus)**2
    size[errs] = max_size
    
    my_rowvar = np.zeros(counts.shape[0])
    for i in range(counts.shape[0]):
        if isinstance(counts, pd.DataFrame):
            row_data = counts.iloc[i, :].values
        else:
            row_data = counts[i, :]
        my_rowvar[i] = np.var(row_data - mus[i])
    
    return {
        'var_obs': my_rowvar,
        'sizes': size,
        'vals': vals
    }

def NBumiCheckFit(counts, fit, suppress_plot=False):
    """Check the fit of the negative binomial model."""
    vals = fit['vals']
    
    row_ps = np.zeros(counts.shape[0])
    col_ps = np.zeros(counts.shape[1])
    
    for i in range(counts.shape[0]):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        p_is = (1 + mu_is / fit['sizes'][i])**(-fit['sizes'][i])
        row_ps[i] = np.sum(p_is)
        col_ps += p_is
    
    if not suppress_plot:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(vals['djs'], row_ps)
        plt.plot([0, max(vals['djs'])], [0, max(vals['djs'])], 'r-')
        plt.xlabel('Observed')
        plt.ylabel('Fit')
        plt.title('Gene-specific Dropouts')
        
        plt.subplot(1, 2, 2)
        plt.scatter(vals['dis'], col_ps)
        plt.plot([0, max(vals['dis'])], [0, max(vals['dis'])], 'r-')
        plt.xlabel('Observed')
        plt.ylabel('Expected')
        plt.title('Cell-specific Dropouts')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'gene_error': np.sum((vals['djs'] - row_ps)**2),
        'cell_error': np.sum((vals['dis'] - col_ps)**2),
        'rowPs': row_ps,
        'colPs': col_ps
    }

def NBumiFitDispVsMean(fit, suppress_plot=True):
    """Fits a linear model to the log-dispersion vs log-mean of gene expression."""
    vals = fit['vals']
    size_g = fit['sizes'] if isinstance(fit['sizes'], np.ndarray) else np.asarray(fit['sizes'])
    tjs = vals['tjs'].values if isinstance(vals['tjs'], pd.Series) else vals['tjs']
    nc = vals['nc']
    mean_expression = tjs / nc
    forfit = (np.isfinite(size_g)) & (size_g < 1e6) & (mean_expression > 1e-3) & (size_g > 0)
    log2_mean_expr = np.log2(mean_expression, where=(mean_expression > 0))
    higher = log2_mean_expr > 4
    if np.sum(higher & forfit) > 2000:
        forfit = higher & forfit
    y = np.log(size_g[forfit])
    x = np.log(mean_expression[forfit])
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    if not suppress_plot:
        plt.figure(figsize=(7, 6))
        plt.scatter(x, y, alpha=0.5, label="Data Points")
        plt.plot(x, model.fittedvalues, color='red', label='Regression Fit')
        plt.title('Dispersion vs. Mean Expression')
        plt.xlabel("Log Mean Expression")
        plt.ylabel("Log Size (Dispersion)")
        plt.legend()
        plt.grid(True)
        plt.show()
    return model.params

def NBumiCheckFitFS(counts, fit, suppress_plot=False):
    """Check fit with feature selection."""
    vals = fit['vals']
    size_coeffs = NBumiFitDispVsMean(fit, suppress_plot=True)
    smoothed_size = np.exp(size_coeffs[0] + size_coeffs[1] * np.log(vals['tjs'] / vals['nc']))
    
    row_ps = np.zeros(counts.shape[0])
    col_ps = np.zeros(counts.shape[1])
    
    for i in range(counts.shape[0]):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        p_is = (1 + mu_is / smoothed_size[i])**(-smoothed_size[i])
        row_ps[i] = np.sum(p_is)
        col_ps += p_is
    
    if not suppress_plot:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(vals['djs'], row_ps)
        plt.plot([0, max(vals['djs'])], [0, max(vals['djs'])], 'r-')
        plt.xlabel('Observed')
        plt.ylabel('Fit')
        plt.title('Gene-specific Dropouts')
        
        plt.subplot(1, 2, 2)
        plt.scatter(vals['dis'], col_ps)
        plt.plot([0, max(vals['dis'])], [0, max(vals['dis'])], 'r-')
        plt.xlabel('Observed')
        plt.ylabel('Expected')
        plt.title('Cell-specific Dropouts')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'gene_error': np.sum((vals['djs'] - row_ps)**2),
        'cell_error': np.sum((vals['dis'] - col_ps)**2),
        'rowPs': row_ps,
        'colPs': col_ps
    }

def NBumiCompareModels(counts, size_factor=None):
    """Compare different normalization models."""
    if size_factor is None:
        col_sums = np.sum(counts, axis=0)
        size_factor = col_sums / np.median(col_sums)
    
    if np.max(counts) < np.max(size_factor):
        raise ValueError("Error: size factors are too large")
    
    # Normalize counts
    if isinstance(counts, pd.DataFrame):
        norm = counts.div(size_factor, axis=1)
    else:
        norm = counts / size_factor[np.newaxis, :]
    
    norm = NBumiConvertToInteger(norm)
    
    # Fit models
    fit_adjust = NBumiFitModel(counts)
    fit_basic = NBumiFitBasicModel(norm)
    
    check_adjust = NBumiCheckFitFS(counts, fit_adjust, suppress_plot=True)
    check_basic = NBumiCheckFitFS(norm, fit_basic, suppress_plot=True)
    
    nc = fit_adjust['vals']['nc']
    
    # Plotting
    plt.figure(figsize=(10, 6))
    xes = np.log10(fit_adjust['vals']['tjs'] / nc)
    
    plt.scatter(xes, fit_adjust['vals']['djs'] / nc, c='black', s=20, alpha=0.7, label='Observed')
    plt.scatter(xes, check_adjust['rowPs'] / nc, c='goldenrod', s=10, alpha=0.7, label='Depth-Adjusted')
    plt.scatter(xes, check_basic['rowPs'] / nc, c='purple', s=10, alpha=0.7, label='Basic')
    
    plt.xscale('log')
    plt.xlabel('Expression')
    plt.ylabel('Dropout Rate')
    plt.legend()
    
    err_adj = np.sum(np.abs(check_adjust['rowPs'] / nc - fit_adjust['vals']['djs'] / nc))
    err_bas = np.sum(np.abs(check_basic['rowPs'] / nc - fit_adjust['vals']['djs'] / nc))
    
    plt.text(0.02, 0.98, f'Depth-Adjusted Error: {err_adj:.2f}\nBasic Error: {err_bas:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()
    
    out = {'Depth-Adjusted': err_adj, 'Basic': err_bas}
    return {
        'errors': out,
        'basic_fit': fit_basic,
        'adjusted_fit': fit_adjust
    }

def hidden_shift_size(mu_all, size_all, mu_group, coeffs):
    """Shift size parameter based on mean expression change."""
    b = np.log(size_all) - coeffs[1] * np.log(mu_all)
    size_group = np.exp(coeffs[1] * np.log(mu_group) + b)
    return size_group

#### Feature Selection ####

def NBumiFeatureSelectionHighVar(fit: dict) -> pd.DataFrame:
    """
    Selects features (genes) with higher variance than expected.
    Returns a DataFrame sorted by residual.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionHighVar()")
    print("Phase [1/1]: Calculating residuals for high variance selection...")
    vals = fit['vals']
    coeffs = NBumiFitDispVsMean(fit, suppress_plot=True)
    mean_expression = (vals['tjs'].values if isinstance(vals['tjs'], pd.Series) else vals['tjs']) / vals['nc']
    with np.errstate(divide='ignore', invalid='ignore'):
        log_mean_expression = np.log(mean_expression)
        log_mean_expression[np.isneginf(log_mean_expression)] = 0
        exp_size = np.exp(coeffs[0] + coeffs[1] * log_mean_expression)
    with np.errstate(divide='ignore', invalid='ignore'):
        sizes_vals = fit['sizes'].values if isinstance(fit['sizes'], pd.Series) else fit['sizes']
        res = np.log(sizes_vals) - np.log(exp_size)
    gene_index = fit['sizes'].index if isinstance(fit['sizes'], pd.Series) else list(range(len(res)))
    results_df = pd.DataFrame({
        'Gene': gene_index,
        'Residual': res
    })
    final_table = results_df.sort_values(by='Residual', ascending=True)
    print("Phase [1/1]: COMPLETE")
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.4f} seconds.\n")
    return final_table

def NBumiFeatureSelectionCombinedDrop(
    fit: dict,
    cleaned_filename: str = None,
    chunk_size: int = 5000,
    method: str = "fdr_bh",
    qval_thresh: float = 0.05,
    suppress_plot: bool = True,
    ntop: int = None
) -> pd.DataFrame:
    """
    Selects features with a significantly higher dropout rate than expected,
    using a GPU-accelerated approach on summary statistics.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionCombinedDrop() | FILE: {cleaned_filename}")
    print("Phase [1/3]: Initializing arrays and calculating expected dispersion...")
    vals = fit['vals']
    coeffs = NBumiFitDispVsMean(fit, suppress_plot=True)
    tjs_vals = vals['tjs'].values if isinstance(vals['tjs'], pd.Series) else vals['tjs']
    tis_vals = vals['tis'].values if isinstance(vals['tis'], pd.Series) else vals['tis']
    total = vals['total']
    nc = vals['nc']
    ng = vals['ng']
    tjs_gpu = cupy.asarray(tjs_vals)
    tis_gpu = cupy.asarray(tis_vals)
    mean_expression_cpu = tjs_vals / nc
    with np.errstate(divide='ignore'):
        exp_size_cpu = np.exp(coeffs[0] + coeffs[1] * np.log(mean_expression_cpu))
    exp_size_gpu = cupy.asarray(exp_size_cpu)
    p_sum_gpu = cupy.zeros(ng, dtype=cupy.float64)
    p_var_sum_gpu = cupy.zeros(ng, dtype=cupy.float64)
    print("Phase [1/3]: COMPLETE")
    print("Phase [2/3]: Calculating expected dropout sums from data chunks...")
    for i in range(0, nc, chunk_size):
        end_col = min(i + chunk_size, nc)
        print(f"Phase [2/3]: Processing: {end_col} of {nc} cells.", end='\r')
        tis_chunk_gpu = tis_gpu[i:end_col]
        mu_chunk_gpu = tjs_gpu[:, cupy.newaxis] * tis_chunk_gpu[cupy.newaxis, :] / total
        p_is_chunk_gpu = cupy.power(1 + mu_chunk_gpu / exp_size_gpu[:, cupy.newaxis], -exp_size_gpu[:, cupy.newaxis])
        p_var_is_chunk_gpu = p_is_chunk_gpu * (1 - p_is_chunk_gpu)
        p_sum_gpu += p_is_chunk_gpu.sum(axis=1)
        p_var_sum_gpu += p_var_is_chunk_gpu.sum(axis=1)
    print(f"Phase [2/3]: COMPLETE {' ' * 50}")
    print("Phase [3/3]: Performing statistical test and adjusting p-values...")
    p_sum_cpu = p_sum_gpu.get()
    p_var_sum_cpu = p_var_sum_gpu.get()
    droprate_exp = p_sum_cpu / nc
    droprate_exp_err = np.sqrt(p_var_sum_cpu / (nc**2))
    djs_vals = vals['djs'].values if isinstance(vals['djs'], pd.Series) else vals['djs']
    droprate_obs = djs_vals / nc
    diff = droprate_obs - droprate_exp
    combined_err = np.sqrt(droprate_exp_err**2 + (droprate_obs * (1 - droprate_obs) / nc))
    with np.errstate(divide='ignore', invalid='ignore'):
        Zed = diff / combined_err
    pvalue = norm.sf(Zed)
    gene_index = vals['tjs'].index if isinstance(vals['tjs'], pd.Series) else list(range(len(pvalue)))
    results_df = pd.DataFrame({
        'Gene': gene_index,
        'p.value': pvalue,
        'effect_size': diff
    })
    results_df = results_df.sort_values(by='p.value')
    # Backward-compat for method name
    adj_method = 'fdr_bh' if method in (None, 'fdr', 'bh') else method
    qval = multipletests(results_df['p.value'].fillna(1), method=adj_method)[1]
    results_df['q.value'] = qval
    if ntop is None:
        final_table = results_df[results_df['q.value'] < qval_thresh].copy()
    else:
        final_table = results_df.head(ntop).copy()
    # Add aliases for backward compatibility
    final_table['p_value'] = final_table['p.value']
    final_table['q_value'] = final_table['q.value']
    print("Phase [3/3]: COMPLETE")
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
    return final_table[['Gene', 'effect_size', 'p.value', 'q.value', 'p_value', 'q_value']]

def NBumiCombinedDropVolcano(
    results_df: pd.DataFrame,
    qval_thresh: float = 0.05,
    effect_size_thresh: float = 0.25,
    top_n_genes: int = 10,
    suppress_plot: bool = False,
    plot_filename: str = None
):
    """
    Generates a volcano plot from the results of feature selection.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCombinedDropVolcano()")
    print("Phase [1/1]: Preparing data for visualization...")
    df = results_df.copy()
    non_zero_min = df[df['q.value'] > 0]['q.value'].min()
    df['q.value'] = df['q.value'].replace(0, non_zero_min)
    df['-log10_qval'] = -np.log10(df['q.value'])
    df['color'] = 'grey'
    sig_up = (df['q.value'] < qval_thresh) & (df['effect_size'] > effect_size_thresh)
    sig_down = (df['q.value'] < qval_thresh) & (df['effect_size'] < -effect_size_thresh)
    df.loc[sig_up, 'color'] = 'red'
    df.loc[sig_down, 'color'] = 'blue'
    print("Phase [1/1]: COMPLETE")
    print("Phase [2/2]: Generating plot...")
    plt.figure(figsize=(10, 8))
    plt.scatter(df['effect_size'], df['-log10_qval'], c=df['color'], s=10, alpha=0.6)
    plt.axvline(x=effect_size_thresh, linestyle='--', color='grey', linewidth=0.8)
    plt.axvline(x=-effect_size_thresh, linestyle='--', color='grey', linewidth=0.8)
    plt.axhline(y=-np.log10(qval_thresh), linestyle='--', color='grey', linewidth=0.8)
    top_genes = df.nsmallest(top_n_genes, 'q.value')
    for _, row in top_genes.iterrows():
        plt.text(row['effect_size'], row['-log10_qval'], row['Gene'],
                 fontsize=9, ha='left', va='bottom', alpha=0.8)
    plt.title('Volcano Plot of Dropout Feature Selection')
    plt.xlabel('Effect Size (Observed - Expected Dropout Rate)')
    plt.ylabel('-log10 (Adjusted p-value)')
    plt.grid(True, linestyle='--', alpha=0.3)
    ax = plt.gca()
    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"STATUS: Volcano plot saved to '{plot_filename}'")
    if not suppress_plot:
        plt.show()
    plt.close()
    print("Phase [2/2]: COMPLETE")
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
    return ax

def PoissonUMIFeatureSelectionDropouts(fit):
    """Feature selection using Poisson model for dropouts."""
    vals = fit['vals']
    
    droprate_exp = np.zeros(vals['ng'])
    droprate_exp_err = np.zeros(vals['ng'])
    
    for i in range(vals['ng']):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        p_is = np.exp(-mu_is)
        p_var_is = p_is * (1 - p_is)
        droprate_exp[i] = np.sum(p_is) / vals['nc']
        droprate_exp_err[i] = np.sqrt(np.sum(p_var_is) / (vals['nc']**2))
    
    droprate_exp[droprate_exp < 1/vals['nc']] = 1/vals['nc']
    droprate_obs = vals['djs'] / vals['nc']
    
    diff = droprate_obs - droprate_exp
    combined_err = droprate_exp_err
    zed = diff / combined_err
    pvalue = 1 - stats.norm.cdf(zed)
    
    gene_names = list(range(len(pvalue)))
    sorted_indices = np.argsort(pvalue)
    
    return {gene_names[i]: pvalue[i] for i in sorted_indices}

#### Normalization and Imputation ####

def NBumiImputeNorm(counts, fit, total_counts_per_cell=None):
    """Impute and normalize counts."""
    if total_counts_per_cell is None:
        total_counts_per_cell = np.median(fit['vals']['tis'])
    
    # Preserve gene/cell names if input is DataFrame
    if isinstance(counts, pd.DataFrame):
        gene_names = counts.index
        cell_names = counts.columns
        counts_array = counts.values
    else:
        gene_names = None
        cell_names = None
        counts_array = counts
    
    coeffs = NBumiFitDispVsMean(fit, suppress_plot=True)
    vals = fit['vals']
    norm = np.copy(counts_array)
    normed_ti = total_counts_per_cell
    normed_mus = vals['tjs'] / vals['total']
    
    from scipy.stats import nbinom
    
    for i in range(counts_array.shape[0]):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        p_orig = nbinom.cdf(counts_array[i, :], n=fit['sizes'][i], p=fit['sizes'][i]/(fit['sizes'][i] + mu_is))
        
        new_size = hidden_shift_size(np.mean(mu_is), fit['sizes'][i], normed_mus[i] * normed_ti, coeffs)
        normed = nbinom.ppf(p_orig, n=new_size, p=new_size/(new_size + normed_mus[i] * normed_ti))
        norm[i, :] = normed
    
    # Return as DataFrame if input was DataFrame
    if gene_names is not None and cell_names is not None:
        return pd.DataFrame(norm, index=gene_names, columns=cell_names)
    else:
        return norm

def NBumiConvertData(input_data, is_log=False, is_counts=False, pseudocount=1, preserve_sparse=True):
    """Convert various input formats to counts matrix."""
    
    # Store gene and cell names for later use
    gene_names = None
    cell_names = None
    
    # Handle different input types
    if hasattr(input_data, 'X'):  # AnnData object
        # AnnData stores data as cells x genes, we need genes x cells for M3Drop
        # So var_names are the genes, obs_names are the cells
        gene_names = input_data.var_names.copy()  # These are the actual gene names
        cell_names = input_data.obs_names.copy()  # These are the actual cell names
        
        if is_log:
            if sp.issparse(input_data.X) and preserve_sparse:
                # Keep sparse, transpose to genes x cells
                lognorm = input_data.X.T.tocsr()
            else:
                lognorm = input_data.X.toarray() if sp.issparse(input_data.X) else input_data.X
                # Create DataFrame with gene and cell names (transpose: cells x genes -> genes x cells)
                lognorm = pd.DataFrame(lognorm.T, index=input_data.var_names, columns=input_data.obs_names)
        else:
            if sp.issparse(input_data.X) and preserve_sparse:
                # Keep sparse, transpose to genes x cells  
                counts = input_data.X.T.tocsr()
            else:
                counts = input_data.X.toarray() if sp.issparse(input_data.X) else input_data.X
                # Create DataFrame with gene and cell names (transpose: cells x genes -> genes x cells)
                counts = pd.DataFrame(counts.T, index=input_data.var_names, columns=input_data.obs_names)
    elif isinstance(input_data, pd.DataFrame):
        if is_log:
            lognorm = input_data.copy()
        elif is_counts:
            counts = input_data.copy()
        else:
            norm = input_data.copy()
    elif isinstance(input_data, np.ndarray):
        # Create gene and cell names
        gene_names = np.array([f"Gene_{i}" for i in range(input_data.shape[0])])
        cell_names = np.array([f"Cell_{i}" for i in range(input_data.shape[1])])
        
        if preserve_sparse:
            # Convert to sparse for memory efficiency
            if is_log:
                lognorm = sp.csr_matrix(input_data)
            elif is_counts:
                counts = sp.csr_matrix(input_data)
            else:
                norm = sp.csr_matrix(input_data)
        else:
            if is_log:
                lognorm = pd.DataFrame(input_data, index=gene_names, columns=cell_names)
            elif is_counts:
                counts = pd.DataFrame(input_data, index=gene_names, columns=cell_names)
            else:
                norm = pd.DataFrame(input_data, index=gene_names, columns=cell_names)
    elif sp.issparse(input_data):
        if preserve_sparse:
            if is_log:
                lognorm = input_data.tocsr()
            elif is_counts:
                counts = input_data.tocsr()
            else:
                norm = input_data.tocsr()
        else:
            # Convert to DataFrame 
            if is_log:
                lognorm = pd.DataFrame(input_data.toarray())
            elif is_counts:
                counts = pd.DataFrame(input_data.toarray())
            else:
                norm = pd.DataFrame(input_data.toarray())
    else:
        raise ValueError(f"Error: Unrecognized input format: {type(input_data)}")
    
    def remove_undetected_genes(mat, genes=None, cells=None):
        """Remove genes with no detected expression."""
        if sp.issparse(mat):
            # Efficient sparse operations
            detected = np.array(mat.sum(axis=1)).flatten() > 0
            filtered = mat[detected, :]
            if not detected.all():
                print(f"Removing {(~detected).sum()} undetected genes.")
            if genes is not None:
                genes = genes[detected]
            return filtered, genes
        elif isinstance(mat, pd.DataFrame):
            detected = mat.sum(axis=1) > 0
            filtered = mat[detected]
            if not detected.all():
                print(f"Removing {(~detected).sum()} undetected genes.")
            return filtered, None
        else:
            # Fallback for numpy arrays
            detected = np.sum(mat > 0, axis=1) > 0
            print(f"Removing {(~detected).sum()} undetected genes.")
            filtered = mat[detected, :]
            if genes is not None:
                genes = genes[detected]
            return filtered, genes
    
    # Prefer raw counts to lognorm
    if 'counts' in locals():
        if sp.issparse(counts):
            # Handle sparse integer conversion
            counts = counts.copy()
            counts.data = np.ceil(counts.data)
            filtered_counts, filtered_genes = remove_undetected_genes(counts, gene_names, cell_names)
            filtered_counts.data = filtered_counts.data.astype(int)
            
            if preserve_sparse:
                # Import SparseMat3Drop from basics module
                from .basics import SparseMat3Drop
                return SparseMat3Drop(filtered_counts, gene_names=filtered_genes, cell_names=cell_names)
            else:
                # Convert to DataFrame for compatibility
                if filtered_genes is not None and cell_names is not None:
                    return pd.DataFrame(filtered_counts.toarray(), 
                                      index=filtered_genes, 
                                      columns=cell_names)
                else:
                    return pd.DataFrame(filtered_counts.toarray())
        else:
            counts = np.ceil(counts)
            filtered_counts, _ = remove_undetected_genes(counts)
            return filtered_counts.astype(int)
    
    # If normalized, rescale
    if 'lognorm' in locals():
        if sp.issparse(lognorm):
            # Handle sparse log transformation
            norm = lognorm.copy()
            norm.data = 2**norm.data - pseudocount
        else:
            norm = 2**lognorm - pseudocount
    
    if 'norm' in locals():
        if sp.issparse(norm):
            # Sparse matrix operations for scaling
            sf = np.array(norm.min(axis=0)).flatten()
            sf[sf == 0] = 1  # Avoid division by zero
            sf = 1 / sf
            # Create diagonal matrix for efficient scaling
            sf_diag = sp.diags(sf, format='csr')
            counts = norm @ sf_diag
            counts.data = np.ceil(counts.data)
            
            filtered_counts, filtered_genes = remove_undetected_genes(counts, gene_names, cell_names)
            filtered_counts.data = filtered_counts.data.astype(int)
            
            if preserve_sparse:
                from .basics import SparseMat3Drop
                return SparseMat3Drop(filtered_counts, gene_names=filtered_genes, cell_names=cell_names)
            else:
                if filtered_genes is not None and cell_names is not None:
                    return pd.DataFrame(filtered_counts.toarray(), 
                                      index=filtered_genes, 
                                      columns=cell_names)
                else:
                    return pd.DataFrame(filtered_counts.toarray())
        else:
            sf = norm.min(axis=0)
            sf[sf == 0] = 1  # Avoid division by zero
            sf = 1 / sf
            counts = (norm.multiply(sf, axis=1) if isinstance(norm, pd.DataFrame) else norm * sf[np.newaxis, :])
            counts = np.ceil(counts)
            filtered_counts, _ = remove_undetected_genes(counts)
            return filtered_counts.astype(int)

