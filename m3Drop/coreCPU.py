import numpy as np
import anndata
import h5py
import pandas as pd
import time

from scipy.sparse import csr_matrix as sp_csr_matrix

import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
def ConvertDataSparseCPU(
    input_filename: str,
    output_filename: str,
    row_chunk_size: int = 5000
):
    """
    Performs out-of-core data cleaning on a standard (cell, gene) sparse
    .h5ad file. It correctly identifies and removes genes with zero counts
    across all cells. CPU-only implementation.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: ConvertDataSparseCPU() | FILE: {input_filename}")

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

            indices_slice = np.array(h5_indices[start_idx:end_idx])
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
                data_slice = np.array(h5_data[start_idx:end_idx])
                indices_slice = np.array(h5_indices[start_idx:end_idx])
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


def hidden_calc_valsCPU(
    filename: str,
    chunk_size: int = 5000
) -> dict:
    """
    Calculates key statistics from a large, sparse (cell, gene) .h5ad file
    using a memory-safe, CPU-only, single-pass algorithm.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: hidden_calc_valsCPU() | FILE: {filename}")

    adata_meta = anndata.read_h5ad(filename, backed='r')
    print("Phase [1/3]: Finding nc and ng...")
    nc, ng = adata_meta.shape
    print("Phase [1/3]: COMPLETE")

    tis = np.zeros(nc, dtype='float64')
    cell_non_zeros = np.zeros(nc, dtype='int64')
    tjs = np.zeros(ng, dtype='float64')
    gene_non_zeros = np.zeros(ng, dtype='int64')

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
            if start_idx == end_idx:
                continue

            data_slice = np.array(h5_data[start_idx:end_idx], dtype=np.float64)
            indices_slice = np.array(h5_indices[start_idx:end_idx], dtype=np.int64)
            indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

            chunk = sp_csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end_row-i, ng))

            tis[i:end_row] = np.asarray(chunk.sum(axis=1)).ravel()
            cell_non_zeros[i:end_row] = np.diff(indptr_slice)

            np.add.at(tjs, indices_slice, data_slice)
            unique_indices, counts = np.unique(indices_slice, return_counts=True)
            gene_non_zeros[unique_indices] += counts

    tjs_series = pd.Series(tjs, index=adata_meta.var.index)
    tis_series = pd.Series(tis, index=adata_meta.obs.index)
    print(f"Phase [2/3]: COMPLETE{' ' * 50}")

    print("Phase [3/3]: Calculating dis, djs, and total...")
    dis = ng - cell_non_zeros
    djs = nc - gene_non_zeros
    total = tjs.sum()
    print("Phase [3/3]: COMPLETE")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return {
        "tis": tis_series,
        "tjs": tjs_series,
        "dis": pd.Series(dis, index=adata_meta.obs.index),
        "djs": pd.Series(djs, index=adata_meta.var.index),
        "total": total,
        "nc": nc,
        "ng": ng
    }


def NBumiFitModelCPU(
    cleaned_filename: str,
    stats: dict,
    chunk_size: int = 5000
) -> dict:
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitModelCPU() | FILE: {cleaned_filename}")

    tjs_series = stats['tjs']
    tis_series = stats['tis']
    tjs = tjs_series.values.astype(np.float64)
    tis = tis_series.values.astype(np.float64)
    nc, ng = stats['nc'], stats['ng']
    total = stats['total']

    sum_x_sq = np.zeros(ng, dtype=np.float64)
    sum_2xmu = np.zeros(ng, dtype=np.float64)

    print("Phase [1/3]: Pre-calculating sum of squared expectations...")
    sum_tis_sq = np.sum(tis**2)
    sum_mu_sq = (tjs**2 / total**2) * sum_tis_sq
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

            data_slice = np.array(h5_data[start_idx:end_idx], dtype=np.float64)
            indices_slice = np.array(h5_indices[start_idx:end_idx], dtype=np.int64)
            indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

            np.add.at(sum_x_sq, indices_slice, data_slice**2)

            row_lengths = np.diff(indptr_slice)
            if row_lengths.sum() == 0:
                continue
            cell_indices = np.repeat(np.arange(i, end_row), row_lengths)

            tis_per_nz = tis[cell_indices]
            tjs_per_nz = tjs[indices_slice]
            term_vals = 2 * data_slice * tjs_per_nz * tis_per_nz / total
            np.add.at(sum_2xmu, indices_slice, term_vals)

    print(f"Phase [2/3]: COMPLETE {' ' * 50}")

    print("Phase [3/3]: Finalizing dispersion and variance calculations...")
    sum_sq_dev = sum_x_sq - sum_2xmu + sum_mu_sq
    var_obs = sum_sq_dev / max(nc - 1, 1)

    sizes = np.full(ng, 10000.0, dtype=np.float64)
    numerator = (tjs**2 / total**2) * sum_tis_sq
    denominator = sum_sq_dev - tjs
    stable_mask = denominator > 1e-6
    sizes[stable_mask] = numerator[stable_mask] / denominator[stable_mask]
    sizes[np.isnan(sizes) | (sizes <= 0)] = 10000.0

    print("Phase [3/3]: COMPLETE")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return {
        'var_obs': pd.Series(var_obs, index=tjs_series.index),
        'sizes': pd.Series(sizes, index=tjs_series.index),
        'vals': stats
    }


def NBumiFitDispVsMeanCPU(fit, suppress_plot=True):
    """
    Fits a linear model to the log-dispersion vs log-mean of gene expression.
    """
    vals = fit['vals']
    size_g = fit['sizes'].values
    tjs = vals['tjs'].values

    mean_expression = tjs / vals['nc']
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


def NBumiFeatureSelectionHighVarCPU(fit: dict) -> pd.DataFrame:
    """
    Selects features (genes) with higher variance than expected.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionHighVarCPU()")

    print("Phase [1/1]: Calculating residuals for high variance selection...")
    vals = fit['vals']
    coeffs = NBumiFitDispVsMeanCPU(fit, suppress_plot=True)

    mean_expression = vals['tjs'].values / vals['nc']

    with np.errstate(divide='ignore', invalid='ignore'):
        log_mean_expression = np.log(mean_expression)
        log_mean_expression[np.isneginf(log_mean_expression)] = 0
        exp_size = np.exp(coeffs[0] + coeffs[1] * log_mean_expression)

    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.log(fit['sizes'].values) - np.log(exp_size)

    results_df = pd.DataFrame({
        'Gene': fit['sizes'].index,
        'Residual': res
    })

    final_table = results_df.sort_values(by='Residual', ascending=True)
    print("Phase [1/1]: COMPLETE")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.4f} seconds.\n")

    return final_table


def NBumiFeatureSelectionCombinedDropCPU(
    fit: dict,
    cleaned_filename: str,
    chunk_size: int = 5000,
    method="fdr_bh",
    qval_thresh=0.05
) -> pd.DataFrame:
    """
    Selects features with a significantly higher dropout rate than expected,
    using an out-of-core, CPU-only approach.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionCombinedDropCPU() | FILE: {cleaned_filename}")

    print("Phase [1/3]: Initializing arrays and calculating expected dispersion...")
    vals = fit['vals']
    coeffs = NBumiFitDispVsMeanCPU(fit, suppress_plot=True)

    tjs = vals['tjs'].values.astype(np.float64)
    tis = vals['tis'].values.astype(np.float64)
    total = vals['total']
    nc = vals['nc']
    ng = vals['ng']

    mean_expression = tjs / nc
    with np.errstate(divide='ignore'):
        exp_size = np.exp(coeffs[0] + coeffs[1] * np.log(mean_expression, where=(mean_expression > 0)))
    exp_size = np.nan_to_num(exp_size, nan=1.0, posinf=1e6, neginf=1.0)

    p_sum = np.zeros(ng, dtype=np.float64)
    p_var_sum = np.zeros(ng, dtype=np.float64)
    print("Phase [1/3]: COMPLETE")

    print("Phase [2/3]: Calculating expected dropout sums from data chunks...")
    for i in range(0, nc, chunk_size):
        end_col = min(i + chunk_size, nc)
        print(f"Phase [2/3]: Processing: {end_col} of {nc} cells.", end='\r')

        tis_chunk = tis[i:end_col]
        if tis_chunk.size == 0:
            continue

        mu_chunk = tjs[:, np.newaxis] * tis_chunk[np.newaxis, :] / total
        base = 1 + mu_chunk / exp_size[:, np.newaxis]
        base = np.maximum(base, 1e-12)
        p_is_chunk = np.power(base, -exp_size[:, np.newaxis])
        p_is_chunk = np.nan_to_num(p_is_chunk, nan=0.0, posinf=1.0, neginf=0.0)

        p_var_is_chunk = p_is_chunk * (1 - p_is_chunk)

        p_sum += np.sum(p_is_chunk, axis=1)
        p_var_sum += np.sum(p_var_is_chunk, axis=1)

    print(f"Phase [2/3]: COMPLETE {' ' * 50}")

    print("Phase [3/3]: Performing statistical test and adjusting p-values...")

    droprate_exp = p_sum / nc
    droprate_exp_err = np.sqrt(p_var_sum / (nc**2))

    droprate_obs = vals['djs'].values / nc

    diff = droprate_obs - droprate_exp
    combined_err = np.sqrt(droprate_exp_err**2 + (droprate_obs * (1 - droprate_obs) / nc))

    with np.errstate(divide='ignore', invalid='ignore'):
        Zed = diff / combined_err

    pvalue = norm.sf(Zed)

    results_df = pd.DataFrame({
        'Gene': vals['tjs'].index,
        'p.value': pvalue,
        'effect_size': diff
    })
    results_df = results_df.sort_values(by='p.value')

    qval = multipletests(results_df['p.value'].fillna(1), method=method)[1]
    results_df['q.value'] = qval
    final_table = results_df[results_df['q.value'] < qval_thresh]
    print("Phase [3/3]: COMPLETE")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return final_table[['Gene', 'effect_size', 'p.value', 'q.value']]


def NBumiCombinedDropVolcanoCPU(
    results_df: pd.DataFrame,
    qval_thresh: float = 0.05,
    effect_size_thresh: float = 0.25,
    top_n_genes: int = 10,
    suppress_plot: bool = False,
    plot_filename: str = None
):
    """
    Generates a volcano plot from the results of feature selection (CPU version).
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCombinedDropVolcanoCPU()")

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
