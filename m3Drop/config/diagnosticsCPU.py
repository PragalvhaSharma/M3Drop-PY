from coreCPU import hidden_calc_valsCPU, NBumiFitModelCPU, NBumiFitDispVsMeanCPU
import numpy as np
import anndata
import h5py
import pandas as pd
import time
import os

from scipy.sparse import csr_matrix as sp_csr_matrix

import matplotlib.pyplot as plt


def NBumiFitBasicModelCPU(
    cleaned_filename: str,
    stats: dict,
    is_logged=False,
    chunk_size: int = 5000
) -> dict:
    """
    Fits a simpler, unadjusted NB model out-of-core using a CPU-only algorithm.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitBasicModelCPU() | FILE: {cleaned_filename}")

    print("Phase [1/2]: Initializing parameters on CPU...")
    tjs_series = stats['tjs']
    tjs = tjs_series.values.astype(np.float64)
    nc, ng = stats['nc'], stats['ng']

    sum_x_sq = np.zeros(ng, dtype=np.float64)
    print("Phase [1/2]: COMPLETE")

    print("Phase [2/2]: Calculating variance from data chunks...")
    with h5py.File(cleaned_filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']

        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"Phase [2/2]: Processing: {end_row} of {nc} cells.", end='\r')

            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx:
                continue

            data_slice = np.array(h5_data[start_idx:end_idx], dtype=np.float64)
            indices_slice = np.array(h5_indices[start_idx:end_idx], dtype=np.int64)

            np.add.at(sum_x_sq, indices_slice, data_slice**2)

    print(f"Phase [2/2]: COMPLETE{' '*40}")

    if is_logged:
        raise NotImplementedError("Logged data variance calculation is not implemented for out-of-core CPU.")
    else:
        mean_x_sq = sum_x_sq / nc
        mean_mu = tjs / nc
        my_rowvar = mean_x_sq - mean_mu**2

        numerator = mean_mu**2
        denominator = my_rowvar - mean_mu

        sizes = np.full(ng, np.nan, dtype=np.float64)
        valid_mask = denominator > 1e-12
        sizes[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    finite_sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    max_size_val = np.max(finite_sizes) * 10 if finite_sizes.size else 1000
    sizes[~np.isfinite(sizes) | (sizes <= 0)] = max_size_val
    sizes[sizes < 1e-10] = 1e-10

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return {
        'var_obs': pd.Series(my_rowvar, index=tjs_series.index),
        'sizes': pd.Series(sizes, index=tjs_series.index),
        'vals': stats
    }


def NBumiCheckFitFSCPU(
    cleaned_filename: str,
    fit: dict,
    chunk_size: int = 5000,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    CPU-only version of NBumiCheckFitFS. Computes expected dropouts for genes
    and cells to compare observed vs fitted values.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCheckFitFSCPU() | FILE: {cleaned_filename}")

    print("Phase [1/2]: Initializing parameters on CPU...")
    vals = fit['vals']
    size_coeffs = NBumiFitDispVsMeanCPU(fit, suppress_plot=True)

    tjs = vals['tjs'].values.astype(np.float64)
    tis = vals['tis'].values.astype(np.float64)
    total = vals['total']
    nc, ng = vals['nc'], vals['ng']

    mean_expression = tjs / nc
    log_mean_expression = np.log(mean_expression, where=(mean_expression > 0))
    smoothed_size = np.exp(size_coeffs[0] + size_coeffs[1] * log_mean_expression)
    smoothed_size = np.nan_to_num(smoothed_size, nan=1.0, posinf=1e6, neginf=1.0)

    row_ps = np.zeros(ng, dtype=np.float64)
    col_ps = np.zeros(nc, dtype=np.float64)
    print("Phase [1/2]: COMPLETE")

    print("Phase [2/2]: Calculating expected dropouts from data chunks...")
    for i in range(0, nc, chunk_size):
        end_col = min(i + chunk_size, nc)
        print(f"Phase [2/2]: Processing: {end_col} of {nc} cells.", end='\r')

        tis_chunk = tis[i:end_col]
        if tis_chunk.size == 0:
            continue

        mu_chunk = tjs[:, np.newaxis] * tis_chunk[np.newaxis, :] / total
        base = 1 + mu_chunk / smoothed_size[:, np.newaxis]
        base = np.maximum(base, 1e-12)
        p_is_chunk = np.power(base, -smoothed_size[:, np.newaxis])
        p_is_chunk = np.nan_to_num(p_is_chunk, nan=0.0, posinf=1.0, neginf=0.0)

        row_ps += np.sum(p_is_chunk, axis=1)
        col_ps[i:end_col] = np.sum(p_is_chunk, axis=0)

    print(f"Phase [2/2]: COMPLETE{' ' * 50}")

    djs = vals['djs'].values
    dis = vals['dis'].values

    if not suppress_plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(djs, row_ps, alpha=0.5, s=10)
        plt.title("Gene-specific Dropouts (Smoothed)")
        plt.xlabel("Observed")
        plt.ylabel("Fit")
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0, label="y=x line")
        plt.grid(True); plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(dis, col_ps, alpha=0.5, s=10)
        plt.title("Cell-specific Dropouts (Smoothed)")
        plt.xlabel("Observed")
        plt.ylabel("Expected")
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0, label="y=x line")
        plt.grid(True); plt.legend()

        plt.tight_layout()
        if plot_filename:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"STATUS: Diagnostic plot saved to '{plot_filename}'")
        plt.show()
        plt.close()

    gene_error = np.sum((djs - row_ps)**2)
    cell_error = np.sum((dis - col_ps)**2)

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return {
        'gene_error': gene_error,
        'cell_error': cell_error,
        'rowPs': pd.Series(row_ps, index=fit['vals']['tjs'].index),
        'colPs': pd.Series(col_ps, index=fit['vals']['tis'].index)
    }


def NBumiCompareModelsCPU(
    raw_filename: str,
    cleaned_filename: str,
    stats: dict,
    fit_adjust: dict,
    chunk_size: int = 5000,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    CPU-only comparison between the depth-adjusted NB model and a basic model.
    """
    pipeline_start_time = time.time()
    print(f"FUNCTION: NBumiCompareModelsCPU() | Comparing models for {cleaned_filename}")

    print("Phase [1/4]: Creating temporary 'basic' normalized data file...")
    basic_norm_filename = cleaned_filename.replace('.h5ad', '_basic_norm.h5ad')

    adata_meta = anndata.read_h5ad(cleaned_filename, backed='r')
    nc, ng = adata_meta.shape
    obs_df = adata_meta.obs.copy()
    var_df = adata_meta.var.copy()

    cell_sums = stats['tis'].values.astype(np.float64)
    positive_mask = cell_sums > 0
    median_sum = np.median(cell_sums[positive_mask]) if np.any(positive_mask) else 1.0
    size_factors = np.ones_like(cell_sums, dtype=np.float32)
    size_factors[positive_mask] = cell_sums[positive_mask] / median_sum

    adata_out = anndata.AnnData(obs=obs_df, var=var_df)
    adata_out.write_h5ad(basic_norm_filename, compression="gzip")

    with h5py.File(basic_norm_filename, 'a') as f_out:
        if 'X' in f_out:
            del f_out['X']
        x_group_out = f_out.create_group('X')
        x_group_out.attrs['encoding-type'] = 'csr_matrix'
        x_group_out.attrs['encoding-version'] = '0.1.0'
        x_group_out.attrs['shape'] = np.array([nc, ng], dtype='int64')

        out_data = x_group_out.create_dataset('data', shape=(0,), maxshape=(None,), dtype='float32')
        out_indices = x_group_out.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int32')
        out_indptr = x_group_out.create_dataset('indptr', shape=(nc + 1,), dtype='int64')
        out_indptr[0] = 0
        current_nnz = 0

        with h5py.File(cleaned_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']

            for i in range(0, nc, chunk_size):
                end_row = min(i + chunk_size, nc)
                print(f"Phase [1/4]: Normalizing: {end_row} of {nc} cells.", end='\r')

                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                if start_idx == end_idx:
                    out_indptr[i + 1 : end_row + 1] = current_nnz
                    continue

                data_slice = np.array(h5_data[start_idx:end_idx], dtype=np.float32)
                indices_slice = np.array(h5_indices[start_idx:end_idx], dtype=np.int32)
                abs_indptr = h5_indptr[i:end_row + 1]
                indptr_slice = abs_indptr - abs_indptr[0]
                row_lengths = np.diff(indptr_slice)

                norm_factors = np.repeat(size_factors[i:end_row], row_lengths)
                norm_factors[norm_factors == 0] = 1.0
                normalized_data = data_slice / norm_factors
                normalized_data = np.round(normalized_data).astype(np.float32)

                chunk_sp = sp_csr_matrix((normalized_data, indices_slice, indptr_slice),
                                         shape=(end_row - i, ng))

                nnz_chunk = chunk_sp.nnz
                out_data.resize(current_nnz + nnz_chunk, axis=0)
                out_data[current_nnz:] = chunk_sp.data

                out_indices.resize(current_nnz + nnz_chunk, axis=0)
                out_indices[current_nnz:] = chunk_sp.indices

                new_indptr_list = chunk_sp.indptr[1:].astype(np.int64) + current_nnz
                out_indptr[i + 1 : end_row + 1] = new_indptr_list

                current_nnz += nnz_chunk

    print(f"Phase [1/4]: COMPLETE{' '*50}")

    print("Phase [2/4]: Fitting Basic Model on normalized data...")
    stats_basic = hidden_calc_valsCPU(basic_norm_filename, chunk_size=chunk_size)
    fit_basic = NBumiFitBasicModelCPU(basic_norm_filename, stats_basic, chunk_size=chunk_size)
    print("Phase [2/4]: COMPLETE")

    print("Phase [3/4]: Evaluating fits of both models on ORIGINAL data...")
    check_adjust = NBumiCheckFitFSCPU(cleaned_filename, fit_adjust, suppress_plot=True, chunk_size=chunk_size)

    fit_basic_for_eval = {
        'sizes': fit_basic['sizes'],
        'vals': stats,
        'var_obs': fit_basic['var_obs']
    }
    check_basic = NBumiCheckFitFSCPU(cleaned_filename, fit_basic_for_eval, suppress_plot=True, chunk_size=chunk_size)
    print("Phase [3/4]: COMPLETE")

    print("Phase [4/4]: Generating final comparison...")
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
                c='purple', s=3, alpha=0.6, label=f'Basic Fit (Error: {err_bas:.2f})')
    plt.scatter(mean_expr.iloc[sorted_idx], adj_dropout_fit.iloc[sorted_idx],
                c='goldenrod', s=3, alpha=0.7, label=f'Depth-Adjusted Fit (Error: {err_adj:.2f})')

    plt.xscale('log')
    plt.xlabel("Mean Expression")
    plt.ylabel("Dropout Rate")
    plt.title("M3Drop Model Comparison (CPU)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"STATUS: Model comparison plot saved to '{plot_filename}'")

    if not suppress_plot:
        plt.show()

    plt.close()
    print("Phase [4/4]: COMPLETE")

    pipeline_end_time = time.time()

    adata_meta.file.close()

    os.remove(basic_norm_filename)
    print(f"STATUS: Temporary file '{basic_norm_filename}' removed.")
    print(f"Total time: {pipeline_end_time - pipeline_start_time:.2f} seconds.\n")

    return {
        "errors": {"Depth-Adjusted": err_adj, "Basic": err_bas},
        "comparison_df": comparison_df
    }


def NBumiPlotDispVsMeanCPU(
    fit: dict,
    suppress_plot: bool = False,
    plot_filename: str = None
):
    """
    Generates a diagnostic plot of the dispersion vs. mean expression (CPU version).
    """
    print("FUNCTION: NBumiPlotDispVsMeanCPU()")

    mean_expression = fit['vals']['tjs'].values / fit['vals']['nc']
    sizes = fit['sizes'].values
    coeffs = NBumiFitDispVsMeanCPU(fit, suppress_plot=True)
    intercept, slope = coeffs[0], coeffs[1]

    positive_means = mean_expression[mean_expression > 0]
    if positive_means.size == 0:
        raise ValueError("Mean expression contains no positive values for plotting.")

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
    plt.title('Dispersion vs. Mean Expression (CPU)')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.6)

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"STATUS: Diagnostic plot saved to '{plot_filename}'")

    if not suppress_plot:
        plt.show()

    plt.close()
    print("FUNCTION: NBumiPlotDispVsMeanCPU() COMPLETE\n")
