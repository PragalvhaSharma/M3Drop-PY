import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import warnings

# Import NBumi functions
from .NB_UMI import NBumiFitModel, hidden_calc_vals

# Normalization Functions

def hidden_UQ(x):
    """Calculate 75th percentile of positive values"""
    positive_values = x[x > 0]
    if len(positive_values) == 0:
        return 0
    return np.percentile(positive_values, 75)

def bg_filter_cells(expr_mat, labels=None, suppress_plot=False, min_detected_genes=None):
    """Filter low-quality cells based on number of detected genes"""
    if issparse(expr_mat):
        num_detected = np.array((expr_mat > 0).sum(axis=0)).flatten()
        num_zero = np.array((expr_mat == 0).sum(axis=0)).flatten()
    else:
        num_detected = np.sum(expr_mat > 0, axis=0)
        num_zero = np.sum(expr_mat == 0, axis=0)
    
    if min_detected_genes is not None:
        low_quality = num_detected < min_detected_genes
    else:
        cell_zero = num_zero
        mu = np.mean(cell_zero)
        sigma = np.std(cell_zero)
        
        # Deal with bi-modal distribution
        within_sigma = (cell_zero > mu - sigma) & (cell_zero < mu + sigma)
        if np.sum(within_sigma) < 0.5:  # should be 0.68 theoretically
            below_median = cell_zero < np.median(cell_zero)
            mu = np.mean(cell_zero[below_median])
            sigma = np.std(cell_zero[below_median])
        
        # Calculate p-values and adjust for multiple testing
        p_values = 1 - stats.norm.cdf((cell_zero - mu) / sigma)
        from statsmodels.stats.multitest import multipletests
        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        low_quality = p_adjusted < 0.05
        
        if not suppress_plot:
            plt.figure(figsize=(8, 6))
            plt.hist(cell_zero, bins=30, density=True, color='grey', alpha=0.75)
            x_range = np.linspace(cell_zero.min(), cell_zero.max(), 100)
            plt.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'b-')
            if np.sum(low_quality) > 0:
                plt.axvline(x=np.min(cell_zero[low_quality]), color='red', linestyle='--')
            plt.xlabel("Number of zeros (per cell)")
            plt.title("Cell Quality Assessment")
            plt.show()
    
    if np.sum(low_quality) > 0:
        if labels is not None and len(labels) == expr_mat.shape[1]:
            labels = labels[~low_quality]
        expr_mat = expr_mat[:, ~low_quality]
    
    return {'data': expr_mat, 'labels': labels}

def hidden_normalize(data):
    """
    Combine UQ and detection rate adjusted normalization 
    Stephanie Hick, Mingziang Teng, Rafael A Irizarry 
    "On the widespread and critical impact of systematic single-cell RNA-Seq data"
    """
    if issparse(data):
        cell_zero = np.array((data == 0).sum(axis=0)).flatten() / data.shape[0]
        uq = np.array([hidden_UQ(data[:, i].toarray().flatten()) for i in range(data.shape[1])])
    else:
        cell_zero = np.sum(data == 0, axis=0) / data.shape[0]
        uq = np.array([hidden_UQ(data[:, i]) for i in range(data.shape[1])])
    
    normfactor = (uq / np.median(uq)) * (np.median(cell_zero) / cell_zero)
    
    if issparse(data):
        # Handle sparse matrix normalization
        data_normalized = data.copy()
        for i in range(data.shape[1]):
            data_normalized[:, i] = data_normalized[:, i] / normfactor[i]
    else:
        data_normalized = data / normfactor[np.newaxis, :]
    
    return data_normalized

def M3DropCleanData(expr_mat, labels=None, is_counts=True, suppress_plot=False, 
                    pseudo_genes=None, min_detected_genes=None):
    """Main data cleaning function for M3Drop"""
    
    # Replace NaN values with 0
    if issparse(expr_mat):
        expr_mat.data = np.nan_to_num(expr_mat.data)
    else:
        expr_mat = np.nan_to_num(expr_mat)
    
    # Remove pseudo genes if specified
    if pseudo_genes is not None and len(pseudo_genes) > 1:
        if hasattr(expr_mat, 'index'):  # pandas DataFrame
            gene_names = expr_mat.index
            is_pseudo = gene_names.isin(pseudo_genes)
            expr_mat = expr_mat[~is_pseudo]
        else:  # assume we have row names somehow accessible
            warnings.warn("Cannot access gene names for pseudo gene filtering")
    
    # Filter cells
    data_list = bg_filter_cells(expr_mat, labels, suppress_plot=suppress_plot, 
                               min_detected_genes=min_detected_genes)
    
    # Filter genes (detected in more than 3 cells)
    if issparse(data_list['data']):
        detected = np.array((data_list['data'] > 0).sum(axis=1)).flatten() > 3
    else:
        detected = np.sum(data_list['data'] > 0, axis=1) > 3
    
    expr_mat = data_list['data'][detected, :]
    labels = data_list['labels']
    
    # Handle ERCC spike-ins
    if hasattr(expr_mat, 'index'):
        gene_names = expr_mat.index
        spikes = gene_names.str.contains('ercc', case=False, na=False)
        spike_indices = np.where(spikes)[0]
    else:
        spike_indices = []  # Cannot identify spikes without gene names
    
    if is_counts:
        if len(spike_indices) > 1:
            if issparse(expr_mat):
                non_spike_mask = np.ones(expr_mat.shape[0], dtype=bool)
                non_spike_mask[spike_indices] = False
                totreads = np.array(expr_mat[non_spike_mask, :].sum(axis=0)).flatten()
            else:
                non_spike_mask = np.ones(expr_mat.shape[0], dtype=bool)
                non_spike_mask[spike_indices] = False
                totreads = np.sum(expr_mat[non_spike_mask, :], axis=0)
        else:
            if issparse(expr_mat):
                totreads = np.array(expr_mat.sum(axis=0)).flatten()
            else:
                totreads = np.sum(expr_mat, axis=0)
        
        # Handle cells with zero total reads to avoid division by zero
        zero_read_cells = totreads == 0
        if np.any(zero_read_cells):
            warnings.warn(f"Found {np.sum(zero_read_cells)} cells with zero total reads. These will be filtered out.")
            # Filter out cells with zero reads
            if issparse(expr_mat):
                expr_mat = expr_mat[:, ~zero_read_cells]
            else:
                expr_mat = expr_mat[:, ~zero_read_cells]
            totreads = totreads[~zero_read_cells]
            if labels is not None:
                labels = labels[~zero_read_cells] if hasattr(labels, '__getitem__') else labels
        
        # Convert to CPM (counts per million)
        if issparse(expr_mat):
            cpm = expr_mat.copy()
            for i in range(expr_mat.shape[1]):
                if totreads[i] > 0:  # Additional safety check
                    cpm[:, i] = cpm[:, i] * (1000000 / totreads[i])
        else:
            # Use safe division to avoid any remaining division by zero
            safe_totreads = np.where(totreads > 0, totreads, 1)
            cpm = expr_mat * (1000000 / safe_totreads[np.newaxis, :])
            # Set CPM to 0 for any cells that had zero reads (though they should be filtered already)
            if np.any(totreads == 0):
                cpm[:, totreads == 0] = 0
        
        # Filter low expression genes (mean CPM < 1e-5)
        if issparse(cpm):
            low_expr = np.array(cpm.mean(axis=1)).flatten() < 1e-5
        else:
            low_expr = np.mean(cpm, axis=1) < 1e-5
        
        cpm = cpm[~low_expr, :]
        return {'data': cpm, 'labels': labels}
    
    # For non-count data - filter low expression and return original data
    if issparse(expr_mat):
        low_expr = np.array(expr_mat.mean(axis=1)).flatten() < 1e-5
    else:
        low_expr = np.mean(expr_mat, axis=1) < 1e-5
    
    data = expr_mat[~low_expr, :]
    return {'data': data, 'labels': labels}

#### Pearson Residuals ####
# February 16, 2023

def NBumiPearsonResiduals(counts, fits=None):
    """Calculate Pearson residuals for negative binomial model"""
    if fits is None:
        fits = NBumiFitModel(counts)
    
    # Convert to numpy for calculations but preserve gene/cell names
    if isinstance(counts, pd.DataFrame):
        gene_names = counts.index
        cell_names = counts.columns
        counts_array = counts.values
    else:
        gene_names = None
        cell_names = None
        counts_array = counts
    
    # Calculate expected values (mu) - matrix multiplication as in R
    # R: mus <- t(t(fits$vals$tjs/fits$vals$total)) %*% fits$vals$tis
    # This creates a genes x cells matrix where mus[i,j] = (tjs[i]/total) * tis[j]
    tjs_normalized = fits['vals']['tjs'] / fits['vals']['total']
    mus = np.outer(tjs_normalized, fits['vals']['tis'])
    
    # Calculate Pearson residuals
    pearson = (counts_array - mus) / np.sqrt(mus + mus**2 / fits['sizes'][:, np.newaxis])
    
    # Return as DataFrame if input was DataFrame
    if gene_names is not None and cell_names is not None:
        return pd.DataFrame(pearson, index=gene_names, columns=cell_names)
    else:
        return pearson

def NBumiPearsonResidualsApprox(counts, fits=None):
    """Calculate approximate Pearson residuals (Poisson approximation)"""
    if fits is None:
        vals = hidden_calc_vals(counts)
    else:
        vals = fits['vals']
    
    # Convert to numpy for calculations but preserve gene/cell names
    if isinstance(counts, pd.DataFrame):
        gene_names = counts.index
        cell_names = counts.columns
        counts_array = counts.values
    else:
        gene_names = None
        cell_names = None
        counts_array = counts
    
    # Calculate expected values (mu) - matrix multiplication as in R
    # R: mus <- t(t(vals$tjs/vals$total)) %*% vals$tis
    tjs_normalized = vals['tjs'] / vals['total']
    mus = np.outer(tjs_normalized, vals['tis'])
    
    # Calculate approximate Pearson residuals (Poisson)
    pearson = (counts_array - mus) / np.sqrt(mus)
    
    # Return as DataFrame if input was DataFrame
    if gene_names is not None and cell_names is not None:
        return pd.DataFrame(pearson, index=gene_names, columns=cell_names)
    else:
        return pearson

#############################
