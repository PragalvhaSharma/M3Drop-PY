
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.stats import nbinom
import matplotlib.pyplot as plt

#### Fitting #####

def hidden_calc_vals(counts):
    """Calculate hidden values for fitting models."""
    if np.any(counts < 0):
        raise ValueError("Expression matrix contains negative values! Please provide raw UMI counts!")
    
    # Check if counts are integers (with some tolerance for floating point)
    if not np.allclose(counts, np.round(counts)):
        raise ValueError("Error: Expression matrix is not integers! Please provide raw UMI counts.")
    
    # Ensure we have row names (gene names)
    if isinstance(counts, pd.DataFrame):
        if counts.index.empty:
            counts.index = [str(i) for i in range(counts.shape[0])]
    elif isinstance(counts, np.ndarray):
        # Convert to DataFrame if numpy array
        counts = pd.DataFrame(counts, index=[str(i) for i in range(counts.shape[0])])
    
    # Convert to sparse matrix if not already
    if not sp.issparse(counts):
        counts_sparse = sp.csr_matrix(counts.values if isinstance(counts, pd.DataFrame) else counts)
    else:
        counts_sparse = counts
    
    # Total molecules/gene
    tjs = np.array(counts_sparse.sum(axis=1)).flatten()
    no_detect = np.sum(tjs <= 0)
    if no_detect > 0:
        raise ValueError(f"Error: contains {no_detect} undetected genes.")
    
    # Total molecules/cell
    tis = np.array(counts_sparse.sum(axis=0)).flatten()
    if np.any(tis <= 0):
        raise ValueError("Error: all cells must have at least one detected molecule.")
    
    # Observed Dropouts per gene
    djs = counts_sparse.shape[1] - np.array((counts_sparse > 0).sum(axis=1)).flatten()
    
    # Observed Dropouts per cell
    dis = counts_sparse.shape[0] - np.array((counts_sparse > 0).sum(axis=0)).flatten()
    
    nc = counts_sparse.shape[1]  # Number of cells
    ng = counts_sparse.shape[0]  # Number of genes
    total = np.sum(tis)  # Total molecules sampled
    
    return {
        'tis': tis,
        'tjs': tjs,
        'dis': dis,
        'djs': djs,
        'total': total,
        'nc': nc,
        'ng': ng
    }

def NBumiConvertToInteger(mat):
    """Convert matrix to integer format."""
    mat = np.ceil(np.asarray(mat)).astype(int)
    # Remove genes with zero total counts
    row_sums = np.sum(mat, axis=1)
    mat = mat[row_sums > 0, :]
    return mat

def NBumiFitModel(counts):
    """Fit negative binomial model to UMI counts."""
    vals = hidden_calc_vals(counts)
    
    min_size = 1e-10
    
    # Calculate row-wise variance
    my_rowvar = np.zeros(counts.shape[0])
    for i in range(counts.shape[0]):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        if isinstance(counts, pd.DataFrame):
            row_data = counts.iloc[i, :].values
        else:
            row_data = counts[i, :]
        my_rowvar[i] = np.var(row_data - mu_is)
    
    # Calculate size parameter
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
    """Fit dispersion vs mean relationship."""
    vals = fit['vals']
    size_g = fit['sizes']
    
    forfit = (fit['sizes'] < np.max(size_g)) & (vals['tjs'] > 0) & (size_g > 0)
    higher = np.log2(vals['tjs'] / vals['nc']) > 4  # As per Grun et al.
    
    if np.sum(higher) > 2000:
        forfit = higher & forfit
    
    x = np.log(vals['tjs'][forfit] / vals['nc'])
    y = np.log(size_g[forfit])
    
    # Linear regression
    coeffs = np.polyfit(x, y, 1)
    
    if not suppress_plot:
        plt.scatter(x, y)
        plt.plot(x, np.polyval(coeffs, x), 'r-')
        plt.xlabel('Log Mean Expression')
        plt.ylabel('Log Size')
        plt.show()
    
    return [coeffs[1], coeffs[0]]  # Return as [intercept, slope] to match R

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

def NBumiFeatureSelectionHighVar(fit):
    """Feature selection based on high variance."""
    vals = fit['vals']
    coeffs = NBumiFitDispVsMean(fit, suppress_plot=True)
    exp_size = np.exp(coeffs[0] + coeffs[1] * np.log(vals['tjs'] / vals['nc']))
    res = np.log(fit['sizes']) - np.log(exp_size)
    
    # Create a sorted dictionary-like structure
    gene_names = list(range(len(res)))  # Replace with actual gene names if available
    sorted_indices = np.argsort(res)[::-1]  # Sort in descending order
    
    return {gene_names[i]: res[i] for i in sorted_indices}

def NBumiFeatureSelectionCombinedDrop(fit, ntop=None, method="fdr", qval_thresh=0.05, suppress_plot=True):
    """Feature selection based on combined dropout analysis."""
    vals = fit['vals']
    
    coeffs = NBumiFitDispVsMean(fit, suppress_plot=True)
    exp_size = np.exp(coeffs[0] + coeffs[1] * np.log(vals['tjs'] / vals['nc']))
    
    droprate_exp = np.zeros(vals['ng'])
    droprate_exp_err = np.zeros(vals['ng'])
    
    for i in range(vals['ng']):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        p_is = (1 + mu_is / exp_size[i])**(-exp_size[i])
        p_var_is = p_is * (1 - p_is)
        droprate_exp[i] = np.sum(p_is) / vals['nc']
        droprate_exp_err[i] = np.sqrt(np.sum(p_var_is) / (vals['nc']**2))
    
    droprate_exp[droprate_exp < 1/vals['nc']] = 1/vals['nc']
    droprate_obs = vals['djs'] / vals['nc']
    droprate_obs_err = np.sqrt(droprate_obs * (1 - droprate_obs) / vals['nc'])
    
    diff = droprate_obs - droprate_exp
    combined_err = np.sqrt(droprate_exp_err**2 + droprate_obs_err**2)
    zed = diff / combined_err
    pvalue = 1 - stats.norm.cdf(zed)  # One-tailed test
    
    # Handle gene names
    gene_names = list(range(len(pvalue)))  # Replace with actual gene names if available
    
    # Sort by p-value and effect size for ties
    reorder = np.lexsort((droprate_exp - droprate_obs, pvalue))
    
    out = pvalue[reorder]
    diff = diff[reorder]
    gene_names = [gene_names[i] for i in reorder]
    
    # Multiple testing correction
    qval = stats.false_discovery_control(out, method='bh') if method == "fdr" else out
    
    if ntop is None:
        mask = qval < qval_thresh
        out = out[mask]
        diff = diff[mask]
        qval = qval[mask]
        gene_names = [gene_names[i] for i in range(len(mask)) if mask[i]]
    else:
        out = out[:ntop]
        diff = diff[:ntop]
        qval = qval[:ntop]
        gene_names = gene_names[:ntop]
    
    outTABLE = pd.DataFrame({
        'Gene': gene_names,
        'effect_size': diff,
        'p_value': out,
        'q_value': qval
    })
    
    if not suppress_plot:
        xes = np.log10(vals['tjs'] / vals['nc'])
        
        # Create density colors (simplified)
        plt.figure(figsize=(10, 6))
        plt.scatter(xes, droprate_obs, c='gray', alpha=0.6, s=20)
        
        # Highlight selected genes
        toplot = np.isin(list(range(len(vals['tjs']))), 
                        [reorder[i] for i in range(len(gene_names))])
        plt.scatter(xes[toplot], droprate_obs[toplot], c='darkorange', s=20)
        plt.scatter(xes, droprate_exp, c='dodgerblue', s=20)
        
        plt.xlabel('log10(expression)')
        plt.ylabel('Dropout Rate')
        plt.show()
    
    return outTABLE

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

