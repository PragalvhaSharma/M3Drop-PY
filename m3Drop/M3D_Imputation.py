import numpy as np
import pandas as pd
from .basics import bg__calc_variables
from .Curve_fitting import bg__fit_MM


def hidden_m3dropImputeZeros(expr_mat):
    """
    Imputes dropout values in single-cell RNA-seq data using the M3Drop method.
    
    This function uses the Michaelis-Menten model to predict expected expression
    values and imputes dropout values when the observed expression is below the
    expected expression.
    
    Parameters
    ----------
    expr_mat : pd.DataFrame or np.ndarray
        Expression matrix with genes as rows and cells as columns.
        Should be normalized but not log-transformed.
    
    Returns
    -------
    pd.DataFrame or np.ndarray
        Imputed expression matrix with the same dimensions as input.
    """
    # Convert to DataFrame if numpy array
    if isinstance(expr_mat, np.ndarray):
        expr_mat = pd.DataFrame(expr_mat)
        return_array = True
    else:
        return_array = False
    
    # Calculate gene-specific variables (mean expression, dropout rate, etc.)
    BasePlot = {'gene_info': bg__calc_variables(expr_mat)}
    
    # Fit Michaelis-Menten model to the relationship between mean expression and dropout rate
    MM = bg__fit_MM(BasePlot['gene_info']['p'], BasePlot['gene_info']['s'])
    
    # Handle zero dropout rates for calculation
    no_zero_p = BasePlot['gene_info']['p'].copy()
    n_cells = expr_mat.shape[1]
    
    # Replace zero dropout rates with small value: 1/(2*n_cells)
    zero_mask = no_zero_p == 0
    no_zero_p[zero_mask] = 1.0 / (n_cells * 2)
    
    # Calculate expected expression using Michaelis-Menten model
    # Expected_S = K * (1/p - 1), where p is dropout rate and K is MM parameter
    Expected_S = MM['K'] * (1.0 / no_zero_p - 1.0)
    
    # Create matrix of expected values replicated across cells
    expect_mat = np.tile(Expected_S.values.reshape(-1, 1), (1, n_cells))
    expect_mat = pd.DataFrame(expect_mat, index=expr_mat.index, columns=expr_mat.columns)
    
    # Calculate imputed values as average of expected and observed
    imp_mat = (expect_mat + expr_mat) / 2.0
    
    # Start with original matrix
    new_mat = expr_mat.copy()
    
    # Replace values where observed < expected with imputed values
    mask = expr_mat < expect_mat
    new_mat[mask] = imp_mat[mask]
    
    # Return as array if input was array
    if return_array:
        return new_mat.values
    else:
        return new_mat


def M3DropImputation(expr_mat, suppress_plot=True):
    """
    Public interface for M3Drop imputation.
    
    Parameters
    ----------
    expr_mat : pd.DataFrame or np.ndarray
        Expression matrix with genes as rows and cells as columns.
    suppress_plot : bool, default=True
        Whether to suppress plotting (for compatibility).
    
    Returns
    -------
    pd.DataFrame or np.ndarray
        Imputed expression matrix.
    """
    return hidden_m3dropImputeZeros(expr_mat)


# Alias for backwards compatibility
hidden_m3drop_impute_zeros = hidden_m3dropImputeZeros
hidden_m3dropImputeZeros = hidden_m3dropImputeZeros
