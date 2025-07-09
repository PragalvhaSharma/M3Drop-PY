from anndata import AnnData
import pandas as pd
import numpy as np
from scipy.sparse import issparse

from .utils import NBumiFitModel
from .normalization import NBumiImputeNorm, NBumiConvertToInteger
from .feature_selection import Consensus_FS

def nbumi_normalize(
    adata: AnnData,
    copy: bool = False,
) -> AnnData | None:
    """
    Normalize count data using the NBumi method from M3Drop.

    This function wraps `NBumiFitModel` and `NBumiImputeNorm`.
    It fits a depth-adjusted negative binomial model to the raw counts and then
    normalizes the data based on this model. The normalized data is stored
    in `adata.X`. This method is suitable for UMI-based scRNA-seq data.

    The input `AnnData` object should contain raw counts. It is recommended to
    store the raw counts in a layer before normalization. For example, by
    running `adata.layers['counts'] = adata.X.copy()`. This function will
    look for `'counts'` in `adata.layers` and use it if available.

    Note that the output matrix is dense, which may consume a lot of memory
    for large datasets.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    copy
        If `True`, returns a new `AnnData` object. Otherwise, updates `adata`.

    Returns
    -------
    If `copy=True`, returns a new `AnnData` object. Otherwise, modifies `adata` inplace
    and returns `None`.
    """
    adata_to_work_on = adata.copy() if copy else adata

    if 'counts' in adata_to_work_on.layers:
        counts_mat = adata_to_work_on.layers['counts']
    else:
        counts_mat = adata_to_work_on.X

    is_sparse_input = issparse(counts_mat)
    if is_sparse_input:
        if not (counts_mat.data % 1 == 0).all():
            counts_mat = NBumiConvertToInteger(counts_mat.toarray())
            is_sparse_input = False
    else:
        if not (np.all(counts_mat % 1 == 0)):
            counts_mat = NBumiConvertToInteger(counts_mat)

    counts_mat_T = counts_mat.T

    if is_sparse_input:
        counts_df = pd.DataFrame.sparse.from_spmatrix(
            counts_mat_T, index=adata_to_work_on.var_names, columns=adata_to_work_on.obs_names
        )
    else:
            counts_df = pd.DataFrame(
            counts_mat_T, index=adata_to_work_on.var_names, columns=adata_to_work_on.obs_names
        )

    fit = NBumiFitModel(counts_df)

    counts_for_norm = counts_mat_T.toarray() if issparse(counts_mat_T) else counts_mat_T
    norm_mat = NBumiImputeNorm(counts_for_norm, fit)

    adata_to_work_on.X = norm_mat.T

    if copy:
        return adata_to_work_on

def m3drop_highly_variable_genes(
    adata: AnnData,
    ntop: int = 2000,
    copy: bool = False,
) -> AnnData | None:
    """
    Feature selection using the M3Drop consensus method.

    This function uses `Consensus_FS` which combines seven different
    feature selection methods to get a consensus ranking. The results are
    stored in `adata.var`.

    The input `AnnData` object should contain raw counts in `adata.layers['counts']`
    and normalized data in `adata.X`.

    Parameters
    ----------
    adata
        The annotated data matrix.
    ntop
        Number of top highly variable genes to select.
    copy
        If `True`, returns a new `AnnData` object. Otherwise, updates `adata`.

    Returns
    -------
    If `copy=True`, returns a new `AnnData` object. Otherwise, modifies `adata` inplace
    and returns `None`. The results are stored in `adata.var`:
    - `highly_variable` (boolean)
    - `m3drop_consensus_rank` (int)
    And other rankings from individual methods.
    """
    adata_to_work_on = adata.copy() if copy else adata

    if 'counts' not in adata_to_work_on.layers:
        raise ValueError("Raw counts not found in adata.layers['counts']. Please store raw counts there.")

    counts_mat = adata_to_work_on.layers['counts']
    norm_mat = adata_to_work_on.X

    counts_df = pd.DataFrame(
        counts_mat.toarray().T if issparse(counts_mat) else counts_mat.T,
        index=adata_to_work_on.var_names
    )
    norm_df = pd.DataFrame(
        norm_mat.toarray().T if issparse(norm_mat) else norm_mat.T,
        index=adata_to_work_on.var_names
    )

    rank_table = Consensus_FS(counts=counts_df, norm=norm_df)

    rank_table = rank_table.rename(columns={'Cons': 'm3drop_consensus_rank'})
    for col in rank_table.columns:
        if col != 'm3drop_consensus_rank':
            rank_table = rank_table.rename(columns={col: f'm3drop_{col}_rank'})

    adata_to_work_on.var = adata_to_work_on.var.join(rank_table)

    top_genes = rank_table.nsmallest(ntop, 'm3drop_consensus_rank').index
    adata_to_work_on.var['highly_variable'] = adata_to_work_on.var.index.isin(top_genes)

    if copy:
        return adata_to_work_on 