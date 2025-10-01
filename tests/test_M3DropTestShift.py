import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest
import scanpy as sc

from m3Drop.Extremes import M3DropTestShift

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "GSM8267529_G-P28_raw_matrix.h5ad"
MAX_CELLS = 200
MAX_BACKGROUND_GENES = 200
MIN_GENES_TO_TEST = 10


@pytest.fixture(scope="module")
def small_raw_counts():
    if not DATA_FILE.exists():
        pytest.skip(f"Missing test fixture file: {DATA_FILE}")

    adata = sc.read_h5ad(DATA_FILE, backed="r")
    subset = None
    try:
        max_cells = min(adata.n_obs, MAX_CELLS)
        background_genes = list(adata.var_names[:min(adata.n_vars, MAX_BACKGROUND_GENES)])
        if max_cells == 0 or not background_genes:
            pytest.skip("Dataset too small for M3DropTestShift test.")

        subset = adata[:max_cells, background_genes].to_memory()
    finally:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

    if subset is None:
        pytest.fail("Failed to build subset for M3DropTestShift test.")

    raw_counts = subset.to_df().astype("float32").T
    raw_counts.index = raw_counts.index.astype(str)
    return raw_counts


def test_m3drop_test_shift_runs_on_small_subset(small_raw_counts):
    if len(small_raw_counts.index) < MIN_GENES_TO_TEST:
        pytest.skip("Not enough genes available for M3DropTestShift test.")

    genes_to_test = small_raw_counts.index[:MIN_GENES_TO_TEST].tolist()
    shift_results = M3DropTestShift(
        small_raw_counts,
        genes_to_test=genes_to_test,
        name="First10",
        suppress_plot=True,
    )

    assert isinstance(shift_results, pd.DataFrame)
    assert not shift_results.empty
