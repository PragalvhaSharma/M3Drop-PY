def NBumiCompareModelsGPU(
    raw_filename: str,
    cleaned_filename: str,
    stats: dict,
    fit_adjust: dict,
    chunk_size: int = None,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    OPTIMIZED VERSION (IN-MEMORY):
    - Eliminates the 46GB '_basic_norm.h5ad' temporary file.
    - Performs depth normalization and variance calculation on-the-fly in GPU VRAM.
    """
    pipeline_start_time = time.time()
    print(f"FUNCTION: NBumiCompareModels() | Comparing models for {cleaned_filename}")

    # [GOVERNOR] High multiplier (12.0) because we hold Raw + Norm + Square in VRAM
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(raw_filename, multiplier=12.0, is_dense=False)

    # --- Phase 1: In-Memory "Basic Fit" (Normalization + Variance) ---
    print("Phase [1/3]: Calculating Basic Model (Depth-Normalized) variance on-the-fly...")
    
    # 1. Prepare Size Factors (CPU)
    tjs = stats['tjs'].values # Gene sums (needed for final dataframe)
    tis = stats['tis'].values # Cell sums (needed for size factors)
    nc, ng = stats['nc'], stats['ng']
    
    median_sum = np.median(tis[tis > 0])
    size_factors = np.ones_like(tis, dtype=np.float32)
    non_zero_mask = tis > 0
    size_factors[non_zero_mask] = tis[non_zero_mask] / median_sum
    
    # 2. Prepare GPU Arrays
    sum_x_sq_gpu = cp.zeros(ng, dtype=cp.float64)
    sum_x_gpu = cp.zeros(ng, dtype=cp.float64) # Need sum(x) to calc mean(x) for variance
    
    # 3. GPU Loop (Raw Data -> Normalize -> Accumulate)
    with h5py.File(raw_filename, 'r') as f_in:
        h5_indptr = f_in['X']['indptr']
        h5_data = f_in['X']['data']
        h5_indices = f_in['X']['indices']

        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"Phase [1/3]: Processing: {end_row} of {nc} cells.", end='\r')
            
            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx: continue

            # Load Raw Chunk
            data_gpu = cp.asarray(h5_data[start_idx:end_idx], dtype=cp.float32)
            indices_gpu = cp.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu = cp.asarray(h5_indptr[i:end_row + 1] - start_idx)

            # Expand Size Factors to match Data Structure
            # (Map cell's size factor to every non-zero gene in that cell)
            nnz_in_chunk = indptr_gpu[-1].item()
            cell_boundary_markers = cp.zeros(nnz_in_chunk, dtype=cp.int32)
            if len(indptr_gpu) > 1:
                cell_boundary_markers[indptr_gpu[:-1]] = 1
            # row_indices maps every data point to its cell index (0 to chunk_size)
            row_indices = cp.cumsum(cell_boundary_markers, axis=0) - 1
            
            # Get size factors for this chunk
            sf_chunk = cp.asarray(size_factors[i:end_row])
            
            # --- THE MAGIC: On-the-Fly Normalization ---
            # data_norm = data_raw / size_factor
            data_gpu /= sf_chunk[row_indices]
            
            # Accumulate for Variance: E[X^2] and E[X]
            cp.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
            cp.add.at(sum_x_gpu, indices_gpu, data_gpu)
            
            # Clean up VRAM
            del data_gpu, indices_gpu, indptr_gpu, row_indices, sf_chunk, cell_boundary_markers
            cp.get_default_memory_pool().free_all_blocks()

    print(f"Phase [1/3]: COMPLETE{' '*50}")

    # 4. Finalize Basic Statistics
    # Var(X) = E[X^2] - (E[X])^2
    mean_x_sq_gpu = sum_x_sq_gpu / nc
    mean_mu_gpu = sum_x_gpu / nc
    my_rowvar_gpu = mean_x_sq_gpu - mean_mu_gpu**2
    
    # Dispersion = Mean^2 / (Var - Mean)
    size_gpu = mean_mu_gpu**2 / (my_rowvar_gpu - mean_mu_gpu)
    
    # Safety Clamping (Same as original)
    max_size_val = cp.nanmax(size_gpu) * 10
    if cp.isnan(max_size_val): max_size_val = 1000
    size_gpu[cp.isnan(size_gpu) | (size_gpu <= 0)] = max_size_val
    size_gpu[size_gpu < 1e-10] = 1e-10
    
    # Construct "Basic Fit" Object
    fit_basic = {
        'sizes': pd.Series(size_gpu.get(), index=stats['tjs'].index),
        'vals': stats,
        'var_obs': pd.Series(my_rowvar_gpu.get(), index=stats['tjs'].index)
    }
    
    # --- Phase 2: Check Fit (Calculate Errors) ---
    print("Phase [2/3]: Evaluating fit errors on ORIGINAL data...")
    
    # Check Adjust (M3Drop)
    check_adjust = NBumiCheckFitFSGPU(
        cleaned_filename, fit_adjust, suppress_plot=True, chunk_size=chunk_size
    )
    
    # Check Basic (Depth-Norm)
    check_basic = NBumiCheckFitFSGPU(
        cleaned_filename, fit_basic, suppress_plot=True, chunk_size=chunk_size
    )
    print("Phase [2/3]: COMPLETE")

    # --- Phase 3: Plotting & Comparison ---
    print("Phase [3/3]: Generating comparison...")
    
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
    plt.title("M3Drop Model Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"STATUS: Model comparison plot saved to '{plot_filename}'")

    if not suppress_plot:
        plt.show()
    
    plt.close()
    
    pipeline_end_time = time.time()
    print(f"Total time: {pipeline_end_time - pipeline_start_time:.2f} seconds.\n")
    
    return {
        "errors": {"Depth-Adjusted": err_adj, "Basic": err_bas},
        "comparison_df": comparison_df
    }
