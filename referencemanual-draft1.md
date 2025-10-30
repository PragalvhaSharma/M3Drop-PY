# M3DropGPU Reference Manual

The purpose of this manual is to provide a comprehensive description of each pipeline and their functions. 

## Table of Contents

1.  [I. Core Pipeline](#i-core-pipeline)
    * [Stage 1: Data Cleaning](#stage-1-data-cleaning)
    * [Stage 2: Total and Dropout Counts Calculation](#stage-2-total-and-dropout-counts-calculation)
    * [Stage 3: Model Fitting](#stage-3-model-fitting)
    * [Stage 4: Feature Selection](#stage-4-feature-selection)
    * [Stage 5: Visualization](#stage-5-visualization)
2.  [II. Diagnostics Pipeline](#ii-diagnostics-pipeline)
    * [Stage 1: Data Cleaning](#stage-1-data-cleaning-1)
    * [Stage 2: Statistics Calculation](#stage-2-statistics-calculation)
    * [Stage 3: Adjusted Model Fitting](#stage-3-adjusted-model-fitting)
    * [Stage 4: Dispersion vs. Mean Plot](#stage-4-dispersion-vs-mean-plot)
    * [Stage 5: Model Comparison](#stage-5-model-comparison)
3.  [III. Normalization Pipeline](#iii-normalization-pipeline)
    * [Stage 1: Data Cleaning](#stage-1-data-cleaning-2)
    * [Stage 2: Statistics Calculation](#stage-2-statistics-calculation-1)
    * [Stage 3: Model Fitting](#stage-3-model-fitting-1)
    * [Stage 4: Pearson Residuals Normalization](#stage-4-pearson-residuals-normalization)
4.  [IV. Plot Pipeline](#iv-plot-pipeline)
    * [Stage 1: Data Cleaning](#stage-1-data-cleaning-3)
    * [Stage 2: Statistics Calculation](#stage-2-statistics-calculation-2)
    * [Stage 3: Adjusted Model Fitting](#stage-3-adjusted-model-fitting-1)
    * [Stage 4: Generate Dispersion vs. Mean Plot](#stage-4-generate-dispersion-vs-mean-plot)
    * [Stage 5: Generate Model Comparison Plot](#stage-5-generate-model-comparison-plot)
    * [Stage 6: Feature Selection for Volcano Plot](#stage-6-feature-selection-for-volcano-plot)
    * [Stage 7: Generate Volcano Plot](#stage-7-generate-volcano-plot)
    * [Stage 8: Cleanup](#stage-8-cleanup)

## I. Core Pipeline

The `core_pipeline.py` script is the primary **feature selection workflow**. It processes a raw `.h5ad` count matrix by cleaning the data, calculating statistics, and fitting a Negative Binomial model. Using this model, it identifies significant genes based on high variance and high dropout rates. The final outputs are `.csv` files listing these genes and a volcano plot visualizing the results.

### Stage 1: Data Cleaning
This stage reads a raw .h5ad file, identifies and removes any genes (columns) with zero counts across all cells, and rounds up any decimal counts to the nearest integer. It then saves this processed matrix as a new, "cleaned" .h5ad file.

#### ConvertDataSparse()

Performs out-of-core cleaning on a sparse (cell, gene) .h5ad file. It filters out all-zero genes and ensures all data is ceiled to integer values.

```python
def ConvertDataSparse(input_filename: str, output_filename: str, row_chunk_size: int = 5000):
```

Parameters:

* `input_filename (str)`: Path to the raw .h5ad file.
* `output_filename (str)`: Path to save the cleaned .h5ad file.
* `row_chunk_size (int)`: The number of cells (rows) to process at a time during the filtering and writing passes.

Output: 

* `None`. A new .h5ad file is saved to the `output_filename` path.

### Stage 2: Total and Dropout Counts Calculation
This stage performs a single, GPU-accelerated pass over the cleaned data file to calculate fundamental count statistics (e.g., total counts per-cell, total counts per-gene) required for model fitting.

#### hidden_calc_vals()

A GPU-accelerated, single-pass function that calculates total counts per-cell (tis), total counts per-gene (tjs), dropouts per-cell (dis), and dropouts per-gene (djs) from a sparse .h5ad file.

```python
def hidden_calc_vals(filename: str, chunk_size: int = 5000) -> dict:
```

Parameters:

* `filename (str)`: Path to the cleaned sparse .h5ad file.
* `chunk_size (int)`: The number of rows (cells) to process at a time.

Output:

* `A dictionary` containing tis, tjs, dis, djs (as pd.Series), total count, nc (cell count), and ng (gene count). This dictionary is then saved as a .pkl file.

### Stage 3: Model Fitting

This stage fits the depth-adjusted Negative Binomial (NB) model to the data. It uses the pre-calculated statistics and the cleaned data file to determine the observed variance and dispersion parameter (sizes) for every gene.

#### NBumiFitModel()

Fits a Negative Binomial model to the count data using a GPU-accelerated, chunked algorithm. It calculates the observed variance (var_obs) and the dispersion parameter (sizes) for each gene.

```python
def NBumiFitModel(cleaned_filename: str, stats: dict, chunk_size: int = 5000) -> dict:
```

Parameters:

* `cleaned_filename (str)`: Path to the cleaned sparse .h5ad file.
* `stats (dict)`: The statistics dictionary output from `hidden_calc_vals`.
* `chunk_size (int)`: The number of rows (cells) to process at a time.

Output:

* `A dictionary` containing var_obs (observed variance) and sizes (dispersion parameter) as pd.Series, and the original stats dictionary nested under the 'vals' key. This "fit" dictionary is saved as a .pkl file.

### Stage 4: Feature Selection
This stage identifies significant genes using two different methods based on the fitted model.

#### Method 1: Variance

#### NBumiFeatureSelectionHighVar()

Selects genes by calculating the residual between the observed log-dispersion (log(sizes)) and the log-dispersion expected from the mean-dispersion trend. Genes with a large positive residual are more variable than expected.

```python
def NBumiFeatureSelectionHighVar(fit: dict) -> pd.DataFrame:
```

Parameters:

* `fit (dict)`: The 'fit' object output from `NBumiFitModel`.

Output: 

* `A pd.DataFrame`. A DataFrame of all genes, sorted by their 'Residual' value. This DataFrame is saved as a .csv file.

#### Method 2: Combined Dropout

#### NBumiFeatureSelectionCombinedDrop()

Calculates the expected dropout rate (proportion of zeros) for each gene based on the model's smoothed dispersion parameters. It performs a Z-test to compare the observed dropout rate to the expected rate, generating p-values and q-values for significance.

```python
def NBumiFeatureSelectionCombinedDrop(
    fit: dict,
    cleaned_filename: str,
    chunk_size: int = 5000,
    method="fdr_bh",
    qval_thresh=0.05
) -> pd.DataFrame:
```

Parameters:

* `fit (dict)`: The 'fit' object output from `NBumiFitModel`.
* `cleaned_filename (str)`: Path to the cleaned sparse .h5ad file.
* `chunk_size (int)`: The number of cells to process at a time when calculating expected dropouts.
* `method (str)`: The p-value correction method (default: "fdr_bh").
* `qval_thresh (float)`: The q-value threshold for filtering the final table (default: 0.05).

Output:

* `A pd.DataFrame`. A DataFrame containing 'Gene', 'effect_size', 'p.value', and 'q.value' for genes below the `qval_thresh`. This DataFrame is saved as a .csv file.

### Stage 5: Visualization

This stage generates a volcano plot from the results of the "Combined Dropout" feature selection.

#### NBumiCombinedDropVolcano()

Creates a volcano plot, plotting effect size (Observed - Expected Dropout Rate) versus statistical significance (-log10(q.value)).

```python
def NBumiCombinedDropVolcano(
    results_df: pd.DataFrame,
    qval_thresh: float = 0.05,
    effect_size_thresh: float = 0.25,
    top_n_genes: int = 10,
    suppress_plot: bool = False,
    plot_filename: str = None
):
```

Parameters:

* `results_df (pd.DataFrame)`: The DataFrame output from `NBumiFeatureSelectionCombinedDrop`.
* `qval_thresh (float)`: The q-value threshold for drawing the significance line (default: 0.05).
* `effect_size_thresh (float)`: The effect size threshold for coloring significant points (default: 0.25).
* `top_n_genes (int)`: The number of top significant genes to label on the plot (default: 10).
* `suppress_plot (bool)`: If True, prevents `plt.show()` from being called (default: False).
* `plot_filename (str, optional)`: If provided, saves the plot to this file path (default: None).

Output:

* `A matplotlib.axes.Axes` object for the plot. The pipeline calls this function to save the plot as a `.png` file.

## II. Diagnostics Pipeline

The `diagnostics_pipeline.py` script is a model validation workflow. Its purpose is to compare the performance of the depth-adjusted Negative Binomial (NB) model (used in the core pipeline) against a simpler, "basic" NB model. It also visualizes the mean-dispersion fit. The final outputs are `.png` plots for model comparison and dispersion analysis.

### Stage 1: Data Cleaning
This stage is identical to Stage 1 of the Core Pipeline. It reads a raw .h5ad file, identifies and removes any genes (columns) with zero counts across all cells, and rounds up any decimal counts to the nearest integer. It then saves this processed matrix as a new, "cleaned" .h5ad file.

#### ConvertDataSparse()

Performs out-of-core cleaning on a sparse (cell, gene) .h5ad file. It filters out all-zero genes and ensures all data is ceiled to integer values.

```python
def ConvertDataSparse(input_filename: str, output_filename: str, row_chunk_size: int = 5000):
```

Parameters:

* `input_filename (str)`: Path to the raw .h5ad file.
* `output_filename (str)`: Path to save the cleaned .h5ad file.
* `row_chunk_size (int)`: The number of cells (rows) to process at a time during the filtering and writing passes.

Output: 

* `None`. A new .h5ad file is saved to the `output_filename` path.

### Stage 2: Statistics Calculation
This stage is identical to Stage 2 of the Core Pipeline. It performs a single, GPU-accelerated pass over the cleaned data file to calculate fundamental count statistics required for model fitting.

#### hidden_calc_vals()

A GPU-accelerated, single-pass function that calculates total counts per-cell (tis), total counts per-gene (tjs), dropouts per-cell (dis), and dropouts per-gene (djs) from a sparse .h5ad file.

```python
def hidden_calc_vals(filename: str, chunk_size: int = 5000) -> dict:
```

Parameters:

* `filename (str)`: Path to the cleaned sparse .h5ad file.
* `chunk_size (int)`: The number of rows (cells) to process at a time.

Output:

* `A dictionary` containing tis, tjs, dis, djs (as pd.Series), total count, nc (cell count), and ng (gene count). This dictionary is then saved as a .pkl file.

### Stage 3: Adjusted Model Fitting

This stage is identical to Stage 3 of the Core Pipeline. It fits the depth-adjusted Negative Binomial (NB) model to the data to determine the observed variance and dispersion parameter (sizes) for every gene.

#### NBumiFitModel()

Fits a Negative Binomial model to the count data using a GPU-accelerated, chunked algorithm. It calculates the observed variance (var_obs) and the dispersion parameter (sizes) for each gene.

```python
def NBumiFitModel(cleaned_filename: str, stats: dict, chunk_size: int = 5000) -> dict:
```

Parameters:

* `cleaned_filename (str)`: Path to the cleaned sparse .h5ad file.
* `stats (dict)`: The statistics dictionary output from `hidden_calc_vals`.
* `chunk_size (int)`: The number of rows (cells) to process at a time.

Output:

* `A dictionary` containing var_obs (observed variance) and sizes (dispersion parameter) as pd.Series, and the original stats dictionary nested under the 'vals' key. This "fit" dictionary is saved as a .pkl file.

### Stage 4: Dispersion vs. Mean Plot

This stage generates a diagnostic plot showing the relationship between gene expression mean and the fitted dispersion parameter, along with the fitted regression line.

#### NBumiPlotDispVsMean()

Generates and saves a log-log plot of mean expression vs. dispersion, with the fitted regression line from `NBumiFitDispVsMean` overlaid.

```python
def NBumiPlotDispVsMean(
    fit: dict,
    suppress_plot: bool = False,
    plot_filename: str = None
):
```

Parameters:

* `fit (dict)`: The 'fit' object output from `NBumiFitModel`.
* `suppress_plot (bool)`: If True, prevents `plt.show()` from being called (default: False).
* `plot_filename (str, optional)`: If provided, saves the plot to this file path (default: None).

Output: 

* `None`. The plot is saved to the `plot_filename` path.

### Stage 5: Model Comparison

This stage runs a full comparison. It creates a "basic" normalized dataset, fits a "basic" NB model, and then generates a plot comparing the observed dropout rates to the expected dropout rates from both the "basic" and "depth-adjusted" models.

#### NBumiCompareModels()

Compares the depth-adjusted model (fit_adjust) to a basic NB model fit on standard normalized data. It generates a plot comparing observed vs. expected dropout rates for both models.

```python
def NBumiCompareModels(
    raw_filename: str,
    cleaned_filename: str,
    stats: dict,
    fit_adjust: dict,
    chunk_size: int = 5000,
    suppress_plot=False,
    plot_filename=None
) -> dict:
```

Parameters:

* `raw_filename (str)`: Path to the original raw .h5ad file.
* `cleaned_filename (str)`: Path to the cleaned sparse .h5ad file.
* `stats (dict)`: The statistics dictionary output from `hidden_calc_vals`.
* `fit_adjust (dict)`: The "fit" object from the depth-adjusted `NBumiFitModel`.
* `chunk_size (int)`: The number of rows to process at a time (default: 5000).
* `suppress_plot (bool)`: If True, prevents `plt.show()` from being called (default: False).
* `plot_filename (str, optional)`: If provided, saves the plot to this file path (default: None).

Output:

* `A dictionary` containing the error values (`"errors"`) and the (`"comparison_df"`) used for plotting. The plot is also saved to `plot_filename`.

## III. Normalization Pipeline

The `normalization_pipeline.py` script is a **data normalization workflow**. It uses the fitted Negative Binomial model to transform the raw, cleaned count matrix into Pearson residuals. It produces two distinct `.h5ad` files: one with full, precise residuals and one with a computationally faster, approximate version.

### Stage 1: Data Cleaning
This stage is identical to Stage 1 of the Core Pipeline. It reads a raw .h5ad file, identifies and removes any genes (columns) with zero counts across all cells, and rounds up any decimal counts to the nearest integer.

#### ConvertDataSparse()

Performs out-of-core cleaning on a sparse (cell, gene) .h5ad file. It filters out all-zero genes and ensures all data is ceiled to integer values.

```python
def ConvertDataSparse(input_filename: str, output_filename: str, row_chunk_size: int = 5000):
```

Parameters:

* `input_filename (str)`: Path to the raw .h5ad file.
* `output_filename (str)`: Path to save the cleaned .h5ad file.
* `row_chunk_size (int)`: The number of cells (rows) to process at a time during the filtering and writing passes.

Output: 

* `None`. A new .h5ad file is saved to the `output_filename` path.

### Stage 2: Statistics Calculation
This stage is identical to Stage 2 of the Core Pipeline. It performs a single, GPU-accelerated pass over the cleaned data file to calculate fundamental count statistics required for model fitting.

#### hidden_calc_vals()

A GPU-accelerated, single-pass function that calculates total counts per-cell (tis), total counts per-gene (tjs), dropouts per-cell (dis), and dropouts per-gene (djs) from a sparse .h5ad file.

```python
def hidden_calc_vals(filename: str, chunk_size: int = 5000) -> dict:
```

Parameters:

* `filename (str)`: Path to the cleaned sparse .h5ad file.
* `chunk_size (int)`: The number of rows (cells) to process at a time.

Output:

* `A dictionary` containing tis, tjs, dis, djs (as pd.Series), total count, nc (cell count), and ng (gene count). This dictionary is then saved as a .pkl file.

### Stage 3: Model Fitting

This stage is identical to Stage 3 of the Core Pipeline. It fits the depth-adjusted Negative Binomial (NB) model to the data to determine the observed variance and dispersion parameter (sizes) for every gene.

#### NBumiFitModel()

Fits a Negative Binomial model to the count data using a GPU-accelerated, chunked algorithm. It calculates the observed variance (var_obs) and the dispersion parameter (sizes) for each gene.

```python
def NBumiFitModel(cleaned_filename: str, stats: dict, chunk_size: int = 5000) -> dict:
```

Parameters:

* `cleaned_filename (str)`: Path to the cleaned sparse .h5ad file.
* `stats (dict)`: The statistics dictionary output from `hidden_calc_vals`.
* `chunk_size (int)`: The number of rows (cells) to process at a time.

Output:

* `A dictionary` containing var_obs (observed variance) and sizes (dispersion parameter) as pd.Series, and the original stats dictionary nested under the 'vals' key. This "fit" dictionary is saved as a .pkl file.

### Stage 4: Pearson Residuals Normalization

This stage computes the normalized data.

#### Method 1: Full Pearson Residuals

#### NBumiPearsonResiduals()

Calculates full, precise Pearson residuals using the fitted model's dispersion (`sizes`) parameter. This method is slower but more accurate, as it uses the full NB variance `(V = mu + mu^2 / size)`.

```python
def NBumiPearsonResiduals(
    cleaned_filename: str,
    fit_filename: str,
    output_filename: str,
    chunk_size: int = 5000
):
```

Parameters:

* `cleaned_filename (str)`: Path to the cleaned sparse .h5ad input file.
* `fit_filename (str)`: Path to the saved 'fit' object (`_fit.pkl`) from Stage 3.
* `output_filename (str)`: Path to save the output .h5ad file containing dense residuals.
* `chunk_size (int)`: The number of cells (rows) to process at a time.

Output:

* `None`. A new .h5ad file containing a *dense* matrix of residuals is saved to the `output_filename` path.

#### Method 2: Approximate Pearson Residuals

#### NBumiPearsonResidualsApprox()

Calculates approximate Pearson residuals. This method ignores the dispersion parameter and assumes a simpler variance model `(V = mu)`, similar to a Poisson model. It is computationally faster but less precise.

```python
def NBumiPearsonResidualsApprox(
    cleaned_filename: str,
    stats_filename: str,
    output_filename: str,
    chunk_size: int = 5000
):
```

Parameters:

* `cleaned_filename (str)`: Path to the cleaned sparse .h5ad input file.
* `stats_filename (str)`: Path to the saved 'stats' object (`_stats.pkl`) from Stage 2.
* `output_filename (str)`: Path to save the output .h5ad file containing dense residuals.
* `chunk_size (int)`: The number of cells (rows) to process at a time.

Output:

* `None`. A new .h5ad file containing a *dense* matrix of residuals is saved to the `output_filename` path.

## IV. Plot Pipeline

The `plot_pipeline.py` is an **all-in-one visualization workflow**. It executes the core feature selection, model diagnostics, and visualization steps in a single script. Its purpose is to generate all three primary plots (Dispersion vs. Mean, Model Comparison, and Volcano) from a single dataset, and then clean up all intermediate `.h5ad` and `.pkl` files.

### Stage 1: Data Cleaning

#### ConvertDataSparse()
* (See Core Pipeline, Stage 1 for full description)

### Stage 2: Statistics Calculation

#### hidden_calc_vals()
* (See Core Pipeline, Stage 2 for full description)

### Stage 3: Adjusted Model Fitting

#### NBumiFitModel()
* (See Core Pipeline, Stage 3 for full description)

### Stage 4: Generate Dispersion vs. Mean Plot

#### NBumiPlotDispVsMean()
* (See Diagnostics Pipeline, Stage 4 for full description)

### Stage 5: Generate Model Comparison Plot

#### NBumiCompareModels()
* (See Diagnostics Pipeline, Stage 5 for full description)

### Stage 6: Feature Selection for Volcano Plot

#### NBumiFeatureSelectionCombinedDrop()
* (See Core Pipeline, Stage 4, Method 2 for full description)

### Stage 7: Generate Volcano Plot

#### NBumiCombinedDropVolcano()
* (See Core Pipeline, Stage 5 for full description)

### Stage 8: Cleanup
This stage is not a function but a `finally` block in the script. It automatically deletes all intermediate files (`_cleaned.h5ad`, `_stats.pkl`, `_adjusted_fit.pkl`) created during the run, leaving only the raw data and the final `.png` plots.
