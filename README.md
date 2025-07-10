# M3DropPy

A Python implementation of the M3Drop single-cell RNA-seq analysis tool, originally developed as an R package.

## About

M3DropPy is a Python conversion of the popular M3Drop R package for dropout-based feature selection in single-cell RNA sequencing data. This package provides powerful methods for identifying highly variable and differentially expressed genes by leveraging the high frequency of dropout events (zero expression values) that are characteristic of single-cell RNA-seq data.

## Background

Single-cell RNA sequencing often results in a large number of dropouts (genes with zero expression in particular cells) due to the technical challenges of reverse-transcribing and amplifying small quantities of RNA from individual cells. M3Drop takes advantage of this characteristic by modeling the relationship between dropout rate and mean expression using the Michaelis-Menten equation:

**P_i = 1 - S_i/(S_i + K)**

Where:
- P_i is the proportion of cells where gene i drops out
- S_i is the mean expression of gene i  
- K is the Michaelis constant

## Key Features

### M3Drop Method
- **Michaelis-Menten Modeling**: Models dropout rates using enzyme kinetics principles
- **Feature Selection**: Identifies differentially expressed genes by detecting outliers from the fitted curve
- **Optimized for Smart-seq2**: Works best with full-transcript protocols without UMIs

### DANB (Depth-Adjusted Negative Binomial) Method  
- **UMI Compatibility**: Specifically designed for UMI-tagged data (10X Chromium, etc.)
- **Depth Adjustment**: Accounts for sequencing depth variations across cells
- **Negative Binomial Modeling**: Models count data with appropriate variance structure

### Additional Methods
- **Brennecke Method**: Implementation of highly variable gene detection
- **Consensus Feature Selection**: Combines multiple feature selection approaches
- **Pearson Residuals**: Alternative normalization for UMI data

## Installation

You can install M3DropPy using pip:

```bash
pip install m3drop
```

## Imports

You can import specific functions from different modules:

```python
from m3Drop.basics import your_function_name
from m3Drop.M3D_Imputation import another_function
```

## Usage

```python
from m3Drop.basics import M3DropConvertData, M3DropFeatureSelection
from m3Drop.NB_UMI import NBumiFitModel, NBumiFeatureSelectionCombinedDrop, NBumiPearsonResiduals

# Load your single-cell expression data
# counts should be a genes x cells matrix

# For non-UMI data (Smart-seq2, etc.)
# Convert and normalize data
norm_data = M3DropConvertData(counts, is_counts=True)

# Perform M3Drop feature selection
selected_genes = M3DropFeatureSelection(norm_data, mt_method="fdr", mt_threshold=0.01)

# For UMI data (10X Chromium, etc.)
# Fit DANB model
danb_fit = NBumiFitModel(counts)

# Perform dropout-based feature selection
selected_genes = NBumiFeatureSelectionCombinedDrop(danb_fit, method="fdr", qval_thres=0.01)

# Calculate Pearson residuals for normalization
pearson_residuals = NBumiPearsonResiduals(counts, danb_fit)
```

## When to Use Each Method

- **M3Drop**: Use for Smart-seq2 and other full-transcript protocols without UMIs
- **DANB/NBumi**: Use for UMI-tagged data like 10X Chromium
- **Consensus**: Use when you want to combine multiple feature selection approaches

## Original R Package

This Python implementation is based on the M3Drop R package developed by Tallulah Andrews and converted to Python by Pragalvha Sharma.

## Citation

If you use M3DropPy in your research, please cite the original M3Drop paper:
- Andrews, T.S. and Hemberg, M. (2019). M3Drop: Dropout-based feature selection for scRNASeq. Bioinformatics, 35(16), 2865-2867.
