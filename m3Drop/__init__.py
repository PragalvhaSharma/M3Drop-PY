# Core data conversion and preprocessing
from .basics import (
    M3DropConvertData,
    M3DropGetMarkers,
    bg__calc_variables,
    bg__horizontal_residuals_MM_log10
)

# Imputation
from .M3D_Imputation import (
    M3DropImputation,
)

# Data normalization and cleaning
from .Normalization import (
    M3DropCleanData,
    NBumiPearsonResiduals,
    NBumiPearsonResidualsApprox,
)

# Feature selection and extreme gene identification
from .Extremes import (
    M3DropFeatureSelection,
    M3DropGetExtremes,
    M3DropTestShift,
)

# Negative binomial UMI modeling
from .NB_UMI import (
    NBumiFitModel,
    NBumiFitBasicModel,
    NBumiCheckFit,
    NBumiFitDispVsMean,
    NBumiConvertToInteger,
    NBumiConvertData,
    NBumiCheckFitFS,
    NBumiFeatureSelectionHighVar,
    NBumiFeatureSelectionCombinedDrop,
    NBumiCompareModels,
    NBumiImputeNorm,
    hidden_calc_vals,
)

# Curve fitting and dropout modeling
from .Curve_fitting import (
    M3DropDropoutModels,
    bg__fit_MM,
    bg__fit_logistic,
    bg__fit_ZIFA,
)

# Plotting and visualization
from .Plotting_fxns import (
    M3DropExpressionHeatmap,
    M3DropGetHeatmapClusters,
    M3DropGetHeatmapNames,
    bg__dropout_plot_base,
    bg__add_model_to_plot,
    bg__highlight_genes,
    bg__expression_heatmap,
)

# Alternative feature selection methods
from .Other_FS_functions import (
    Consensus_fs,
    gini_fs,
    cor_fs,
    irlba_pca_fs,
)

# Highly variable genes and co-expression analysis
from .DANB_HVG import (
    NBumiHVG,
)

from .DANB_Coexpression import (
    NBumiCoexpression,
)

# Brennecke method for HVG identification
from .Brennecke_implementation import (
    BrenneckeGetVariableGenes,
)

# Venn diagram plotting
from .Threeway_ProportionalArea_VennDiagrams import (
    m3drop_three_set_venn,
)

# Simulation functions
from .Simulations_Functions import (
    NBumiSimulationTrifecta,
    M3DropSimulationTrifecta,
    Make_Sim,
    bg__default_mean2disp,
    bg__MakeSimData,
    bg__MakeSimDE,
    bg__MakeSimDVar,
    bg__MakeSimHVar,
)

# Import scanpy integration module
from .scanpy import (
    nbumi_normalize,
    m3drop_highly_variable_genes,
)

# All public functions and classes
__all__ = [
    # Core functions
    'M3DropConvertData',
    'M3DropGetMarkers',
    'bg__calc_variables',
    'bg__horizontal_residuals_MM_log10',

    # Imputation
    'M3DropImputation',

    # Normalization and cleaning
    'M3DropCleanData',
    'NBumiPearsonResiduals',
    'NBumiPearsonResidualsApprox',

    # Feature selection
    'M3DropFeatureSelection',
    'M3DropGetExtremes',
    'M3DropTestShift',

    # NB-UMI modeling
    'NBumiFitModel',
    'NBumiFitBasicModel',
    'NBumiCheckFit',
    'NBumiFitDispVsMean',
    'NBumiConvertToInteger',
    'NBumiConvertData',
    'NBumiCheckFitFS',
    'NBumiFeatureSelectionHighVar',
    'NBumiFeatureSelectionCombinedDrop',
    'NBumiCompareModels',
    'NBumiImputeNorm',
    'hidden_calc_vals',

    # Curve fitting
    'M3DropDropoutModels',
    'bg__fit_MM',
    'bg__fit_logistic',
    'bg__fit_ZIFA',

    # Plotting
    'M3DropExpressionHeatmap',
    'M3DropGetHeatmapClusters',
    'M3DropGetHeatmapNames',
    'bg__dropout_plot_base',
    'bg__add_model_to_plot',
    'bg__highlight_genes',
    'bg__expression_heatmap',

    # Alternative feature selection
    'Consensus_fs',
    'gini_fs',
    'cor_fs',
    'irlba_pca_fs',

    # HVG and co-expression
    'NBumiHVG',
    'NBumiCoexpression',
    'BrenneckeGetVariableGenes',

    # Visualization
    'm3drop_three_set_venn',

    # Simulations
    'NBumiSimulationTrifecta',
    'M3DropSimulationTrifecta',
    'Make_Sim',
    'bg__default_mean2disp',
    'bg__MakeSimData',
    'bg__MakeSimDE',
    'bg__MakeSimDVar',
    'bg__MakeSimHVar',
    
    # Scanpy integration
    'nbumi_normalize',
    'm3drop_highly_variable_genes',
]