# M3Drop (Refactored) __init__.py
# This file imports all CPU and GPU functions to make them
# directly accessible from the main package.

# --- CPU Functions ---

# From coreCPU.py
from .coreGPU import (
    ConvertDataSparseGPU,
    hidden_calc_valsGPU,
    NBumiFitModelGPU,
    NBumiFitDispVsMeanGPU,
    NBumiFeatureSelectionHighVarGPU,
    NBumiFeatureSelectionCombinedDropGPU,
    NBumiCombinedDropVolcanoGPU,
    get_optimal_chunk_size
)

from .diagnosticsGPU import (
    NBumiFitBasicModelGPU,
    NBumiCheckFitFSGPU,
    NBumiCompareModelsGPU,
    NBumiPlotDispVsMeanGPU
)

from .normalizationGPU import (
    NBumiPearsonResidualsGPU,
    NBumiPearsonResidualsApproxGPU
)


# --- GPU Functions ---

# From coreGPU.py
from .coreGPU import (
    ConvertDataSparseGPU,
    hidden_calc_valsGPU,
    NBumiFitModelGPU,
    NBumiFitDispVsMeanGPU,
    NBumiFeatureSelectionHighVarGPU,
    NBumiFeatureSelectionCombinedDropGPU,
    NBumiCombinedDropVolcanoGPU,
)

# From diagnosticsGPU.py
from .diagnosticsGPU import (
    NBumiFitBasicModelGPU,
    NBumiCheckFitFSGPU,
    NBumiCompareModelsGPU,
    NBumiPlotDispVsMeanGPU,
)

# From normalizationGPU.py
from .normalizationGPU import (
    NBumiPearsonResidualsGPU,
    NBumiPearsonResidualsApproxGPU,
)


# --- Public API (`__all__`) ---
# Defines what `from m3Drop import *` will import.

__all__ = [
    # --- CPU ---
    # coreCPU
    'ConvertDataSparseCPU',
    'hidden_calc_valsCPU',
    'NBumiFitModelCPU',
    'NBumiFitDispVsMeanCPU',
    'NBumiFeatureSelectionHighVarCPU',
    'NBumiFeatureSelectionCombinedDropCPU',
    'NBumiCombinedDropVolcanoCPU',
    
    # diagnosticsCPU
    'NBumiFitBasicModelCPU',
    'NBumiCheckFitFSCPU',
    'NBumiCompareModelsCPU',
    'NBumiPlotDispVsMeanCPU',
    
    # normalizationCPU
    'NBumiPearsonResidualsCPU',
    'NBumiPearsonResidualsApproxCPU',

    # --- GPU ---
    # coreGPU
    'ConvertDataSparseGPU',
    'hidden_calc_valsGPU',
    'NBumiFitModelGPU',
    'NBumiFitDispVsMeanGPU',
    'NBumiFeatureSelectionHighVarGPU',
    'NBumiFeatureSelectionCombinedDropGPU',
    'NBumiCombinedDropVolcanoGPU',
    
    # diagnosticsGPU
    'NBumiFitBasicModelGPU',
    'NBumiCheckFitFSGPU',
    'NBumiCompareModelsGPU',
    'NBumiPlotDispVsMeanGPU',
    
    # normalizationGPU
    'NBumiPearsonResidualsGPU',
    'NBumiPearsonResidualsApproxGPU',
]
