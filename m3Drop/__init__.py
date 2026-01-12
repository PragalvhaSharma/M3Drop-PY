# M3Drop (Refactored) __init__.py
# This file imports all CPU and GPU functions to make them
# directly accessible from the main package.

# --- CPU Functions ---

# From coreCPU.py
from .coreCPU import (
    ConvertDataSparseCPU,
    hidden_calc_valsCPU,
    NBumiFitModelCPU,
    NBumiFitDispVsMeanCPU,
    NBumiFeatureSelectionHighVarCPU,
    NBumiFeatureSelectionCombinedDropCPU,
    NBumiCombinedDropVolcanoCPU,
)

# From diagnosticsCPU.py
from .diagnosticsCPU import (
    NBumiFitBasicModelCPU,
    NBumiCheckFitFSCPU,
    NBumiCompareModelsCPU,
    NBumiPlotDispVsMeanCPU,
)

# From normalizationCPU.py
from .normalizationCPU import (
    NBumiPearsonResidualsCPU,
    NBumiPearsonResidualsApproxCPU,
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
    get_io_chunk_size,      # <--- NEW: For Safe Disk I/O
    get_compute_tile_size,  # <--- NEW: For Safe GPU Memory
    get_optimal_chunk_size  # Kept for backward compatibility
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
    'get_io_chunk_size',
    'get_compute_tile_size',
    'get_optimal_chunk_size',
    
    # diagnosticsGPU
    'NBumiFitBasicModelGPU',
    'NBumiCheckFitFSGPU',
    'NBumiCompareModelsGPU',
    'NBumiPlotDispVsMeanGPU',
    
    # normalizationGPU
    'NBumiPearsonResidualsGPU',
    'NBumiPearsonResidualsApproxGPU',
]
