# M3Drop/__init__.py

# --- CPU Functions ---

# From CoreCPU.py
from .CoreCPU import (
    ConvertDataSparseCPU,
    hidden_calc_valsCPU,
    NBumiFitModelCPU,
    NBumiFitDispVsMeanCPU,
    NBumiFeatureSelectionHighVarCPU,
    NBumiFeatureSelectionCombinedDropCPU,
    NBumiCombinedDropVolcanoCPU,
)

# From DiagnosticsCPU.py
from .DiagnosticsCPU import (
    NBumiFitBasicModelCPU,
    NBumiCheckFitFSCPU,
    NBumiCompareModelsCPU,
    NBumiPlotDispVsMeanCPU,
)

# From NormalizationCPU.py
from .NormalizationCPU import (
    NBumiPearsonResidualsCombinedCPU,
)

# --- GPU Functions (Placeholders based on your request) ---

# From CoreGPU.py
try:
    from .CoreGPU import (
        ConvertDataSparseGPU,
        hidden_calc_valsGPU,
        NBumiFitModelGPU,
        NBumiFitDispVsMeanGPU,
        NBumiFeatureSelectionHighVarGPU,
        NBumiFeatureSelectionCombinedDropGPU,
        NBumiCombinedDropVolcanoGPU,
    )
except ImportError:
    pass 

# From DiagnosticsGPU.py
try:
    from .DiagnosticsGPU import (
        NBumiFitBasicModelGPU,
        NBumiCheckFitFSGPU,
        NBumiCompareModelsGPU,
        NBumiPlotDispVsMeanGPU,
    )
except ImportError:
    pass

# From NormalizationGPU.py
try:
    from .NormalizationGPU import (
        NBumiPearsonResidualsCombinedGPU,
    )
except ImportError:
    pass
    
# --- Public API ---
__all__ = [
    # --- CPU ---
    'ConvertDataSparseCPU',
    'hidden_calc_valsCPU',
    'NBumiFitModelCPU',
    'NBumiFitDispVsMeanCPU',
    'NBumiFeatureSelectionHighVarCPU',
    'NBumiFeatureSelectionCombinedDropCPU',
    'NBumiCombinedDropVolcanoCPU',
    
    'NBumiFitBasicModelCPU',
    'NBumiCheckFitFSCPU',
    'NBumiCompareModelsCPU',
    'NBumiPlotDispVsMeanCPU',
    
    'NBumiPearsonResidualsCombinedCPU',

    # --- GPU ---
    'ConvertDataSparseGPU',
    'hidden_calc_valsGPU',
    'NBumiFitModelGPU',
    'NBumiFitDispVsMeanGPU',
    'NBumiFeatureSelectionHighVarGPU',
    'NBumiFeatureSelectionCombinedDropGPU',
    'NBumiCombinedDropVolcanoGPU',
    
    'NBumiFitBasicModelGPU',
    'NBumiCheckFitFSGPU',
    'NBumiCompareModelsGPU',
    'NBumiPlotDispVsMeanGPU',
    
    'NBumiPearsonResidualsCombinedGPU',
]
