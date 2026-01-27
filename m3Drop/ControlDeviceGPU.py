import numpy as np
import h5py
import os
import subprocess
import re
import sys

# Try to import CuPy for robust VRAM detection
try:
    import cupy
    HAS_CUPY = True
except ImportError:
    cupy = None
    HAS_CUPY = False

# Try to import cpuinfo for L3 cache detection
try:
    import cpuinfo
except ImportError:
    cpuinfo = None

# Try to import psutil for System RAM detection
try:
    import psutil
except ImportError:
    psutil = None

class ControlDevice:
    def __init__(
        self, 
        indptr: np.ndarray, 
        total_rows: int,
        n_genes: int,
        l3_cache_mb: float = None, 
        vram_limit_mb: float = None, 
        os_floor: int = 2048,
        mode: str = "auto",
        manual_target: int = 5000,
        **kwargs 
    ):
        self.indptr = indptr
        self.total_rows = total_rows
        self.n_genes = n_genes 
        self.mode = mode.lower()
        self.manual_target = manual_target
        
        # --- AUTO-DETECT HARDWARE (SLURM PRIORITY) ---
        self.sys_ram_limit_mb = self._detect_real_memory_limit()

        if l3_cache_mb is None: 
            self.l3_cache_mb = self._detect_l3_cache()
        else: 
            self.l3_cache_mb = l3_cache_mb

        if vram_limit_mb is None: 
            self.vram_limit_mb = self._detect_vram()
        else: 
            self.vram_limit_mb = vram_limit_mb

        # --- BUDGETS ---
        # 1. L3 Budget (Default/Baseline) - 90% of L3
        self.l3_budget_bytes = (self.l3_cache_mb * 1024 * 1024) * 0.90
        
        # 2. VRAM Budget (Safety Net) - 95% of Total
        self.vram_budget_bytes = (self.vram_limit_mb * 1024 * 1024) * 0.95
        
        self.os_floor = os_floor
        self.bytes_per_item = 16 # Float64 Sparse estimate

        # --- DIAGNOSTICS PRINT ---
        print(f"\n-------------- CONTROL DEVICE --------------")
        
        # UI UPDATE: Clean "Plane Cockpit" Feel
        if self.mode == "manual":
            print(f"  > Mode:          MANUAL")
        else:
            print(f"  > Mode:          AUTO (L3 Optimized)")
            
        print(f"  > L3 Cache:      {self.l3_budget_bytes / (1024**2):.2f} MB / {self.l3_cache_mb:.2f} MB")

        # VRAM Reporting
        vram_mb = self.vram_limit_mb
        vram_budget_mb = self.vram_budget_bytes / (1024**2)
        print(f"  > VRAM Budget:   {vram_budget_mb:.2f} MB / {vram_mb:.2f} MB")
        print(f"--------------------------------------------\n")

    def _detect_real_memory_limit(self) -> float:
        """ Detects the TRUE memory limit, prioritizing SLURM Env Vars. """
        limits = []
        if 'SLURM_MEM_PER_NODE' in os.environ:
            try: limits.append(float(os.environ['SLURM_MEM_PER_NODE'])) 
            except: pass
        if 'SLURM_MEM_PER_CPU' in os.environ and 'SLURM_CPUS_ON_NODE' in os.environ:
            try: limits.append(float(os.environ['SLURM_MEM_PER_CPU']) * float(os.environ['SLURM_CPUS_ON_NODE']))
            except: pass
        if os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
            try:
                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                    val = float(f.read().strip())
                    if val < 1e15: limits.append(val / (1024**2))
            except: pass
        if os.path.exists('/sys/fs/cgroup/memory.max'):
            try:
                with open('/sys/fs/cgroup/memory.max', 'r') as f:
                    val_str = f.read().strip()
                    if val_str != "max":
                         val = float(val_str)
                         if val < 1e15: limits.append(val / (1024**2))
            except: pass
        if psutil: limits.append(psutil.virtual_memory().total / (1024**2))
        if not limits: return 4096.0 
        return min(limits)

    def _detect_l3_cache(self) -> float:
        try:
            if cpuinfo:
                info = cpuinfo.get_cpu_info()
                if 'l3_cache_size' in info:
                    value = info['l3_cache_size']
                    if isinstance(value, int): return value / (1024 * 1024)
                    elif isinstance(value, str):
                        digits = float(re.findall(r"[\d\.]+", value)[0])
                        if "KB" in value.upper(): return digits / 1024
                        elif "MB" in value.upper(): return digits
        except Exception: pass
        return 16.0 

    def _detect_vram(self) -> float:
        if HAS_CUPY:
            try:
                mempool = cupy.get_default_memory_pool()
                mempool.free_all_blocks()
                return float(cupy.cuda.Device(0).mem_info[1]) / (1024 * 1024)
            except Exception: pass
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], 
                encoding="utf-8"
            )
            return float(result.strip().split('\n')[0])
        except Exception: return 4000.0

    @classmethod
    def from_h5ad(cls, filepath: str, mode: str = "auto", manual_target: int = 5000, **kwargs):
        if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
        with h5py.File(filepath, "r") as f:
            if isinstance(f['X'], h5py.Group) and 'indptr' in f['X']:
                indptr_loaded = f['X']['indptr'][:]
                if 'shape' in f['X'].attrs:
                    shape = f['X'].attrs['shape']
                    total_rows, n_genes = shape[0], shape[1]
                else:
                    total_rows = len(indptr_loaded) - 1
                    n_genes = len(f['var']) if 'var' in f else 1
                return cls(
                    indptr=indptr_loaded, 
                    total_rows=total_rows, 
                    n_genes=n_genes, 
                    mode=mode, 
                    manual_target=manual_target, 
                    **kwargs
                )
            else: raise ValueError("ControlDevice requires SPARSE (CSR) data.")

    def get_next_chunk(self, start_row: int, mode: str = 'sparse', overhead_multiplier: float = 1.0) -> int:
        if start_row >= self.total_rows: return None
        
        overhead_multiplier = max(overhead_multiplier, 1.0)

        # ==========================================
        #  STEP 1: DETERMINE TENTATIVE END ROW
        # ==========================================
        
        if self.mode == "manual" and self.manual_target > 0:
            # --- MANUAL MODE ---
            # Set tentative end based on user input
            end_row = start_row + self.manual_target
        else:
            # --- AUTO MODE (L3 Optimized) ---
            limit_bytes = self.l3_budget_bytes
            max_items_capacity = int(limit_bytes / self.bytes_per_item)
            current_ptr = self.indptr[start_row]
            target_ptr = current_ptr + max_items_capacity
            
            # Binary search for the row containing the target pointer
            soft_limit_row = np.searchsorted(self.indptr, target_ptr, side='right') - 1
            if soft_limit_row <= start_row: soft_limit_row = start_row + 1
            end_row = soft_limit_row

        # --- HARD CONSTRAINTS (Bounds Check) ---
        if (end_row - start_row) < self.os_floor:
            end_row = min(start_row + self.os_floor, self.total_rows)
             
        if end_row > self.total_rows:
            end_row = self.total_rows

        # Align to Warp 32
        count = end_row - start_row
        if count > 32:
            aligned_count = (count // 32) * 32
            end_row = start_row + aligned_count

        # ==========================================
        #  STEP 2: VRAM SAFETY CHECK (THE GATEKEEPER)
        # ==========================================
        # This logic is non-negotiable. It scales down ANY request (Manual or Auto)
        # if it exceeds the VRAM budget.
        
        chunk_rows = end_row - start_row
        
        if mode == 'dense':
            total_vram_cost = (chunk_rows * self.n_genes * 8) * overhead_multiplier
            if total_vram_cost > self.vram_budget_bytes:
                # Calculate safe max rows
                bytes_per_row = self.n_genes * 8 * overhead_multiplier
                max_vram_rows = int(self.vram_budget_bytes / bytes_per_row)
                
                # Override the tentative end_row
                end_row = start_row + max_vram_rows
                
                # Re-align
                if (end_row - start_row) > 32:
                    end_row = start_row + ((end_row - start_row) // 32 * 32)

        else: # Sparse
            actual_nnz = self.indptr[end_row] - self.indptr[start_row]
            sparse_cost = (actual_nnz * 16) * overhead_multiplier
            if sparse_cost > self.vram_budget_bytes:
                # Calculate safe ratio
                ratio = self.vram_budget_bytes / sparse_cost
                new_count = int(chunk_rows * ratio)
                
                # Override the tentative end_row
                end_row = start_row + max(32, new_count)

        return end_row
