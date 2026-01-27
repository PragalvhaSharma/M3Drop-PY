import numpy as np
import h5py
import os
import re
import sys

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
        ram_limit_mb: float = None, 
        os_floor: int = 2048,
        mode: str = "auto",
        manual_target: int = 5000,
        **kwargs 
    ):
        """
        CPU-Optimized Resource Governor.
        Manages chunk sizes based on L3 Cache (Speed) and System RAM (Safety).
        """
        self.indptr = indptr
        self.total_rows = total_rows
        self.n_genes = n_genes 
        self.mode = mode.lower()
        self.manual_target = manual_target
        
        # --- AUTO-DETECT HARDWARE (SLURM PRIORITY) ---
        # On CPU, the "Limit" is the System RAM (Host Memory)
        if ram_limit_mb is None:
            self.ram_limit_mb = self._detect_real_memory_limit()
        else:
            self.ram_limit_mb = ram_limit_mb

        if l3_cache_mb is None: 
            self.l3_cache_mb = self._detect_l3_cache()
        else: 
            self.l3_cache_mb = l3_cache_mb

        # --- BUDGETS ---
        # 1. L3 Budget (Speed Target) - 90% of L3
        # Keeping chunks inside L3 prevents cache-thrashing, speeding up numpy ops.
        self.l3_budget_bytes = (self.l3_cache_mb * 1024 * 1024) * 0.90
        
        # 2. RAM Budget (Safety Net) - 85% of Total Available
        # Slightly more conservative on CPU to leave room for OS/Python overhead
        self.ram_budget_bytes = (self.ram_limit_mb * 1024 * 1024) * 0.85
        
        self.os_floor = os_floor
        self.bytes_per_item = 16 # Float64 Sparse estimate

        # --- DIAGNOSTICS PRINT ---
        print(f"\n-------------- CONTROL DEVICE (CPU) --------------")
        if self.mode == "manual":
            print(f"  > Mode:          MANUAL")
        else:
            print(f"  > Mode:          AUTO (L3 Cache Optimized)")
            
        print(f"  > L3 Cache:      {self.l3_budget_bytes / (1024**2):.2f} MB / {self.l3_cache_mb:.2f} MB")
        
        # RAM Reporting
        ram_total_mb = self.ram_limit_mb
        ram_budget_mb = self.ram_budget_bytes / (1024**2)
        print(f"  > RAM Budget:    {ram_budget_mb:.2f} MB / {ram_total_mb:.2f} MB")
        print(f"--------------------------------------------------\n")

    def _detect_real_memory_limit(self) -> float:
        """ 
        Detects the TRUE memory limit, prioritizing SLURM Env Vars. 
        Crucial for HPC environments where psutil sees the whole node, not the job limit.
        """
        limits = []
        # 1. Check SLURM (HPC)
        if 'SLURM_MEM_PER_NODE' in os.environ:
            try: limits.append(float(os.environ['SLURM_MEM_PER_NODE'])) 
            except: pass
        if 'SLURM_MEM_PER_CPU' in os.environ and 'SLURM_CPUS_ON_NODE' in os.environ:
            try: limits.append(float(os.environ['SLURM_MEM_PER_CPU']) * float(os.environ['SLURM_CPUS_ON_NODE']))
            except: pass
            
        # 2. Check Cgroups (Docker/Containers)
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
            
        # 3. Check Physical RAM (Laptop/Desktop)
        if psutil: 
            # Use .total because we calculate budget as % of total.
            # On a shared laptop, you might prefer .available, but .total is safer for consistency.
            limits.append(psutil.virtual_memory().total / (1024**2))
            
        if not limits: return 4096.0 # Default fallback
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
        return 8.0 # Conservative default for Laptops

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
            end_row = start_row + self.manual_target
        else:
            # --- AUTO MODE (L3 Optimized) ---
            # Try to fit the working set into L3 Cache for speed
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

        # ==========================================
        #  STEP 2: RAM SAFETY CHECK (THE GATEKEEPER)
        # ==========================================
        # This prevents OOM (Out of Memory) kills.
        
        chunk_rows = end_row - start_row
        
        if mode == 'dense':
            # Dense cost: Rows * Genes * 8 bytes (float64)
            total_ram_cost = (chunk_rows * self.n_genes * 8) * overhead_multiplier
            if total_ram_cost > self.ram_budget_bytes:
                # Calculate safe max rows
                bytes_per_row = self.n_genes * 8 * overhead_multiplier
                max_ram_rows = int(self.ram_budget_bytes / bytes_per_row)
                
                # Override the tentative end_row
                end_row = start_row + max_ram_rows

        else: # Sparse
            # Sparse cost: NNZ * 16 bytes (val+idx)
            actual_nnz = self.indptr[end_row] - self.indptr[start_row]
            sparse_cost = (actual_nnz * 16) * overhead_multiplier
            if sparse_cost > self.ram_budget_bytes:
                # Calculate safe ratio
                ratio = self.ram_budget_bytes / sparse_cost
                new_count = int(chunk_rows * ratio)
                
                # Override the tentative end_row
                end_row = start_row + max(32, new_count)

        return end_row
