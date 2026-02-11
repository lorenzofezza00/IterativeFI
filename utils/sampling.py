
import torch
from itertools import product, islice
from collections.abc import Sequence
import random
import math
from typing import Optional
from .wald import _initial_n_infinite, plan_n_with_fpc
import numpy as np
import bisect

def _unpack_half(h):
    if isinstance(h, (tuple, list)) and h:
        return float(h[0])
    if isinstance(h, (int, float, np.floating)):
        return float(h)
    return float("nan")

class FaultIndexer:
    def __init__(self, model, bit_depth=32):
        self.bit_depth = bit_depth
        self.layers = []
        self.cumulative = []

        total = 0

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                w = module.weight
                num_weights = w.numel()
                faults = num_weights * bit_depth

                self.layers.append({
                    "name": name,
                    "shape": w.shape,
                    "num_weights": num_weights,
                    "start": total,
                    "end": total + faults
                })

                total += faults
                self.cumulative.append(total)

        self.total_faults = total

    def __len__(self):
        return self.total_faults

    def __getitem__(self, index):
        if index >= self.total_faults:
            raise IndexError

        layer_idx = bisect.bisect_right(self.cumulative, index)
        layer = self.layers[layer_idx]

        local_index = index - layer["start"]

        weight_index = local_index // self.bit_depth
        bit = local_index % self.bit_depth

        idx = np.unravel_index(weight_index, layer["shape"])

        return (layer["name"], idx, bit)

def _next_wor_fault(fault_sites, k1_order, k1_ptr):
    """Returns a fault site (WOR) or None if the entire population is processed."""
    if k1_order is None or k1_ptr >= len(k1_order):
        return None, k1_ptr
    idx = k1_order[k1_ptr]
    k1_ptr += 1
    return fault_sites[idx], k1_ptr

def simple_random_sampling(pool, total, seed=None, max_yield=None):
    rnd = random.Random(seed)
    n = total
    if n == 0:
        return
    produced = 0
    while max_yield is None or produced < max_yield:
        i = rnd.randrange(n)
        yield pool[i]
        produced += 1
        
def decide_sampling_policy(num_faults: int, K: int, e_goal: float, conf: float, p0: float = 0.5,
                           exhaustive_cap: Optional[int] = None):
    """
    Returns: (N_pop, use_fpc, force_exhaustive, n_inf, n_fpc_or_None, ratio)
    """
    N_pop = math.comb(num_faults, K)
    if N_pop <= 0:
        return 0, False, False, 0, None, 0.0

    n_inf = _initial_n_infinite(e_goal, conf, p0)
    ratio = n_inf / N_pop

    # Use finite population correction if ratio > 5%
    use_fpc = (ratio > 0.05)
    n_fpc = None
    if use_fpc:
        # Use p0 for initial planning; iterative updates use p_hat
        n_fpc = plan_n_with_fpc(p_hat=p0, e_target=e_goal, conf=conf, N_pop=N_pop)
        
    # Force exhaustive calculation when n_FPC effectively covers the whole population
    force_exhaustive = False
    if use_fpc and n_fpc is not None:
        if n_fpc >= 0.95 * N_pop:
            force_exhaustive = True
            
    # Optional: also set an absolute N_pop limit to force exhaustive calculation
    if exhaustive_cap is not None and N_pop <= exhaustive_cap:
        force_exhaustive = True

    return N_pop, use_fpc, force_exhaustive, n_inf, n_fpc, ratio

def _sci_format_comb(n: int, k: int) -> str:
    if k < 0 or k > n:
        return "0"
    k = min(k, n - k)
    if k == 0:
        return "1"
    ln10 = math.log(10.0)
    log10_val = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / ln10
    exp = int(math.floor(log10_val))
    mant = 10 ** (log10_val - exp)
    return f"{mant:.3f}e+{exp:d}"