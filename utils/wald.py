
from typing import Optional
import math

def _z_from_conf(conf: float) -> float:
    table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576, 0.999: 3.291}
    return table.get(conf, 1.96)

def fpc_factor(N_pop: Optional[int], n: int) -> float:
    if not N_pop or N_pop <= 0:
        return 1.0
    if n <= 1 or n >= N_pop:
        val = max(0.0, (N_pop - n) / max(1, N_pop - 1))
        return math.sqrt(val)
    return math.sqrt((N_pop - n) / (N_pop - 1))

def halfwidth_normal_fpc(p_hat: float, n: int, conf: float, N_pop: Optional[int]) -> float:
    z = _z_from_conf(conf)
    p = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    hw = z * math.sqrt(p * (1.0 - p) / max(1, n))
    return hw * fpc_factor(N_pop, n)

def _initial_n_infinite(e_goal: float, conf: float, p0: float = 0.5) -> int:
    z = _z_from_conf(conf)
    return int(math.ceil((z*z*p0*(1.0-p0)) / (e_goal*e_goal)))

def plan_n_with_fpc(p_hat: float, e_target: float, conf: float, N_pop: Optional[int]) -> int:
    z = _z_from_conf(conf)
    p = min(max(p_hat, 1e-6), 1.0 - 1e-6)
    if not N_pop or N_pop <= 0:
        n = math.ceil((z * z * p * (1.0 - p)) / (e_target * e_target))
        return max(1, n)
    denom = 1.0 + (e_target * e_target * (N_pop - 1.0)) / (z * z * p * (1.0 - p))
    n = N_pop / denom
    return int(min(N_pop, max(1, math.ceil(n))))

def ep_next_error(e_goal: float, E_hat: float, p_hat: float) -> float:
    if E_hat <= 0.0:
        return e_goal
    thresh = E_hat / 3.0
    if thresh <= e_goal:
        return e_goal
    k = 4.0 * (thresh - e_goal)
    val = -k * (p_hat ** 2) + k * p_hat + e_goal
    return float(min(max(val, e_goal), thresh))

def wald_ci(p_hat: float, n: int, conf: float, N_pop: Optional[int]):
    half = halfwidth_normal_fpc(p_hat, n, conf, N_pop)
    z = _z_from_conf(conf)
    se = half / max(z, 1e-12)
    low = max(0.0, p_hat - half)
    high = min(1.0, p_hat + half)
    return low, high, half, se