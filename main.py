import csv
import os
import argparse
import torch
from datasets import get_dataset, datasets
from networks import get_network
from injector import WeightFaultInjector, WeightFault
import torch.nn as nn
from utils import get_preds
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from itertools import product
import math
from typing import Optional

parser = argparse.ArgumentParser(description="Argparse")

available_datasets = datasets.keys()

parser.add_argument("-data", choices=available_datasets, required=True, help=f"{','.join(available_datasets)}")
parser.add_argument("-root", type=str, help="root datapath", default="../../data/")
parser.add_argument("-net", type=str, help="network", default=None)
parser.add_argument("-weights_path", help="weights_path", default="./networks/weights")
parser.add_argument("-results_path", help="results_path", default="./results/")
# parser.add_argument("-preds_path", help="preds_path", default="./preds/")

parser.add_argument("-pilot", type=int, help="pilot", default = 200)
parser.add_argument("-eps", type=float, help="epsilon", default = 0.005)
parser.add_argument("-conf", type=float, help="confidence", default = 0.95)
parser.add_argument("-p0", type=float, help="epsilon", default = 0.5)
parser.add_argument("-block", type=int, help="pilot", default=50)
parser.add_argument("-budget_cap", type=int, help="budget_cap", default=None)
parser.add_argument("-seed", type=int, help="seed", default=0)

def _get_all_fault_sites(model, as_list=True):
    def _iter():
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                try:
                    w = module.weight
                except Exception:
                    print(f"{name} module does not contain weight")
                    continue
                shape = w.shape
                for idx in product(*[range(s) for s in shape]):
                    for bit in range(32):
                        yield (name, idx, bit)
    return list(_iter()) if as_list else _iter()

def _next_wor_combo(fault_sites, k1_order, k1_ptr):
    """Ritorna un combo unico (WOR) oppure None se popolazione esaurita."""
    if k1_order is None or k1_ptr >= len(k1_order):
        return None
    idx = k1_order[k1_ptr]
    k1_ptr += 1
    return fault_sites[idx], k1_ptr

def srs_combinations(pool, r, seed=None, max_yield=None):
    rnd = random.Random(seed)
    n = len(pool)
    if n == 0:
        return
    r = min(r, n)
    produced = 0
    while max_yield is None or produced < max_yield:
        idxs = rnd.sample(range(n), r)
        yield tuple(pool[i] for i in idxs)
        produced += 1

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

def decide_sampling_policy(num_faults: int, K: int, e_goal: float, conf: float, p0: float = 0.5,
                           exhaustive_cap: Optional[int] = None):
    """
    Ritorna: (N_pop, use_fpc, force_exhaustive, n_inf, n_fpc_or_None, ratio)
    """
    N_pop = math.comb(num_faults, K)
    if N_pop <= 0:
        return 0, False, False, 0, None, 0.0

    n_inf = _initial_n_infinite(e_goal, conf, p0)
    ratio = n_inf / N_pop

    # Regola 5% STRETTAMENTE maggiore
    use_fpc = (ratio > 0.05)
    n_fpc = None
    if use_fpc:
        # usa p0 per il planning iniziale; l'iterativo poi aggiorna con p_hat
        n_fpc = plan_n_with_fpc(p_hat=p0, e_target=e_goal, conf=conf, N_pop=N_pop)

    # forza esaustiva se la n_FPC copre sostanzialmente tutta la popolazione
    force_exhaustive = False
    if use_fpc and n_fpc is not None:
        if n_fpc >= 0.95 * N_pop:
            force_exhaustive = True

    # opzionale: anche un 'tetto' assoluto a N_pop per andare esaustivo
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

def main(args):
    
    # carico dataset e rete
    dataset_name = args.data
    data_root = args.root
    network_name = args.net
    weights_path = args.weights_path
    
    if dataset_name=="cifar10":
        train_loader, val_loader, test_loader = get_dataset(dataset_name, root=data_root, train_batch_size=128, eval_batch_size=128)
    elif dataset_name=="banknote":
        train_loader, val_loader, test_loader = get_dataset(dataset_name)
    elif dataset_name=='mnist':
        train_loader, val_loader, test_loader = get_dataset(dataset_name, root=data_root)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_network(network_name, load_path=weights_path, train_loader=train_loader, test_loader=test_loader)
    model.to(device)
    # calcolo l'accuracy della rete
    # g_acc, golden_preds = get_preds(test_loader, model, verb=True)
    clean_by_batch = []
    with torch.inference_mode():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            clean_by_batch.append(torch.argmax(logits, dim=1).cpu())

    clean_flat = torch.cat(clean_by_batch, dim=0) if len(clean_by_batch) else torch.tensor([], dtype=torch.long)
    golden_preds = clean_flat
    num_classes = int(clean_flat.max().item()) + 1 if clean_flat.numel() > 0 else 2
    baseline_hist = np.bincount(clean_flat.numpy(), minlength=num_classes) if clean_flat.numel() > 0 else np.zeros(2, dtype=int)
    baseline_dist = baseline_hist / max(1, baseline_hist.sum())
    print(f"[BASELINE] pred dist = {baseline_dist.tolist()}")
    
    nome_file = "file.txt"
    os.makedirs(args.results_path, exist_ok=True)
    output_file = os.path.join(args.results_path, nome_file)
    # preds_path = args.preds_path
    # os.makedirs(res_path, exist_ok=True)
    # os.makedirs(preds_path, exist_ok=True)
        
    # Global accumulators
    global_fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
    mism_by_clean_sum = np.zeros(num_classes, dtype=np.int64)
    cnt_by_clean_sum  = np.zeros(num_classes, dtype=np.int64)
    global_cm_cf      = np.zeros((num_classes, num_classes), dtype=np.int64)
    maj_shares, kls   = [], []

    # Defaults
    pilot = args.pilot
    
    eps = args.eps
    e_goal = eps
    
    conf = args.conf
    p0 = args.p0
    exhaustive_cap = None
    
    block = args.block
    budget_cap = args.budget_cap
    
    seed = args.seed
    
    fault_sites = _get_all_fault_sites(model=model, as_list=True)
    num_faults = len(fault_sites)
    K = 1
    BER = K/num_faults
    print(f"BER: {BER}")
    print(f"K: {K}")
    print(f"Tot Bits: {num_faults}")
    
    total_possible, use_fpc_auto, _, n_inf, n_fpc, ratio = \
        decide_sampling_policy(num_faults, K, e_goal, conf, p0, exhaustive_cap)
    
    # sampling policy
    design = "WOR" if use_fpc_auto else "WR"
    print(f"[POLICY] K={K} | N_pop={total_possible} | n_inf={n_inf} | ratio={ratio:.3%} | FPC_auto={use_fpc_auto} | n_fpc0={n_fpc} | DESIGN={design}")

    comb_str = _sci_format_comb(num_faults, K)
    ep_control = "wald"
    print(
        f"[INFO] STATISTICAL (PROPOSED): C({num_faults},{K})≈{comb_str}. "
        f"Launching iterative EP (ctrl={ep_control}){' + FPC' if use_fpc_auto else ''} [{design}]."
    )
    
    use_fpc = use_fpc_auto
    N_pop = total_possible
    
    # --- WOR/WR design ---
    use_wor = bool(use_fpc and N_pop and K >= 1)
    rng = random.Random(seed)
    
    # WOR K=1: permutazione dei singoli fault
    k1_order = None
    k1_ptr = 0
    if use_wor and K == 1:
        k1_order = list(range(len(fault_sites)))
        rng.shuffle(k1_order)

    
    pilot = min(pilot, len(fault_sites))
    
    # Fault Injector
    injector = WeightFaultInjector(model)
    total_samples = len(golden_preds)

    # Generators WR (vecchio comportamento) quando FPC=OFF
    if not use_wor:
        gen = srs_combinations(num_faults, r=K, seed=seed,   max_yield=pilot)
        gen_more = srs_combinations(num_faults, r=K, seed=seed+1, max_yield=None)
    else:
        gen = None
        gen_more = None
    
    # Clamp pilot se WOR e abbiamo N_pop
    if use_wor and N_pop:
        pilot = min(pilot, int(N_pop))
        if K == 1:
            pilot = min(pilot, len(fault_sites))
    
    
    # ------------------ PILOT ------------------
    injected = 0
    mean_frcrit = 0.0
    m2_frcrit   = 0.0
    count_fr    = 0
    sum_fr, n_inj, inj_id = 0.0, 0, 0
    pbar = tqdm(total=pilot, desc=f"[PROP] pilot K={K} (WOR)")
    while injected < pilot:
        combo, k1_ptr = _next_wor_combo(fault_sites, k1_order, k1_ptr)
        if combo is None:
            break  # popolazione esaurita
        inj_id += 1
        ln, ti, bt = combo
        fault = WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bit=bt)
        injector.inject_bit_flip(fault)
        _, faulty_preds = get_preds(test_loader, model, device=device, verb=False)    
        injector.restore_golden()
        
        frcrit = (golden_preds==faulty_preds).sum()/total_samples
        sum_fr += frcrit
        n_inj  += 1
        injected += 1
        pbar.update(1)

        count_fr += 1
        delta = frcrit - mean_frcrit
        mean_frcrit += delta / count_fr
        m2_frcrit += delta * (frcrit - mean_frcrit)
    
    # Stima iniziale
    p_hat = (sum_fr / n_inj) if n_inj else 0.0
    E_wald = halfwidth_normal_fpc(p_hat, n_inj, conf, (N_pop if use_fpc else None))
    E_seq = [(n_inj, E_wald)]
    E_hat = E_wald
    print(f"[PROP] after pilot: p̂={p_hat:.6f}  E_wald={E_wald:.6f}  (ctrl={ep_control})")

    # ------------------ ITERATIVE STEPS ------------------
    steps = 0
    while True:
        steps += 1
        if E_hat <= e_goal:
            break
        if budget_cap and n_inj >= budget_cap:
            break

        # EP controller: prossimo target di errore
        e_next = ep_next_error(e_goal=e_goal, E_hat=E_hat, p_hat=p_hat)

        # Pianifica n totale
        n_tot_desired = plan_n_with_fpc(
            p_hat=p_hat, e_target=e_next, conf=conf,
            N_pop=(N_pop if use_fpc else None)
        )
        
        # Cap FPC: n <= N_pop
        if use_fpc and N_pop:
            n_tot_desired = min(n_tot_desired, int(N_pop))

        add_needed = max(0, n_tot_desired - n_inj)
        if budget_cap:
            add_needed = min(add_needed, max(0, budget_cap - n_inj))
        if add_needed <= 0:
            break

        pbar2 = tqdm(total=add_needed, desc=f"[PROP/{ep_control}] step#{steps} to n={n_tot_desired} (e_next={e_next:.6g})")
        to_do = add_needed
        while to_do > 0:
            if use_wor:
                combo, k1_ptr = _next_wor_combo(fault_sites, k1_order, k1_ptr)
                if combo is None:
                    # popolazione esaurita: non posso crescere oltre
                    to_do = 0
                    break
            else:
                combo = next(gen_more)

            inj_id += 1
            ln, ti, bt = combo
            fault = WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bit=bt)
            
            injector.inject_bit_flip(fault)
            faulty_preds, _ = get_preds(test_loader, model, device=device, verb=False)
            injector.restore_golden()
        
            frcrit = (golden_preds==faulty_preds).sum()/total_samples
            sum_fr += frcrit
            n_inj  += 1
            to_do  -= 1
            pbar2.update(1)

            count_fr += 1
            delta = frcrit - mean_frcrit
            mean_frcrit += delta / count_fr
            m2_frcrit += delta * (frcrit - mean_frcrit)

            if (n_inj % block) == 0:
                p_hat = (sum_fr / n_inj)
                E_wald = halfwidth_normal_fpc(p_hat, n_inj, conf, (N_pop if use_fpc else None))
                E_hat  = E_wald
                E_seq.append((n_inj, E_wald))
                pbar2.set_postfix_str(f"p̂={p_hat:.6f} Ew={E_wald:.6f}")
                if E_hat <= e_goal or (budget_cap and n_inj >= budget_cap):
                    break

        # Update dopo il batch
        p_hat = (sum_fr / n_inj)
        E_wald = halfwidth_normal_fpc(p_hat, n_inj, conf, (N_pop if use_fpc else None))
        E_hat  = E_wald
        E_seq.append((n_inj, E_wald))
        print(f"[PROP/{ep_control}] update: n={n_inj} p̂={p_hat:.6f}  Ew={E_wald:.6f}")

        if E_hat <= e_goal or (budget_cap and n_inj >= budget_cap):
            break

        # Se WOR e popolazione esaurita, stop
        if use_wor and K == 1 and k1_order is not None and k1_ptr >= len(k1_order):
            break
    
    # ------------------ FINAL STATS & SAVE ------------------
    avg_frcrit = (sum_fr / n_inj) if n_inj else 0.0
    half_norm = halfwidth_normal_fpc(avg_frcrit, n_inj, conf, (N_pop if use_fpc else None))
    w_low, w_high, w_half, w_se = wald_ci(avg_frcrit, n_inj, conf, (N_pop if use_fpc else None))
    sample_std = math.sqrt(m2_frcrit / (count_fr - 1)) if count_fr > 1 else 0.0

    total_preds = int(global_fault_hist.sum())
    global_fault_dist = (global_fault_hist / total_preds) if total_preds > 0 else baseline_dist
    tiny = 1e-12
    global_delta_max = float(np.max(np.abs(global_fault_dist - baseline_dist)))
    global_kl = float(np.sum(global_fault_dist * np.log((global_fault_dist + tiny)/(baseline_dist + tiny))))
    global_tv = 0.5 * float(np.sum(np.abs(global_fault_dist - baseline_dist)))
    H = lambda p: float(-np.sum(p * np.log(p + tiny)))
    entropy_baseline = H(baseline_dist)
    entropy_global   = H(global_fault_dist)
    entropy_drop     = entropy_baseline - entropy_global
    ber_per_class = []
    for c in range(num_classes):
        ber_c = (mism_by_clean_sum[c] / max(1, cnt_by_clean_sum[c]))
        ber_per_class.append(float(ber_c))
    BER = float(np.mean(ber_per_class)) if ber_per_class else 0.0
    agree_global = float(np.trace(global_cm_cf)) / max(1, int(global_cm_cf.sum()))
    off_sum = int(global_cm_cf.sum() - np.trace(global_cm_cf))
    diff = np.abs(global_cm_cf - global_cm_cf.T)
    asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum())
    flip_asym_global = float(asym_num) / max(1, off_sum)
    maj_shares_arr = np.array(maj_shares) if maj_shares else np.array([])
    mean_share = float(maj_shares_arr.mean()) if maj_shares_arr.size else 0.0
    p90_share  = float(np.percentile(maj_shares_arr, 90)) if maj_shares_arr.size else 0.0
    frac_collapse_080 = float(np.mean(maj_shares_arr >= 0.80)) if maj_shares_arr.size else 0.0
    mean_kl = float(np.mean(kls)) if kls else 0.0

    design = "WOR" if use_wor else "WR"
    # salvataggio dati
    design = "WOR" if use_wor else "WR"
    with open(output_file, "w") as f:
        f.write(f"[PROP/{ep_control}] K={K}  FRcrit_avg={avg_frcrit:.8f}  conf={conf}  steps={steps}  n={n_inj}\n")
        f.write(f"half_norm={half_norm:.8f}  e_goal={e_goal:.8f}  FPC={'on' if use_fpc else 'off'}  DESIGN={design}  N_pop={N_pop}\n")
        f.write(f"EP_control={ep_control}  E_wald_final={E_wald:.8f}\n")
        f.write(f"Wald_CI({int(conf*100)}%):   [{w_low:.8f}, {w_high:.8f}]   half={w_half:.8f}   se={w_se:.8f}   FPC={'on' if use_fpc else 'off'}\n")
        f.write(f"SampleStdDev_FRcrit_across_injections: s={sample_std:.8f} (n={n_inj})\n")
        f.write(f"injections_used={n_inj}  pilot={pilot}  block={block}  budget_cap={budget_cap}\n")
        f.write(f"baseline_pred_dist={baseline_dist.tolist()}\n")
        f.write(
            "global_summary_over_injections: "
            f"fault_pred_dist={global_fault_dist.tolist()} "
            f"Δmax={global_delta_max:.3f} KL={global_kl:.3f} TV={global_tv:.3f} "
            f"H_baseline={entropy_baseline:.3f} H_global={entropy_global:.3f} ΔH={entropy_drop:.3f} "
            f"BER={BER:.4f} per_class={ber_per_class} "
            f"agree={agree_global:.3f} flip_asym={flip_asym_global:.3f}\n"
        )
        f.write("\n[E_seq] n, E_wald, E_wilson\n")
        for n_i, ew_i in E_seq:
            f.write(f"{n_i},{ew_i:.8f}\n")
        f.write(f"\n[E_final] E_wald={E_seq[-1][1]:.8f}\n\n")

    print(f"[PROP/{ep_control}] K={K}  avgFRcrit={avg_frcrit:.6f}  Ew_final={E_wald:.6f}  n={n_inj}  DESIGN={design} → {output_file}")
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)