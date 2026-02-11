import csv
import os
import argparse
import torch
from datasets import get_dataloaders, datasets
from networks import get_network
from injector import WeightFaultInjector, WeightFault, _evaluate_faulty_model
from utils import _z_from_conf, fpc_factor, halfwidth_normal_fpc, \
    _initial_n_infinite, plan_n_with_fpc, ep_next_error, wald_ci, \
        _next_wor_fault, simple_random_sampling, decide_sampling_policy, \
            _sci_format_comb, FaultIndexer, _unpack_half
import numpy as np
from tqdm import tqdm
import random
import math
import heapq
import time

parser = argparse.ArgumentParser(description="Argparse")

available_datasets = datasets.keys()

parser.add_argument("-data", choices=available_datasets, required=True, help=f"{','.join(available_datasets)}")
parser.add_argument("-root", type=str, help="root datapath", default="../../data/")
parser.add_argument("-net", type=str, help="network", default=None)
parser.add_argument("-weights_path", help="weights_path", default="./networks/weights")
parser.add_argument("-results_path", help="results_path", default="./results/")
parser.add_argument("-pilot", type=int, help="pilot", default = 200)
parser.add_argument("-eps", type=float, help="epsilon", default = 0.005)
parser.add_argument("-conf", type=float, help="confidence", default = 0.95)
parser.add_argument("-p0", type=float, help="epsilon", default = 0.5)
parser.add_argument("-block", type=int, help="pilot", default=50)
parser.add_argument("-budget_cap", type=int, help="budget_cap", default=None)
parser.add_argument("-seed", type=int, help="seed", default=0)

def main(args):
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # dataset and network loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_name = args.data
    data_root = args.root
    network_name = args.net
    weights_path = args.weights_path
    
    if dataset_name=="cifar10":
        train_loader, val_loader, test_loader = get_dataloaders(dataset_name, root=data_root, train_batch_size=128, eval_batch_size=128)
    elif dataset_name=="banknote":
        train_loader, val_loader, test_loader = get_dataloaders(dataset_name)
    elif dataset_name=='mnist':
        train_loader, val_loader, test_loader = get_dataloaders(dataset_name, root=data_root)
    
    model = get_network(network_name, load_path=weights_path, train_loader=train_loader, test_loader=test_loader)
    model.to(device)
    model.eval()
    
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
    
    print(f"[BASELINE] predictions disttribution = {baseline_dist.tolist()}")
    
    # - Build the fault list
    # - Automatically determine FPC based on n_inf / N_pop (> 5% rule)
    # - Run the iterative campaign (+FPC if enabled) and saves results
    
    t0 = time.time()
    
    nome_file = f"net_{network_name}_data_{dataset_name}_iterative_fi.txt"
    os.makedirs(args.results_path, exist_ok=True)
    output_file = os.path.join(args.results_path, nome_file)    
    
    indexer = FaultIndexer(model)
    fault_sites = indexer
    num_faults = len(indexer)
    if num_faults == 0:
        raise RuntimeError("Empty fault list: cannot proceed.")
    
    # Defaults
    # --- Automatic Policy (> 5% rule) ---
    pilot = args.pilot
    
    eps = args.eps
    e_goal = eps
    
    conf = args.conf
    p0 = args.p0
    exhaustive_cap = None
    
    block = args.block
    budget_cap = args.budget_cap
    
    K = 1
    BER = K/num_faults
    print(f"BER: {BER}")
    print(f"K: {K}")
    print(f"Tot Bits: {num_faults}")
    
    total_possible, use_fpc_auto, _, n_inf, n_fpc, ratio = \
        decide_sampling_policy(num_faults, K, e_goal, conf, p0, exhaustive_cap)
    
    design = "WOR" if use_fpc_auto else "WR"
    print(f"[POLICY] K={K} | N_pop={total_possible} | n_inf={n_inf} | ratio={ratio:.3%} | FPC_auto={use_fpc_auto} | n_fpc0={n_fpc} | DESIGN={design}")

    comb_str = _sci_format_comb(num_faults, K)
    ep_control = "wald"
    print(
        f"[INFO] STATISTICAL (PROPOSED): C({num_faults},{K})≈{comb_str}. "
        f"Launching iterative EP (ctrl={ep_control}){' + FPC' if use_fpc_auto else ''} [{design}]."
    )
    
    # Injector & basic checks
    injector = WeightFaultInjector(model)
    total_samples = len(golden_preds)
    if total_samples == 0:
        raise RuntimeError("Empty test loader.")

    # Global accumulators
    global_fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
    mism_by_clean_sum = np.zeros(num_classes, dtype=np.int64)
    cnt_by_clean_sum  = np.zeros(num_classes, dtype=np.int64)
    global_cm_cf      = np.zeros((num_classes, num_classes), dtype=np.int64)
    maj_shares, kls   = [], []

    mean_frcrit = 0.0
    m2_frcrit   = 0.0
    count_fr    = 0
    top_heap    = []  # min-heap for top-100 worst injections
    
    use_fpc = use_fpc_auto
    N_pop = total_possible
    
    # --- WOR/WR design ---
    use_wor = bool(use_fpc and N_pop and K >= 1)
    rng = random.Random(seed)
    
    # WOR K=1: single bit faults permutations
    k1_order = None
    k1_ptr = 0

    if use_wor:
        k1_order = list(range(num_faults))
        rng.shuffle(k1_order)
    
    # Generators WR when FPC=OFF
    if not use_wor:
        gen = simple_random_sampling(fault_sites, num_faults, seed=seed,   max_yield=pilot)
        gen_more = simple_random_sampling(fault_sites, num_faults, seed=seed+1, max_yield=None)
    else:
        gen = None
        gen_more = None
        
    # Clamp pilot if using WOR and N_pop is defined
    if use_wor and N_pop:
        pilot = min(pilot, len(fault_sites))
    
    # ------------------ PILOT ------------------
    sum_fr, n_inj, inj_id = 0.0, 0, 0
    if use_wor:
        pbar = tqdm(total=pilot, desc=f"[PROP] pilot K={K} (WOR)")
        injected = 0
        while injected < pilot:
            fault_site, k1_ptr = _next_wor_fault(fault_sites, k1_order, k1_ptr)
            if fault_site is None:
                break  # Entire population has been processed
            inj_id += 1
            frcrit, fault, bias, fh, mbc, cbc, cm = _evaluate_faulty_model(
                model, device, test_loader, clean_by_batch, injector, fault_site, inj_id, total_samples,
                baseline_hist, baseline_dist, num_classes
            )
            sum_fr += frcrit
            n_inj  += 1
            injected += 1
            pbar.update(1)

            count_fr += 1
            delta = frcrit - mean_frcrit
            mean_frcrit += delta / count_fr
            m2_frcrit += delta * (frcrit - mean_frcrit)

            global_fault_hist += fh
            mism_by_clean_sum += mbc
            cnt_by_clean_sum  += cbc
            global_cm_cf      += cm
            maj_shares.append(bias["maj_share"])
            kls.append(bias["kl"])

            if len(top_heap) < 100:
                heapq.heappush(top_heap, (frcrit, inj_id, fault, bias))
            elif frcrit > top_heap[0][0]:
                heapq.heapreplace(top_heap, (frcrit, inj_id, fault, bias))

    else:
        pbar = tqdm(gen, total=pilot, desc=f"[PROP] pilot K={K} (WR)")
        for fault_site in pbar:
            inj_id += 1
            frcrit, fault, bias, fh, mbc, cbc, cm = _evaluate_faulty_model(
                model, device, test_loader, clean_by_batch, injector, fault_site, inj_id, total_samples,
                baseline_hist, baseline_dist, num_classes
            )
            sum_fr += frcrit
            n_inj  += 1

            count_fr += 1
            delta = frcrit - mean_frcrit
            mean_frcrit += delta / count_fr
            m2_frcrit += delta * (frcrit - mean_frcrit)

            global_fault_hist += fh
            mism_by_clean_sum += mbc
            cnt_by_clean_sum  += cbc
            global_cm_cf      += cm
            maj_shares.append(bias["maj_share"])
            kls.append(bias["kl"])

            if len(top_heap) < 100:
                heapq.heappush(top_heap, (frcrit, inj_id, fault, bias))
            elif frcrit > top_heap[0][0]:
                heapq.heapreplace(top_heap, (frcrit, inj_id, fault, bias))
    
    # Initial estimation
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

        # EP controller: error target
        e_next = ep_next_error(e_goal=e_goal, E_hat=E_hat, p_hat=p_hat)

        # Plan total sample size
        n_tot_desired = plan_n_with_fpc(
            p_hat=p_hat, e_target=e_next, conf=conf,
            N_pop=(N_pop if use_fpc else None)
        )
        
        # Limit FPC: n must not exceed N_pop
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
                fault_site, k1_ptr = _next_wor_fault(fault_sites, k1_order, k1_ptr)
                if fault_site is None:
                    # Entire population has been processed
                    to_do = 0
                    break
            else:
                fault_site = next(gen_more)

            inj_id += 1
            frcrit, fault, bias, fh, mbc, cbc, cm = _evaluate_faulty_model(
                model, device, test_loader, clean_by_batch, injector, fault_site, inj_id, total_samples,
                baseline_hist, baseline_dist, num_classes
            )
            sum_fr += frcrit
            n_inj  += 1
            to_do  -= 1
            pbar2.update(1)

            count_fr += 1
            delta = frcrit - mean_frcrit
            mean_frcrit += delta / count_fr
            m2_frcrit += delta * (frcrit - mean_frcrit)

            global_fault_hist += fh
            mism_by_clean_sum += mbc
            cnt_by_clean_sum  += cbc
            global_cm_cf      += cm
            maj_shares.append(bias["maj_share"])
            kls.append(bias["kl"])

            if len(top_heap) < 100:
                heapq.heappush(top_heap, (frcrit, inj_id, fault, bias))
            elif frcrit > top_heap[0][0]:
                heapq.heapreplace(top_heap, (frcrit, inj_id, fault, bias))

            if (n_inj % block) == 0:
                p_hat = (sum_fr / n_inj)
                E_wald = halfwidth_normal_fpc(p_hat, n_inj, conf, (N_pop if use_fpc else None))
                E_hat  = E_wald
                E_seq.append((n_inj, E_wald))
                pbar2.set_postfix_str(f"p̂={p_hat:.6f} Ew={E_wald:.6f}")
                if E_hat <= e_goal or (budget_cap and n_inj >= budget_cap):
                    break

        # Update estimates
        p_hat = (sum_fr / n_inj)
        E_wald = halfwidth_normal_fpc(p_hat, n_inj, conf, (N_pop if use_fpc else None))
        E_hat  = E_wald
        E_seq.append((n_inj, E_wald))
        print(f"[PROP/{ep_control}] update: n={n_inj} p̂={p_hat:.6f}  Ew={E_wald:.6f}")

        if E_hat <= e_goal or (budget_cap and n_inj >= budget_cap):
            break

        # if WOR and entire population processed, stop
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

    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
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
            f"fault_pred_dist = {global_fault_dist.tolist()} "
            f"Δmax = {global_delta_max:.3f} KL = {global_kl:.3f} TV = {global_tv:.3f} "
            f"H_baseline = {entropy_baseline:.3f} H_global = {entropy_global:.3f} ΔH = {entropy_drop:.3f} "
            f"BER = {BER:.4f} per_class = {ber_per_class} "
            f"agree = {agree_global:.3f} flip_asym = {flip_asym_global:.3f}\n"
        )
        f.write("\n[E_seq] n, E_wald\n")
        for n_i, ew_i in E_seq:
            f.write(f"{n_i},{ew_i:.8f}\n")
        f.write(f"\n[E_final] E_wald={E_seq[-1][1]:.8f}\n\n")

        f.write(f"Top-{min(100, len(top_sorted))} worst injections (proposed iter EP)\n")
        for rank, (frcrit, inj, fault, bias) in enumerate(top_sorted, 1):
            desc = f"{fault.layer_name} {[int(tdx) for tdx in fault.tensor_index]} bit {fault.bit}"
            f.write(
                f"{rank:3d}) Inj {inj:6d} | FRcrit={frcrit:.6f} | "
                f"maj={bias['maj_cls']}@{bias['maj_share']:.2f} Δmax={bias['delta_max']:.3f} "
                f"KL={bias['kl']:.3f} | {desc}\n"
            )

    print(f"[PROP/{ep_control}] K={K}  avgFRcrit={avg_frcrit:.6f}  Ew_final={E_wald:.6f}  n={n_inj}  DESIGN={design} → {output_file}")
    
    dt_min = (time.time() - t0) / 60.0
    print(f"[PROP/{ep_control}] saved {output_file} – {dt_min:.2f} min "
          f"(avg FRcrit={avg_frcrit:.6f}, half_norm={half_norm:.6f}, n={n_inj}, steps={steps})")
    
    avg_w, half_tuple_w, n_w, _, out_w = avg_frcrit, (half_norm,), n_inj, top_sorted, output_file
    half_w = _unpack_half(half_tuple_w)
    print(f"[DONE/WALD] N={K} | FR={avg_w:.6g} | half={half_w:.6g} | injections={n_w} | file={out_w}")

    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)