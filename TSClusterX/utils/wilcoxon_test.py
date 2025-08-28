import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def wilcoxon_vs_baseline(baseline_file: str,
                         folder: str,
                         metric: str = "RI",
                         alpha: float = 0.01,
                         alternative: str = "greater",
                         print_table: bool = True) -> pd.DataFrame:

    base_df = pd.read_csv(os.path.join(folder, baseline_file))
    base_vals = base_df[metric].values
    algorithms, p_raw, w_stat, n_pos, n_zero, n_neg = [], [], [], [], [], []

    for csv in os.listdir(folder):
        if csv == baseline_file or not csv.endswith(".csv"):
            continue

        algo_df  = pd.read_csv(os.path.join(folder, csv))
        algo_vals = algo_df[metric].values

        if len(algo_vals) != len(base_vals):
            raise ValueError(f"{csv}: length {len(algo_vals)} "
                             f"≠ baseline length {len(base_vals)}")

        stat, p = wilcoxon(algo_vals, base_vals, alternative=alternative)

        diff = algo_vals - base_vals
        n_pos.append((diff >  0).sum())
        n_zero.append((diff == 0).sum())
        n_neg.append((diff <  0).sum())

        algorithms.append(csv.replace(".csv", ""))
        w_stat.append(stat)
        p_raw.append(p)

    reject, p_holm, _, _ = multipletests(p_raw, alpha=alpha, method="holm")

    results = pd.DataFrame({
        "Algorithm"  : algorithms,
        "n_pos"      : n_pos,
        "n_zero"     : n_zero,
        "n_neg"      : n_neg,
        "W_stat"     : w_stat,
        "p_raw"      : p_raw,
        "p_Holm"     : p_holm,
        f"Reject_H0@α={alpha}": reject
    }).sort_values("p_Holm")

    if print_table:
        print("\nWilcoxon vs baseline ({})  —  Holm–Bonferroni α = {}\n"
              .format(alternative, alpha))
        print(results.to_string(index=False, float_format="%.4g"))

    return results
