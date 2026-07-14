"""
cpcv.py — Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu, 2015).

Given a performance matrix (observations x strategy-configs), estimate the
probability that the configuration chosen as best in-sample underperforms the
median configuration out-of-sample. Targets the (signal_quantile,
threshold_window) grid selection in walk_forward.sweep_params.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def cscv_pbo(perf_matrix: np.ndarray, n_splits: int = 16) -> dict:
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return {"pbo": float("nan"), "logits": [], "n_combinations": 0,
                "reason": "need >= 2 configs"}
    T, N = M.shape

    # Clamp n_splits to an even number <= T; require a meaningful minimum.
    S = min(n_splits, T - (T % 2))
    if S % 2 == 1:
        S -= 1
    if T < 8 or S < 4:
        return {"pbo": float("nan"), "logits": [], "n_combinations": 0,
                "reason": f"insufficient observations for PBO (got {T}, need >= 8)"}

    blocks = np.array_split(np.arange(T), S)
    half = S // 2
    logits = []
    for is_blocks in combinations(range(S), half):
        is_set = set(is_blocks)
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in is_set])

        is_perf = M[is_rows].mean(axis=0)
        oos_perf = M[oos_rows].mean(axis=0)
        n_star = int(np.argmax(is_perf))

        # OOS relative rank of the IS-best config (higher perf -> higher rank).
        order = oos_perf.argsort()
        ranks = np.empty(N)
        ranks[order] = np.arange(1, N + 1)
        w = ranks[n_star] / (N + 1)                 # in (0, 1)
        w = min(max(w, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(w / (1.0 - w))))

    logits_arr = np.array(logits)
    pbo = float(np.mean(logits_arr < 0.0))
    return {"pbo": pbo, "logits": logits, "n_combinations": len(logits),
            "reason": ""}


if __name__ == "__main__":
    # ponytail: runnable self-check — noise -> ~0.5, single genuine edge -> ~0.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((40, 20))
    assert 0.3 < cscv_pbo(noise, 10)["pbo"] < 0.7
    edged = rng.standard_normal((40, 20)) * 0.1
    edged[:, 3] += 1.0
    assert cscv_pbo(edged, 10)["pbo"] < 0.1
    print("cpcv self-check OK")
