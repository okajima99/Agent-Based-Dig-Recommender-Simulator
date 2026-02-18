from __future__ import annotations

import numpy as np


def update_like_history_gi(agent, g_vec, i_vec, step):
    agent.like_history_G.append((step, g_vec))
    agent.like_history_I.append((step, i_vec))


def compute_pseudo(
    history,
    current_step,
    *,
    pseudo_history_window_steps: int,
    pseudo_discount_gamma: float,
):
    if not history:
        return None
    cutoff = int(pseudo_history_window_steps)
    valid_vecs = []
    weights = []

    for s, vec in history:
        if vec is None:
            continue
        age = int(current_step) - int(s)
        if age < 0 or age > cutoff:
            continue
        arr = np.asarray(vec, dtype=np.float64).ravel()
        if arr.size == 0:
            continue
        valid_vecs.append(arr)
        weights.append(float(pseudo_discount_gamma) ** age)

    if not weights:
        return None
    stacked = np.stack(valid_vecs, axis=0)
    w = np.asarray(weights, dtype=np.float64)
    wsum = float(w.sum())
    if (not np.isfinite(wsum)) or (wsum <= 0):
        return None
    weighted = (w[:, None] * stacked).sum(axis=0) / wsum
    return weighted.tolist()


def compute_pseudo_vector_g(agent, step, *, compute_pseudo_fn):
    if agent.pseudo_g_cache is None:
        agent.pseudo_g_cache = compute_pseudo_fn(agent.like_history_G, step)
    return agent.pseudo_g_cache


def compute_pseudo_vector_i(agent, step, *, compute_pseudo_fn):
    if agent.pseudo_i_cache is None:
        agent.pseudo_i_cache = compute_pseudo_fn(agent.like_history_I, step)
    return agent.pseudo_i_cache
