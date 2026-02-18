from __future__ import annotations

import numpy as np
import torch


def update_global_pop_scores(isc_obj, *, device: torch.device, torch_softmax_rank, lambda_popularity: float):
    likes = np.asarray([c.likes for c in isc_obj.pool], dtype=np.float32)
    t = torch.tensor(likes, device=device)
    probs = torch_softmax_rank(t, lam=float(lambda_popularity))
    isc_obj._GLOBAL_POP_SCORES_T = probs


def update_global_trend_scores(isc_obj, *, device: torch.device, torch_softmax_rank, lambda_trend: float):
    arr = np.asarray([c.trend_ema for c in isc_obj.pool], dtype=np.float32)
    t = torch.tensor(arr, device=device)
    probs = torch_softmax_rank(t, lam=float(lambda_trend))
    isc_obj._GLOBAL_TREND_SCORES_T = probs


def update_global_buzz_scores(
    isc_obj,
    step: int,
    *,
    device: torch.device,
    torch_softmax_rank,
    lambda_buzz: float,
    buzz_window: int,
    buzz_gamma: float,
):
    num_contents = len(isc_obj.pool)
    scores = np.zeros(num_contents, dtype=np.float32)

    for i, content in enumerate(isc_obj.pool):
        if not content.like_history:
            continue
        ts_arr = np.fromiter((ts for ts, _uid in content.like_history), dtype=np.int64)
        age = step - ts_arr
        mask = age <= int(buzz_window)
        if mask.any():
            scores[i] = float(np.sum(np.power(float(buzz_gamma), age[mask], dtype=np.float64)))

    scores_t = torch.tensor(scores, dtype=torch.float32, device=device)
    probs_t = torch_softmax_rank(scores_t, lam=float(lambda_buzz))
    isc_obj._GLOBAL_BUZZ_SCORES_T = probs_t
