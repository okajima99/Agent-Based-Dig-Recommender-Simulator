from __future__ import annotations

from typing import Callable

import numpy as np
import torch

try:
    from .algorithms_common import (
        _clear_agent_cache,
        _normalize_probs_t,
        _pick_from_agent_cache_with_unseen_mask_gpu,
        _store_cache_tensors,
    )
except ImportError:
    from engine.algorithms_common import (
        _clear_agent_cache,
        _normalize_probs_t,
        _pick_from_agent_cache_with_unseen_mask_gpu,
        _store_cache_tensors,
    )

def sample_from_global_probs(
    engine,
    global_probs,
    agents,
    *,
    cache_attr: str,
    timer_attr: str,
    cache_duration: int,
    unseen_mask_for_ids: Callable[[int, np.ndarray, int], np.ndarray],
    pick_from_probs: Callable[[np.ndarray, np.ndarray], int],
    get_content_ids: Callable[[], np.ndarray | None],
    pool_size: int,
):
    del unseen_mask_for_ids, pick_from_probs
    content_ids = get_content_ids()
    if content_ids is None:
        raise RuntimeError("_CONTENT_IDS is not initialized")
    content_ids_t = torch.as_tensor(content_ids, dtype=torch.long, device=engine.device)
    num_contents = int(content_ids_t.numel())
    if num_contents <= 0:
        raise RuntimeError("content index is empty")

    if global_probs is None:
        raise RuntimeError("global score tensor is not initialized")
    base_probs_t = torch.as_tensor(global_probs, dtype=torch.float32, device=engine.device)
    valid_len = min(int(base_probs_t.numel()), num_contents)
    if valid_len <= 0:
        raise RuntimeError("global score tensor has no valid rows")

    ids_eval_t = content_ids_t[:valid_len]
    probs_eval_t = base_probs_t[:valid_len]
    probs_eval_t = torch.where(torch.isfinite(probs_eval_t), probs_eval_t, torch.zeros_like(probs_eval_t))
    probs_eval_t = torch.clamp(probs_eval_t, min=0.0)

    picked_ids: list[int] = []
    for agent in agents:
        if getattr(agent, timer_attr, 0) > 0:
            setattr(agent, timer_attr, int(getattr(agent, timer_attr)) - 1)

        if getattr(agent, timer_attr, 0) > 0:
            cached_cid = _pick_from_agent_cache_with_unseen_mask_gpu(
                engine,
                agent,
                cache_attr=cache_attr,
                num_contents=num_contents,
            )
            if cached_cid is not None:
                picked_ids.append(int(cached_cid))
                continue
            _clear_agent_cache(agent, cache_attr)

        mask_row = engine.get_seen_mask_row(agent.id, num_contents)
        if mask_row is None:
            unseen_t = torch.ones(valid_len, dtype=torch.bool, device=engine.device)
        else:
            seen_t = torch.as_tensor(mask_row, dtype=torch.bool, device=engine.device)
            unseen_t = ~seen_t[ids_eval_t]

        probs_masked_t = torch.where(unseen_t, probs_eval_t, torch.zeros_like(probs_eval_t))
        probs_norm_t = _normalize_probs_t(probs_masked_t)
        if probs_norm_t is not None:
            keep_t = unseen_t & torch.isfinite(probs_norm_t) & (probs_norm_t > 0.0)
            if bool(keep_t.any()):
                _store_cache_tensors(
                    agent,
                    cache_attr,
                    ids_eval_t[keep_t],
                    probs_norm_t[keep_t],
                )
                cid = _pick_from_agent_cache_with_unseen_mask_gpu(
                    engine,
                    agent,
                    cache_attr=cache_attr,
                    num_contents=num_contents,
                )
                if cid is not None:
                    picked_ids.append(int(cid))
                    setattr(agent, timer_attr, int(cache_duration))
                    continue

        _clear_agent_cache(agent, cache_attr)
        picked_ids.append(int(agent.next_unseen_random_cid(pool_size)))
        setattr(agent, timer_attr, int(cache_duration))

    return torch.tensor(picked_ids, device=engine.device, dtype=torch.long)
