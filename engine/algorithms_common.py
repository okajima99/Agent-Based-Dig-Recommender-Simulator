from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def _content_count(get_content_ids: Callable[[], np.ndarray | None]) -> int:
    content_ids = get_content_ids()
    if content_ids is None:
        raise RuntimeError("_CONTENT_IDS is not initialized")
    return int(len(content_ids))


def _clear_agent_cache(agent, cache_attr: str) -> None:
    setattr(agent, cache_attr, [])


def _cache_to_tensors(cache, *, device: torch.device):
    if isinstance(cache, tuple) and len(cache) == 2 and torch.is_tensor(cache[0]) and torch.is_tensor(cache[1]):
        ids_t = cache[0].to(device=device, dtype=torch.long)
        probs_t = cache[1].to(device=device, dtype=torch.float32)
        return ids_t, probs_t

    if isinstance(cache, list) and cache:
        ids, probs = zip(*cache)
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)
        probs_t = torch.as_tensor(probs, dtype=torch.float32, device=device)
        return ids_t, probs_t

    return (
        torch.empty(0, dtype=torch.long, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
    )


def _normalize_probs_t(probs_t: torch.Tensor):
    if probs_t.numel() == 0:
        return None
    p = torch.as_tensor(probs_t, dtype=torch.float32, device=probs_t.device)
    p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))
    p = torch.clamp(p, min=0.0)
    s = torch.sum(p)
    if (not torch.isfinite(s)) or (float(s.item()) <= 0.0):
        return None
    return p / s


def _store_cache_tensors(agent, cache_attr: str, ids_t: torch.Tensor, probs_t: torch.Tensor) -> None:
    setattr(
        agent,
        cache_attr,
        (
            ids_t.detach().to(dtype=torch.long),
            probs_t.detach().to(dtype=torch.float32),
        ),
    )


def _pick_from_agent_cache_with_unseen_mask_gpu(engine, agent, *, cache_attr: str, num_contents: int):
    cache = getattr(agent, cache_attr, [])
    if not cache:
        return None

    ids_t, probs_t = _cache_to_tensors(cache, device=engine.device)
    if ids_t.numel() == 0:
        _clear_agent_cache(agent, cache_attr)
        return None

    valid_t = (ids_t >= 0) & (ids_t < int(num_contents))
    if not bool(valid_t.any()):
        _clear_agent_cache(agent, cache_attr)
        return None

    ids_t = ids_t[valid_t]
    probs_t = probs_t[valid_t]

    mask_row = engine.get_seen_mask_row(agent.id, int(num_contents))
    if mask_row is None:
        unseen_t = torch.ones_like(ids_t, dtype=torch.bool, device=engine.device)
    else:
        seen_t = torch.as_tensor(mask_row, dtype=torch.bool, device=engine.device)
        unseen_t = ~seen_t[ids_t]

    masked_probs_t = torch.where(unseen_t, probs_t, torch.zeros_like(probs_t))
    probs_norm_t = _normalize_probs_t(masked_probs_t)
    if probs_norm_t is None:
        _clear_agent_cache(agent, cache_attr)
        return None

    sampled_idx = int(torch.multinomial(probs_norm_t, num_samples=1, replacement=True).item())
    cid = int(ids_t[sampled_idx].item())

    keep_t = unseen_t & (ids_t != cid)
    if bool(keep_t.any()):
        next_ids_t = ids_t[keep_t]
        next_probs_t = _normalize_probs_t(probs_t[keep_t])
        if next_probs_t is None:
            _clear_agent_cache(agent, cache_attr)
        else:
            _store_cache_tensors(agent, cache_attr, next_ids_t, next_probs_t)
    else:
        _clear_agent_cache(agent, cache_attr)

    return cid


def _torch_softmax_rank_t(x_t: torch.Tensor, lam: float = 1.0):
    x = torch.as_tensor(x_t, dtype=torch.float32, device=x_t.device)
    if x.numel() == 0:
        return x
    finite_t = torch.isfinite(x)
    if not bool(finite_t.any()):
        n = max(1, int(x.numel()))
        return torch.ones(n, device=x.device, dtype=torch.float32) / float(n)
    temperature = float(lam)
    if (not np.isfinite(temperature)) or (temperature <= 0.0):
        temperature = 1e-6
    max_finite = torch.max(x[finite_t])
    z = x - max_finite
    y = torch.exp(z / temperature)
    y = torch.where(finite_t, y, torch.zeros_like(y))
    denom = torch.sum(y)
    if (not torch.isfinite(denom)) or (float(denom.item()) <= 0.0):
        n = max(1, int(x.numel()))
        return torch.ones(n, device=x.device, dtype=torch.float32) / float(n)
    return y / denom


def sim_on_active_alpha(u_vec, content, alpha: float):
    idx = content.active_idx
    if idx.size == 0:
        return 0.0
    u = np.asarray(u_vec, dtype=np.float64)[idx]
    c = np.asarray(content.g_active, dtype=np.float64)
    dot = float(u @ c)
    if float(alpha) == 0.0:
        return dot
    nu = float(np.linalg.norm(u))
    nc = float(np.linalg.norm(c))
    if nu <= 1e-9 or nc <= 1e-9:
        return 0.0
    denom = (nu * nc) ** float(alpha)
    return dot / denom
