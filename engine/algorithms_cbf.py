from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch

try:
    from .algorithms_common import (
        _clear_agent_cache,
        _content_count,
        _pick_from_agent_cache_with_unseen_mask_gpu,
        _store_cache_tensors,
    )
except ImportError:
    from engine.algorithms_common import (
        _clear_agent_cache,
        _content_count,
        _pick_from_agent_cache_with_unseen_mask_gpu,
        _store_cache_tensors,
    )

def vectorized_cbf_faced(
    pseudo_g: np.ndarray,
    pseudo_i: np.ndarray,
    beta: float,
    face: str,
    w_g: float,
    w_i: float,
    *,
    exclude_ids=None,
    exclude_mask=None,
    top_k: int = 0,
    get_content_ids: Callable[[], np.ndarray | None],
    get_id2row: Callable[[], dict[int, int] | None],
    ensure_content_index: Callable[[list], None],
    resolve_pool: Callable[[], list | None],
    proj_sim_alpha_t: Callable[[bool, object, float], torch.Tensor],
    torch_softmax_rank: Callable[[torch.Tensor, float], torch.Tensor],
    to_np: Callable[[torch.Tensor], np.ndarray],
    device: torch.device,
):
    del to_np
    content_ids = get_content_ids()
    id2row = get_id2row()
    if content_ids is None or id2row is None:
        pool = resolve_pool()
        if pool is not None:
            ensure_content_index(pool)
            content_ids = get_content_ids()
            id2row = get_id2row()
    if content_ids is None or id2row is None:
        raise RuntimeError(
            "Content index is not built; ensure_content_index(pool) must be called before CBF scoring."
        )
    content_ids_t = torch.as_tensor(content_ids, dtype=torch.long, device=device)

    sim_g = proj_sim_alpha_t(True, pseudo_g, 0.0)
    sim_i = proj_sim_alpha_t(False, pseudo_i, 0.0)
    score = float(w_g) * sim_g + float(w_i) * sim_i

    face_str = str(face).lower() if face is not None else "affinity"
    if face_str == "novelty":
        score = (-float(w_g) * sim_g) + (-float(w_i) * sim_i)

    if exclude_mask is not None:
        try:
            mask_t = torch.as_tensor(exclude_mask, device=device, dtype=torch.bool)
            score = score.masked_fill(mask_t, -float("inf"))
        except Exception:
            pass
    elif exclude_ids:
        ridxs = [id2row.get(int(cid)) for cid in exclude_ids if id2row.get(int(cid)) is not None]
        if ridxs:
            ridxs_t = torch.as_tensor(ridxs, device=device, dtype=torch.long)
            score[ridxs_t] = -float("inf")

    vals, idx = torch.sort(score, descending=True)
    if top_k and top_k > 0:
        k = min(int(top_k), vals.numel())
        vals = vals[:k]
        idx = idx[:k]

    probs_t = torch_softmax_rank(vals, lam=float(beta))
    cids_t = content_ids_t[idx]
    return cids_t, probs_t.to(dtype=torch.float32)


def _cbf_fallback_cid(content_ids: np.ndarray | None, pool_size: int) -> int:
    if content_ids is not None and len(content_ids) > 0:
        ridx = random.randrange(len(content_ids))
        return int(content_ids[ridx])
    return int(random.randrange(max(1, pool_size)))


def cbf_affinity(
    engine,
    isc_obj,
    agents,
    step,
    *,
    lam: float,
    face: str | None,
    default_face: str,
    ensure_content_index: Callable[[list], None],
    seen_mask_np: Callable[[int, int], np.ndarray],
    unseen_mask_for_ids: Callable[[int, np.ndarray, int], np.ndarray],
    pick_from_probs: Callable[[np.ndarray, np.ndarray], int],
    vectorized_cbf_faced: Callable,
    get_content_ids: Callable[[], np.ndarray | None],
    cbf_w_g: float,
    cbf_w_i: float,
    cbf_top_k: int,
):
    del seen_mask_np, unseen_mask_for_ids, pick_from_probs
    use_face = default_face if face is None else face

    ensure_content_index(isc_obj.pool)
    num_contents = _content_count(get_content_ids)
    picked_ids: list[int] = []

    for agent in agents:
        if getattr(isc_obj, "pseudo_cache_timer", 0) > 0:
            cached_cid = _pick_from_agent_cache_with_unseen_mask_gpu(
                engine,
                agent,
                cache_attr="cbf_score_cache",
                num_contents=num_contents,
            )
            if cached_cid is not None:
                picked_ids.append(int(cached_cid))
                continue

        content_ids = get_content_ids()
        if content_ids is None:
            ensure_content_index(isc_obj.pool)
            content_ids = get_content_ids()
        content_len = 0 if content_ids is None else len(content_ids)
        mask_row = engine.get_seen_mask_row(agent.id, content_len)
        if mask_row is None:
            exclude_mask = torch.zeros(content_len, dtype=torch.bool, device=engine.device)
        else:
            exclude_mask = torch.as_tensor(mask_row, dtype=torch.bool, device=engine.device)

        scores_t = getattr(isc_obj, "cbf_pseudo_g_t", None)
        scores_i = getattr(isc_obj, "cbf_pseudo_i_t", None)

        cand_ids_t = torch.empty(0, dtype=torch.long, device=engine.device)
        cand_probs_t = torch.empty(0, dtype=torch.float32, device=engine.device)
        built = False
        if (scores_t is not None) and (scores_i is not None):
            try:
                g_vec = scores_t[agent.id]
                i_vec = scores_i[agent.id]
                cand_ids_t, cand_probs_t = vectorized_cbf_faced(
                    g_vec,
                    i_vec,
                    float(lam),
                    face=use_face,
                    w_g=cbf_w_g,
                    w_i=cbf_w_i,
                    exclude_mask=exclude_mask,
                    top_k=cbf_top_k,
                )
                built = True
            except Exception:
                built = False

        if not built:
            pseudo_g = agent.compute_pseudo_vector_G(step)
            pseudo_i = agent.compute_pseudo_vector_I(step)
            cand_ids_t, cand_probs_t = vectorized_cbf_faced(
                pseudo_g,
                pseudo_i,
                float(lam),
                face=use_face,
                w_g=cbf_w_g,
                w_i=cbf_w_i,
                exclude_mask=exclude_mask,
                top_k=cbf_top_k,
            )

        cand_ids_t = torch.as_tensor(cand_ids_t, dtype=torch.long, device=engine.device)
        cand_probs_t = torch.as_tensor(cand_probs_t, dtype=torch.float32, device=engine.device)
        if cand_ids_t.numel() > 0:
            _store_cache_tensors(agent, "cbf_score_cache", cand_ids_t, cand_probs_t)
            cid = _pick_from_agent_cache_with_unseen_mask_gpu(
                engine,
                agent,
                cache_attr="cbf_score_cache",
                num_contents=num_contents,
            )
            if cid is not None:
                picked_ids.append(int(cid))
                continue

        _clear_agent_cache(agent, "cbf_score_cache")
        picked_ids.append(int(agent.next_unseen_random_cid(len(isc_obj.pool))))

    return torch.tensor(picked_ids, device=engine.device, dtype=torch.long)

