from __future__ import annotations

from typing import Callable

import numpy as np
import torch

try:
    from .algorithms_common import (
        _clear_agent_cache,
        _content_count,
        _pick_from_agent_cache_with_unseen_mask_gpu,
        _torch_softmax_rank_t,
    )
except ImportError:
    from engine.algorithms_common import (
        _clear_agent_cache,
        _content_count,
        _pick_from_agent_cache_with_unseen_mask_gpu,
        _torch_softmax_rank_t,
    )

def _cf_affinity_with_agent_cache(
    engine,
    step,
    isc_obj,
    agents,
    *,
    face: str,
    cache_attr: str,
    compute_method_name: str,
    unseen_mask_for_ids: Callable[[int, np.ndarray, int], np.ndarray],
    get_content_ids: Callable[[], np.ndarray | None],
    pick_from_probs: Callable[[np.ndarray, np.ndarray], int],
    cf_cache_duration: int,
):
    del unseen_mask_for_ids, pick_from_probs
    picked_ids: list[int] = []
    num_contents = _content_count(get_content_ids)

    for agent in agents:
        if getattr(agent, "cf_cache_timer", 0) > 0:
            agent.cf_cache_timer -= 1

        if agent.cf_cache_timer > 0:
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

        compute_method = getattr(agent, compute_method_name)
        compute_method(step, agents, isc_obj, face=face)
        cid = _pick_from_agent_cache_with_unseen_mask_gpu(
            engine,
            agent,
            cache_attr=cache_attr,
            num_contents=num_contents,
        )
        if cid is None:
            cid = agent.next_unseen_random_cid(len(isc_obj.pool))
        picked_ids.append(int(cid))

        agent.cf_cache_timer = cf_cache_duration

    return torch.tensor(picked_ids, device=engine.device, dtype=torch.long)


def cf_user_affinity(
    engine,
    step,
    isc_obj,
    agents,
    *,
    cf_user_face: str,
    unseen_mask_for_ids: Callable[[int, np.ndarray, int], np.ndarray],
    get_content_ids: Callable[[], np.ndarray | None],
    pick_from_probs: Callable[[np.ndarray, np.ndarray], int],
    cf_cache_duration: int,
):
    return _cf_affinity_with_agent_cache(
        engine,
        step,
        isc_obj,
        agents,
        face=cf_user_face,
        cache_attr="cf_user_score_cache",
        compute_method_name="compute_and_cache_cf_user_scores",
        unseen_mask_for_ids=unseen_mask_for_ids,
        get_content_ids=get_content_ids,
        pick_from_probs=pick_from_probs,
        cf_cache_duration=cf_cache_duration,
    )


def cf_item_affinity(
    engine,
    step,
    isc_obj,
    agents,
    *,
    cf_item_face: str,
    unseen_mask_for_ids: Callable[[int, np.ndarray, int], np.ndarray],
    get_content_ids: Callable[[], np.ndarray | None],
    pick_from_probs: Callable[[np.ndarray, np.ndarray], int],
    cf_cache_duration: int,
):
    return _cf_affinity_with_agent_cache(
        engine,
        step,
        isc_obj,
        agents,
        face=cf_item_face,
        cache_attr="cf_item_score_cache",
        compute_method_name="compute_and_cache_cf_item_scores",
        unseen_mask_for_ids=unseen_mask_for_ids,
        get_content_ids=get_content_ids,
        pick_from_probs=pick_from_probs,
        cf_cache_duration=cf_cache_duration,
    )


def cf_user_candidates_gpu(
    isc_obj,
    target_agent,
    step: int,
    *,
    lam: float,
    face: str,
    neigh_k: int,
    cand_k: int,
    ensure_content_index: Callable[[list], None],
    get_content_ids: Callable[[], np.ndarray | None],
    seen_mask_np: Callable[[int, int], np.ndarray],
    get_seen_mask_row: Callable[[int, int], torch.Tensor | None],
    softmax_arr: Callable[[np.ndarray, float], np.ndarray],
):
    del step, seen_mask_np, softmax_arr
    ensure_content_index(isc_obj.pool)
    content_ids = get_content_ids()
    if (content_ids is None) or (len(content_ids) == 0):
        raise RuntimeError("CF: content index is empty")

    uv_matrix = getattr(isc_obj, "UV_matrix", None)
    vu_matrix = getattr(isc_obj, "VU_matrix", None)
    if uv_matrix is None or vu_matrix is None:
        raise RuntimeError("CF: UV/VU sparse matrices are missing")

    device = uv_matrix.device
    content_ids_t = torch.as_tensor(content_ids, dtype=torch.long, device=device)

    def _fallback_single():
        cid = int(target_agent.next_unseen_random_cid(len(isc_obj.pool)))
        return (
            torch.tensor([cid], dtype=torch.long, device=device),
            torch.tensor([1.0], dtype=torch.float32, device=device),
        )

    uid = int(target_agent.id)
    mask_row = get_seen_mask_row(uid, len(content_ids))
    if mask_row is None:
        seen_t = torch.zeros(len(content_ids), dtype=torch.bool, device=device)
    else:
        seen_t = torch.as_tensor(mask_row, dtype=torch.bool, device=device)
    unseen_t = ~seen_t

    k_neigh = int(neigh_k)
    k_cand = int(cand_k)

    if getattr(isc_obj, "cf_user_sim_matrix_t", None) is not None:
        sim_users = isc_obj.cf_user_sim_matrix_t[:, uid]
    else:
        vu_col = vu_matrix.to_dense()[:, uid:uid + 1]
        sim_users = torch.sparse.mm(uv_matrix, vu_col).squeeze(1)

    sim_users = sim_users.to(torch.float32)
    if sim_users.numel() == 0:
        raise RuntimeError("CF: sim_users is empty")
    if uid < sim_users.numel():
        sim_users[uid] = 0.0

    if k_neigh > 0 and sim_users.numel() > k_neigh:
        vals_top, idx_top = torch.topk(sim_users, k=k_neigh)
        mask_top = vals_top > 0
        sim_vec = torch.zeros_like(sim_users)
        if mask_top.any():
            sim_vec[idx_top[mask_top]] = vals_top[mask_top]
    else:
        sim_vec = torch.where(sim_users > 0, sim_users, torch.zeros_like(sim_users))

    if float(sim_vec.sum().item()) <= 0.0:
        return _fallback_single()

    cand_scores = torch.sparse.mm(vu_matrix, sim_vec.view(-1, 1)).squeeze(1)
    finite_t = torch.isfinite(cand_scores) & unseen_t
    if not bool(finite_t.any()):
        return _fallback_single()

    face_str = str(face).lower() if face is not None else "affinity"
    work_scores = cand_scores
    if face_str == "novelty":
        finite_vals_t = cand_scores[finite_t]
        if finite_vals_t.numel() > 0:
            max_v_t = torch.max(finite_vals_t)
            min_v_t = torch.min(finite_vals_t)
            work_scores = cand_scores.clone()
            if float(max_v_t.item()) != float(min_v_t.item()):
                work_scores[finite_t] = max_v_t - cand_scores[finite_t]
            else:
                work_scores[finite_t] = 1.0

    if k_cand > 0:
        finite_count = int(finite_t.sum().item())
        k = min(int(k_cand), finite_count)
        masked_scores = torch.where(finite_t, work_scores, torch.full_like(work_scores, -torch.inf))
        vals_t, ridx_t = torch.topk(masked_scores, k=k, largest=True)
        keep_t = torch.isfinite(vals_t)
        vals_t = vals_t[keep_t]
        ridx_t = ridx_t[keep_t]
    else:
        ridx_t = torch.nonzero(finite_t, as_tuple=False).squeeze(1)
        vals_t = work_scores[ridx_t]

    if ridx_t.numel() == 0:
        return _fallback_single()

    probs_t = _torch_softmax_rank_t(vals_t, lam=float(lam))
    ids_t = content_ids_t[ridx_t]
    if ids_t.numel() == 0:
        return _fallback_single()
    return ids_t, probs_t


def cf_item_candidates_gpu(
    isc_obj,
    target_agent,
    step: int,
    *,
    lam: float,
    face: str,
    neigh_k: int,
    cand_k: int,
    ensure_content_index: Callable[[list], None],
    get_content_ids: Callable[[], np.ndarray | None],
    seen_mask_np: Callable[[int, int], np.ndarray],
    get_seen_mask_row: Callable[[int, int], torch.Tensor | None],
    softmax_arr: Callable[[np.ndarray, float], np.ndarray],
    cf_history_window_steps: int,
    device: torch.device,
):
    del seen_mask_np, softmax_arr
    ensure_content_index(isc_obj.pool)
    content_ids = get_content_ids()
    if (content_ids is None) or (len(content_ids) == 0):
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    uv_matrix = getattr(isc_obj, "UV_matrix", None)
    vu_matrix = getattr(isc_obj, "VU_matrix", None)
    sim_item = getattr(isc_obj, "cf_item_sim_matrix_t", None)
    if uv_matrix is None or vu_matrix is None:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    content_ids_t = torch.as_tensor(content_ids, dtype=torch.long, device=device)

    def _fallback_single():
        cid = int(target_agent.next_unseen_random_cid(len(isc_obj.pool)))
        return (
            torch.tensor([cid], dtype=torch.long, device=device),
            torch.tensor([1.0], dtype=torch.float32, device=device),
        )

    uid = int(target_agent.id)
    row_self = isc_obj.user_like_w.get(uid, {})
    if not row_self:
        return _fallback_single()

    liked_items: list[tuple[int, float]] = []
    for ridx, dq in list(row_self.items()):
        while dq and (step - dq[0] > cf_history_window_steps):
            dq.popleft()
        if dq:
            liked_items.append((int(ridx), 1.0))
        else:
            row_self.pop(ridx, None)

    if not liked_items:
        return _fallback_single()

    k_neigh = int(neigh_k)
    if k_neigh > 0 and len(liked_items) > k_neigh:
        liked_items.sort(key=lambda t: t[1], reverse=True)
        liked_items = liked_items[:k_neigh]

    num_items = len(content_ids)
    try:
        w = torch.zeros(num_items, device=device, dtype=torch.float32)
        for ridx, _wgt in liked_items:
            if 0 <= ridx < num_items:
                w[ridx] = 1.0

        if sim_item is not None:
            if sim_item.is_sparse:
                scores_t = torch.sparse.mm(sim_item, w.view(-1, 1)).squeeze(1)
            else:
                scores_t = (sim_item @ w.view(-1, 1)).squeeze(1)
        else:
            z = torch.sparse.mm(uv_matrix, w.view(-1, 1)).squeeze(1)
            scores_t = torch.sparse.mm(vu_matrix, z.view(-1, 1)).squeeze(1)
    except Exception:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    mask_row = get_seen_mask_row(uid, len(content_ids))
    if mask_row is None:
        seen_t = torch.zeros(len(content_ids), dtype=torch.bool, device=device)
    else:
        seen_t = torch.as_tensor(mask_row, dtype=torch.bool, device=device)
    unseen_t = ~seen_t

    finite_t = torch.isfinite(scores_t) & unseen_t
    if not bool(finite_t.any()):
        return _fallback_single()

    face_str = str(face).lower() if face is not None else "affinity"
    work_scores = scores_t
    if face_str == "novelty":
        finite_vals_t = scores_t[finite_t]
        if finite_vals_t.numel() > 0:
            max_v_t = torch.max(finite_vals_t)
            min_v_t = torch.min(finite_vals_t)
            work_scores = scores_t.clone()
            if float(max_v_t.item()) != float(min_v_t.item()):
                work_scores[finite_t] = max_v_t - scores_t[finite_t]
            else:
                work_scores[finite_t] = 1.0

    k_cand = int(cand_k)
    if k_cand > 0:
        finite_count = int(finite_t.sum().item())
        k = min(int(k_cand), finite_count)
        masked_scores = torch.where(finite_t, work_scores, torch.full_like(work_scores, -torch.inf))
        vals_t, ridx_t = torch.topk(masked_scores, k=k, largest=True)
        keep_t = torch.isfinite(vals_t)
        vals_t = vals_t[keep_t]
        ridx_t = ridx_t[keep_t]
    else:
        ridx_t = torch.nonzero(finite_t, as_tuple=False).squeeze(1)
        vals_t = work_scores[ridx_t]

    if ridx_t.numel() == 0:
        return _fallback_single()

    probs_t = _torch_softmax_rank_t(vals_t, lam=float(lam))
    ids_t = content_ids_t[ridx_t]
    if ids_t.numel() == 0:
        return _fallback_single()
    return ids_t, probs_t


def cf_user_candidates(
    isc_obj,
    uid: int,
    step: int,
    agents,
    *,
    face: str = "affinity",
    lam: float,
    neigh_k: int,
    cand_k: int,
    ensure_content_index: Callable[[list], None],
    get_content_ids: Callable[[], np.ndarray | None],
    seen_mask_np: Callable[[int, int], np.ndarray],
    get_seen_mask_row: Callable[[int, int], torch.Tensor | None],
    softmax_arr: Callable[[np.ndarray, float], np.ndarray],
):
    target_agent = agents[int(uid)]
    return cf_user_candidates_gpu(
        isc_obj,
        target_agent,
        step=step,
        lam=lam,
        face=face,
        neigh_k=neigh_k,
        cand_k=cand_k,
        ensure_content_index=ensure_content_index,
        get_content_ids=get_content_ids,
        seen_mask_np=seen_mask_np,
        get_seen_mask_row=get_seen_mask_row,
        softmax_arr=softmax_arr,
    )


def cf_item_candidates(
    isc_obj,
    uid: int,
    step: int,
    agents,
    *,
    face: str = "affinity",
    lam: float,
    neigh_k: int,
    cand_k: int,
    ensure_content_index: Callable[[list], None],
    get_content_ids: Callable[[], np.ndarray | None],
    seen_mask_np: Callable[[int, int], np.ndarray],
    get_seen_mask_row: Callable[[int, int], torch.Tensor | None],
    softmax_arr: Callable[[np.ndarray, float], np.ndarray],
    cf_history_window_steps: int,
    device: torch.device,
):
    target_agent = agents[int(uid)]
    return cf_item_candidates_gpu(
        isc_obj,
        target_agent,
        step=step,
        lam=lam,
        face=face,
        neigh_k=neigh_k,
        cand_k=cand_k,
        ensure_content_index=ensure_content_index,
        get_content_ids=get_content_ids,
        seen_mask_np=seen_mask_np,
        get_seen_mask_row=get_seen_mask_row,
        softmax_arr=softmax_arr,
        cf_history_window_steps=cf_history_window_steps,
        device=device,
    )

