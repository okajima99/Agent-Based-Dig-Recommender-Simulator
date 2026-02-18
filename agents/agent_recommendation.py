from __future__ import annotations

import torch


def compute_and_cache_cbf_scores(
    agent,
    step,
    content_pool,
    *,
    ensure_content_index,
    seen_mask_np,
    vectorized_cbf_faced,
    lambda_cbf: float,
    cbf_face: str,
    cbf_w_g: float,
    cbf_w_i: float,
    cbf_top_k: int,
):
    ensure_content_index(content_pool)
    pg = agent.compute_pseudo_vector_G(step)
    pi = agent.compute_pseudo_vector_I(step)
    exclude_mask = seen_mask_np(agent.id, len(content_pool))
    cids, probs = vectorized_cbf_faced(
        pg,
        pi,
        lambda_cbf,
        face=cbf_face,
        w_g=cbf_w_g,
        w_i=cbf_w_i,
        exclude_mask=exclude_mask,
        top_k=cbf_top_k,
    )
    if torch.is_tensor(cids):
        if cids.numel() == 0:
            agent.cbf_score_cache = []
        else:
            agent.cbf_score_cache = (
                cids.detach().to(dtype=torch.long),
                probs.detach().to(dtype=torch.float32),
            )
        return

    agent.cbf_score_cache = list(zip(cids.tolist(), probs.tolist()))


def compute_and_cache_cf_user_scores(agent, step, agents, isc_obj, face, *, cf_user_candidates):
    ids, probs = cf_user_candidates(isc_obj, agent.id, step, agents, face=face)
    if torch.is_tensor(ids):
        if ids.numel() == 0:
            cid = agent.next_unseen_random_cid(len(isc_obj.pool))
            agent.cf_user_score_cache = [(int(cid), 1.0)]
        else:
            agent.cf_user_score_cache = (
                ids.detach().to(dtype=torch.long),
                probs.detach().to(dtype=torch.float32),
            )
        return

    if ids.size == 0:
        cid = agent.next_unseen_random_cid(len(isc_obj.pool))
        agent.cf_user_score_cache = [(int(cid), 1.0)]
    else:
        agent.cf_user_score_cache = list(zip(ids.tolist(), probs.tolist()))


def compute_and_cache_cf_item_scores(agent, step, agents, isc_obj, face, *, cf_item_candidates):
    ids, probs = cf_item_candidates(isc_obj, agent.id, step, agents, face=face)
    if torch.is_tensor(ids):
        if ids.numel() == 0:
            cid = agent.next_unseen_random_cid(len(isc_obj.pool))
            agent.cf_item_score_cache = [(int(cid), 1.0)]
        else:
            agent.cf_item_score_cache = (
                ids.detach().to(dtype=torch.long),
                probs.detach().to(dtype=torch.float32),
            )
        return

    if ids.size == 0:
        cid = agent.next_unseen_random_cid(len(isc_obj.pool))
        agent.cf_item_score_cache = [(int(cid), 1.0)]
    else:
        agent.cf_item_score_cache = list(zip(ids.tolist(), probs.tolist()))
