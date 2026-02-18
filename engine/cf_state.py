from __future__ import annotations

from collections import deque

import numpy as np
import torch


def apply_pending_cf_likes(isc_obj, *, require_content_ids):
    pending = getattr(isc_obj, "pending_cf_likes", None)
    if not pending:
        return

    content_ids = require_content_ids()
    for uid, ridx, ts in pending:
        row = isc_obj.user_like_w.setdefault(int(uid), {})
        dq_u = row.setdefault(int(ridx), deque())
        dq_u.append(int(ts))

        col = isc_obj.item_liked_by_w.setdefault(int(content_ids[int(ridx)]), {})
        dq_i = col.setdefault(int(uid), deque())
        dq_i.append(int(ts))

    isc_obj.pending_cf_likes = []


def rebuild_cf_sync(
    isc_obj,
    step: int,
    agents,
    *,
    device: torch.device,
    cf_history_window_steps: int,
    cf_discount_gamma: float,
    cf_cache_duration: int,
):
    user_count = len(agents)
    item_count = len(isc_obj.pool)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for uid, items in list(isc_obj.user_like_w.items()):
        for ridx, dq in list(items.items()):
            while dq and (step - dq[0] > int(cf_history_window_steps)):
                dq.popleft()
            if dq:
                w = sum(float(cf_discount_gamma ** (step - ts)) for ts in dq)
                if w > 0.0:
                    rows.append(int(uid))
                    cols.append(int(ridx))
                    vals.append(w)
            else:
                items.pop(ridx, None)
        if not items:
            isc_obj.user_like_w.pop(uid, None)

    if len(vals) > 0:
        rows_np = np.asarray(rows, dtype=np.int64)
        cols_np = np.asarray(cols, dtype=np.int64)
        vals_np = np.asarray(vals, dtype=np.float32)
        indices = torch.as_tensor(np.stack([rows_np, cols_np]), device=device, dtype=torch.long)
        values = torch.as_tensor(vals_np, device=device, dtype=torch.float32)
        uv = torch.sparse_coo_tensor(indices, values, size=(user_count, item_count), device=device)
        isc_obj.UV_matrix = uv.coalesce()
        isc_obj.VU_matrix = uv.transpose(0, 1).coalesce()

        try:
            uv_coal = isc_obj.UV_matrix.coalesce()
            uv_idx = uv_coal.indices()[0]
            uv_val = uv_coal.values()
            user_norm_sq = torch.zeros(user_count, device=device, dtype=torch.float32)
            user_norm_sq.scatter_add_(0, uv_idx, uv_val * uv_val)
            isc_obj.cf_user_row_norm_t = torch.sqrt(user_norm_sq.clamp_min(1e-12))
        except Exception:
            isc_obj.cf_user_row_norm_t = None

        try:
            vu_coal = isc_obj.VU_matrix.coalesce()
            vu_idx = vu_coal.indices()[0]
            vu_val = vu_coal.values()
            item_norm_sq = torch.zeros(item_count, device=device, dtype=torch.float32)
            item_norm_sq.scatter_add_(0, vu_idx, vu_val * vu_val)
            isc_obj.cf_item_row_norm_t = torch.sqrt(item_norm_sq.clamp_min(1e-12))
        except Exception:
            isc_obj.cf_item_row_norm_t = None

        try:
            sim_mat = torch.sparse.mm(isc_obj.UV_matrix, isc_obj.VU_matrix).to(torch.float32)
            if getattr(isc_obj, "cf_user_row_norm_t", None) is not None:
                rn = isc_obj.cf_user_row_norm_t.view(-1, 1)
                denom = rn @ rn.t()
                sim_mat = sim_mat / denom.clamp_min(1e-12)
            diag = torch.arange(sim_mat.shape[0], device=device)
            sim_mat[diag, diag] = 0.0
            isc_obj.cf_user_sim_matrix_t = sim_mat
        except Exception:
            isc_obj.cf_user_sim_matrix_t = None

        try:
            item_sim_mat = torch.sparse.mm(isc_obj.VU_matrix, isc_obj.UV_matrix).to(torch.float32)
            if getattr(isc_obj, "cf_item_row_norm_t", None) is not None:
                rn_i = isc_obj.cf_item_row_norm_t.view(-1, 1)
                denom_i = rn_i @ rn_i.t()
                item_sim_mat = item_sim_mat / denom_i.clamp_min(1e-12)
            diag_i = torch.arange(item_sim_mat.shape[0], device=device)
            item_sim_mat[diag_i, diag_i] = 0.0
            isc_obj.cf_item_sim_matrix_t = item_sim_mat
        except Exception:
            isc_obj.cf_item_sim_matrix_t = None
    else:
        isc_obj.UV_matrix = None
        isc_obj.VU_matrix = None
        isc_obj.cf_user_sim_matrix_t = None
        isc_obj.cf_item_sim_matrix_t = None

    isc_obj.cf_matrix_cache_timer = int(cf_cache_duration)
    for agent in agents:
        agent.cf_score_cache = []
        agent.cf_user_score_cache = []
        agent.cf_item_score_cache = []
        agent.cf_cache_timer = 0


def stage_cf_like(isc_obj, uid: int, cid: int, step_like: int, *, get_id2row):
    id2row = get_id2row()
    if id2row is None:
        return
    ridx = id2row.get(int(cid))
    if ridx is None:
        return
    isc_obj.pending_cf_likes.append((int(uid), int(ridx), int(step_like)))
