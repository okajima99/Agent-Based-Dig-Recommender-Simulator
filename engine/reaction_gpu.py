from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch


def like_and_dig_batch(
    engine,
    *,
    step,
    isc,
    agents,
    picked_idx_t,
    torch_dtype: torch.dtype,
    content_mat_dtype: str,
    agent_alpha: float,
    like_w_cg: float,
    like_w_cv: float,
    like_w_ci: float,
    logit_k: float,
    logit_x0: float,
    like_divisor: float,
    mu0: float,
    mu_slope: float,
    mu_alpha_c: float,
    mu_beta_v: float,
    sigma0: float,
    sigma_lamda: float,
    sigma_alpha_c: float,
    sigma_beta_v: float,
    a0: float,
    a_lamda: float,
    a_alpha_c: float,
    a_beta_v: float,
    dig_logit_k: float,
    dig_logit_x0: float,
    dig_divisor: float,
):
    del step, agents

    device = engine.device
    num_agents = engine.num_agents

    picked_idx_t = picked_idx_t.to(device=device, dtype=torch.long)

    ug = engine.ug_matrix_t
    uv = engine.uv_matrix_t
    ui = engine.ui_matrix_t

    if engine.cg_matrix_half is not None:
        cg = engine.cg_matrix_half[picked_idx_t].to(torch_dtype)
        ci = engine.ci_matrix_half[picked_idx_t].to(torch_dtype)
    else:
        cg = engine.cg_matrix_t[picked_idx_t]
        ci = engine.ci_matrix_t[picked_idx_t]

    eps = 1e-8
    autocast_enabled = (device.type == "cuda")
    autocast_dtype = torch.bfloat16 if str(content_mat_dtype).lower() == "bf16" else torch.float16
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True)
        if autocast_enabled
        else nullcontext()
    )

    with autocast_ctx:
        def _sim_torch(u, v, alpha_val: float):
            dot = (u * v).sum(dim=1)
            if alpha_val == 0.0:
                return dot
            nu = torch.linalg.norm(u, dim=1).clamp_min(eps)
            nv = torch.linalg.norm(v, dim=1).clamp_min(eps)
            denom = (nu * nv).pow(alpha_val)
            return dot / denom

        alpha_like = float(agent_alpha)
        s_cg = _sim_torch(ug, cg, alpha_like)
        s_cv = _sim_torch(uv, cg, alpha_like)
        s_ci = _sim_torch(ui, ci, alpha_like)

        like_score = (
            float(like_w_cg) * s_cg
            + float(like_w_cv) * s_cv
            + float(like_w_ci) * s_ci
        )

        likes_np = np.asarray([c.likes for c in isc.pool], dtype=np.float32)
        if likes_np.size == 0:
            social = torch.zeros_like(like_score, dtype=torch.float32)
        else:
            content_ids_np = np.asarray([c.id for c in isc.pool], dtype=np.int64)
            # likes降順、同率時は新しいcontent_id(大きいID)を優先。
            order_np = np.lexsort((-content_ids_np, -likes_np))
            rank0_np = np.empty(likes_np.size, dtype=np.float32)
            rank0_np[order_np] = np.arange(likes_np.size, dtype=np.float32)
            denom = max(1, likes_np.size - 1)
            social_all = 1.0 - (rank0_np / float(denom))
            social_all_t = torch.as_tensor(social_all, device=device, dtype=torch.float32)
            social = social_all_t[picked_idx_t]

        alpha_t = getattr(engine, "social_alpha_t", None)
        if (alpha_t is None) or (alpha_t.numel() != num_agents):
            alpha_t = torch.zeros((num_agents,), device=device, dtype=torch.float32)
        boost = (1.0 + (alpha_t * social)).to(dtype=like_score.dtype)
        like_score = like_score * boost

        like_raw = torch.sigmoid(float(logit_k) * (like_score - float(logit_x0)))
        p_like = like_raw / float(like_divisor)

        gap = cg - ug
        c = cg
        v = uv

        mu = float(mu0) + float(mu_slope) * (float(mu_alpha_c) * c + float(mu_beta_v) * v)
        sigma = float(sigma0) * torch.exp(
            float(sigma_lamda) * (float(sigma_alpha_c) * c + float(sigma_beta_v) * v)
        )
        a = float(a0) * torch.exp(
            float(a_lamda) * (float(a_alpha_c) * c + float(a_beta_v) * v)
        )

        pos_mask = gap > 0.0
        sigma = sigma.clamp_min(1e-6)
        z = (gap - mu) / sigma
        gauss = a * torch.exp(-0.5 * z * z)
        gauss = torch.where(pos_mask, gauss, torch.full_like(gauss, -1e9))

        best_score, j_dim = gauss.max(dim=1)
        best_score = best_score * boost

        dig_raw = torch.sigmoid(float(dig_logit_k) * (best_score - float(dig_logit_x0)))
        p_dig = dig_raw / float(dig_divisor)

        u_like = torch.rand(num_agents, device=device)
        u_dig = torch.rand(num_agents, device=device)

    like_flags_t = (u_like < p_like)
    dig_flags_t = (u_dig < p_dig)

    like_flags = like_flags_t.detach().cpu().numpy().astype(np.bool_)
    dig_flags = dig_flags_t.detach().cpu().numpy().astype(np.bool_)
    j_dims = j_dim.detach().cpu().numpy().astype(np.int64)
    p_like_np = p_like.detach().cpu().numpy()
    p_dig_np = p_dig.detach().cpu().numpy()

    return like_flags, dig_flags, j_dims, p_like_np, p_dig_np
