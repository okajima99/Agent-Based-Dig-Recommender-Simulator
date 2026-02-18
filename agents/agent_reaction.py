from __future__ import annotations

import numpy as np


def like_prob_scores(
    agent,
    uid: int,
    content,
    *,
    sim_on_active_alpha,
    sim_alpha,
    agent_alpha: float,
    like_w_cg: float,
    like_w_cv: float,
    like_w_ci: float,
    sigmoid01,
    logit_k: float,
    logit_x0: float,
    like_divisor: float,
    random_module,
):
    del uid
    s_cg = sim_on_active_alpha(agent.interests, content, agent_alpha)
    s_cv = sim_on_active_alpha(agent.V, content, agent_alpha)
    s_ci = sim_alpha(agent.I, content.i_vector, agent_alpha)

    score = (like_w_cg * s_cg) + (like_w_cv * s_cv) + (like_w_ci * s_ci)
    p_like_raw = sigmoid01(score, k=logit_k, x0=logit_x0)
    p_like = p_like_raw / float(like_divisor)

    u = random_module.random()
    return p_like, u, float(s_cg), float(s_cv), float(s_ci)


def like_and_dig_scores(
    agent,
    uid,
    content,
    *,
    sigmoid01,
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
    del uid
    idx = content.active_idx
    if idx.size == 0:
        j_dim = int(np.argmax(agent.interests))
        return 0.0, 0.0, j_dim

    g_u_full = np.asarray(agent.interests, dtype=np.float64)
    v_full = np.asarray(agent.V, dtype=np.float64)
    c_act = np.asarray(content.g_active, dtype=np.float64)
    idx_act = idx

    gap = c_act - g_u_full[idx_act]
    pos_mask = gap > 0.0
    if not pos_mask.any():
        j_dim = int(np.argmax(agent.interests))
        return 0.0, 0.0, j_dim

    v_act = v_full[idx_act]

    mu_vec = float(mu0) + float(mu_slope) * (float(mu_alpha_c) * c_act + float(mu_beta_v) * v_act)
    sigma_vec = float(sigma0) * np.exp(float(sigma_lamda) * (float(sigma_alpha_c) * c_act + float(sigma_beta_v) * v_act))
    a_vec = float(a0) * np.exp(float(a_lamda) * (float(a_alpha_c) * c_act + float(a_beta_v) * v_act))

    z = (gap - mu_vec) / sigma_vec
    gauss = a_vec * np.exp(-0.5 * z * z)
    score_vec = gauss
    score_vec[~pos_mask] = -np.inf

    j_local = int(np.argmax(score_vec))
    best_score = float(score_vec[j_local])
    j_dim = int(idx_act[j_local])
    if not np.isfinite(best_score):
        return 0.0, 0.0, j_dim

    p_dig_raw = sigmoid01(best_score, k=float(dig_logit_k), x0=float(dig_logit_x0))
    p_dig = p_dig_raw / float(dig_divisor)
    return 0.0, max(0.0, min(1.0, p_dig)), j_dim
