from __future__ import annotations


def bind_core_runtime(
    *,
    context: dict[str, object],
    clip01_fn,
    clip11_fn,
    sim_alpha_fn,
    safe_sim_alpha_fn,
    truncnorm_fn,
    l2_norm_fn,
    seen_mask_np_fn,
    exclude_ids_from_mask_fn,
    unseen_mask_for_ids_fn,
    gen_agent_vector_legacy_fn,
    gen_agent_g_vector_fn,
    gen_agent_i_vector_fn,
    gen_agent_v_vector_fn,
    gen_content_vector_unified_fn,
    gen_content_g_vector_unified_fn,
    gen_content_i_vector_unified_fn,
    get_content_ids_impl_fn,
    get_id2row_impl_fn,
    ensure_content_index_impl_fn,
    proj_sim_alpha_t_impl_fn,
    run_sim_on_active_alpha_fn,
    run_vectorized_cbf_faced_fn,
    update_global_pop_scores_state_fn,
    update_global_trend_scores_state_fn,
    update_global_buzz_scores_state_fn,
):
    def _ctx(name: str):
        return context[name]

    def sigmoid01(x, k=None, x0=None):
        k = _ctx("LOGIT_K") if k is None else k
        x0 = _ctx("LOGIT_X0") if x0 is None else x0
        return _ctx("sigmoid01_impl")(x, k=k, x0=x0)

    def softmax_arr(x, lam=1.0):
        return _ctx("softmax_arr_impl")(x, lam=lam)

    def _sim_alpha(x_vec, y_vec, alpha: float):
        return sim_alpha_fn(x_vec, y_vec, alpha)

    def _safe_sim_alpha(a_vec, b_vec, alpha: float):
        return safe_sim_alpha_fn(a_vec, b_vec, alpha)

    def _tnorm(mu, sigma, lo, hi):
        return truncnorm_fn(mu, sigma, lo, hi)

    def _l2(x):
        return l2_norm_fn(x)

    def _clip01(x: float):
        return clip01_fn(x)

    def _clip11(x: float):
        return clip11_fn(x)

    def _seen_mask_np(agent_idx: int, num_contents: int):
        eng = context.get("engine")
        return seen_mask_np_fn(eng, agent_idx, num_contents)

    def _exclude_ids_from_mask(agent_idx: int, num_contents: int):
        eng = context.get("engine")
        return exclude_ids_from_mask_fn(eng, agent_idx, num_contents)

    def _unseen_mask_for_ids(agent_idx: int, ids_arr, num_contents: int):
        eng = context.get("engine")
        return unseen_mask_for_ids_fn(eng, agent_idx, ids_arr, num_contents)

    def _engine_seen_mask_row(agent_idx: int, num_contents: int):
        eng = context.get("engine")
        if eng is None:
            return None
        try:
            return eng.get_seen_mask_row(agent_idx, num_contents)
        except Exception:
            return None

    def _generate_agent_vector_legacy(length, mu_e, sig_e, norm_mu, norm_sig):
        return gen_agent_vector_legacy_fn(length, mu_e, sig_e, norm_mu, norm_sig)

    def generate_agent_g_vector(length, mu_e, sig_e, norm_mu, norm_sig):
        return gen_agent_g_vector_fn(
            length,
            mu_e,
            sig_e,
            norm_mu,
            norm_sig,
            mode=_ctx("AGENT_G_MODE"),
        )

    def generate_agent_i_vector(length, mu_e, sig_e, norm_mu, norm_sig):
        return gen_agent_i_vector_fn(
            length,
            mu_e,
            sig_e,
            norm_mu,
            norm_sig,
            mode=_ctx("AGENT_I_MODE"),
        )

    def generate_agent_v_vector(length, mu_e, sig_e, norm_mu, norm_sig):
        return gen_agent_v_vector_fn(
            length,
            mu_e,
            sig_e,
            norm_mu,
            norm_sig,
            mode=_ctx("AGENT_V_MODE"),
        )

    def _generate_content_vector_unified(
        length: int,
        mode: str,
        mu_e: float,
        sig_e: float,
        norm_mu: float,
        norm_sig: float,
        active_count: int,
    ):
        return gen_content_vector_unified_fn(
            length=length,
            mode=mode,
            mu_e=mu_e,
            sig_e=sig_e,
            norm_mu=norm_mu,
            norm_sig=norm_sig,
            active_count=active_count,
        )

    def generate_content_g_vector_unified(length, params, active_count):
        return gen_content_g_vector_unified_fn(
            length=length,
            params=params,
            active_count=active_count,
            mode=_ctx("CONTENT_G_MODE"),
        )

    def generate_content_i_vector_unified(length, params, active_count):
        return gen_content_i_vector_unified_fn(
            length=length,
            params=params,
            active_count=active_count,
            mode=_ctx("CONTENT_I_MODE"),
        )

    def _get_content_ids():
        return get_content_ids_impl_fn()

    def _get_id2row():
        return get_id2row_impl_fn()

    def _require_content_ids():
        content_ids = _get_content_ids()
        if content_ids is None:
            raise RuntimeError("Content index is not built")
        return content_ids

    def _proj_sim_alpha_t(is_g: bool, p_vec, alpha: float):
        return proj_sim_alpha_t_impl_fn(
            is_g,
            p_vec,
            alpha,
            to_t=_ctx("_to_t"),
            device=_ctx("DEVICE"),
            torch_module=_ctx("torch"),
        )

    def ensure_content_index(pool):
        ensure_content_index_impl_fn(
            pool,
            to_t=_ctx("_to_t"),
            device=_ctx("DEVICE"),
            torch_module=_ctx("torch"),
            content_mat_dtype=_ctx("CONTENT_MAT_DTYPE"),
        )

    def _sim_on_active_alpha(u_vec, content, alpha: float):
        return run_sim_on_active_alpha_fn(u_vec, content, alpha)

    def vectorized_cbf_faced(
        pseudo_g,
        pseudo_i,
        beta: float,
        face: str,
        w_g: float,
        w_i: float,
        exclude_ids=None,
        exclude_mask=None,
        top_k: int = 0,
    ):
        return run_vectorized_cbf_faced_fn(
            pseudo_g,
            pseudo_i,
            beta,
            face,
            w_g,
            w_i,
            exclude_ids=exclude_ids,
            exclude_mask=exclude_mask,
            top_k=top_k,
            get_content_ids=_get_content_ids,
            get_id2row=_get_id2row,
            ensure_content_index=ensure_content_index,
            resolve_pool=lambda: getattr(context.get("isc"), "pool", None),
            proj_sim_alpha_t=_proj_sim_alpha_t,
            torch_softmax_rank=_ctx("torch_softmax_rank"),
            to_np=_ctx("_to_np"),
            device=_ctx("DEVICE"),
        )

    def update_global_pop_scores(isc):
        update_global_pop_scores_state_fn(
            isc,
            device=_ctx("DEVICE"),
            torch_softmax_rank=_ctx("torch_softmax_rank"),
            lambda_popularity=_ctx("LAMBDA_POPULARITY"),
        )

    def update_global_trend_scores_global(isc):
        update_global_trend_scores_state_fn(
            isc,
            device=_ctx("DEVICE"),
            torch_softmax_rank=_ctx("torch_softmax_rank"),
            lambda_trend=_ctx("LAMBDA_TREND"),
        )

    def update_global_buzz_scores(isc, step):
        update_global_buzz_scores_state_fn(
            isc,
            step,
            device=_ctx("DEVICE"),
            torch_softmax_rank=_ctx("torch_softmax_rank"),
            lambda_buzz=_ctx("LAMBDA_BUZZ"),
            buzz_window=_ctx("BUZZ_WINDOW"),
            buzz_gamma=_ctx("BUZZ_GAMMA"),
        )

    def _random_interval_active(step: int) -> bool:
        if not _ctx("RANDOM_INTERVAL_ON"):
            return False
        if _ctx("RANDOM_RANDOM_BLOCK_LEN") <= 0:
            return False
        cycle = _ctx("RANDOM_RANDOM_BLOCK_LEN") + max(0, _ctx("RANDOM_NORMAL_BLOCK_LEN"))
        if cycle <= 0:
            return False
        if step < _ctx("INITIAL_RANDOM_STEPS"):
            return True
        rel = step - _ctx("INITIAL_RANDOM_STEPS")
        return (rel % cycle) < _ctx("RANDOM_RANDOM_BLOCK_LEN")

    return {
        "sigmoid01": sigmoid01,
        "softmax_arr": softmax_arr,
        "_sim_alpha": _sim_alpha,
        "_safe_sim_alpha": _safe_sim_alpha,
        "_tnorm": _tnorm,
        "_l2": _l2,
        "_clip01": _clip01,
        "_clip11": _clip11,
        "_seen_mask_np": _seen_mask_np,
        "_exclude_ids_from_mask": _exclude_ids_from_mask,
        "_unseen_mask_for_ids": _unseen_mask_for_ids,
        "_engine_seen_mask_row": _engine_seen_mask_row,
        "_generate_agent_vector_legacy": _generate_agent_vector_legacy,
        "generate_agent_g_vector": generate_agent_g_vector,
        "generate_agent_i_vector": generate_agent_i_vector,
        "generate_agent_v_vector": generate_agent_v_vector,
        "_generate_content_vector_unified": _generate_content_vector_unified,
        "generate_content_g_vector_unified": generate_content_g_vector_unified,
        "generate_content_i_vector_unified": generate_content_i_vector_unified,
        "_get_content_ids": _get_content_ids,
        "_get_id2row": _get_id2row,
        "_require_content_ids": _require_content_ids,
        "_proj_sim_alpha_t": _proj_sim_alpha_t,
        "ensure_content_index": ensure_content_index,
        "_sim_on_active_alpha": _sim_on_active_alpha,
        "vectorized_cbf_faced": vectorized_cbf_faced,
        "update_global_pop_scores": update_global_pop_scores,
        "update_global_trend_scores_global": update_global_trend_scores_global,
        "update_global_buzz_scores": update_global_buzz_scores,
        "_random_interval_active": _random_interval_active,
    }
