from __future__ import annotations


def build_gpu_display_engine_class(
    *,
    context: dict[str, object],
    initialize_engine_state_fn,
    ensure_engine_seen_capacity_fn,
    mark_engine_seen_fn,
    mark_engine_seen_batch_fn,
    get_engine_seen_mask_row_fn,
    load_engine_agents_fn,
    load_engine_contents_fn,
    run_cf_user_affinity_fn,
    run_cf_item_affinity_fn,
    run_cbf_affinity_fn,
    run_like_and_dig_batch_fn,
    run_pick_contents_fn,
    unseen_mask_for_ids_fn,
    get_content_ids_fn,
    pick_from_probs_fn,
    ensure_content_index_fn,
    seen_mask_np_fn,
    vectorized_cbf_faced_fn,
):
    def _ctx(name: str):
        return context[name]

    class GPUDisplayEngine:
        def __init__(self, num_agents, num_contents, g_dim, device):
            initialize_engine_state_fn(
                self,
                num_agents=num_agents,
                num_contents=num_contents,
                g_dim=g_dim,
                device=device,
                num_instinct_dim=_ctx("NUM_INSTINCT_DIM"),
                torch_dtype=_ctx("TORCH_DTYPE"),
            )

        def _ensure_seen_capacity(self, num_contents: int):
            ensure_engine_seen_capacity_fn(self, num_contents)

        def mark_seen(self, agent_idx: int, cid: int):
            mark_engine_seen_fn(self, agent_idx, cid)

        def mark_seen_batch(self, agent_idx_t, cid_t):
            mark_engine_seen_batch_fn(self, agent_idx_t, cid_t)

        def increment_views_batch(self, cid_t):
            del cid_t
            raise AttributeError(
                "increment_views_batch is not implemented in GPUDisplayEngine "
                "(views are updated on Content objects)"
            )

        def get_seen_mask_row(self, agent_idx: int, num_contents: int):
            return get_engine_seen_mask_row_fn(self, agent_idx, num_contents)

        def load_agents(self, agents):
            load_engine_agents_fn(self, agents)

        def load_contents(self, pool):
            load_engine_contents_fn(self, pool, torch_dtype=_ctx("TORCH_DTYPE"))

        def cf_user_affinity(self, step, isc_obj, agents):
            return run_cf_user_affinity_fn(
                self,
                step,
                isc_obj,
                agents,
                cf_user_face=_ctx("CF_USER_FACE"),
                unseen_mask_for_ids=unseen_mask_for_ids_fn,
                get_content_ids=get_content_ids_fn,
                pick_from_probs=pick_from_probs_fn,
                cf_cache_duration=_ctx("CF_CACHE_DURATION"),
            )

        def cf_item_affinity(self, step, isc_obj, agents):
            return run_cf_item_affinity_fn(
                self,
                step,
                isc_obj,
                agents,
                cf_item_face=_ctx("CF_ITEM_FACE"),
                unseen_mask_for_ids=unseen_mask_for_ids_fn,
                get_content_ids=get_content_ids_fn,
                pick_from_probs=pick_from_probs_fn,
                cf_cache_duration=_ctx("CF_CACHE_DURATION"),
            )

        def cbf_affinity(self, isc_obj, agents, step, lam=30.0, face=None):
            return run_cbf_affinity_fn(
                self,
                isc_obj,
                agents,
                step,
                lam=lam,
                face=face,
                default_face=_ctx("CBF_FACE"),
                ensure_content_index=ensure_content_index_fn,
                seen_mask_np=seen_mask_np_fn,
                unseen_mask_for_ids=unseen_mask_for_ids_fn,
                pick_from_probs=pick_from_probs_fn,
                vectorized_cbf_faced=vectorized_cbf_faced_fn,
                get_content_ids=get_content_ids_fn,
                cbf_w_g=_ctx("CBF_W_G"),
                cbf_w_i=_ctx("CBF_W_I"),
                cbf_top_k=_ctx("CBF_TOP_K"),
            )

        def like_and_dig_batch(self, step, isc, agents, picked_idx_t):
            return run_like_and_dig_batch_fn(
                self,
                step=step,
                isc=isc,
                agents=agents,
                picked_idx_t=picked_idx_t,
                torch_dtype=_ctx("TORCH_DTYPE"),
                content_mat_dtype=_ctx("CONTENT_MAT_DTYPE"),
                agent_alpha=_ctx("AGENT_ALPHA"),
                like_w_cg=_ctx("LIKE_W_CG"),
                like_w_cv=_ctx("LIKE_W_CV"),
                like_w_ci=_ctx("LIKE_W_CI"),
                logit_k=_ctx("LOGIT_K"),
                logit_x0=_ctx("LOGIT_X0"),
                like_divisor=_ctx("LIKE_DIVISOR"),
                mu0=_ctx("MU0"),
                mu_slope=_ctx("MU_SLOPE"),
                mu_alpha_c=_ctx("MU_ALPHA_C"),
                mu_beta_v=_ctx("MU_BETA_V"),
                sigma0=_ctx("SIGMA0"),
                sigma_lamda=_ctx("SIGMA_LAMDA"),
                sigma_alpha_c=_ctx("SIGMA_ALPHA_C"),
                sigma_beta_v=_ctx("SIGMA_BETA_V"),
                a0=_ctx("A0"),
                a_lamda=_ctx("A_LAMDA"),
                a_alpha_c=_ctx("A_ALPHA_C"),
                a_beta_v=_ctx("A_BETA_V"),
                dig_logit_k=_ctx("DIG_LOGIT_K"),
                dig_logit_x0=_ctx("DIG_LOGIT_X0"),
                dig_divisor=_ctx("DIG_DIVISOR"),
            )

        def pick_contents(self, display_algorithm, step, isc, agents):
            return run_pick_contents_fn(
                self,
                display_algorithm,
                step,
                isc,
                agents,
                lambda_cbf=_ctx("LAMBDA_CBF"),
                cbf_face=_ctx("CBF_FACE"),
                unseen_mask_for_ids=unseen_mask_for_ids_fn,
                pick_from_probs=pick_from_probs_fn,
                get_content_ids=get_content_ids_fn,
                pop_cache_duration=_ctx("POP_CACHE_DURATION"),
                trend_cache_duration=_ctx("TREND_CACHE_DURATION"),
                buzz_cache_duration=_ctx("BUZZ_CACHE_DURATION"),
            )

    return GPUDisplayEngine
