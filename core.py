# ============================================================================
# Imports & 基本セットアップ
# ============================================================================
import random
import time
import numpy as np
try:
    from .core.core_imports import load_core_symbols
except ImportError:
    from core.core_imports import load_core_symbols

globals().update(load_core_symbols())
# ----------------------------
# 設定ロード（既定値は config.py 側に集約）
# ----------------------------
_CORE_PARAMS = load_core_params()
globals().update(_CORE_PARAMS)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# ログ出力（コンソールの代わりにテキストへ集約）
# ============================================================================
globals().update(
    setup_logging(
        out_prefix=OUT_PREFIX,
        display_algorithm=DISPLAY_ALGORITHM,
        cbf_face=CBF_FACE,
        cf_item_face=CF_ITEM_FACE,
        cf_user_face=CF_USER_FACE,
        log_progress_every=LOG_PROGRESS_EVERY,
    )
)

# ============================================================================
# PyTorch / GPU 設定
# ============================================================================
globals().update(
    setup_torch_bindings(
        seed=RANDOM_SEED,
        strict_cuda=STRICT_CUDA,
        log_fn=log_and_print,
    )
)

# ============================================================================
# GPU Batch Engine（メインループ一括化のコア）
# ============================================================================
# class wiring is assigned after helper functions are defined.

# ============================================================================
# ユーティリティ関数群
# ============================================================================
globals().update(
    bind_core_runtime(
        context=globals(),
        clip01_fn=clip01,
        clip11_fn=clip11,
        sim_alpha_fn=sim_alpha,
        safe_sim_alpha_fn=safe_sim_alpha,
        truncnorm_fn=truncnorm,
        l2_norm_fn=l2_norm,
        seen_mask_np_fn=seen_mask_np,
        exclude_ids_from_mask_fn=exclude_ids_from_mask,
        unseen_mask_for_ids_fn=unseen_mask_for_ids,
        gen_agent_vector_legacy_fn=gen_agent_vector_legacy,
        gen_agent_g_vector_fn=gen_agent_g_vector,
        gen_agent_i_vector_fn=gen_agent_i_vector,
        gen_agent_v_vector_fn=gen_agent_v_vector,
        gen_content_vector_unified_fn=gen_content_vector_unified,
        gen_content_g_vector_unified_fn=gen_content_g_vector_unified,
        gen_content_i_vector_unified_fn=gen_content_i_vector_unified,
        get_content_ids_impl_fn=get_content_ids_impl,
        get_id2row_impl_fn=get_id2row_impl,
        ensure_content_index_impl_fn=ensure_content_index_impl,
        proj_sim_alpha_t_impl_fn=proj_sim_alpha_t_impl,
        run_sim_on_active_alpha_fn=run_sim_on_active_alpha,
        run_vectorized_cbf_faced_fn=run_vectorized_cbf_faced,
        update_global_pop_scores_state_fn=update_global_pop_scores_state,
        update_global_trend_scores_state_fn=update_global_trend_scores_state,
        update_global_buzz_scores_state_fn=update_global_buzz_scores_state,
    )
)

# ============================================================================
# 補充設定
# ============================================================================
# 周期補充だけを採用
_REPLENISH_MAP = build_replenish_map(
    replenish_every=REPLENISH_EVERY,
    replenish_count=REPLENISH_COUNT,
    replenish_start_step=REPLENISH_START_STEP,
    replenish_end_step=REPLENISH_END_STEP,
    max_steps=MAX_STEPS,
)

GPUDisplayEngine = build_gpu_display_engine_class(
    context=globals(),
    initialize_engine_state_fn=initialize_engine_state,
    ensure_engine_seen_capacity_fn=ensure_engine_seen_capacity,
    mark_engine_seen_fn=mark_engine_seen,
    mark_engine_seen_batch_fn=mark_engine_seen_batch,
    get_engine_seen_mask_row_fn=get_engine_seen_mask_row,
    load_engine_agents_fn=load_engine_agents,
    load_engine_contents_fn=load_engine_contents,
    run_cf_user_affinity_fn=run_cf_user_affinity,
    run_cf_item_affinity_fn=run_cf_item_affinity,
    run_cbf_affinity_fn=run_cbf_affinity,
    run_like_and_dig_batch_fn=run_like_and_dig_batch,
    run_pick_contents_fn=run_pick_contents,
    unseen_mask_for_ids_fn=_unseen_mask_for_ids,
    get_content_ids_fn=_get_content_ids,
    pick_from_probs_fn=torch_pick_from_probs,
    ensure_content_index_fn=ensure_content_index,
    seen_mask_np_fn=_seen_mask_np,
    vectorized_cbf_faced_fn=vectorized_cbf_faced,
)

# ============================================================================
# ISCクラスとコンテンツ定義
# ============================================================================
ISC = build_isc_class(
    context=globals(),
    domain_content_cls=DomainContent,
    build_initial_pool_fn=build_initial_pool,
    build_id2content_fn=build_id2content,
    replenish_pool_fn=replenish_pool,
    ensure_content_index_fn=ensure_content_index,
    apply_pending_cf_likes_state_fn=apply_pending_cf_likes_state,
    rebuild_cf_sync_state_fn=rebuild_cf_sync_state,
    rebuild_cbf_pseudo_batch_state_fn=rebuild_cbf_pseudo_batch_state,
    stage_cf_like_state_fn=stage_cf_like_state,
    tick_and_refresh_pseudo_if_needed_state_fn=tick_and_refresh_pseudo_if_needed_state,
    require_content_ids_fn=_require_content_ids,
    get_id2row_fn=_get_id2row,
    random_interval_active_fn=_random_interval_active,
)


def _configure_user_agent_hooks():
    configure_user_agent_hooks_for_core(
        configure_user_agent_hooks_fn=configure_user_agent_hooks,
        context=globals(),
        sim_on_active_alpha_fn=_sim_on_active_alpha,
        sim_alpha_fn=_sim_alpha,
        sigmoid01_fn=sigmoid01,
        ensure_content_index_fn=ensure_content_index,
        vectorized_cbf_faced_fn=vectorized_cbf_faced,
        seen_mask_np_fn=_seen_mask_np,
        unseen_mask_for_ids_fn=_unseen_mask_for_ids,
        softmax_arr_fn=softmax_arr,
        pick_from_probs_fn=torch_pick_from_probs,
        get_seen_mask_row_fn=_engine_seen_mask_row,
        get_content_ids_fn=_get_content_ids,
        require_content_ids_fn=_require_content_ids,
        get_isc_fn=lambda: globals().get("isc"),
        random_module=random,
        random_choice=random.choice,
    )

# ============================================================================
# 実行セットアップ
# ============================================================================

def run_core():
    isc, agents, engine = build_world(
        isc_cls=ISC,
        user_agent_cls=UserAgent,
        engine_cls=GPUDisplayEngine,
        num_genres=NUM_GENRES,
        content_g_params=content_G_PARAMS,
        num_contents=NUM_CONTENTS,
        num_agents=NUM_AGENTS,
        device=DEVICE,
        configure_user_agent_hooks=_configure_user_agent_hooks,
    )
    globals()["isc"] = isc
    globals()["agents"] = agents
    globals()["engine"] = engine

    t0 = time.time()
    prev_force_random = False
    for step in range(MAX_STEPS):
        log_on = (LOG_PROGRESS_EVERY > 0) and (step % LOG_PROGRESS_EVERY == 0)
        t_step_start = time.time() if log_on else None

        force_random = prepare_step(
            step,
            prev_force_random=prev_force_random,
            isc=isc,
            agents=agents,
            display_algorithm=DISPLAY_ALGORITHM,
            initial_random_steps=INITIAL_RANDOM_STEPS,
            random_interval_active=_random_interval_active,
            pop_cache_duration=POP_CACHE_DURATION,
            trend_cache_duration=TREND_CACHE_DURATION,
            buzz_cache_duration=BUZZ_CACHE_DURATION,
            replenish_map=_REPLENISH_MAP,
            engine=engine,
            update_global_pop_scores=update_global_pop_scores,
            update_global_trend_scores=update_global_trend_scores_global,
            update_global_buzz_scores=update_global_buzz_scores,
        )

        picked_idx = pick_indices(
            step,
            display_algorithm=DISPLAY_ALGORITHM,
            force_random=force_random,
            isc=isc,
            agents=agents,
            engine=engine,
            num_agents=NUM_AGENTS,
            device=DEVICE,
            torch_module=torch,
        )
        t_after_pick = time.time() if log_on else None

        reaction_result = run_reaction_batch(
            step,
            isc=isc,
            agents=agents,
            picked_idx_t=picked_idx,
            engine=engine,
            device=DEVICE,
            torch_module=torch,
        )
        t_after_like = time.time() if log_on else None

        apply_step_updates(
            step,
            isc=isc,
            agents=agents,
            engine=engine,
            picked_idx_t=picked_idx,
            reaction_result=reaction_result,
            device=DEVICE,
            torch_module=torch,
            num_agents=NUM_AGENTS,
            dig_g_step=DIG_G_STEP,
            dig_v_range=DIG_V_RANGE,
            cf_history_window_steps=CF_HISTORY_WINDOW_STEPS,
            get_id2row=_get_id2row,
            random_module=random,
        )

        if (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            log_and_print(f"[GPU batch] step {step+1}/{MAX_STEPS} 経過 ({elapsed:.1f}s)", flush=False)

        if log_on:
            t_end = time.time()
            log_and_print(
                f"[step {step}] pick={t_after_pick - t_step_start:.3f}s, like/dig={t_after_like - t_after_pick:.3f}s, "
                f"book={t_end - t_after_like:.3f}s, total={t_end - t_step_start:.3f}s",
                flush=False,
            )

        prev_force_random = force_random

    log_and_print("🔥 GPU バッチ版シミュレーション完了")
    return {
        "isc": isc,
        "agents": agents,
        "engine": engine,
        "OUT_PREFIX": OUT_PREFIX,
        "OUT_PREFIX_FACE": OUT_PREFIX_FACE,
        "LOG_FILE_PATH": LOG_FILE_PATH,
    }


if __name__ == "__main__":
    run_core()
