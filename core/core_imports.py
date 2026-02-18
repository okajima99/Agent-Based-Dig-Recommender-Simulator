from __future__ import annotations

try:
    from ..agents import agent_history, agent_hooks, agent_model, agent_random, agent_reaction, agent_recommendation
    from ..config import load_core_params
    from ..engine import (
        cbf_state,
        cf_state,
        engine_algorithms,
        engine_gpu,
        engine_state,
        global_scores,
        reaction_gpu,
    )
    from ..steps import step_flow
    from ..utils import log_buffer, math_utils, runtime, vector_generators
    from . import bootstrap, content_domain, content_index, core_helpers, isc_model
except ImportError:
    from agents import agent_history, agent_hooks, agent_model, agent_random, agent_reaction, agent_recommendation
    from config import load_core_params
    from engine import cbf_state, cf_state, engine_algorithms, engine_gpu, engine_state, global_scores, reaction_gpu
    from steps import step_flow
    from utils import log_buffer, math_utils, runtime, vector_generators
    from core import bootstrap, content_domain, content_index, core_helpers, isc_model


def load_core_symbols() -> dict[str, object]:
    return {
        "compute_agent_pseudo": agent_history.compute_pseudo,
        "compute_agent_pseudo_vector_g": agent_history.compute_pseudo_vector_g,
        "compute_agent_pseudo_vector_i": agent_history.compute_pseudo_vector_i,
        "update_agent_like_history_gi": agent_history.update_like_history_gi,
        "configure_user_agent_hooks_for_core": agent_hooks.configure_user_agent_hooks_for_core,
        "init_agent_perm_params": agent_random.init_perm_params,
        "next_agent_perm_cid": agent_random.next_perm_cid,
        "next_agent_unseen_random_cid": agent_random.next_unseen_random_cid,
        "on_agent_pool_grew": agent_random.on_pool_grew,
        "run_agent_like_and_dig_scores": agent_reaction.like_and_dig_scores,
        "run_agent_like_prob_scores": agent_reaction.like_prob_scores,
        "run_compute_and_cache_cbf_scores": agent_recommendation.compute_and_cache_cbf_scores,
        "run_compute_and_cache_cf_item_scores": agent_recommendation.compute_and_cache_cf_item_scores,
        "run_compute_and_cache_cf_user_scores": agent_recommendation.compute_and_cache_cf_user_scores,
        "UserAgent": agent_model.UserAgent,
        "configure_user_agent_hooks": agent_model.configure_user_agent_hooks,
        "rebuild_cbf_pseudo_batch_state": cbf_state.rebuild_cbf_pseudo_batch,
        "tick_and_refresh_pseudo_if_needed_state": cbf_state.tick_and_refresh_pseudo_if_needed,
        "apply_pending_cf_likes_state": cf_state.apply_pending_cf_likes,
        "rebuild_cf_sync_state": cf_state.rebuild_cf_sync,
        "stage_cf_like_state": cf_state.stage_cf_like,
        "ensure_content_index_impl": content_index.ensure_content_index,
        "get_content_ids_impl": content_index.get_content_ids,
        "get_id2row_impl": content_index.get_id2row,
        "proj_sim_alpha_t_impl": content_index.proj_sim_alpha_t,
        "DomainContent": content_domain.Content,
        "build_id2content": content_domain.build_id2content,
        "build_initial_pool": content_domain.build_initial_pool,
        "replenish_pool": content_domain.replenish_pool,
        "bind_core_runtime": core_helpers.bind_core_runtime,
        "run_cbf_affinity": engine_algorithms.cbf_affinity,
        "run_cf_item_affinity": engine_algorithms.cf_item_affinity,
        "run_cf_item_candidates": engine_algorithms.cf_item_candidates,
        "run_cf_user_affinity": engine_algorithms.cf_user_affinity,
        "run_cf_user_candidates": engine_algorithms.cf_user_candidates,
        "run_pick_contents": engine_algorithms.pick_contents,
        "run_sim_on_active_alpha": engine_algorithms.sim_on_active_alpha,
        "run_vectorized_cbf_faced": engine_algorithms.vectorized_cbf_faced,
        "build_gpu_display_engine_class": engine_gpu.build_gpu_display_engine_class,
        "update_global_buzz_scores_state": global_scores.update_global_buzz_scores,
        "update_global_pop_scores_state": global_scores.update_global_pop_scores,
        "update_global_trend_scores_state": global_scores.update_global_trend_scores,
        "build_world": bootstrap.build_world,
        "setup_logging": log_buffer.setup_logging,
        "load_core_params": load_core_params,
        "build_isc_class": isc_model.build_isc_class,
        "ensure_engine_seen_capacity": engine_state.ensure_seen_capacity,
        "get_engine_seen_mask_row": engine_state.get_seen_mask_row,
        "initialize_engine_state": engine_state.initialize_engine_state,
        "load_engine_agents": engine_state.load_agents,
        "load_engine_contents": engine_state.load_contents,
        "mark_engine_seen": engine_state.mark_seen,
        "mark_engine_seen_batch": engine_state.mark_seen_batch,
        "exclude_ids_from_mask": engine_state.exclude_ids_from_mask,
        "seen_mask_np": engine_state.seen_mask_np,
        "unseen_mask_for_ids": engine_state.unseen_mask_for_ids,
        "clip01": math_utils.clip01,
        "clip11": math_utils.clip11,
        "safe_sim_alpha": math_utils.safe_sim_alpha,
        "sigmoid01_impl": math_utils.sigmoid01_impl,
        "sim_alpha": math_utils.sim_alpha,
        "softmax_arr_impl": math_utils.softmax_arr_impl,
        "build_replenish_map": step_flow.build_replenish_map,
        "run_like_and_dig_batch": reaction_gpu.like_and_dig_batch,
        "setup_torch_bindings": runtime.setup_torch_bindings,
        "apply_step_updates": step_flow.apply_step_updates,
        "prepare_step": step_flow.prepare_step,
        "pick_indices": step_flow.pick_indices,
        "run_reaction_batch": step_flow.run_reaction_batch,
        "gen_agent_g_vector": vector_generators.generate_agent_g_vector,
        "gen_agent_i_vector": vector_generators.generate_agent_i_vector,
        "gen_agent_v_vector": vector_generators.generate_agent_v_vector,
        "gen_content_g_vector_unified": vector_generators.generate_content_g_vector_unified,
        "gen_content_i_vector_unified": vector_generators.generate_content_i_vector_unified,
        "gen_content_vector_unified": vector_generators.generate_content_vector_unified,
        "gen_agent_vector_legacy": vector_generators.generate_agent_vector_legacy,
        "l2_norm": vector_generators.l2,
        "truncnorm": vector_generators.tnorm,
    }
