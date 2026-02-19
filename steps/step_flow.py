from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cache_needed_for_algorithm(cache_name: str, display_algorithm: str) -> bool:
    algo = str(display_algorithm).lower()
    if cache_name == "cf_matrix":
        return algo in {"cf_user", "cf_item"}
    if cache_name == "cbf_pseudo":
        return algo == "cbf"
    if cache_name == "pop":
        return algo == "popularity"
    if cache_name == "trend":
        return algo == "trend"
    if cache_name == "buzz":
        return algo == "buzz"
    return True


def _log_cache_reset_with_state(
    *,
    logger,
    step: int,
    cache_name: str,
    display_algorithm: str,
    reason: str,
    timer_before: int | None,
    random_interval_active: bool,
    isc,
    include_snapshot_state: bool,
) -> None:
    if logger is None or not getattr(logger, "enabled", False):
        return
    if not _cache_needed_for_algorithm(cache_name, display_algorithm):
        return

    event_id = logger.log_cache_refresh_event(
        step=step,
        cache_name=cache_name,
        reason=reason,
        action="reset",
        timer_before=timer_before,
        timer_after=0,
        random_interval_active=random_interval_active,
    )

    if cache_name == "cf_matrix":
        uv = getattr(isc, "UV_matrix", None) if include_snapshot_state else None
        logger.log_cache_state_cf(event_id=event_id, step=step, uv_matrix=uv)
        return

    if cache_name == "cbf_pseudo":
        g_t = getattr(isc, "cbf_pseudo_g_t", None) if include_snapshot_state else None
        i_t = getattr(isc, "cbf_pseudo_i_t", None) if include_snapshot_state else None
        logger.log_cache_state_cbf(event_id=event_id, step=step, pseudo_g_t=g_t, pseudo_i_t=i_t)
        return

    if cache_name in {"pop", "trend", "buzz"}:
        if include_snapshot_state:
            if cache_name == "pop":
                scores = getattr(isc, "_GLOBAL_POP_SCORES_T", None)
            elif cache_name == "trend":
                scores = getattr(isc, "_GLOBAL_TREND_SCORES_T", None)
            else:
                scores = getattr(isc, "_GLOBAL_BUZZ_SCORES_T", None)
            if scores is None:
                content_ids = []
            else:
                content_ids = [int(c.id) for c in getattr(isc, "pool", [])]
        else:
            scores = None
            content_ids = []
        logger.log_cache_state_global(
            event_id=event_id,
            step=step,
            cache_name=cache_name,
            content_ids=content_ids,
            scores=scores,
        )


def build_replenish_map(
    *,
    replenish_every: int,
    replenish_count: int,
    replenish_start_step: int,
    replenish_end_step: int,
    max_steps: int,
):
    replenish_map = {}
    if (replenish_every > 0) and (replenish_count > 0):
        start = max(0, int(replenish_start_step))
        end = min(int(replenish_end_step), int(max_steps))
        step_point = start
        while step_point <= end:
            replenish_map[step_point] = int(replenish_count)
            step_point += int(replenish_every)
    return replenish_map


def prepare_step(
    step: int,
    *,
    prev_force_random: bool,
    isc,
    agents,
    display_algorithm: str,
    initial_random_steps: int,
    random_interval_active,
    pop_cache_duration: int,
    trend_cache_duration: int,
    buzz_cache_duration: int,
    replenish_map,
    engine,
    update_global_pop_scores,
    update_global_trend_scores,
    update_global_buzz_scores,
    logger=None,
) -> tuple[bool, bool]:
    random_interval_active_flag = bool(random_interval_active(step))
    force_random = (step < initial_random_steps) or random_interval_active_flag

    if prev_force_random and (not force_random):
        pseudo_before = int(getattr(isc, "pseudo_cache_timer", 0))
        cf_before = int(getattr(isc, "cf_matrix_cache_timer", 0))
        pop_before = int(getattr(isc, "pop_cache_timer", 0))
        trend_before = int(getattr(isc, "trend_cache_timer", 0))
        buzz_before = int(getattr(isc, "buzz_cache_timer", 0))
        isc.pseudo_cache_timer = 0
        isc.cf_matrix_cache_timer = 0
        isc.pop_cache_timer = 0
        isc.trend_cache_timer = 0
        isc.buzz_cache_timer = 0
        for agent in agents:
            agent.pop_score_cache = []
            agent.trend_score_cache = []
            agent.buzz_score_cache = []
            agent.pop_cache_timer = 0
            agent.trend_cache_timer = 0
            agent.buzz_cache_timer = 0

        _log_cache_reset_with_state(
            logger=logger,
            step=step,
            cache_name="cbf_pseudo",
            display_algorithm=display_algorithm,
            reason="random_window_exit",
            timer_before=pseudo_before,
            random_interval_active=random_interval_active_flag,
            isc=isc,
            include_snapshot_state=True,
        )
        _log_cache_reset_with_state(
            logger=logger,
            step=step,
            cache_name="cf_matrix",
            display_algorithm=display_algorithm,
            reason="random_window_exit",
            timer_before=cf_before,
            random_interval_active=random_interval_active_flag,
            isc=isc,
            include_snapshot_state=True,
        )
        _log_cache_reset_with_state(
            logger=logger,
            step=step,
            cache_name="pop",
            display_algorithm=display_algorithm,
            reason="random_window_exit",
            timer_before=pop_before,
            random_interval_active=random_interval_active_flag,
            isc=isc,
            include_snapshot_state=True,
        )
        _log_cache_reset_with_state(
            logger=logger,
            step=step,
            cache_name="trend",
            display_algorithm=display_algorithm,
            reason="random_window_exit",
            timer_before=trend_before,
            random_interval_active=random_interval_active_flag,
            isc=isc,
            include_snapshot_state=True,
        )
        _log_cache_reset_with_state(
            logger=logger,
            step=step,
            cache_name="buzz",
            display_algorithm=display_algorithm,
            reason="random_window_exit",
            timer_before=buzz_before,
            random_interval_active=random_interval_active_flag,
            isc=isc,
            include_snapshot_state=True,
        )

    isc.tick_and_refresh_pseudo_if_needed(step, agents)

    if (not force_random) and display_algorithm == "popularity":
        if isc.pop_cache_timer <= 0:
            timer_before = int(getattr(isc, "pop_cache_timer", 0))
            update_global_pop_scores(
                isc,
                step=step,
                logger=logger,
                cache_reason="ttl",
                timer_before=timer_before,
                timer_after=int(pop_cache_duration),
                random_interval_active=random_interval_active_flag,
            )
            isc.pop_cache_timer = pop_cache_duration
        else:
            isc.pop_cache_timer -= 1

    if (not force_random) and display_algorithm == "trend":
        if isc.trend_cache_timer <= 0:
            for content in isc.pool:
                content.update_trend_score(step)
            timer_before = int(getattr(isc, "trend_cache_timer", 0))
            update_global_trend_scores(
                isc,
                step=step,
                logger=logger,
                cache_reason="ttl",
                timer_before=timer_before,
                timer_after=int(trend_cache_duration),
                random_interval_active=random_interval_active_flag,
            )
            isc.trend_cache_timer = trend_cache_duration
        else:
            isc.trend_cache_timer -= 1

    if (not force_random) and display_algorithm == "buzz":
        if isc.buzz_cache_timer <= 0:
            timer_before = int(getattr(isc, "buzz_cache_timer", 0))
            update_global_buzz_scores(
                isc,
                step,
                logger=logger,
                cache_reason="ttl",
                timer_before=timer_before,
                timer_after=int(buzz_cache_duration),
                random_interval_active=random_interval_active_flag,
            )
            isc.buzz_cache_timer = buzz_cache_duration
        else:
            isc.buzz_cache_timer -= 1

    if step in replenish_map:
        pseudo_before = int(getattr(isc, "pseudo_cache_timer", 0))
        cf_before = int(getattr(isc, "cf_matrix_cache_timer", 0))
        pop_before = int(getattr(isc, "pop_cache_timer", 0))
        trend_before = int(getattr(isc, "trend_cache_timer", 0))
        buzz_before = int(getattr(isc, "buzz_cache_timer", 0))
        isc.replenish(replenish_map[step])
        engine.load_contents(isc.pool)
        for agent in agents:
            agent.pop_score_cache = []
            agent.trend_score_cache = []
            agent.buzz_score_cache = []
            agent.pop_cache_timer = 0
            agent.trend_cache_timer = 0
            agent.buzz_cache_timer = 0

        _log_cache_reset_with_state(
            logger=logger,
            step=step,
            cache_name="cbf_pseudo",
            display_algorithm=display_algorithm,
            reason="replenish",
            timer_before=pseudo_before,
            random_interval_active=random_interval_active_flag,
            isc=isc,
            include_snapshot_state=False,
        )
        _log_cache_reset_with_state(
            logger=logger,
            step=step,
            cache_name="cf_matrix",
            display_algorithm=display_algorithm,
            reason="replenish",
            timer_before=cf_before,
            random_interval_active=random_interval_active_flag,
            isc=isc,
            include_snapshot_state=False,
        )
        _log_cache_reset_with_state(
            logger=logger,
            step=step,
            cache_name="pop",
            display_algorithm=display_algorithm,
            reason="replenish",
            timer_before=pop_before,
            random_interval_active=random_interval_active_flag,
            isc=isc,
            include_snapshot_state=False,
        )
        _log_cache_reset_with_state(
            logger=logger,
            step=step,
            cache_name="trend",
            display_algorithm=display_algorithm,
            reason="replenish",
            timer_before=trend_before,
            random_interval_active=random_interval_active_flag,
            isc=isc,
            include_snapshot_state=False,
        )
        _log_cache_reset_with_state(
            logger=logger,
            step=step,
            cache_name="buzz",
            display_algorithm=display_algorithm,
            reason="replenish",
            timer_before=buzz_before,
            random_interval_active=random_interval_active_flag,
            isc=isc,
            include_snapshot_state=False,
        )

    return force_random, random_interval_active_flag


def pick_indices(
    step: int,
    *,
    display_algorithm: str,
    force_random: bool,
    isc,
    agents,
    engine,
    num_agents: int,
    device,
    torch_module,
):
    if display_algorithm == "random" or force_random:
        picked_idx_cpu = np.zeros(num_agents, dtype=np.int64)
        for agent_id, agent in enumerate(agents):
            picked_idx_cpu[agent_id] = int(agent.next_unseen_random_cid(len(isc.pool)))
        return torch_module.from_numpy(picked_idx_cpu).to(device, dtype=torch_module.long)

    picked_idx = engine.pick_contents(
        display_algorithm,
        step,
        isc,
        agents,
    )
    if picked_idx is None:
        raise ValueError(f"Unsupported DISPLAY_ALGORITHM for GPU path: {display_algorithm}")
    return picked_idx


@dataclass(slots=True)
class ReactionBatchResult:
    like_flags: object
    dig_flags: object
    j_dims: object


def run_reaction_batch(
    step: int,
    *,
    isc,
    agents,
    picked_idx_t,
    engine,
    device,
    torch_module,
) -> ReactionBatchResult:
    del device
    del torch_module
    like_flags, dig_flags, j_dims, _, _ = engine.like_and_dig_batch(
        step,
        isc,
        agents,
        picked_idx_t,
    )

    return ReactionBatchResult(
        like_flags=like_flags,
        dig_flags=dig_flags,
        j_dims=j_dims,
    )


def apply_step_updates(
    step: int,
    *,
    isc,
    agents,
    engine,
    picked_idx_t,
    reaction_result,
    device,
    torch_module,
    num_agents: int,
    dig_g_step: float,
    dig_v_range: float,
    cf_history_window_steps: int,
    get_id2row,
    random_module,
    logger=None,
    random_interval_active: bool = False,
):
    picked_idx_cpu = picked_idx_t.detach().cpu().numpy()
    engine.mark_seen_batch(torch_module.arange(num_agents, device=device, dtype=torch_module.long), picked_idx_t)

    liked_flag_np = np.asarray(reaction_result.like_flags, dtype=np.int64)
    dig_flag_np = np.asarray(reaction_result.dig_flags, dtype=np.int64)
    j_dims = np.asarray(reaction_result.j_dims, dtype=np.int64)

    g_updates = []
    v_updates = []

    liked_mask_idx = np.where(liked_flag_np == 1)[0]
    liked_cids = picked_idx_cpu[liked_mask_idx] if liked_mask_idx.size > 0 else np.array([], dtype=np.int64)
    if liked_cids.size > 0:
        like_counts = np.bincount(liked_cids, minlength=len(isc.pool))
        nz = np.nonzero(like_counts)[0]
        for cid in nz:
            isc.pool[int(cid)].likes += int(like_counts[cid])

    for agent_id, cid in enumerate(picked_idx_cpu):
        content = isc.pool[int(cid)]
        agent = agents[agent_id]
        g_pre = np.asarray(agent.interests, dtype=np.float32).copy()
        v_pre = np.asarray(agent.V, dtype=np.float32).copy()
        i_pre = np.asarray(agent.I, dtype=np.float32).copy()
        dig_dim = int(j_dims[agent_id])
        d_v_logged = 0.0

        liked_flag_int = int(liked_flag_np[agent_id])

        if bool(dig_flag_np[agent_id]):
            j_dim = dig_dim

            if 0 <= j_dim < len(agent.interests):
                prev_g = agent.interests[j_dim]
                new_g = prev_g + float(dig_g_step)
                delta_g = float(new_g - prev_g)
                agent.interests[j_dim] = new_g
                if delta_g != 0.0:
                    g_updates.append((agent_id, j_dim, delta_g))

            d_v = random_module.uniform(-float(dig_v_range), float(dig_v_range))
            d_v_logged = float(d_v)
            if 0 <= j_dim < len(agent.V):
                prev_v = agent.V[j_dim]
                new_v = prev_v + d_v
                delta_v = float(new_v - prev_v)
                agent.V[j_dim] = new_v
                if delta_v != 0.0:
                    v_updates.append((agent_id, j_dim, delta_v))

        if liked_flag_int:
            content.liked_by.append(agent_id)
            content.like_history.append((step, agent_id))
            agent.total_likes += 1

            g_vec = np.asarray(content.vector, dtype=np.float32)
            i_vec = np.asarray(content.i_vector, dtype=np.float32)
            agent.update_like_history_GI(g_vec, i_vec, step)
            agent.pseudo_g_cache = None
            agent.pseudo_i_cache = None

            id2row = get_id2row()
            ridx = id2row.get(int(content.id)) if (id2row is not None) else None
            if ridx is not None:
                agent.lh_steps = np.append(agent.lh_steps, np.int32(step))
                agent.lh_ridx = np.append(agent.lh_ridx, np.int64(ridx))
                if agent.lh_steps.size:
                    mask = (step - agent.lh_steps) <= cf_history_window_steps
                    agent.lh_steps = agent.lh_steps[mask]
                    agent.lh_ridx = agent.lh_ridx[mask]
                isc.stage_cf_like(agent.id, content.id, step)

        if logger is not None and getattr(logger, "enabled", False):
            logger.log_impression_pre(
                step=step,
                agent_id=agent_id,
                content_id=int(content.id),
                g_pre=g_pre,
                v_pre=v_pre,
                i_pre=i_pre,
                like_flag=liked_flag_int,
                dig_flag=int(dig_flag_np[agent_id]),
                dig_dim=dig_dim,
                d_v=float(d_v_logged),
                random_interval_active=bool(random_interval_active),
            )

    try:
        view_counts = np.bincount(picked_idx_cpu, minlength=len(isc.pool))
        nz = np.nonzero(view_counts)[0]
        for cid in nz:
            isc.pool[int(cid)].views += int(view_counts[cid])
    except Exception:
        for cid in picked_idx_cpu:
            isc.pool[int(cid)].views += 1

    if g_updates:
        a_idx = torch_module.tensor([t[0] for t in g_updates], dtype=torch_module.long, device=device)
        d_idx = torch_module.tensor([t[1] for t in g_updates], dtype=torch_module.long, device=device)
        delta = torch_module.tensor([t[2] for t in g_updates], dtype=torch_module.float32, device=device)
        engine.Ug.index_put_((a_idx, d_idx), delta, accumulate=True)
        engine.ug_matrix_t.index_put_((a_idx, d_idx), delta, accumulate=True)

    if v_updates:
        a_idx = torch_module.tensor([t[0] for t in v_updates], dtype=torch_module.long, device=device)
        d_idx = torch_module.tensor([t[1] for t in v_updates], dtype=torch_module.long, device=device)
        delta = torch_module.tensor([t[2] for t in v_updates], dtype=torch_module.float32, device=device)
        engine.Uv.index_put_((a_idx, d_idx), delta, accumulate=True)
        engine.uv_matrix_t.index_put_((a_idx, d_idx), delta, accumulate=True)
