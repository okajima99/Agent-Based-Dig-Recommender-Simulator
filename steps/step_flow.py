from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
) -> bool:
    force_random = (step < initial_random_steps) or bool(random_interval_active(step))

    if prev_force_random and (not force_random):
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

    isc.tick_and_refresh_pseudo_if_needed(step, agents)

    if (not force_random) and display_algorithm == "popularity":
        if isc.pop_cache_timer <= 0:
            update_global_pop_scores(isc)
            isc.pop_cache_timer = pop_cache_duration
        else:
            isc.pop_cache_timer -= 1

    if (not force_random) and display_algorithm == "trend":
        if isc.trend_cache_timer <= 0:
            for content in isc.pool:
                content.update_trend_score(step)
            update_global_trend_scores(isc)
            isc.trend_cache_timer = trend_cache_duration
        else:
            isc.trend_cache_timer -= 1

    if (not force_random) and display_algorithm == "buzz":
        if isc.buzz_cache_timer <= 0:
            update_global_buzz_scores(isc, step)
            isc.buzz_cache_timer = buzz_cache_duration
        else:
            isc.buzz_cache_timer -= 1

    if step in replenish_map:
        isc.replenish(replenish_map[step])
        engine.load_contents(isc.pool)
        for agent in agents:
            agent.pop_score_cache = []
            agent.trend_score_cache = []
            agent.buzz_score_cache = []
            agent.pop_cache_timer = 0
            agent.trend_cache_timer = 0
            agent.buzz_cache_timer = 0

    return force_random


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

        liked_flag_int = int(liked_flag_np[agent_id])

        if bool(dig_flag_np[agent_id]):
            j_dim = int(j_dims[agent_id])

            if 0 <= j_dim < len(agent.interests):
                prev_g = agent.interests[j_dim]
                new_g = prev_g + float(dig_g_step)
                delta_g = float(new_g - prev_g)
                agent.interests[j_dim] = new_g
                if delta_g != 0.0:
                    g_updates.append((agent_id, j_dim, delta_g))

            d_v = random_module.uniform(-float(dig_v_range), float(dig_v_range))
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
