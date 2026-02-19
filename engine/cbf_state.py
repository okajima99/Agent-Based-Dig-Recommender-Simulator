from __future__ import annotations

import numpy as np
import torch


def rebuild_cbf_pseudo_batch(
    isc_obj,
    step: int,
    agents,
    *,
    num_genres: int,
    num_instinct_dim: int,
    to_t,
    device: torch.device,
    logger=None,
    cache_reason: str = "ttl",
    timer_before: int | None = None,
    timer_after: int | None = None,
    random_interval_active: bool = False,
):
    user_count = len(agents)
    if user_count == 0:
        isc_obj.cbf_pseudo_g_t = None
        isc_obj.cbf_pseudo_i_t = None
        return

    pseudo_g = np.zeros((user_count, num_genres), dtype=np.float32)
    pseudo_i = np.zeros((user_count, num_instinct_dim), dtype=np.float32)
    for i, agent in enumerate(agents):
        pg = agent.compute_pseudo_vector_G(step)
        pi = agent.compute_pseudo_vector_I(step)
        if pg is not None:
            pseudo_g[i, :min(num_genres, len(pg))] = pg[:min(num_genres, len(pg))]
        if pi is not None:
            pseudo_i[i, :min(num_instinct_dim, len(pi))] = pi[:min(num_instinct_dim, len(pi))]

    isc_obj.cbf_pseudo_g_t = to_t(pseudo_g, device=device, dtype=torch.float32)
    isc_obj.cbf_pseudo_i_t = to_t(pseudo_i, device=device, dtype=torch.float32)

    if logger is not None and getattr(logger, "enabled", False):
        event_id = logger.log_cache_refresh_event(
            step=int(step),
            cache_name="cbf_pseudo",
            reason=cache_reason,
            action="rebuild",
            timer_before=timer_before,
            timer_after=timer_after,
            random_interval_active=random_interval_active,
        )
        logger.log_cache_state_cbf(
            event_id=event_id,
            step=int(step),
            pseudo_g_t=isc_obj.cbf_pseudo_g_t,
            pseudo_i_t=isc_obj.cbf_pseudo_i_t,
        )


def tick_and_refresh_pseudo_if_needed(
    isc_obj,
    step: int,
    agents,
    *,
    initial_random_steps: int,
    random_interval_active,
    display_algorithm: str,
    pseudo_cache_duration: int,
    apply_pending_cf_likes,
    rebuild_cf_sync,
    rebuild_cbf_pseudo_batch,
    logger=None,
):
    if (step < int(initial_random_steps)) or random_interval_active(step):
        return

    need_pseudo = display_algorithm == "cbf"
    need_cf = display_algorithm in {"cf_user", "cf_item"}
    if not need_pseudo and not need_cf:
        return

    if need_pseudo:
        if isc_obj.pseudo_cache_timer > 0:
            isc_obj.pseudo_cache_timer -= 1
        else:
            timer_before = int(getattr(isc_obj, "pseudo_cache_timer", 0))
            for agent in agents:
                agent.pseudo_g_cache = None
                agent.pseudo_i_cache = None
                agent.cbf_score_cache = []
            rebuild_cbf_pseudo_batch(
                step,
                agents,
                logger=logger,
                cache_reason="ttl",
                timer_before=timer_before,
                timer_after=int(pseudo_cache_duration),
                random_interval_active=False,
            )
            isc_obj.pseudo_cache_timer = int(pseudo_cache_duration)

    if not need_cf:
        return

    if isc_obj.cf_matrix_cache_timer > 0:
        isc_obj.cf_matrix_cache_timer -= 1
    if isc_obj.cf_matrix_cache_timer > 0:
        return

    timer_before = int(getattr(isc_obj, "cf_matrix_cache_timer", 0))
    apply_pending_cf_likes()
    rebuild_cf_sync(
        step,
        agents,
        logger=logger,
        cache_reason="ttl",
        timer_before=timer_before,
        random_interval_active=False,
    )
