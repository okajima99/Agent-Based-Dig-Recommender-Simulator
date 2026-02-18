from __future__ import annotations

from typing import Callable

import numpy as np

try:
    from .algorithms_global import sample_from_global_probs
except ImportError:
    from engine.algorithms_global import sample_from_global_probs

def pick_contents(
    engine,
    display_algorithm,
    step,
    isc,
    agents,
    *,
    lambda_cbf: float,
    cbf_face: str,
    unseen_mask_for_ids: Callable[[int, np.ndarray, int], np.ndarray],
    pick_from_probs: Callable[[np.ndarray, np.ndarray], int],
    get_content_ids: Callable[[], np.ndarray | None],
    pop_cache_duration: int,
    trend_cache_duration: int,
    buzz_cache_duration: int,
):
    if display_algorithm == "cbf":
        return engine.cbf_affinity(
            isc,
            agents,
            step,
            lam=lambda_cbf,
            face=cbf_face,
        )

    if display_algorithm == "cf_user":
        return engine.cf_user_affinity(step, isc, agents)

    if display_algorithm == "cf_item":
        return engine.cf_item_affinity(step, isc, agents)

    if display_algorithm == "popularity":
        return sample_from_global_probs(
            engine,
            isc._GLOBAL_POP_SCORES_T,
            agents,
            cache_attr="pop_score_cache",
            timer_attr="pop_cache_timer",
            cache_duration=pop_cache_duration,
            unseen_mask_for_ids=unseen_mask_for_ids,
            pick_from_probs=pick_from_probs,
            get_content_ids=get_content_ids,
            pool_size=len(isc.pool),
        )

    if display_algorithm == "trend":
        return sample_from_global_probs(
            engine,
            isc._GLOBAL_TREND_SCORES_T,
            agents,
            cache_attr="trend_score_cache",
            timer_attr="trend_cache_timer",
            cache_duration=trend_cache_duration,
            unseen_mask_for_ids=unseen_mask_for_ids,
            pick_from_probs=pick_from_probs,
            get_content_ids=get_content_ids,
            pool_size=len(isc.pool),
        )

    if display_algorithm == "buzz":
        return sample_from_global_probs(
            engine,
            isc._GLOBAL_BUZZ_SCORES_T,
            agents,
            cache_attr="buzz_score_cache",
            timer_attr="buzz_cache_timer",
            cache_duration=buzz_cache_duration,
            unseen_mask_for_ids=unseen_mask_for_ids,
            pick_from_probs=pick_from_probs,
            get_content_ids=get_content_ids,
            pool_size=len(isc.pool),
        )

    return None
