from __future__ import annotations

import numpy as np


class Content:
    def __init__(
        self,
        content_id: int,
        vector,
        *,
        content_g_active: int,
        content_i_active: int,
        trend_ema_alpha: float,
        buzz_window: int,
        buzz_gamma: float,
    ):
        self.id = int(content_id)
        self.vector = np.asarray(vector, dtype=np.float32)  # G
        self.i_vector = None  # I
        self.views = 0
        self.likes = 0
        self.liked_by = []
        self.like_history = []

        self.active_idx = np.flatnonzero(self.vector)
        self.g_active = self.vector[self.active_idx]

        self.trend_score = 0.0
        self.trend_ema = 0.0
        self.prev_likes = 0

        self.trend_score_cache = []
        self.buzz_score_cache = []

        self.g_active_count = int(content_g_active)
        self.i_active_count = int(content_i_active)

        self._trend_ema_alpha = float(trend_ema_alpha)
        self._buzz_window = int(buzz_window)
        self._buzz_gamma = float(buzz_gamma)

    def update_trend_score(self, step):
        del step
        delta_likes = self.likes - self.prev_likes
        alpha = self._trend_ema_alpha
        self.trend_ema = alpha * delta_likes + (1.0 - alpha) * self.trend_ema
        self.trend_score = self.trend_ema
        self.prev_likes = self.likes

    def get_buzz_score(self, step):
        score = 0.0
        for step_liked, _uid in self.like_history:
            if step - step_liked <= self._buzz_window:
                score += self._buzz_gamma ** (step - step_liked)
        return score


def build_id2content(pool):
    return {c.id: c for c in pool}


def _create_content(
    content_id: int,
    *,
    content_class,
    generate_content_g_vector_unified,
    generate_content_i_vector_unified,
    num_genres: int,
    num_instinct_dim: int,
    content_g_params,
    content_i_params,
    content_g_active: int,
    content_i_active: int,
    trend_ema_alpha: float,
    buzz_window: int,
    buzz_gamma: float,
):
    g_vec, g_active_idx = generate_content_g_vector_unified(
        length=num_genres,
        params=content_g_params,
        active_count=content_g_active,
    )
    content = content_class(
        content_id,
        g_vec,
        content_g_active=content_g_active,
        content_i_active=content_i_active,
        trend_ema_alpha=trend_ema_alpha,
        buzz_window=buzz_window,
        buzz_gamma=buzz_gamma,
    )

    i_vec, _ = generate_content_i_vector_unified(
        length=num_instinct_dim,
        params=content_i_params,
        active_count=content_i_active,
    )
    content.i_vector = i_vec
    content.active_idx = np.asarray(g_active_idx, dtype=np.int64)
    content.g_active = content.vector[content.active_idx]
    return content


def build_initial_pool(
    *,
    dim: int,
    num_contents: int,
    content_class,
    generate_content_g_vector_unified,
    generate_content_i_vector_unified,
    content_g_params,
    content_i_params,
    content_g_active: int,
    content_i_active: int,
    num_instinct_dim: int,
    trend_ema_alpha: float,
    buzz_window: int,
    buzz_gamma: float,
):
    pool = []
    for content_id in range(int(num_contents)):
        content = _create_content(
            content_id,
            content_class=content_class,
            generate_content_g_vector_unified=generate_content_g_vector_unified,
            generate_content_i_vector_unified=generate_content_i_vector_unified,
            num_genres=int(dim),
            num_instinct_dim=int(num_instinct_dim),
            content_g_params=content_g_params,
            content_i_params=content_i_params,
            content_g_active=int(content_g_active),
            content_i_active=int(content_i_active),
            trend_ema_alpha=float(trend_ema_alpha),
            buzz_window=int(buzz_window),
            buzz_gamma=float(buzz_gamma),
        )
        pool.append(content)
    return pool


def replenish_pool(
    pool,
    n_new: int,
    *,
    content_class,
    generate_content_g_vector_unified,
    generate_content_i_vector_unified,
    num_genres: int,
    num_instinct_dim: int,
    content_g_params,
    content_i_params,
    content_g_active: int,
    content_i_active: int,
    trend_ema_alpha: float,
    buzz_window: int,
    buzz_gamma: float,
):
    n_new = int(n_new)
    if n_new <= 0:
        return

    start_id = len(pool)
    for offset in range(n_new):
        content = _create_content(
            start_id + offset,
            content_class=content_class,
            generate_content_g_vector_unified=generate_content_g_vector_unified,
            generate_content_i_vector_unified=generate_content_i_vector_unified,
            num_genres=int(num_genres),
            num_instinct_dim=int(num_instinct_dim),
            content_g_params=content_g_params,
            content_i_params=content_i_params,
            content_g_active=int(content_g_active),
            content_i_active=int(content_i_active),
            trend_ema_alpha=float(trend_ema_alpha),
            buzz_window=int(buzz_window),
            buzz_gamma=float(buzz_gamma),
        )
        pool.append(content)
