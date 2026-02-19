from __future__ import annotations


def build_isc_class(
    *,
    context: dict[str, object],
    domain_content_cls,
    build_initial_pool_fn,
    build_id2content_fn,
    replenish_pool_fn,
    ensure_content_index_fn,
    apply_pending_cf_likes_state_fn,
    rebuild_cf_sync_state_fn,
    rebuild_cbf_pseudo_batch_state_fn,
    stage_cf_like_state_fn,
    tick_and_refresh_pseudo_if_needed_state_fn,
    require_content_ids_fn,
    get_id2row_fn,
    random_interval_active_fn,
):
    def _ctx(name: str):
        return context[name]

    class ISC:
        Content = domain_content_cls

        def __init__(self, dim, content_g_params, num_contents):
            self.pool = build_initial_pool_fn(
                dim=dim,
                num_contents=num_contents,
                content_class=self.Content,
                generate_content_g_vector_unified=_ctx("generate_content_g_vector_unified"),
                generate_content_i_vector_unified=_ctx("generate_content_i_vector_unified"),
                content_g_params=content_g_params,
                content_i_params=_ctx("content_I_PARAMS"),
                content_g_active=_ctx("content_g_active"),
                content_i_active=_ctx("content_i_active"),
                num_instinct_dim=_ctx("NUM_INSTINCT_DIM"),
                trend_ema_alpha=_ctx("TREND_EMA_ALPHA"),
                buzz_window=_ctx("BUZZ_WINDOW"),
                buzz_gamma=_ctx("BUZZ_GAMMA"),
            )
            self.id2content = build_id2content_fn(self.pool)
            ensure_content_index_fn(self.pool)

            self.pop_cache_timer = 0
            self.trend_cache_timer = 0
            self.buzz_cache_timer = 0
            self.pseudo_cache_timer = 0
            self.cf_matrix_cache_timer = 0
            self.pending_cf_likes = []

            self.user_likes = {}
            self.item_liked_by = {}
            self.cf_user_neighbors = None
            self.cf_item_neighbors = None
            self.user_like_w = {}
            self.item_liked_by_w = {}
            self.UV_matrix = None
            self.VU_matrix = None
            self.cf_last_built_step = -1
            self.cf_user_row_norm_t = None
            self.cf_item_row_norm_t = None
            self.cbf_pseudo_g_t = None
            self.cbf_pseudo_i_t = None

        def display_random_to_agent(self, agent, step):
            del step
            cid = agent.next_unseen_random_cid(len(self.pool))
            return self.id2content[int(cid)]

        def _apply_pending_cf_likes(self):
            apply_pending_cf_likes_state_fn(
                self,
                require_content_ids=require_content_ids_fn,
            )

        def _rebuild_cf_sync(
            self,
            step: int,
            agents,
            *,
            logger=None,
            cache_reason: str = "ttl",
            timer_before: int | None = None,
            random_interval_active: bool = False,
        ):
            self._apply_pending_cf_likes()
            rebuild_cf_sync_state_fn(
                self,
                step,
                agents,
                device=_ctx("DEVICE"),
                cf_history_window_steps=_ctx("CF_HISTORY_WINDOW_STEPS"),
                cf_discount_gamma=_ctx("CF_DISCOUNT_GAMMA"),
                cf_cache_duration=_ctx("CF_CACHE_DURATION"),
                logger=logger,
                cache_reason=cache_reason,
                timer_before=timer_before,
                random_interval_active=random_interval_active,
            )

        def _rebuild_cbf_pseudo_batch(
            self,
            step: int,
            agents,
            *,
            logger=None,
            cache_reason: str = "ttl",
            timer_before: int | None = None,
            timer_after: int | None = None,
            random_interval_active: bool = False,
        ):
            rebuild_cbf_pseudo_batch_state_fn(
                self,
                step,
                agents,
                num_genres=_ctx("NUM_GENRES"),
                num_instinct_dim=_ctx("NUM_INSTINCT_DIM"),
                to_t=_ctx("_to_t"),
                device=_ctx("DEVICE"),
                logger=logger,
                cache_reason=cache_reason,
                timer_before=timer_before,
                timer_after=timer_after,
                random_interval_active=random_interval_active,
            )

        def stage_cf_like(self, uid: int, cid: int, step_like: int):
            stage_cf_like_state_fn(
                self,
                uid,
                cid,
                step_like,
                get_id2row=get_id2row_fn,
            )

        def replenish(self, n_new: int):
            if n_new <= 0:
                return
            replenish_pool_fn(
                self.pool,
                n_new,
                content_class=self.Content,
                generate_content_g_vector_unified=_ctx("generate_content_g_vector_unified"),
                generate_content_i_vector_unified=_ctx("generate_content_i_vector_unified"),
                num_genres=_ctx("NUM_GENRES"),
                num_instinct_dim=_ctx("NUM_INSTINCT_DIM"),
                content_g_params=_ctx("content_G_PARAMS"),
                content_i_params=_ctx("content_I_PARAMS"),
                content_g_active=_ctx("content_g_active"),
                content_i_active=_ctx("content_i_active"),
                trend_ema_alpha=_ctx("TREND_EMA_ALPHA"),
                buzz_window=_ctx("BUZZ_WINDOW"),
                buzz_gamma=_ctx("BUZZ_GAMMA"),
            )

            self.id2content = build_id2content_fn(self.pool)
            ensure_content_index_fn(self.pool)
            self.pop_cache_timer = self.trend_cache_timer = self.buzz_cache_timer = 0
            self.pseudo_cache_timer = 0
            self.cf_matrix_cache_timer = 0

        def tick_and_refresh_pseudo_if_needed(self, step, agents):
            tick_and_refresh_pseudo_if_needed_state_fn(
                self,
                step,
                agents,
                initial_random_steps=_ctx("INITIAL_RANDOM_STEPS"),
                random_interval_active=random_interval_active_fn,
                display_algorithm=_ctx("DISPLAY_ALGORITHM"),
                pseudo_cache_duration=_ctx("PSEUDO_CACHE_DURATION"),
                apply_pending_cf_likes=self._apply_pending_cf_likes,
                rebuild_cf_sync=self._rebuild_cf_sync,
                rebuild_cbf_pseudo_batch=self._rebuild_cbf_pseudo_batch,
                logger=context.get("analysis_logger"),
            )

    return ISC
