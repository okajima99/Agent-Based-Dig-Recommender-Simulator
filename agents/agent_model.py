from __future__ import annotations

import numpy as np

_HOOKS = {}


def configure_user_agent_hooks(**hooks):
    _HOOKS.update(hooks)


def _hook(name: str):
    if name not in _HOOKS:
        raise RuntimeError(f"UserAgent hook '{name}' is not configured")
    return _HOOKS[name]


class UserAgent:
    def __init__(self, id):
        self.id = id
        num_genres = _hook("NUM_GENRES")
        num_instinct_dim = _hook("NUM_INSTINCT_DIM")
        agent_g_params = _hook("Agent_G_PARAMS")
        agent_v_params = _hook("Agent_V_PARAMS")
        agent_i_params = _hook("Agent_I_PARAMS")

        self.interests = _hook("generate_agent_g_vector")(
            num_genres,
            agent_g_params["mu"],
            agent_g_params["sigma"],
            agent_g_params["norm_mu"],
            agent_g_params["norm_sigma"],
        )
        self.V = _hook("generate_agent_v_vector")(
            num_genres,
            agent_v_params["mu"],
            agent_v_params["sigma"],
            agent_v_params["norm_mu"],
            agent_v_params["norm_sigma"],
        )
        self.I = _hook("generate_agent_i_vector")(
            num_instinct_dim,
            agent_i_params["mu"],
            agent_i_params["sigma"],
            agent_i_params["norm_mu"],
            agent_i_params["norm_sigma"],
        )
        self.initial_vector = self.interests[:]
        self.initial_V = self.V[:]
        self.initial_I = self.I[:]

        self.like_history_G = []
        self.like_history_I = []
        self.total_likes = 0
        self.social_alpha = float(np.random.uniform(0.0, 1.0))

        self._perm_N = _hook("NUM_CONTENTS")
        self._perm_start, self._perm_stride = self._init_perm_params(self._perm_N, seed=id)
        self._perm_idx = 0

        self.pseudo_g_cache = None
        self.pseudo_i_cache = None

        self.cbf_score_cache = []
        self.cf_user_score_cache = []
        self.cf_item_score_cache = []
        self.cf_cache_timer = 0
        self.pop_score_cache = []
        self.trend_score_cache = []
        self.buzz_score_cache = []
        self.pop_cache_timer = 0
        self.trend_cache_timer = 0
        self.buzz_cache_timer = 0

        self.lh_steps = np.empty(0, dtype=np.int32)
        self.lh_ridx = np.empty(0, dtype=np.int64)

    def on_pool_grew(self, new_total: int):
        return _hook("on_agent_pool_grew")(self, new_total)

    def _init_perm_params(self, N: int, seed: int):
        return _hook("init_agent_perm_params")(N, seed=seed, random_seed=_hook("RANDOM_SEED"))

    def _next_perm_cid(self):
        return _hook("next_agent_perm_cid")(
            self,
            random_repeat_policy=_hook("RANDOM_REPEAT_POLICY"),
            init_perm_params_fn=lambda n, seed: _hook("init_agent_perm_params")(
                n, seed=seed, random_seed=_hook("RANDOM_SEED")
            ),
        )

    def next_unseen_random_cid(self, num_items):
        return _hook("next_agent_unseen_random_cid")(
            self,
            num_items,
            seen_mask_np=_hook("seen_mask_np"),
            random_repeat_policy=_hook("RANDOM_REPEAT_POLICY"),
            init_perm_params_fn=lambda n, seed: _hook("init_agent_perm_params")(
                n, seed=seed, random_seed=_hook("RANDOM_SEED")
            ),
        )

    def like_prob_scores(self, uid: int, content):
        return _hook("run_agent_like_prob_scores")(
            self,
            uid,
            content,
            sim_on_active_alpha=_hook("sim_on_active_alpha"),
            sim_alpha=_hook("sim_alpha"),
            agent_alpha=_hook("AGENT_ALPHA"),
            like_w_cg=_hook("LIKE_W_CG"),
            like_w_cv=_hook("LIKE_W_CV"),
            like_w_ci=_hook("LIKE_W_CI"),
            sigmoid01=_hook("sigmoid01"),
            logit_k=_hook("LOGIT_K"),
            logit_x0=_hook("LOGIT_X0"),
            like_divisor=_hook("LIKE_DIVISOR"),
            random_module=_hook("random_module"),
        )

    def like_and_dig_scores(self, uid, content):
        return _hook("run_agent_like_and_dig_scores")(
            self,
            uid,
            content,
            sigmoid01=_hook("sigmoid01"),
            mu0=_hook("MU0"),
            mu_slope=_hook("MU_SLOPE"),
            mu_alpha_c=_hook("MU_ALPHA_C"),
            mu_beta_v=_hook("MU_BETA_V"),
            sigma0=_hook("SIGMA0"),
            sigma_lamda=_hook("SIGMA_LAMDA"),
            sigma_alpha_c=_hook("SIGMA_ALPHA_C"),
            sigma_beta_v=_hook("SIGMA_BETA_V"),
            a0=_hook("A0"),
            a_lamda=_hook("A_LAMDA"),
            a_alpha_c=_hook("A_ALPHA_C"),
            a_beta_v=_hook("A_BETA_V"),
            dig_logit_k=_hook("DIG_LOGIT_K"),
            dig_logit_x0=_hook("DIG_LOGIT_X0"),
            dig_divisor=_hook("DIG_DIVISOR"),
        )

    def update_like_history_GI(self, g_vec, i_vec, step):
        return _hook("update_agent_like_history_gi")(self, g_vec, i_vec, step)

    def _compute_pseudo(self, history, current_step, dim_hint=None):
        del dim_hint
        return _hook("compute_agent_pseudo")(
            history,
            current_step,
            pseudo_history_window_steps=_hook("PSEUDO_HISTORY_WINDOW_STEPS"),
            pseudo_discount_gamma=_hook("PSEUDO_DISCOUNT_GAMMA"),
        )

    def compute_and_cache_cbf_scores(self, step, content_pool):
        return _hook("run_compute_and_cache_cbf_scores")(
            self,
            step,
            content_pool,
            ensure_content_index=_hook("ensure_content_index"),
            seen_mask_np=_hook("seen_mask_np"),
            vectorized_cbf_faced=_hook("vectorized_cbf_faced"),
            lambda_cbf=_hook("LAMBDA_CBF"),
            cbf_face=_hook("CBF_FACE"),
            cbf_w_g=_hook("CBF_W_G"),
            cbf_w_i=_hook("CBF_W_I"),
            cbf_top_k=_hook("CBF_TOP_K"),
        )

    def compute_and_cache_cf_user_scores(self, step, agents, isc_obj, face):
        return _hook("run_compute_and_cache_cf_user_scores")(
            self,
            step,
            agents,
            isc_obj,
            face,
            cf_user_candidates=_hook("cf_user_candidates"),
        )

    def compute_and_cache_cf_item_scores(self, step, agents, isc_obj, face):
        return _hook("run_compute_and_cache_cf_item_scores")(
            self,
            step,
            agents,
            isc_obj,
            face,
            cf_item_candidates=_hook("cf_item_candidates"),
        )

    def compute_pseudo_vector_G(self, step):
        return _hook("compute_agent_pseudo_vector_g")(
            self,
            step,
            compute_pseudo_fn=lambda history, s: self._compute_pseudo(history, s, dim_hint=_hook("NUM_GENRES")),
        )

    def compute_pseudo_vector_I(self, step):
        return _hook("compute_agent_pseudo_vector_i")(
            self,
            step,
            compute_pseudo_fn=lambda history, s: self._compute_pseudo(
                history, s, dim_hint=_hook("NUM_INSTINCT_DIM")
            ),
        )
