from __future__ import annotations

import math

import numpy as np


def init_perm_params(n: int, *, seed: int, random_seed: int):
    rng = np.random.default_rng(int(random_seed) + int(seed))
    start = int(rng.integers(0, n))
    stride = int(rng.integers(1, min(n, 10_000))) * 2 + 1
    while math.gcd(stride, n) != 1:
        stride += 2
        if stride >= n:
            stride = 1
    return start, stride


def on_pool_grew(agent, new_total: int):
    new_total = int(new_total)
    if new_total <= agent._perm_N:
        return
    agent._perm_N = new_total
    agent._perm_start = int(agent._perm_start % new_total)

    stride = int(agent._perm_stride)
    if stride % 2 == 0:
        stride += 1
    while math.gcd(stride, new_total) != 1:
        stride += 2
        if stride >= new_total:
            stride = 1
    agent._perm_stride = stride


def next_perm_cid(agent, *, random_repeat_policy: str, init_perm_params_fn):
    cid = (agent._perm_start + agent._perm_stride * agent._perm_idx) % agent._perm_N
    agent._perm_idx += 1
    if agent._perm_idx >= agent._perm_N:
        if random_repeat_policy == "reset_when_exhausted":
            agent._perm_start, agent._perm_stride = init_perm_params_fn(
                agent._perm_N, seed=agent.id + agent._perm_idx
            )
            agent._perm_idx = 0
        elif random_repeat_policy == "allow_when_exhausted":
            agent._perm_idx = 0
        else:
            raise RuntimeError(f"Agent {agent.id} has no unseen content left.")
    return int(cid)


def next_unseen_random_cid(
    agent,
    num_items: int,
    *,
    seen_mask_np,
    random_repeat_policy: str,
    init_perm_params_fn,
):
    if int(num_items) != int(agent._perm_N):
        on_pool_grew(agent, int(num_items))
    for _ in range(agent._perm_N):
        cid = next_perm_cid(
            agent,
            random_repeat_policy=random_repeat_policy,
            init_perm_params_fn=init_perm_params_fn,
        )
        mask_row = seen_mask_np(agent.id, num_items)
        if not mask_row[cid]:
            return cid
    return next_perm_cid(
        agent,
        random_repeat_policy=random_repeat_policy,
        init_perm_params_fn=init_perm_params_fn,
    )
