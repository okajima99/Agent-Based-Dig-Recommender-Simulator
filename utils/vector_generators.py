from __future__ import annotations

import math
import random

import numpy as np


def tnorm(mu: float, sigma: float, lo: float, hi: float) -> float:
    while True:
        v = random.gauss(mu, sigma)
        if lo <= v <= hi:
            return v


def l2(x) -> float:
    return math.sqrt(sum(t * t for t in x)) + 1e-12


def generate_agent_vector_legacy(length, mu_e, sig_e, norm_mu, norm_sig):
    vec = [tnorm(mu_e, sig_e, 0.0, 1.0) for _ in range(length)]
    base = l2(vec)
    target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
    s = target / base
    return [min(1.0, max(0.0, x * s)) for x in vec]


def generate_agent_g_vector(length, mu_e, sig_e, norm_mu, norm_sig, *, mode: str):
    mode_l = str(mode).lower()
    if mode_l == "legacy":
        return generate_agent_vector_legacy(length, mu_e, sig_e, norm_mu, norm_sig)
    if mode_l == "element":
        vec = [tnorm(mu_e, sig_e, 0.0, 1.0) for _ in range(length)]
        return [min(1.0, max(0.0, x)) for x in vec]
    if mode_l == "norm":
        vec = [1.0] * length
        base = l2(vec)
        target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base if base > 0 else 1.0
        return [min(1.0, max(0.0, x * s)) for x in vec]
    if mode_l == "random":
        return [random.random() for _ in range(length)]
    return generate_agent_vector_legacy(length, mu_e, sig_e, norm_mu, norm_sig)


def generate_agent_i_vector(length, mu_e, sig_e, norm_mu, norm_sig, *, mode: str):
    mode_l = str(mode).lower()
    if mode_l == "legacy":
        vec = [tnorm(mu_e, sig_e, 0.0, 1.0) for _ in range(length)]
        base = l2(vec)
        target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base
        return [min(1.0, max(0.0, x * s)) for x in vec]
    if mode_l == "element":
        return [min(1.0, max(0.0, tnorm(mu_e, sig_e, 0.0, 1.0))) for _ in range(length)]
    if mode_l == "norm":
        vec = [1.0] * length
        base = l2(vec)
        target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base if base > 0 else 1.0
        return [min(1.0, max(0.0, x * s)) for x in vec]
    if mode_l == "random":
        return [random.random() for _ in range(length)]
    vec = [tnorm(mu_e, sig_e, 0.0, 1.0) for _ in range(length)]
    base = l2(vec)
    target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
    s = target / base
    return [min(1.0, max(0.0, x * s)) for x in vec]


def generate_agent_v_vector(length, mu_e, sig_e, norm_mu, norm_sig, *, mode: str):
    mode_l = str(mode).lower()
    if mode_l == "legacy":
        vec = [tnorm(mu_e, sig_e, -1.0, 1.0) for _ in range(length)]
        base = l2(vec)
        target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base
        return [max(-1.0, min(1.0, x * s)) for x in vec]
    if mode_l == "element":
        return [max(-1.0, min(1.0, tnorm(mu_e, sig_e, -1.0, 1.0))) for _ in range(length)]
    if mode_l == "norm":
        vec = [1.0] * length
        base = l2(vec)
        target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base if base > 0 else 1.0
        return [max(-1.0, min(1.0, x * s)) for x in vec]
    if mode_l == "random":
        return [random.uniform(-1.0, 1.0) for _ in range(length)]
    vec = [tnorm(mu_e, sig_e, -1.0, 1.0) for _ in range(length)]
    base = l2(vec)
    target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
    s = target / base
    return [max(-1.0, min(1.0, x * s)) for x in vec]


def generate_content_vector_unified(
    length: int,
    mode: str,
    mu_e: float,
    sig_e: float,
    norm_mu: float,
    norm_sig: float,
    active_count: int,
):
    mode_l = str(mode).lower()
    k = max(1, min(int(active_count), length))
    vec = np.zeros(length, dtype=np.float64)
    active_idx = np.asarray(random.sample(range(length), k), dtype=np.int64)

    if mode_l == "element":
        for i in active_idx:
            vec[i] = tnorm(mu_e, sig_e, 0.0, 1.0)

    elif mode_l == "norm":
        vec[active_idx] = 1.0
        base = l2(vec)
        target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(k))
        s = target / base if base > 0 else 1.0
        vec = np.clip(vec * s, 0.0, 1.0)

    elif mode_l == "legacy":
        for i in active_idx:
            vec[i] = tnorm(mu_e, sig_e, 0.0, 1.0)
        base = l2(vec)
        target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(k))
        s = target / base if base > 0 else 1.0
        vec = np.clip(vec * s, 0.0, 1.0)

    elif mode_l == "random":
        for i in active_idx:
            vec[i] = random.random()

    else:
        for i in active_idx:
            vec[i] = tnorm(mu_e, sig_e, 0.0, 1.0)
        base = l2(vec)
        target = tnorm(norm_mu, norm_sig, 0.0, math.sqrt(k))
        s = target / base if base > 0 else 1.0
        vec = np.clip(vec * s, 0.0, 1.0)

    return vec, active_idx


def generate_content_g_vector_unified(length, params, active_count, *, mode: str):
    vec, active_idx = generate_content_vector_unified(
        length=length,
        mode=mode,
        mu_e=params["mu"],
        sig_e=params["sigma"],
        norm_mu=params["norm_mu"],
        norm_sig=params["norm_sigma"],
        active_count=active_count,
    )
    return vec.astype(np.float32), active_idx


def generate_content_i_vector_unified(length, params, active_count, *, mode: str):
    vec, active_idx = generate_content_vector_unified(
        length=length,
        mode=mode,
        mu_e=params["mu"],
        sig_e=params["sigma"],
        norm_mu=params["norm_mu"],
        norm_sig=params["norm_sigma"],
        active_count=active_count,
    )
    return vec.astype(np.float32), active_idx
