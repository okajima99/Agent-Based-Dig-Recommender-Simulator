from __future__ import annotations

import math

import numpy as np


def sigmoid01(x, k: float, x0: float) -> float:
    z = max(-60.0, min(60.0, float(k) * (float(x) - float(x0))))
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid01_impl(x, k: float, x0: float) -> float:
    return sigmoid01(x, k, x0)


def softmax_arr(x, lam: float = 1.0):
    arr = np.asarray(x, dtype=np.float64)
    temperature = float(lam)
    if (not np.isfinite(temperature)) or (temperature <= 0.0):
        temperature = 1e-6
    arr = np.exp((arr - np.max(arr)) / temperature)
    s = arr.sum()
    return arr / s if s > 0 else np.ones_like(arr) / len(arr)


def softmax_arr_impl(x, lam: float = 1.0):
    return softmax_arr(x, lam=lam)


def sim_alpha(x_vec, y_vec, alpha: float) -> float:
    x = np.asarray(x_vec, dtype=np.float64).ravel()
    y = np.asarray(y_vec, dtype=np.float64).ravel()
    dot = float(np.dot(x, y))
    if float(alpha) == 0.0:
        return dot
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= 1e-9 or ny <= 1e-9:
        return 0.0
    denom = (nx * ny) ** float(alpha)
    return dot / denom


def safe_sim_alpha(a_vec, b_vec, alpha: float):
    if a_vec is None or b_vec is None:
        return ""
    a = np.asarray(a_vec, dtype=np.float64).ravel()
    b = np.asarray(b_vec, dtype=np.float64).ravel()
    dot = float(np.dot(a, b))
    if float(alpha) == 0.0:
        return dot
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return ""
    return dot / ((na * nb) ** float(alpha))


def clip01(x: float) -> float:
    return 1.0 if x >= 1.0 else (0.0 if x <= 0.0 else x)


def clip11(x: float) -> float:
    return 1.0 if x >= 1.0 else (-1.0 if x <= -1.0 else x)
