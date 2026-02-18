from __future__ import annotations

import numpy as np
import torch

_CONTENT_IDS = None
_CONTENT_G_RAW = None
_CONTENT_I_RAW = None
_ID2ROW = None
_CONTENT_G_RAW_T = None
_CONTENT_I_RAW_T = None
_CONTENT_G_NORM_T = None
_CONTENT_I_NORM_T = None
_CONTENT_G_RAW_T_BF16 = None
_CONTENT_I_RAW_T_BF16 = None
_DTYPE_MAT = None


def get_content_ids():
    return _CONTENT_IDS


def get_id2row():
    return _ID2ROW


def build_content_index_from_pool(
    pool,
    *,
    to_t,
    device: torch.device,
    torch_module,
    content_mat_dtype: str,
):
    global _CONTENT_IDS
    global _CONTENT_G_RAW
    global _CONTENT_I_RAW
    global _ID2ROW
    global _CONTENT_G_RAW_T
    global _CONTENT_I_RAW_T
    global _CONTENT_G_NORM_T
    global _CONTENT_I_NORM_T
    global _CONTENT_G_RAW_T_BF16
    global _CONTENT_I_RAW_T_BF16
    global _DTYPE_MAT

    g_mat = np.asarray([c.vector for c in pool], dtype=np.float32)
    i_mat = np.asarray([c.i_vector for c in pool], dtype=np.float32)
    _CONTENT_G_RAW = g_mat
    _CONTENT_I_RAW = i_mat
    _CONTENT_IDS = np.asarray([c.id for c in pool], dtype=np.int64)
    _ID2ROW = {int(cid): int(i) for i, cid in enumerate(_CONTENT_IDS)}

    _CONTENT_G_RAW_T = to_t(g_mat)
    _CONTENT_I_RAW_T = to_t(i_mat)
    _CONTENT_G_NORM_T = torch_module.linalg.vector_norm(_CONTENT_G_RAW_T, dim=1)
    _CONTENT_I_NORM_T = torch_module.linalg.vector_norm(_CONTENT_I_RAW_T, dim=1)

    if device.type == "cuda":
        _DTYPE_MAT = torch_module.bfloat16 if content_mat_dtype == "bf16" else torch_module.float16
    else:
        _DTYPE_MAT = torch_module.bfloat16

    _CONTENT_G_RAW_T_BF16 = _CONTENT_G_RAW_T.to(_DTYPE_MAT).contiguous()
    _CONTENT_I_RAW_T_BF16 = _CONTENT_I_RAW_T.to(_DTYPE_MAT).contiguous()


def ensure_content_index(
    pool,
    *,
    to_t,
    device: torch.device,
    torch_module,
    content_mat_dtype: str,
):
    global _CONTENT_IDS
    if pool is None or len(pool) == 0:
        _CONTENT_IDS = None
        return
    if (_CONTENT_IDS is None) or (len(_CONTENT_IDS) != len(pool)):
        build_content_index_from_pool(
            pool,
            to_t=to_t,
            device=device,
            torch_module=torch_module,
            content_mat_dtype=content_mat_dtype,
        )


def proj_sim_alpha_t(
    is_g: bool,
    p_vec,
    alpha: float,
    *,
    to_t,
    device: torch.device,
    torch_module,
) -> torch.Tensor:
    if p_vec is None:
        if _CONTENT_G_RAW_T is None:
            return torch_module.zeros(0, device=device, dtype=torch_module.float32)
        n = _CONTENT_G_RAW_T.shape[0]
        return torch_module.zeros(n, device=device, dtype=torch_module.float32)

    m_bf16 = _CONTENT_G_RAW_T_BF16 if is_g else _CONTENT_I_RAW_T_BF16
    rn = _CONTENT_G_NORM_T if is_g else _CONTENT_I_NORM_T
    if m_bf16 is None or rn is None:
        raise RuntimeError("content index tensors are not initialized")

    p_t = to_t(p_vec).view(-1, 1)
    p_cast = p_t.to(m_bf16.dtype)
    dot = (m_bf16 @ p_cast).squeeze(1).to(torch_module.float32)

    if alpha == 0.0:
        return dot

    pn = torch_module.linalg.vector_norm(p_t).clamp_min(1e-12)
    denom = (pn * rn).pow(float(alpha))
    out = torch_module.zeros_like(dot)
    m = denom > 1e-12
    out[m] = dot[m] / denom[m]
    return out
