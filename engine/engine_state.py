from __future__ import annotations

import numpy as np
import torch


def initialize_engine_state(
    engine,
    *,
    num_agents: int,
    num_contents: int,
    g_dim: int,
    device: torch.device,
    num_instinct_dim: int,
    torch_dtype: torch.dtype,
) -> None:
    engine.device = device
    engine.sim_dtype = torch.float16 if device.type == "cuda" else torch_dtype
    engine.g_dim = g_dim
    engine.num_agents = num_agents
    engine.num_contents = num_contents

    engine.Ug = torch.zeros((num_agents, g_dim), dtype=torch.float32, device=device)
    engine.Uv = torch.zeros((num_agents, g_dim), dtype=torch.float32, device=device)
    engine.Ui = torch.zeros((num_agents, num_instinct_dim), dtype=torch.float32, device=device)

    engine.Cg = None
    engine.Ci = None

    engine.seen_mask_t = torch.zeros((num_agents, num_contents), dtype=torch.bool, device=device)

    engine.pseudo_G = torch.zeros_like(engine.Ug)
    engine.pseudo_I = torch.zeros_like(engine.Ui)

    engine.Ug_norm = None
    engine.Cg_norm = None
    engine.social_alpha_t = torch.zeros((num_agents,), dtype=torch.float32, device=device)


def ensure_seen_capacity(engine, num_contents: int) -> None:
    if engine.seen_mask_t.shape[1] < num_contents:
        diff = num_contents - engine.seen_mask_t.shape[1]
        pad = torch.zeros((engine.num_agents, diff), dtype=torch.bool, device=engine.device)
        engine.seen_mask_t = torch.cat([engine.seen_mask_t, pad], dim=1)


def mark_seen(engine, agent_idx: int, cid: int) -> None:
    try:
        if cid < engine.seen_mask_t.shape[1]:
            engine.seen_mask_t[agent_idx, cid] = True
    except Exception:
        pass


def mark_seen_batch(engine, agent_idx_t: torch.Tensor, cid_t: torch.Tensor) -> None:
    if cid_t.numel() != agent_idx_t.numel():
        raise RuntimeError("mark_seen_batch: size mismatch")
    engine.seen_mask_t[agent_idx_t, cid_t] = True


def get_seen_mask_row(engine, agent_idx: int, num_contents: int):
    if (engine.seen_mask_t is None) or (engine.seen_mask_t.shape[1] < num_contents):
        return None
    return engine.seen_mask_t[agent_idx, :num_contents]


def load_agents(engine, agents) -> None:
    ug_mat = np.array([a.interests for a in agents], dtype=np.float32)
    uv_mat = np.array([a.V for a in agents], dtype=np.float32)
    ui_mat = np.array([a.I for a in agents], dtype=np.float32)
    alpha_mat = np.array([getattr(a, "social_alpha", 0.0) for a in agents], dtype=np.float32)
    engine.Ug[:] = torch.from_numpy(ug_mat).to(engine.device)
    engine.Uv[:] = torch.from_numpy(uv_mat).to(engine.device)
    engine.Ui[:] = torch.from_numpy(ui_mat).to(engine.device)
    engine.social_alpha_t = torch.from_numpy(alpha_mat).to(engine.device)

    engine.ug_matrix_t = engine.Ug
    engine.uv_matrix_t = engine.Uv
    engine.ui_matrix_t = engine.Ui

    ug = engine.Ug
    engine.Ug_norm = ug / (ug.norm(dim=1, keepdim=True) + 1e-12)


def load_contents(engine, pool, *, torch_dtype: torch.dtype) -> None:
    cg_mat = np.array([c.vector for c in pool], dtype=np.float32)
    ci_mat = np.array([c.i_vector for c in pool], dtype=np.float32)
    engine.Cg = torch.from_numpy(cg_mat).to(engine.device)
    engine.Ci = torch.from_numpy(ci_mat).to(engine.device)
    engine.num_contents = cg_mat.shape[0]
    ensure_seen_capacity(engine, engine.num_contents)

    engine.cg_matrix_t = engine.Cg
    engine.ci_matrix_t = engine.Ci
    if engine.sim_dtype != torch_dtype:
        engine.cg_matrix_half = engine.Cg.to(engine.sim_dtype)
        engine.ci_matrix_half = engine.Ci.to(engine.sim_dtype)
    else:
        engine.cg_matrix_half = None
        engine.ci_matrix_half = None

    cg = engine.Cg
    engine.Cg_norm = cg / (cg.norm(dim=1, keepdim=True) + 1e-12)


def seen_mask_np(engine, agent_idx: int, num_contents: int):
    if engine is None or getattr(engine, "seen_mask_t", None) is None:
        raise RuntimeError("GPU seen_mask が利用できません（CPUフォールバックなし）")
    mask_row = engine.get_seen_mask_row(agent_idx, num_contents)
    if mask_row is None:
        raise RuntimeError("GPU seen_mask 行が取得できませんでした")
    return mask_row.detach().cpu().numpy().astype(bool)


def exclude_ids_from_mask(engine, agent_idx: int, num_contents: int):
    return seen_mask_np(engine, agent_idx, num_contents)


def unseen_mask_for_ids(engine, agent_idx: int, ids_arr, num_contents: int):
    ids_np = np.asarray(ids_arr, dtype=np.int64)
    mask_row = seen_mask_np(engine, agent_idx, num_contents)
    return ~mask_row[ids_np]
