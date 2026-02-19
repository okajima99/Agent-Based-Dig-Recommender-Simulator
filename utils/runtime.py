from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch

TORCH_DTYPE = torch.float32


def ensure_cuda_required(*, strict_cuda: bool = True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if strict_cuda and device.type != "cuda":
        raise RuntimeError("CUDA device is required (CPU fallback is disabled).")
    return device


def initialize_torch_runtime(*, seed: int | None = None, strict_cuda: bool = True) -> torch.device:
    device = ensure_cuda_required(strict_cuda=strict_cuda)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.set_grad_enabled(False)

    return device


def torch_runtime_summary(device: torch.device) -> str:
    return (
        f"[Torch] version={torch.__version__}, "
        f"device={device}, cuda_available={torch.cuda.is_available()}"
    )


def setup_torch_bindings(
    *,
    seed: int | None = None,
    strict_cuda: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, object]:
    device = initialize_torch_runtime(seed=seed, strict_cuda=strict_cuda)
    if log_fn is not None:
        log_fn(torch_runtime_summary(device))

    if not hasattr(np, "long"):
        np.long = np.int64

    def _to_t(x, *, device=device, dtype=TORCH_DTYPE):
        return torch.as_tensor(x, dtype=dtype, device=device)

    def _to_np(x_t: torch.Tensor):
        return x_t.detach().to("cpu").numpy()

    def torch_softmax_rank(x, lam: float = 1.0):
        x_t = torch.as_tensor(x, dtype=TORCH_DTYPE, device=device)
        if not torch.isfinite(x_t).any():
            n = x_t.numel()
            return torch.ones(n, device=device) / n
        temperature = float(lam)
        if (not np.isfinite(temperature)) or (temperature <= 0.0):
            temperature = 1e-6
        max_finite = torch.max(x_t[torch.isfinite(x_t)])
        z = x_t - max_finite
        y = torch.exp(z / temperature)
        denom = torch.sum(y)
        if denom.item() == 0 or not torch.isfinite(denom):
            n = x_t.numel()
            return torch.ones(n, device=device) / n
        return y / denom

    def torch_pick_from_probs(ids, probs):
        ids_np = np.asarray(ids)
        p = np.asarray(probs, dtype=np.float64)
        p[~np.isfinite(p)] = 0.0
        p[p < 0] = 0.0
        s = p.sum()
        if s <= 0:
            idx = np.random.randint(0, len(ids_np))
            return int(ids_np[idx])

        p = p / s
        t = _to_t(p, dtype=torch.float32)
        idx_t = torch.multinomial(t, num_samples=1, replacement=True)
        idx = int(idx_t.item())
        return int(ids_np[idx])

    return {
        "torch": torch,
        "TORCH_DTYPE": TORCH_DTYPE,
        "DEVICE": device,
        "_to_t": _to_t,
        "_to_np": _to_np,
        "torch_softmax_rank": torch_softmax_rank,
        "torch_pick_from_probs": torch_pick_from_probs,
    }
