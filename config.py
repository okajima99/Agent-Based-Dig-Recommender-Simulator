from __future__ import annotations

"""
設定解決の中核。

- 既定値の単一ソースは `CORE_PARAM_DEFAULTS`。
- 上書きは実行時環境変数のみ。
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

def _default_analysis_log_dir() -> str:
    # 実行ディレクトリ基準で保存する。
    return str((Path.cwd() / "analysis_logs").resolve())

# -----------------------------------------------------------------------------
# Core Default Parameters (single source of defaults)
# -----------------------------------------------------------------------------
CORE_PARAM_DEFAULTS: dict[str, object] = {
    # Entry / Runtime
    "DISPLAY_ALGORITHM": "cf_user",
    "CBF_FACE": "affinity",
    "CF_ITEM_FACE": "affinity",
    "CF_USER_FACE": "affinity",
    "OUT_PREFIX": None,
    "STRICT_CUDA": True,
    "RANDOM_SEED": 42,
    "LOG_PROGRESS_EVERY": 100,
    "ANALYSIS_LOG_FLUSH_EVERY_STEPS": 20,
    "ANALYSIS_LOG_IMPRESSION_FLUSH_ROWS": 20000,
    "ANALYSIS_LOG_BUFFER_MAX_ROWS": 100000,
    "ANALYSIS_LOG_COMPRESSION": "zstd",
    # Scale
    "NUM_GENRES": 10,
    "NUM_INSTINCT_DIM": 5,
    "NUM_AGENTS": 1000,
    "NUM_CONTENTS": 30000,
    "MAX_STEPS": 200000,
    "INITIAL_RANDOM_STEPS": 0,
    # Random Schedule
    "RANDOM_INTERVAL_ON": True,
    "RANDOM_RANDOM_BLOCK_LEN": 3000,
    "RANDOM_NORMAL_BLOCK_LEN": 3000,
    "RANDOM_REPEAT_POLICY": "reset_when_exhausted",
    # Replenish
    "REPLENISH_EVERY": 6000,
    "REPLENISH_COUNT": 30000,
    "REPLENISH_START_STEP": 0,
    "REPLENISH_END_STEP": None,
    # Modes
    "AGENT_ALPHA": 0.0,
    "CONTENT_G_ACTIVE": 3,
    "CONTENT_I_ACTIVE": 3,
    "CONTENT_G_MODE": "random",
    "CONTENT_I_MODE": "random",
    "CONTENT_MAT_DTYPE": "fp16",
    "AGENT_G_MODE": "element",
    "AGENT_I_MODE": "random",
    "AGENT_V_MODE": "random",
    # Vector Distribution
    "CONTENT_G_MU": 0.35,
    "CONTENT_G_SIGMA": 0.30,
    "CONTENT_G_NORM_MU": 0.70,
    "CONTENT_G_NORM_SIGMA": 0.27,
    "CONTENT_I_MU": 0.35,
    "CONTENT_I_SIGMA": 0.25,
    "CONTENT_I_NORM_MU": 1.30,
    "CONTENT_I_NORM_SIGMA": 0.35,
    "AGENT_G_MU": 0.15,
    "AGENT_G_SIGMA": 0.07,
    "AGENT_G_NORM_MU": 0.47,
    "AGENT_G_NORM_SIGMA": 0.01,
    "AGENT_V_MU": 0.00,
    "AGENT_V_SIGMA": 0.50,
    "AGENT_V_NORM_MU": 1.00,
    "AGENT_V_NORM_SIGMA": 0.25,
    "AGENT_I_MU": 0.35,
    "AGENT_I_SIGMA": 0.25,
    "AGENT_I_NORM_MU": 1.30,
    "AGENT_I_NORM_SIGMA": 0.35,
    # Like / Dig
    "LOGIT_K": 10.00,
    "LOGIT_X0": 1.218,
    "LIKE_DIVISOR": 3.0,
    "DIG_LOGIT_K": 10.00,
    "DIG_LOGIT_X0": 0.361,
    "DIG_DIVISOR": 60.0,
    "DIG_G_STEP": 0.00115,
    "DIG_V_RANGE": 0.10,
    "LIKE_W_CG": 1.0,
    "LIKE_W_CV": 1.0,
    "LIKE_W_CI": 1.0,
    "MU0": 0.1,
    "MU_SLOPE": 0.2,
    "MU_ALPHA_C": 0.25,
    "MU_BETA_V": 0.75,
    "SIGMA0": 0.05,
    "SIGMA_LAMDA": 2.0,
    "SIGMA_ALPHA_C": 0.25,
    "SIGMA_BETA_V": 0.75,
    "A0": 0.75,
    "A_LAMDA": 0.5,
    "A_ALPHA_C": 0.25,
    "A_BETA_V": 0.75,
    # Metrics / Cache / Candidates
    "TREND_EMA_ALPHA": 0.0002,
    "TREND_CACHE_DURATION": 1000,
    "BUZZ_WINDOW": 10000,
    "BUZZ_GAMMA": 0.9990,
    "BUZZ_CACHE_DURATION": 1000,
    "POP_CACHE_DURATION": 1000,
    "PSEUDO_CACHE_DURATION": 1000,
    "PSEUDO_DISCOUNT_GAMMA": 0.9998,
    "PSEUDO_HISTORY_WINDOW_STEPS": 10000,
    "CBF_TOP_K": 3000,
    "CBF_W_G": 1.0,
    "CBF_W_I": 1.0,
    "CF_CACHE_DURATION": 1000,
    "CF_DISCOUNT_GAMMA": 0.9998,
    "CF_HISTORY_WINDOW_STEPS": 10000,
    "CF_USER_NEIGHBOR_TOP_K": 10,
    "CF_USER_CANDIDATE_TOP_K": 3000,
    # None means fallback to CF_USER_* in resolver.
    "CF_ITEM_NEIGHBOR_TOP_K": None,
    "CF_ITEM_CANDIDATE_TOP_K": None,
    # Softmax temperature T (larger -> flatter, smaller -> sharper).
    # These defaults are reciprocal-adjusted to preserve prior behavior.
    "LAMBDA_POPULARITY": 0.033,
    "LAMBDA_TREND": 0.033,
    "LAMBDA_BUZZ": 0.033,
    "LAMBDA_CBF": 0.033,
    "LAMBDA_CF_USER": 0.033,
    "LAMBDA_CF_ITEM": 0.033,
}


def _env_int(env: Mapping[str, str], key: str, default: int | None) -> int | None:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(env: Mapping[str, str], key: str, default: float | None) -> float | None:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(env: Mapping[str, str], key: str, default: str | None) -> str | None:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return str(raw)


def _resolve_by_default(env: Mapping[str, str], key: str, default: object) -> object:
    if isinstance(default, bool):
        return _env_bool(env, key, default)
    if isinstance(default, int):
        return _env_int(env, key, default)
    if isinstance(default, float):
        return _env_float(env, key, default)
    if isinstance(default, str):
        return _env_str(env, key, default)
    if default is None:
        return _env_str(env, key, None)
    return _env_str(env, key, str(default))


def _merged_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    runtime_env = os.environ if env is None else env
    return {str(k): str(v) for k, v in runtime_env.items()}


def load_core_params(env: Mapping[str, str] | None = None) -> dict[str, object]:
    merged = _merged_env(env)
    resolved: dict[str, object] = {
        key: _resolve_by_default(merged, key, default)
        for key, default in CORE_PARAM_DEFAULTS.items()
    }

    display_algorithm = str(resolved["DISPLAY_ALGORITHM"])
    out_prefix = resolved.get("OUT_PREFIX")
    resolved["OUT_PREFIX"] = str(out_prefix) if out_prefix not in (None, "") else f"simulation_{display_algorithm}"

    replenish_end = resolved.get("REPLENISH_END_STEP")
    resolved["REPLENISH_END_STEP"] = int(resolved["MAX_STEPS"]) if replenish_end in (None, "") else int(replenish_end)

    cf_item_neighbor = resolved.get("CF_ITEM_NEIGHBOR_TOP_K")
    cf_item_candidate = resolved.get("CF_ITEM_CANDIDATE_TOP_K")
    resolved["CF_ITEM_NEIGHBOR_TOP_K"] = (
        int(resolved["CF_USER_NEIGHBOR_TOP_K"]) if cf_item_neighbor in (None, "") else int(cf_item_neighbor)
    )
    resolved["CF_ITEM_CANDIDATE_TOP_K"] = (
        int(resolved["CF_USER_CANDIDATE_TOP_K"]) if cf_item_candidate in (None, "") else int(cf_item_candidate)
    )

    out: dict[str, object] = dict(resolved)
    # 分析ログ出力先は常に実行ディレクトリ配下へ固定（環境変数上書きなし）。
    out["ANALYSIS_LOG_DIR"] = _default_analysis_log_dir()
    out["content_g_active"] = int(resolved["CONTENT_G_ACTIVE"])
    out["content_i_active"] = int(resolved["CONTENT_I_ACTIVE"])
    out["content_G_PARAMS"] = {
        "mu": float(resolved["CONTENT_G_MU"]),
        "sigma": float(resolved["CONTENT_G_SIGMA"]),
        "norm_mu": float(resolved["CONTENT_G_NORM_MU"]),
        "norm_sigma": float(resolved["CONTENT_G_NORM_SIGMA"]),
    }
    out["content_I_PARAMS"] = {
        "mu": float(resolved["CONTENT_I_MU"]),
        "sigma": float(resolved["CONTENT_I_SIGMA"]),
        "norm_mu": float(resolved["CONTENT_I_NORM_MU"]),
        "norm_sigma": float(resolved["CONTENT_I_NORM_SIGMA"]),
    }
    out["Agent_G_PARAMS"] = {
        "mu": float(resolved["AGENT_G_MU"]),
        "sigma": float(resolved["AGENT_G_SIGMA"]),
        "norm_mu": float(resolved["AGENT_G_NORM_MU"]),
        "norm_sigma": float(resolved["AGENT_G_NORM_SIGMA"]),
    }
    out["Agent_V_PARAMS"] = {
        "mu": float(resolved["AGENT_V_MU"]),
        "sigma": float(resolved["AGENT_V_SIGMA"]),
        "norm_mu": float(resolved["AGENT_V_NORM_MU"]),
        "norm_sigma": float(resolved["AGENT_V_NORM_SIGMA"]),
    }
    out["Agent_I_PARAMS"] = {
        "mu": float(resolved["AGENT_I_MU"]),
        "sigma": float(resolved["AGENT_I_SIGMA"]),
        "norm_mu": float(resolved["AGENT_I_NORM_MU"]),
        "norm_sigma": float(resolved["AGENT_I_NORM_SIGMA"]),
    }
    return out


@dataclass(slots=True)
class SimulationConfig:
    display_algorithm: str = str(CORE_PARAM_DEFAULTS["DISPLAY_ALGORITHM"])
    cbf_face: str = str(CORE_PARAM_DEFAULTS["CBF_FACE"])
    cf_item_face: str = str(CORE_PARAM_DEFAULTS["CF_ITEM_FACE"])
    cf_user_face: str = str(CORE_PARAM_DEFAULTS["CF_USER_FACE"])
    out_prefix: str | None = None
    log_progress_every: int = int(CORE_PARAM_DEFAULTS["LOG_PROGRESS_EVERY"])
    cf_item_neighbor_top_k: int | None = None
    cf_item_candidate_top_k: int | None = None
    random_seed: int = int(CORE_PARAM_DEFAULTS["RANDOM_SEED"])
    strict_cuda: bool = bool(CORE_PARAM_DEFAULTS["STRICT_CUDA"])
    core_script_path: Path | None = None

    @property
    def resolved_core_script_path(self) -> Path:
        if self.core_script_path is not None:
            return Path(self.core_script_path).expanduser().resolve()
        return (Path(__file__).resolve().parent / "core.py").resolve()

    def to_env_overrides(self) -> dict[str, str]:
        env: dict[str, str] = {
            "DISPLAY_ALGORITHM": self.display_algorithm,
            "CBF_FACE": self.cbf_face,
            "CF_ITEM_FACE": self.cf_item_face,
            "CF_USER_FACE": self.cf_user_face,
            "LOG_PROGRESS_EVERY": str(self.log_progress_every),
            "RANDOM_SEED": str(self.random_seed),
            "STRICT_CUDA": "1" if self.strict_cuda else "0",
        }
        if self.out_prefix:
            env["OUT_PREFIX"] = self.out_prefix
        if self.cf_item_neighbor_top_k is not None:
            env["CF_ITEM_NEIGHBOR_TOP_K"] = str(self.cf_item_neighbor_top_k)
        if self.cf_item_candidate_top_k is not None:
            env["CF_ITEM_CANDIDATE_TOP_K"] = str(self.cf_item_candidate_top_k)
        return env


def load_config_from_env(env: Mapping[str, str] | None = None) -> SimulationConfig:
    merged = _merged_env(env)
    d = CORE_PARAM_DEFAULTS
    return SimulationConfig(
        display_algorithm=_env_str(merged, "DISPLAY_ALGORITHM", str(d["DISPLAY_ALGORITHM"])) or str(d["DISPLAY_ALGORITHM"]),
        cbf_face=_env_str(merged, "CBF_FACE", str(d["CBF_FACE"])) or str(d["CBF_FACE"]),
        cf_item_face=_env_str(merged, "CF_ITEM_FACE", str(d["CF_ITEM_FACE"])) or str(d["CF_ITEM_FACE"]),
        cf_user_face=_env_str(merged, "CF_USER_FACE", str(d["CF_USER_FACE"])) or str(d["CF_USER_FACE"]),
        out_prefix=_env_str(merged, "OUT_PREFIX", None) or None,
        log_progress_every=_env_int(merged, "LOG_PROGRESS_EVERY", int(d["LOG_PROGRESS_EVERY"]))
        or int(d["LOG_PROGRESS_EVERY"]),
        cf_item_neighbor_top_k=_env_int(merged, "CF_ITEM_NEIGHBOR_TOP_K", None),
        cf_item_candidate_top_k=_env_int(merged, "CF_ITEM_CANDIDATE_TOP_K", None),
        random_seed=_env_int(merged, "RANDOM_SEED", int(d["RANDOM_SEED"])) or int(d["RANDOM_SEED"]),
        strict_cuda=_env_bool(merged, "STRICT_CUDA", bool(d["STRICT_CUDA"])),
        core_script_path=Path(merged["CORE_SCRIPT_PATH"]).expanduser() if merged.get("CORE_SCRIPT_PATH") else None,
    )
