from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any, Iterable

import numpy as np


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))


def _algo_slug(display_algorithm: str) -> str:
    slug = _safe_slug(display_algorithm)
    return slug or "unknown"


def _algo_label(
    *,
    display_algorithm: str,
    cbf_face: str | None,
    cf_item_face: str | None,
    cf_user_face: str | None,
) -> str:
    algo = str(display_algorithm).lower()
    if algo == "cbf":
        return f"{algo}_{str(cbf_face or 'affinity').lower()}"
    if algo == "cf_item":
        return f"{algo}_{str(cf_item_face or 'affinity').lower()}"
    if algo == "cf_user":
        return f"{algo}_{str(cf_user_face or 'affinity').lower()}"
    return algo


def _make_run_id(*, algo_label: str, random_seed: int) -> str:
    # Example: 2026-02-19_13-45-10_cf_item_novelty_seed42_ab12cd34
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algo = _safe_slug(algo_label)
    return f"{stamp}_{algo}_seed{int(random_seed)}_{uuid.uuid4().hex[:8]}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _vec_list(vec: Any) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return arr.astype(np.float32).tolist()


@dataclass(slots=True)
class NullAnalysisLogger:
    enabled: bool = False
    run_id: str | None = None
    run_dir: Path | None = None

    def log_run_config(self, params: dict[str, Any]) -> None:
        del params

    def log_agent_init_batch(self, agents: Iterable[Any]) -> None:
        del agents

    def log_agent_final_batch(self, agents: Iterable[Any]) -> None:
        del agents

    def log_content_final_batch(self, pool: Iterable[Any], *, final_step: int) -> None:
        del pool, final_step

    def log_impression_pre(
        self,
        *,
        step: int,
        agent_id: int,
        content_id: int,
        g_pre,
        v_pre,
        i_pre,
        like_flag: int,
        dig_flag: int,
        dig_dim: int,
        d_v: float,
        random_interval_active: bool,
    ) -> None:
        del step, agent_id, content_id, g_pre, v_pre, i_pre, like_flag, dig_flag, dig_dim, d_v, random_interval_active

    def log_cache_refresh_event(
        self,
        *,
        step: int,
        cache_name: str,
        reason: str,
        action: str,
        timer_before: int | None,
        timer_after: int | None,
        random_interval_active: bool,
    ) -> int:
        del step, cache_name, reason, action, timer_before, timer_after, random_interval_active
        return -1

    def log_cache_state_cf(self, *, event_id: int, step: int, uv_matrix) -> None:
        del event_id, step, uv_matrix

    def log_cache_state_cbf(self, *, event_id: int, step: int, pseudo_g_t, pseudo_i_t) -> None:
        del event_id, step, pseudo_g_t, pseudo_i_t

    def log_cache_state_global(
        self,
        *,
        event_id: int,
        step: int,
        cache_name: str,
        content_ids: Iterable[int],
        scores,
    ) -> None:
        del event_id, step, cache_name, content_ids, scores

    def maybe_flush_step(self, step: int) -> None:
        del step

    def flush_all(self) -> None:
        return None

    def close(self) -> None:
        return None


class ParquetAnalysisLogger:
    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        compression: str,
        flush_every_steps: int,
        impression_flush_rows: int,
        buffer_max_rows: int,
        pyarrow_module,
        pyarrow_parquet_module,
    ):
        self.enabled = True
        self.run_id = str(run_id)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._pa = pyarrow_module
        self._pq = pyarrow_parquet_module
        self._compression = str(compression)
        self._flush_every_steps = int(flush_every_steps)
        self._impression_flush_rows = int(impression_flush_rows)
        self._buffer_max_rows = int(buffer_max_rows)

        self._event_seq = 0
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._writers: dict[str, Any] = {}
        self._row_counts: dict[str, int] = {}

    def _threshold_for(self, name: str) -> int:
        if name == "impression_pre":
            return max(1, self._impression_flush_rows)
        return max(1, self._buffer_max_rows)

    def _append_many(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        buf = self._buffers.setdefault(name, [])
        buf.extend(rows)
        self._row_counts[name] = int(self._row_counts.get(name, 0)) + int(len(rows))
        threshold = self._threshold_for(name)
        if len(buf) >= threshold:
            self._flush_name(name)

    def _append_one(self, name: str, row: dict[str, Any]) -> None:
        self._append_many(name, [row])

    def _flush_name(self, name: str) -> None:
        rows = self._buffers.get(name)
        if not rows:
            return

        table = self._pa.Table.from_pylist(rows)
        writer = self._writers.get(name)
        if writer is None:
            out_path = self.run_dir / f"{name}.parquet"
            writer = self._pq.ParquetWriter(
                str(out_path),
                table.schema,
                compression=self._compression,
            )
            self._writers[name] = writer
        writer.write_table(table)
        rows.clear()

    def flush_all(self) -> None:
        for name in list(self._buffers.keys()):
            self._flush_name(name)

    def maybe_flush_step(self, step: int) -> None:
        if self._flush_every_steps <= 0:
            return
        if ((int(step) + 1) % self._flush_every_steps) == 0:
            self.flush_all()

    def close(self) -> None:
        if int(self._row_counts.get("cache_refresh_event", 0)) <= 0:
            self._append_one(
                "cache_refresh_event",
                {
                    "run_id": self.run_id,
                    "event_id": -1,
                    "step": -1,
                    "cache_name": "none",
                    "reason": "none",
                    "action": "none",
                    "timer_before": None,
                    "timer_after": None,
                    "random_interval_active": False,
                },
            )
        self.flush_all()
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def log_run_config(self, params: dict[str, Any]) -> None:
        rows: list[dict[str, Any]] = []
        for key, value in sorted(params.items(), key=lambda kv: str(kv[0])):
            val = _jsonable(value)
            rows.append(
                {
                    "run_id": self.run_id,
                    "param_key": str(key),
                    "param_type": type(value).__name__,
                    "param_value_json": json.dumps(val, ensure_ascii=True, separators=(",", ":")),
                }
            )
        self._append_many("run_config", rows)

    def log_agent_init_batch(self, agents: Iterable[Any]) -> None:
        rows: list[dict[str, Any]] = []
        for a in agents:
            rows.append(
                {
                    "run_id": self.run_id,
                    "agent_id": int(a.id),
                    "g_init": _vec_list(a.initial_vector),
                    "v_init": _vec_list(a.initial_V),
                    "i_init": _vec_list(a.initial_I),
                    "social_alpha": float(getattr(a, "social_alpha", 0.0)),
                }
            )
        self._append_many("agent_init", rows)

    def log_agent_final_batch(self, agents: Iterable[Any]) -> None:
        rows: list[dict[str, Any]] = []
        for a in agents:
            rows.append(
                {
                    "run_id": self.run_id,
                    "agent_id": int(a.id),
                    "g_final": _vec_list(a.interests),
                    "v_final": _vec_list(a.V),
                    "i_final": _vec_list(a.I),
                    "social_alpha": float(getattr(a, "social_alpha", 0.0)),
                    "total_likes": int(getattr(a, "total_likes", 0)),
                    "total_digs": int(np.asarray(getattr(a, "lh_steps", np.empty(0))).size),
                }
            )
        self._append_many("agent_final", rows)

    def log_content_final_batch(self, pool: Iterable[Any], *, final_step: int) -> None:
        rows: list[dict[str, Any]] = []
        for c in pool:
            try:
                buzz = float(c.get_buzz_score(final_step))
            except Exception:
                buzz = float("nan")
            rows.append(
                {
                    "run_id": self.run_id,
                    "content_id": int(c.id),
                    "g": _vec_list(c.vector),
                    "i": _vec_list(c.i_vector),
                    "views": int(getattr(c, "views", 0)),
                    "likes": int(getattr(c, "likes", 0)),
                    "trend_ema": float(getattr(c, "trend_ema", 0.0)),
                    "buzz_score_final": buzz,
                }
            )
        self._append_many("content_final", rows)

    def log_impression_pre(
        self,
        *,
        step: int,
        agent_id: int,
        content_id: int,
        g_pre,
        v_pre,
        i_pre,
        like_flag: int,
        dig_flag: int,
        dig_dim: int,
        d_v: float,
        random_interval_active: bool,
    ) -> None:
        self._append_one(
            "impression_pre",
            {
                "run_id": self.run_id,
                "step": int(step),
                "agent_id": int(agent_id),
                "content_id": int(content_id),
                "g_pre": _vec_list(g_pre),
                "v_pre": _vec_list(v_pre),
                "i_pre": _vec_list(i_pre),
                "like_flag": int(like_flag),
                "dig_flag": int(dig_flag),
                "dig_dim": int(dig_dim),
                "d_v": float(d_v),
                "random_interval_active": bool(random_interval_active),
            },
        )

    def log_cache_refresh_event(
        self,
        *,
        step: int,
        cache_name: str,
        reason: str,
        action: str,
        timer_before: int | None,
        timer_after: int | None,
        random_interval_active: bool,
    ) -> int:
        self._event_seq += 1
        event_id = int(self._event_seq)
        self._append_one(
            "cache_refresh_event",
            {
                "run_id": self.run_id,
                "event_id": event_id,
                "step": int(step),
                "cache_name": str(cache_name),
                "reason": str(reason),
                "action": str(action),
                "timer_before": None if timer_before is None else int(timer_before),
                "timer_after": None if timer_after is None else int(timer_after),
                "random_interval_active": bool(random_interval_active),
            },
        )
        return event_id

    def log_cache_state_cf(self, *, event_id: int, step: int, uv_matrix) -> None:
        if uv_matrix is None:
            self._append_one(
                "cache_state_cf",
                {
                    "run_id": self.run_id,
                    "event_id": int(event_id),
                    "step": int(step),
                    "row": -1,
                    "col": -1,
                    "val": float("nan"),
                },
            )
            return
        try:
            coal = uv_matrix.coalesce()
            idx = coal.indices().detach().cpu().numpy()
            vals = coal.values().detach().cpu().numpy()
        except Exception:
            self._append_one(
                "cache_state_cf",
                {
                    "run_id": self.run_id,
                    "event_id": int(event_id),
                    "step": int(step),
                    "row": -1,
                    "col": -1,
                    "val": float("nan"),
                },
            )
            return

        nnz = int(vals.size)
        if nnz == 0:
            self._append_one(
                "cache_state_cf",
                {
                    "run_id": self.run_id,
                    "event_id": int(event_id),
                    "step": int(step),
                    "row": -1,
                    "col": -1,
                    "val": float("nan"),
                },
            )
            return

        chunk = max(1, self._buffer_max_rows)
        for off in range(0, nnz, chunk):
            end = min(off + chunk, nnz)
            rows = [
                {
                    "run_id": self.run_id,
                    "event_id": int(event_id),
                    "step": int(step),
                    "row": int(idx[0, i]),
                    "col": int(idx[1, i]),
                    "val": float(vals[i]),
                }
                for i in range(off, end)
            ]
            self._append_many("cache_state_cf", rows)

    def log_cache_state_cbf(self, *, event_id: int, step: int, pseudo_g_t, pseudo_i_t) -> None:
        if (pseudo_g_t is None) or (pseudo_i_t is None):
            self._append_one(
                "cache_state_cbf",
                {
                    "run_id": self.run_id,
                    "event_id": int(event_id),
                    "step": int(step),
                    "agent_id": -1,
                    "pseudo_g": [],
                    "pseudo_i": [],
                }
            )
            return
        try:
            g_np = pseudo_g_t.detach().cpu().numpy()
            i_np = pseudo_i_t.detach().cpu().numpy()
        except Exception:
            self._append_one(
                "cache_state_cbf",
                {
                    "run_id": self.run_id,
                    "event_id": int(event_id),
                    "step": int(step),
                    "agent_id": -1,
                    "pseudo_g": [],
                    "pseudo_i": [],
                }
            )
            return

        n = int(min(len(g_np), len(i_np)))
        rows: list[dict[str, Any]] = []
        for uid in range(n):
            rows.append(
                {
                    "run_id": self.run_id,
                    "event_id": int(event_id),
                    "step": int(step),
                    "agent_id": int(uid),
                    "pseudo_g": _vec_list(g_np[uid]),
                    "pseudo_i": _vec_list(i_np[uid]),
                }
            )
        self._append_many("cache_state_cbf", rows)

    def log_cache_state_global(
        self,
        *,
        event_id: int,
        step: int,
        cache_name: str,
        content_ids: Iterable[int],
        scores,
    ) -> None:
        try:
            scores_np = np.asarray(scores.detach().cpu().numpy() if hasattr(scores, "detach") else scores, dtype=np.float32)
        except Exception:
            self._append_one(
                "cache_state_global",
                {
                    "run_id": self.run_id,
                    "event_id": int(event_id),
                    "step": int(step),
                    "cache_name": str(cache_name),
                    "content_id": -1,
                    "score": float("nan"),
                },
            )
            return
        ids_np = np.asarray(list(content_ids), dtype=np.int64)
        n = int(min(ids_np.size, scores_np.size))
        if n <= 0:
            self._append_one(
                "cache_state_global",
                {
                    "run_id": self.run_id,
                    "event_id": int(event_id),
                    "step": int(step),
                    "cache_name": str(cache_name),
                    "content_id": -1,
                    "score": float("nan"),
                },
            )
            return

        chunk = max(1, self._buffer_max_rows)
        for off in range(0, n, chunk):
            end = min(off + chunk, n)
            rows = [
                {
                    "run_id": self.run_id,
                    "event_id": int(event_id),
                    "step": int(step),
                    "cache_name": str(cache_name),
                    "content_id": int(ids_np[i]),
                    "score": float(scores_np[i]),
                }
                for i in range(off, end)
            ]
            self._append_many("cache_state_global", rows)


def create_analysis_logger(
    *,
    base_dir: str,
    display_algorithm: str,
    cbf_face: str | None,
    cf_item_face: str | None,
    cf_user_face: str | None,
    random_seed: int,
    flush_every_steps: int,
    impression_flush_rows: int,
    buffer_max_rows: int,
    compression: str,
):
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Analysis logging requires 'pyarrow'. Install it and rerun."
        ) from exc

    algo_label = _algo_label(
        display_algorithm=display_algorithm,
        cbf_face=cbf_face,
        cf_item_face=cf_item_face,
        cf_user_face=cf_user_face,
    )
    algo = _algo_slug(algo_label)
    run_id = _make_run_id(algo_label=algo_label, random_seed=random_seed)
    algo_dir = Path(base_dir).expanduser().resolve() / algo
    run_dir = algo_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    marker_text = (
        f"run_id={run_id}\n"
        f"algorithm={algo}\n"
        f"random_seed={int(random_seed)}\n"
        f"created_at_utc={datetime.now(timezone.utc).isoformat()}\n"
    )
    (run_dir / "RUN_ID.txt").write_text(marker_text, encoding="utf-8")
    (algo_dir / "latest_run_id.txt").write_text(f"{run_id}\n", encoding="utf-8")
    (algo_dir / f"{run_id}.runid").write_text(marker_text, encoding="utf-8")

    return ParquetAnalysisLogger(
        run_id=run_id,
        run_dir=run_dir,
        compression=str(compression),
        flush_every_steps=int(flush_every_steps),
        impression_flush_rows=int(impression_flush_rows),
        buffer_max_rows=int(buffer_max_rows),
        pyarrow_module=pa,
        pyarrow_parquet_module=pq,
    )
