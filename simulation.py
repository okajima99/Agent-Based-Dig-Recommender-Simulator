from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

try:
    from .config import SimulationConfig
    from .utils.runtime import initialize_torch_runtime
except ImportError:
    from config import SimulationConfig
    from utils.runtime import initialize_torch_runtime


@dataclass(slots=True)
class SimulationResult:
    script_path: Path
    elapsed_sec: float
    out_prefix: str
    out_prefix_face: str | None
    log_file_path: str | None
    analysis_run_id: str | None
    analysis_log_dir: str | None
    namespace: dict[str, Any]


@contextmanager
def _temporary_environ(overrides: dict[str, str]) -> Iterator[None]:
    old_values: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            old_values[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _load_core_module(path: Path):
    module_name = f"_research_core_runtime_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create import spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    script_dir = str(path.parent)
    inserted = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass
    return module


def run_simulation(config: SimulationConfig) -> SimulationResult:
    initialize_torch_runtime(seed=config.random_seed, strict_cuda=config.strict_cuda)

    script_path = config.resolved_core_script_path
    if not script_path.exists():
        raise FileNotFoundError(f"core script not found: {script_path}")

    env_overrides = config.to_env_overrides()

    started = time.time()
    with _temporary_environ(env_overrides):
        core_module = _load_core_module(script_path)
        run_core = getattr(core_module, "run_core", None)
        if not callable(run_core):
            raise RuntimeError(f"run_core() is missing in module: {script_path}")
        namespace = run_core()
    elapsed = time.time() - started

    out_prefix_face = namespace.get("OUT_PREFIX_FACE")
    out_prefix = str(
        out_prefix_face
        or namespace.get("OUT_PREFIX")
        or config.out_prefix
        or f"simulation_{config.display_algorithm}"
    )
    log_file_path = namespace.get("LOG_FILE_PATH")
    if log_file_path is not None:
        log_file_path = str(log_file_path)
    analysis_run_id = namespace.get("ANALYSIS_RUN_ID")
    if analysis_run_id is not None:
        analysis_run_id = str(analysis_run_id)
    analysis_log_dir = namespace.get("ANALYSIS_LOG_DIR")
    if analysis_log_dir is not None:
        analysis_log_dir = str(analysis_log_dir)

    return SimulationResult(
        script_path=script_path,
        elapsed_sec=elapsed,
        out_prefix=out_prefix,
        out_prefix_face=str(out_prefix_face) if out_prefix_face is not None else None,
        log_file_path=log_file_path,
        analysis_run_id=analysis_run_id,
        analysis_log_dir=analysis_log_dir,
        namespace=namespace,
    )


def emit_all_outputs(result: SimulationResult, config: SimulationConfig) -> None:
    def display_path(path_value: str | Path) -> str:
        p = Path(path_value).expanduser()
        try:
            return str(p.resolve().relative_to(Path.cwd().resolve()))
        except Exception:
            return str(p)

    # core intentionally excludes analysis/CSV section from plot23 line 3282+.
    print(f"[研究] simulation complete in {result.elapsed_sec:.2f}s")
    print(f"[研究] script={display_path(result.script_path)}")
    print(f"[研究] out_prefix={result.out_prefix}")
    if result.log_file_path:
        print(f"[研究] log_file={display_path(result.log_file_path)}")
    if result.analysis_run_id:
        print(f"[研究] analysis_run_id={result.analysis_run_id}")
    if result.analysis_log_dir:
        print(f"[研究] analysis_log_dir={display_path(result.analysis_log_dir)}")
