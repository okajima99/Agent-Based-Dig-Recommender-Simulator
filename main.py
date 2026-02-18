from __future__ import annotations

try:
    from .config import load_config_from_env
    from .simulation import emit_all_outputs, run_simulation
except ImportError:
    from config import load_config_from_env
    from simulation import emit_all_outputs, run_simulation


def main() -> int:
    config = load_config_from_env()
    result = run_simulation(config)
    emit_all_outputs(result, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
