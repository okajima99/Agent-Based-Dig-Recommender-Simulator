## Package Guide

### Suggested Reading Order
1. `main.py`
2. `simulation.py`
3. `core.py`
4. `engine/`

### Responsibilities
- `main.py`: loads config and starts execution
- `simulation.py`: run orchestration and result/report output
- `core.py`: simulation runtime loop
- `agents/`: agent behavior and state update logic
- `engine/`: recommendation and cache computation
- `steps/`: per-step execution flow
- `utils/`: logging, runtime helpers, math/vector utilities
- `core/`: builders and internal wiring helpers

### Batch Run
- Run all algorithms:
  - `python3 run_all_algorithms.py`
- Short smoke run example:
  - `MAX_STEPS=100 NUM_AGENTS=200 NUM_CONTENTS=2000 python3 run_all_algorithms.py`
- Optional output prefix grouping:
  - `OUT_PREFIX_BASE=myrun`

### Analysis Log Layout
- Output root: `./analysis_logs/<algorithm_label>/<run_id>/` (relative to current working directory)
- `algorithm_label` includes face variants where relevant:
  - `cbf_affinity`, `cbf_novelty`, `cf_user_affinity`, `cf_item_novelty`
- Run ID markers:
  - `<run_id>/RUN_ID.txt`
  - `<algorithm_label>/latest_run_id.txt`
  - `<algorithm_label>/<run_id>.runid`
- Console/text log:
  - `<run_id>/<OUT_PREFIX_FACE>_analysis.txt`
