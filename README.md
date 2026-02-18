## Research Package

### Reading Order
1. `main.py`
2. `simulation.py`
3. `engine/`

`main.py` is responsible only for loading configuration and starting execution.  
`simulation.py` controls the execution flow of each simulation step.  
`engine/` contains the core logic for recommendation and reaction computation.

Supporting responsibilities are separated into `agents/`, `steps/`, `utils/`, and `core/`.

## Environment
- Python 3.10+
- NumPy
- PyTorch
- CUDA-capable GPU + NVIDIA Driver (default configuration)
- Optional CPU execution by setting `STRICT_CUDA=0`