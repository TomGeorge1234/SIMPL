# SIMPL 

<img src="simpl.gif" width=850>

## Installation and Usage
This repository contains code for the ICLR 2025 paper "_SIMPL: Scalable and hassle-free optimisation of neural representations from behaviour_" ([ICLR](https://openreview.net/forum?id=9kFaNwX6rv)). Specifically:

* Source code in `src/simpl/` for the SIMPL algorithm.
* A working example in `examples/simpl_demo.ipynb`.

To run the example you will need to install `simpl` by

1. **Clone**: `git clone https://github.com/TomGeorge1234/SIMPL.git` and navigate to the root: `cd SIMPL`
2. _(Recommended)_ Create a virtual environment (e.g. `python -m venv simpl_env` and `source simpl_env/bin/activate`).
3. **Install**: `pip install .[demos]`. This will install the `simpl` package and its dependencies.
4. **Run the demo**: `jupyter notebook examples/simpl_demo.ipynb` !

## Package Structure

```
src/simpl/
├── __init__.py        # Top-level exports: SIMPL, Environment, prepare_data, ...
├── simpl.py           # Core SIMPL class (EM algorithm)
├── environment.py     # Environment class (spatial discretisation)
├── utils.py           # Gaussian helpers, CCA, data prep, I/O
├── kalman.py          # KalmanFilter class + Kalman functions
├── kde.py             # KDE, Poisson log-likelihood, gaussian_kernel
└── data/              # Bundled demo data
```

## Top-level API

```python
from simpl import SIMPL, Environment, prepare_data

# Load and prepare data
data = prepare_data(Y=spikes, Xb=positions, time=timestamps)
env = Environment(X=positions)

# Fit the model
model = SIMPL(data=data, environment=env)
model.train_N_epochs(5)
```

## Development

```bash
# Install for development
pip install -e ".[dev]"

# Lint
ruff check src/
ruff format --check src/

# Run tests
pytest --cov=simpl
```

## Cite
If you use SIMPL in your work, please cite it as:

> Tom George, Pierre Glaser, Kim Stachenfeld, Caswell Barry, & Claudia Clopath (2025). SIMPL: Scalable and hassle-free optimisation of neural representations from behaviour. In The Thirteenth International Conference on Learning Representations.

```
@inproceedings{
george2025simpl,
title={{SIMPL}: Scalable and hassle-free optimisation of neural representations from behaviour},
author={Tom George and Pierre Glaser and Kim Stachenfeld and Caswell Barry and Claudia Clopath},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=9kFaNwX6rv}
}
```
