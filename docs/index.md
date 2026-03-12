# SIMPL

[![Tests](https://github.com/TomGeorge1234/SIMPL/actions/workflows/ci.yml/badge.svg)](https://github.com/TomGeorge1234/SIMPL/actions/workflows/ci.yml)

<img src="https://raw.githubusercontent.com/TomGeorge1234/SIMPL/main/simpl.gif" width=850>

**SIMPL** implements the algorithm from the ICLR 2025 paper [_"Scalable and hassle-free optimisation of neural representations from behaviour"_](https://openreview.net/forum?id=9kFaNwX6rv). It optimises neural representations (place fields) from behavioural initialisation using an EM-style algorithm.

{%
  include-markdown "../README.md"
  start="<!-- docs-intro-start -->"
  end="<!-- docs-intro-end -->"
%}

## Quick Start

```python
from simpl import SIMPL

model = SIMPL(
    speed_prior=0.4,
    kernel_bandwidth=0.02,
    bin_size=0.02,
    env_pad=0.0,
)
model.fit(Y, Xb, time, n_epochs=5)

model.X_        # decoded positions (T, D)
model.F_        # fitted receptive fields
model.results_  # full xarray.Dataset
```

See the [Getting Started](getting-started.md) guide for a full walkthrough, or jump to the [API Reference](api/simpl.md).

## Cite

If you use SIMPL in your work, please cite:

```bibtex
@inproceedings{
    george2025simpl,
    title={{SIMPL}: Scalable and hassle-free optimisation of neural representations from behaviour},
    author={Tom George and Pierre Glaser and Kim Stachenfeld and Caswell Barry and Claudia Clopath},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=9kFaNwX6rv}
}
```
