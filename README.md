# SIMPL

> [!WARNING]
> 🚧 This is currently alpha software, undergoing active development. Some API changes are to be expected as we are work on the first stable release.


<img src="simpl.gif" width=850>

## Installation and Usage
This repository contains code for the ICLR 2025 paper "_SIMPL: Scalable and hassle-free optimisation of neural representations from behaviour_" ([ICLR](https://openreview.net/forum?id=9kFaNwX6rv)). Specifically: 

* Source code in `simpl/` for the SIMPL algorithm.
* A working example in `demos/simpl_demo.ipynb`.

To run the example you will need to install `simpl` by 

1. **Clone**: `git clone https://github.com/TomGeorge1234/SIMPL.git` and navigate to the root: `cd SIMPL`
3. _(Recommended)_ Create a virtual environment (e.g. `python -m venv simpl_env` and `source simpl_env/bin/activate`). 
4. **Install**: `pip install .[demo]`. This will install the `simpl` package and its dependencies.
5. **Run the demo**: `jupyter notebook demos/simpl_demo.ipynb` !


## Cite 
If you use SIMPL in yoor work, please cite it as: 

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
