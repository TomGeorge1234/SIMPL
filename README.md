# SIMPL

> [!WARNING]
> 🚧 This is currently alpha software, undergoing active development. Expect abrupt API changes as we are currently working on the first stable release.


<img src="simpl.gif" width=850>

## Installation and Usage
This repository contains code for the "SIMPL: Scalable and hassle-free optimisation of neural representations from behaviour" ([preprint](https://www.biorxiv.org/content/10.1101/2024.11.11.623030v1)). Specifically: 

* Source code in `simpl/` for the SIMPL algorithm.
* A working example in `demos/simpl_demo.ipynb`.

To run the example you will need to install `simpl` by 

1. Navigating to the root of this repository (assuming you are viewing this from an anonymised website, download the repo first (top right)).
2. (Optional) Create and activate a virtual environment (however you usually do this, for example `python -m venv venv` and `source venv/bin/activate`). 
3. Running `pip install .[demo]`. This will install the `simpl` package and its dependencies.
4. Run the demo `jupyter notebook demos/simpl_demo.ipynb` !

## Cite 

If you use SIMPL in yuor work, please cite it as: 

> George, T. M., Glaser, P., Stachenfeld, K., Barry, C., & Clopath, C. (2024). SIMPL: Scalable and hassle-free optimization of neural representations from behaviour. bioRxiv, 2024-11.

```
@article{george2024simpl,
  title={SIMPL: Scalable and hassle-free optimization of neural representations from behaviour},
  author={George, Tom M and Glaser, Pierre and Stachenfeld, Kimberly and Barry, Caswell and Clopath, Claudia},
  journal={bioRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
