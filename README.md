# SIMPL

<img src="simpl.gif" width=850>

This repository contains code for the "SIMPL: Scalable and hassle-free optimisation of neural representations from behaviour". Specifically: 

* Source code in `simpl/` for the SIMPL algorithm.
* A working example in `demos/simpl_demo.ipynb`.

To run the example you will need to install `simpl` by 

1. Navigating to the root of this repository (assuming you are viewing this from an anonymised website, download the repo first (top right)).
2. (Optional) Create and activate a virtual environment (however you usually do this, for example `python -m venv venv` and `source venv/bin/activate`). 
3. Running `pip install .[demo]`. This will install the `simpl` package and its dependencies.
4. Run the demo `jupyter notebook demos/simpl_demo.ipynb` !