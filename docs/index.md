<p align="center">
  <img src="assets/simpl_logo.png" width="420" alt="SIMPL logo">
</p>

<div align="center">

<a href="https://github.com/TomGeorge1234/SIMPL/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/TomGeorge1234/SIMPL/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://colab.research.google.com/github/TomGeorge1234/SIMPL/blob/main/examples/simpl_demo.ipynb"><img alt="Colab demo" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
<a href="https://openreview.net/pdf?id=9kFaNwX6rv"><img alt="Paper" src="https://img.shields.io/badge/paper-ICLR%202025-blue"></a>
<a href="https://pepy.tech/projects/simpl-neuro"><img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/simpl-neuro"></a>

<br>

<p>A JAX-python package for <strong>optimising latent representations and neural tuning curves</strong> from spike data. It does this by iteratively decoding the latent and fitting the tuning curves, starting from behavior or stimuli. It is lightweight, scalable, and very fast. Published at <a href="https://openreview.net/forum?id=9kFaNwX6rv">ICLR 2025</a>.</p>

</div>

<p align="center">
<img src="assets/simpl.gif" width=850>
</p>

{%
  include-markdown "../README.md"
  rewrite-relative-urls=false
  start="<!-- docs-intro-start -->"
  end="<!-- docs-intro-end -->"
%}

See the [Quickstart](quickstart.md) for the core workflow, [Examples/Demos](examples.md) for complete walkthroughs, or [Code & Docstring Reference](api/simpl.md) for raw docstrings.

{%
  include-markdown "../README.md"
  rewrite-relative-urls=false
  start="<!-- docs-cite-start -->"
  end="<!-- docs-cite-end -->"
%}
