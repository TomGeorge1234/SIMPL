# Results Variable Registry

Every variable stored in `model.results_` (an `xarray.Dataset`) carries rich metadata
describing its name, dimensions, description, and (where applicable) a LaTeX formula.
This metadata is attached as `.attrs` on each `DataArray` and can be inspected at runtime:

```python
model.results_["F"].attrs["description"]
model.results_["F"].attrs["formula"]
```

The full registry is built by the internal helper below.  Expand it to browse
every variable that SIMPL can produce.

::: simpl._variable_registry
    options:
      show_root_heading: false
      members:
        - _build_variable_info_dict
        - _dict_to_dataset
      show_if_no_docstring: false
      docstring_style: numpy
