"""Spatial environment discretisation for SIMPL.

The ``Environment`` manages the spatial grid over which receptive fields
are defined.  It supports arbitrary dimensionality (1-D through n-D) and
provides coordinate arrays, discretised meshgrids, and plotting helpers.

Dimension naming convention:

* 1-D: ``['x']``
* 2-D: ``['x', 'y']``
* 3-D: ``['x', 'y', 'z']``
* Higher: ``['x1', 'x2', ..., 'xD']``

``SIMPL.fit()`` builds an ``Environment`` automatically from the data and the
``bin_size``, ``env_pad``, and ``env_lims`` hyperparameters. ``SIMPL`` does not
accept a pre-built environment. This class remains available directly for
inspecting or plotting regular spatial grids.
"""

import warnings

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

AUTO_BINS_PER_LARGEST_DIM = 25
LARGE_GRID_WARNING_BINS = 100_000


class Environment:
    """Regular rectilinear grid used internally by SIMPL.

    The grid covers either the data range plus ``pad`` or the explicit
    ``force_lims``. Coordinates are bin centres spaced by ``bin_size``. SIMPL
    constructs this class during ``SIMPL.fit``; an ``Environment`` instance
    is not accepted by the ``SIMPL`` constructor.

    Parameters
    ----------
    X : np.ndarray, shape (T, D)
        Latent positions used to infer the grid limits when ``force_lims`` is None.
    pad : float, optional
        Padding outside the data bounds, in the same units as ``X``. Ignored
        when ``force_lims`` is provided. By default 0.1.
    bin_size : float, optional
        Grid spacing in the same units as ``X``. By default 0.02.
    force_lims : tuple or None, optional
        Explicit lower and upper limits formatted as
        ``((min_dim1, ..., min_dimD), (max_dim1, ..., max_dimD))``. These
        replace limits inferred from ``X``. By default None.
    verbose : bool, optional
        Whether to print a grid summary. By default True.

    Notes
    -----
    ``discrete_env_shape`` gives the per-dimension grid shape,
    ``flattened_discretised_coords`` has shape ``(N_bins, D)``, and
    ``coords_dict`` maps dimension names to their one-dimensional coordinates.
    """

    def __init__(
        self,
        X: np.ndarray,
        pad: float = 0.1,
        bin_size: float | str = 0.02,
        force_lims: tuple | None = None,
        verbose: bool = True,
    ) -> None:

        self.data_lims = None
        self.pad = pad
        if force_lims is None:
            if X.ndim != 2:
                raise ValueError("X should be a 2D array of size (T x D).")
            self.data_lims = (tuple(np.floor(X.min(axis=0) * 100) / 100), tuple(np.ceil(X.max(axis=0) * 100) / 100))
            self.lims = (
                tuple(np.floor((X.min(axis=0) - pad) * 100) / 100),
                tuple(np.ceil((X.max(axis=0) + pad) * 100) / 100),
            )
            self.D = X.shape[1]
        else:
            self.lims = force_lims
            self.D = len(force_lims[0])
        self.extent = ()  # like lims but more matplotlib friendly, (minx, maxx, miny, maxy)
        for d in range(self.D):
            for i in range(2):
                self.extent += (self.lims[i][d],)

        if bin_size == "auto":
            bin_size = np.max(np.subtract(self.lims[1], self.lims[0])) / AUTO_BINS_PER_LARGEST_DIM
        self.bin_size = bin_size

        # create dim names

        if self.D == 1:
            self.dim = ["x"]
        elif self.D == 2:
            self.dim = ["x", "y"]
        elif self.D == 3:
            self.dim = ["x", "y", "z"]
        else:
            self.dim = [f"x{i}" for i in range(self.D)]

        # Make the coordinate arrays
        self.coords_dict = {}
        for i, dim in enumerate(self.dim):
            coordinate_array = np.arange(self.lims[0][i] + bin_size / 2, self.lims[1][i], bin_size)
            self.coords_dict[dim] = coordinate_array

        # make the discretised coords
        self.discretised_coords = np.stack(
            np.meshgrid(*self.coords_dict.values(), indexing="ij")
        )  # (D, N_xbins, N_ybins, ...)
        self.discrete_env_shape = self.discretised_coords.shape[1:]
        self.flattened_discretised_coords = self.discretised_coords.reshape(self.D, -1).T

        n_bins = self.flattened_discretised_coords.shape[0]
        if n_bins > LARGE_GRID_WARNING_BINS:
            warnings.warn(
                f"The environment grid contains {n_bins:,} bins, which may require substantial compute and memory. "
                "Consider using a larger bin_size.",
                stacklevel=2,
            )

        if verbose:
            print(
                f"Created a {self.D}D cuboid environment with dimensions "
                f"{self.dim} and discretised shape {self.discrete_env_shape}"
            )
            print(f"Environment limits are {self.lims}")
            print(
                f"The coords of each dimension are stored in "
                f"self.coords_dict and a list of combined {self.dim} "
                f"coords for all bins is stored in "
                f"self.discretised_coords"
            )

    def plot_environment(self, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
        """Plots the environment axes.

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            The axes to plot on. Default is None.

        Returns
        -------
        ax : matplotlib.Axes
            The axes with the environment plotted on it.
        """

        if self.D == 1:
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 1))
            ax.set_xlim(self.lims[0][0], self.lims[1][0])
            ax.set_xlabel(self.dim[0])

        if self.D == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_xlim(self.lims[0][0], self.lims[1][0])
            ax.set_ylim(self.lims[0][1], self.lims[1][1])
            ax.set_aspect("equal")
            # turn of x and y axis
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel(self.dim[0])
            ax.set_ylabel(self.dim[1])
            if self.data_lims is not None:
                ax.plot(
                    [
                        self.data_lims[0][0],
                        self.data_lims[1][0],
                        self.data_lims[1][0],
                        self.data_lims[0][0],
                        self.data_lims[0][0],
                    ],
                    [
                        self.data_lims[0][1],
                        self.data_lims[0][1],
                        self.data_lims[1][1],
                        self.data_lims[1][1],
                        self.data_lims[0][1],
                    ],
                    color="white",
                    linestyle="--",
                    linewidth=1,
                    zorder=2,
                )

        return ax
