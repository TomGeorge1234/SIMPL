import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np


class Environment:
    """Basic environment class.

    Key attributes are
    - lims: the limits of the environment, a tuple of two tuples ((min_dim1, ..., min_dimD),(max_dim1, ..., max_dimD))
    - extent : like lims but more matplotlib friendly, (min_dim1, max_dim1, min_dim2, max_dim2, ...)
    - pad : how much the environment is padded, in m, outside the bounds of the behaviour
    - bin_size: the size of the bins in the environment
    - D: the dimensionality of the environment
    - dim : the names of the dimensions of the environment. Originating from the
            spatial-maps origins of this code we use the following dimension
            naming convention:
            1D: ['x']
            2D: ['x', 'y']
            3D: ['x', 'y', 'z']
            ...
            DD: ['x1', 'x2', 'x3', ..., 'xD']
    - coords_dict: a dictionary mapping the coordinate names to the coordinate
            arrays in the environment. These a strictly increasing arrays of
            the form np.linspace(lims[0][i], lims[1][i], N_bins) for each
            dimension i.
    - dicretised_coords: an array of coordinates discretising the env,
            flattened into shape (N_bins x D) where
            (N_bins = N_xbins, x N_ybins x ...)
    - discrete_env_shape: the shape of the discretised environment.
            Specifically, _any_ array of shape (..., N_bins, ...) can be
            reshaped to (..., N_xbins, N_ybins, N_zbins, ...). We always
            recommend the following:
        ```python
        array = np.moveaxis(array, axis_of_size_N_bins, -1)
        array = array.reshape(array.shape[:-1] + discrete_env_shape)
        ```

    A note on visualising environment variables: A 2D tensor reshaped to
    discrete_env_shape and visualise (e.g. using matplotlib.imshow()) will
    have x going down the rows and y going across the columns which is not
    conventional. Instead you should swap the x and y dimensions then
    reverse the y. Instead of plt.imshow(array) you should use
    plt.imshow(array.T[::-1, :]). A BETTER way to do this is to try and
    always store the array as a xarray with names dimensions and
    coordinates.

    Environments can optionally have a "plot_environment()" - this should
    return an single matplotlib.Axes object with the environment (and
    anything important) plotted on it. This is useful for visualising the
    environment and used by the plotting module.


    Parameters
    ----------
    X : np.ndarray (T, D)
        A sample of the latent variable, used to scale how big the environment is so it fits the data.
    pad : float, optional
        How much the environment is padded outside the bounds of the behaviour. Default is 0.1 m.
    bin_size : float, optional
        The size of the bins in the environment. Default is 0.02 m.
    force_lims : tuple, optional
        The limits of the environment, this will override those calculated
        from Z and pad Z. Should be a two-tuple like
        ((min_dim1, ..., min_dimD),(max_dim1, ..., max_dimD)).
        Default is None."""

    def __init__(
        self,
        X: np.ndarray,
        pad: float = 0.1,
        bin_size: float = 0.02,
        force_lims: tuple | None = None,
        verbose: bool = True,
    ) -> None:

        self.data_lims = None
        self.pad = pad
        if force_lims is None:
            if X.ndim != 2:
                raise ValueError("X should be a 2D array of size (T x D).")
            self.data_lims = (tuple(np.round(X.min(axis=0), 2)), tuple(np.round(X.max(axis=0), 2)))
            self.lims = (tuple(np.round(X.min(axis=0) - pad, 2)), tuple(np.round(X.max(axis=0) + pad, 2)))
            self.D = X.shape[1]
        else:
            self.lims = force_lims
            self.D = len(force_lims[0])
        self.extent = ()  # like lims but more matplotlib friendly, (minx, maxx, miny, maxy)
        for d in range(self.D):
            for i in range(2):
                self.extent += (self.lims[i][d],)

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
