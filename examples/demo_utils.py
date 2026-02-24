# Demo utilities for SIMPL examples
# Ported from kalmax demo_utils.py + make_simulated_dataset

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def animate_over_time(
    plot_function,
    t_start,
    t_end,
    speed_up=10,
    fps=20,
):
    """Animates the plot function over time_range.

    Parameters
    ----------
    plot_function : callable
        Function that takes ax, time and time_start as input and returns ax.
        Must have the following signature:
        def plot_function(ax, time, time_start=0):
            # do something
            return ax
    t_start : float
        Start time
    t_end : float
        End time
    speed_up : float
        Speed multiplier relative to real time
    fps : int
        Frames per second
    """
    time_per_frame = speed_up / fps
    times = np.arange(t_start, t_end, time_per_frame)

    def update(time, ax, fig):
        for ax_ in fig.get_axes():
            ax_.clear()
        ax = plot_function(ax=ax, t_start=t_start, t_end=time)
        return ax

    ax = plot_function()
    from matplotlib.animation import FuncAnimation

    anim = FuncAnimation(plt.gcf(), update, frames=times, fargs=(ax, plt.gcf()), interval=1000 / fps)
    plt.close()
    return anim


def plot_trajectory(
    trajectory,
    time_stamps,
    ax=None,
    t_start=None,
    t_end=None,
    **plot_kwargs,
):
    """Plots a trajectory over a given time range.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory to plot, shape (T, 2)
    time_stamps : np.ndarray
        Time stamps for the trajectory, shape (T,)
    ax : matplotlib axis, optional
        Axis to plot on
    t_start : float, optional
        Start time
    t_end : float, optional
        End time
    plot_kwargs : dict, optional
        Additional plotting arguments

    Returns
    -------
    ax : matplotlib axis
    """
    if t_start is None:
        t_start = time_stamps[0]
    if t_end is None:
        t_end = time_stamps[-1]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    id_start, id_end = np.argmin(np.abs(time_stamps - t_start)), np.argmin(np.abs(time_stamps - t_end))
    trajectory_ = trajectory[id_start:id_end]

    color = plot_kwargs.get("color", "k")
    scatter_points = plot_kwargs.get("scatter_points", True)
    show_line = plot_kwargs.get("show_line", True)
    linewidth = plot_kwargs.get("linewidth", 1)
    title = plot_kwargs.get("title", None)
    xlabel = plot_kwargs.get("xlabel", "x [m]")
    ylabel = plot_kwargs.get("ylabel", "y [m]")
    alpha = plot_kwargs.get("alpha", 1)
    label = plot_kwargs.get("label", None)
    min_x = plot_kwargs.get("min_x", trajectory[:, 0].min().round(1))
    max_x = plot_kwargs.get("max_x", trajectory[:, 0].max().round(1))
    min_y = plot_kwargs.get("min_y", trajectory[:, 1].min().round(1))
    max_y = plot_kwargs.get("max_y", trajectory[:, 1].max().round(1))

    if show_line:
        ax.plot(trajectory_[:, 0], trajectory_[:, 1], color=color, linewidth=linewidth, alpha=alpha, label=label)
    if scatter_points:
        ax.scatter(trajectory_[:, 0], trajectory_[:, 1], color=color, linewidth=0, s=6, alpha=alpha)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", "box")
    ax.set_xticks([min_x, max_x])
    ax.set_yticks([min_y, max_y])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return ax


def plot_ellipse(ax, mean, cov, color):
    """Draw an uncertainty ellipse on the given axes.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    mean : array-like
        Center of the ellipse (x, y)
    cov : array-like
        2x2 covariance matrix
    color : str or color
        Color of the ellipse
    """
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = matplotlib.patches.Ellipse(
        xy=mean,
        width=lambda_[0] * 2,
        height=lambda_[1] * 2,
        angle=np.rad2deg(np.arctan(v[:, 0][1] / v[:, 0][0])),
        lw=1,
        fill=True,
        edgecolor=color,
        facecolor=color,
        alpha=0.5,
    )
    ax.add_artist(ell)
    return ax


def make_simulated_dataset(time_mins=60, n_cells=100, firing_rate=10, random_seed=None, **kwargs):
    """Makes a simulated dataset for an agent randomly foraging a 1 m square box.
    Data generated with the RatInABox package and defaults to place cells.

    Parameters
    ----------
    time_mins : int
        The number of minutes to simulate the agent for
    n_cells : int
        The number of place cells to simulate. Default is 100.
    firing_rate : float
        The maximum firing rate of each place cell in Hz. Default is 10.
    random_seed : int, optional
        If provided, sets the numpy random seed for reproducibility.
    kwargs : dict
        Additional arguments to pass to the RatInABox simulation

    Returns
    -------
    time : jnp.ndarray, shape (N,)
        The time points of the simulation
    position : jnp.ndarray, shape (N, dims)
        The position of the agent at each time point
    spikes : jnp.ndarray, shape (N, N_cells)
        The spikes of the place cells at each time point
    """
    import tqdm as tqdm
    from ratinabox.Agent import Agent
    from ratinabox.Environment import Environment
    from ratinabox.Neurons import PlaceCells

    if random_seed is not None:
        np.random.seed(random_seed)

    env_params = kwargs.get("env_params", {})
    agent_params = kwargs.get("agent_params", {"dt": 0.1})
    place_cell_params = kwargs.get("place_cell_params", {"n": n_cells, "max_fr": firing_rate, "widths": 0.1})

    env = Environment(params=env_params)
    agent = Agent(env, params=agent_params)
    place_cells = PlaceCells(agent, params=place_cell_params)

    for i in tqdm.tqdm(range(int(60 * time_mins / agent.dt))):
        agent.update()
        place_cells.update()

    time = jnp.array(agent.history["t"])
    position = jnp.array(agent.history["pos"])
    spikes = jnp.array(place_cells.history["spikes"])

    return time, position, spikes
