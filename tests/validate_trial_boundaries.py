"""
Test script to validate trial boundaries functionality in SIMPL.

This script compares SIMPL runs with and without trial boundaries to verify
that trial boundaries prevent smoothing across trial boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simpl.utils import load_datafile, prepare_data
from simpl.environment import Environment
from simpl.simpl import SIMPL
import kalmax


def load_and_prepare_demo_data(trial_boundaries=None):
    """Load and prepare data from the demo.
    
    Parameters
    ----------
    trial_boundaries : np.ndarray, optional
        Trial boundaries to pass to prepare_data. If None, no trial boundaries.
    
    Returns
    -------
    data : xr.Dataset
        Prepared data for SIMPL
    """
    # Load data
    data_raw = load_datafile('gridcelldata.npz')
    
    Y = data_raw['Y']
    Xb = data_raw['Xb']
    time = data_raw['time']
    Xt = data_raw['Xt']
    dims = data_raw['dim']
    neuron = data_raw['neuron']
    Ft = data_raw['Ft']
    xbins = data_raw['x']
    ybins = data_raw['y']
    
    # Prepare data
    data = prepare_data(
        Y=Y,
        Xb=Xb,
        time=time,
        dims=dims,
        neurons=neuron,
        Xt=Xt,
        Ft=Ft,
        Ft_coords_dict={'x': xbins, 'y': ybins},
        trial_boundaries=trial_boundaries,
    )
    
    return data


def create_environment(data):
    """Create environment for SIMPL.
    
    Parameters
    ----------
    data : xr.Dataset
        Data containing Xb values
        
    Returns
    -------
    env : Environment
        Environment object
    """
    PAD = 0.0
    DX = 0.02
    env = Environment(
        X=data.Xb.values,
        pad=PAD,
        bin_size=DX,
    )
    return env


def create_simpl_model(data, env):
    """Create and configure SIMPL model.
    
    Trial boundaries are read from data.attrs (set by prepare_data).
    
    Parameters
    ----------
    data : xr.Dataset
        Prepared data (from prepare_data, with trial_boundaries in attrs if applicable)
    env : Environment
        Environment object
        
    Returns
    -------
    simpl_model : SIMPL
        Configured SIMPL model
    """
    kernel = kalmax.kernels.gaussian_kernel
    kernel_bandwidth = 0.02
    speed_prior = 0.4
    behaviour_prior = 0.02
    test_frac = 0.1
    speckle_block_size_seconds = 1
    manifold_align_against = 'behaviour'
    evaluate_each_epoch = True
    save_likelihood_maps = False
    resample_spike_mask = True
    
    simpl_model = SIMPL(
        data=data,
        environment=env,
        kernel=kernel,
        kernel_bandwidth=kernel_bandwidth,
        speed_prior=speed_prior,
        behaviour_prior=behaviour_prior,
        test_frac=test_frac,
        speckle_block_size_seconds=speckle_block_size_seconds,
        manifold_align_against=manifold_align_against,
        evaluate_each_epoch=evaluate_each_epoch,
        save_likelihood_maps=save_likelihood_maps,
        resample_spike_mask=resample_spike_mask,
    )
    
    return simpl_model


def create_trial_boundaries(T, num_trials=5):
    """Create equal-sized trial boundaries.
    
    Parameters
    ----------
    T : int
        Total number of time steps
    num_trials : int, optional
        Number of trials to create, by default 5
        
    Returns
    -------
    trial_boundaries : np.ndarray
        Array of trial boundary indices, starting with 0
    """
    trial_size = T // num_trials
    trial_boundaries = np.array([i * trial_size for i in range(num_trials)])
    return trial_boundaries


def run_simpl_training(simpl_model, num_epochs=5):
    """Run SIMPL training.
    
    Parameters
    ----------
    simpl_model : SIMPL
        SIMPL model to train
    num_epochs : int, optional
        Number of epochs to train, by default 5
    """
    simpl_model.calculate_baselines()
    simpl_model.train_N_epochs(N=num_epochs)


def compare_latent_variables(results_no_boundaries, results_with_boundaries, epoch=None):
    """Compare latent variables between two SIMPL runs.
    
    Parameters
    ----------
    results_no_boundaries : xr.Dataset
        Results from run without trial boundaries
    results_with_boundaries : xr.Dataset
        Results from run with trial boundaries
    epoch : int, optional
        Epoch to compare. If None, uses the maximum training epoch (excluding baselines -2, -1), by default None
        
    Returns
    -------
    comparison : dict
        Dictionary containing comparison metrics
    """
    # Find the last training epoch if not specified
    if epoch is None:
        epochs_no_boundaries = results_no_boundaries.epoch.values
        epochs_with_boundaries = results_with_boundaries.epoch.values
        # Exclude baseline epochs (-2, -1) and find max training epoch
        training_epochs_no_boundaries = epochs_no_boundaries[epochs_no_boundaries >= 0]
        training_epochs_with_boundaries = epochs_with_boundaries[epochs_with_boundaries >= 0]
        epoch = min(np.max(training_epochs_no_boundaries), np.max(training_epochs_with_boundaries))
        print(f"   Comparing at epoch {epoch}")
    # Extract latent variables
    X_no_boundaries = results_no_boundaries.X.sel(epoch=epoch).values
    X_with_boundaries = results_with_boundaries.X.sel(epoch=epoch).values
    
    # Calculate differences
    diff = X_with_boundaries - X_no_boundaries
    abs_diff = np.abs(diff)
    
    comparison = {
        'epoch': epoch,
        'mean_abs_diff': np.mean(abs_diff),
        'max_abs_diff': np.max(abs_diff),
        'mean_diff': np.mean(diff, axis=0),
        'std_diff': np.std(diff, axis=0),
        'X_no_boundaries': X_no_boundaries,
        'X_with_boundaries': X_with_boundaries,
        'diff': diff,
    }
    
    return comparison


def plot_comparison(results_no_boundaries, results_with_boundaries, 
                   trial_boundaries, comparison, save_path=None):
    """Plot comparison of latent variables.
    
    Parameters
    ----------
    results_no_boundaries : xr.Dataset
        Results from run without trial boundaries
    results_with_boundaries : xr.Dataset
        Results from run with trial boundaries
    trial_boundaries : np.ndarray
        Trial boundary indices
    comparison : dict
        Comparison dictionary from compare_latent_variables (must contain 'epoch')
    save_path : str, optional
        Path to save figure, by default None
    """
    epoch = comparison['epoch']
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    time = results_no_boundaries.time.values
    
    # Plot 1: Latent variables overlay
    ax = axes[0]
    X_no_boundaries = comparison['X_no_boundaries']
    X_with_boundaries = comparison['X_with_boundaries']
    
    ax.plot(time, X_no_boundaries[:, 0], 'b-', alpha=0.7, label='No boundaries', linewidth=2)
    ax.plot(time, X_with_boundaries[:, 0], 'r-', alpha=0.7, label='With boundaries', linewidth=2)
    
    # Mark trial boundaries
    for boundary in trial_boundaries[1:]:  # Skip first and last
        ax.axvline(time[boundary], color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('x-position (m)')
    ax.set_title(f'Latent Variables Comparison (Epoch {epoch})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Difference
    ax = axes[1]
    diff = comparison['diff']
    ax.plot(time, diff[:, 0], 'g-', alpha=0.7, linewidth=1.5)
    
    # Mark trial boundaries
    for boundary in trial_boundaries[1:]:
        ax.axvline(time[boundary], color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Difference (m)')
    ax.set_title('Difference: With Boundaries - No Boundaries')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    
    # Plot 3: Absolute difference with boundary markers
    ax = axes[2]
    abs_diff = np.abs(diff)
    ax.plot(time, abs_diff[:, 0], 'purple', alpha=0.7, linewidth=1.5)
    
    # Mark trial boundaries more prominently
    for boundary in trial_boundaries[1:]:
        ax.axvline(time[boundary], color='red', linestyle='--', alpha=0.7, linewidth=2, label='Trial boundary' if boundary == trial_boundaries[1] else '')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Absolute Difference (m)')
    ax.set_title('Absolute Difference (larger at boundaries expected)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_boundary_triggered_average(comparison, trial_boundaries, time, lag=50, save_path=None):
    """Plot trial-boundary triggered average of differences.
    
    Parameters
    ----------
    comparison : dict
        Comparison dictionary from compare_latent_variables (must contain 'diff')
    trial_boundaries : np.ndarray
        Trial boundary indices
    time : np.ndarray
        Time array
    lag : int, optional
        Number of time steps before and after each boundary to include, by default 50
    save_path : str, optional
        Path to save figure, by default None
    """
    diff = comparison['diff']
    T = len(time)
    dt = time[1] - time[0] if len(time) > 1 else 1.0
    
    # Extract windows around each boundary (excluding first and last)
    boundary_indices = trial_boundaries[1:-1]  # Skip first (0) and last (T)
    
    if len(boundary_indices) == 0:
        print("   No trial boundaries to plot (only one trial)")
        return None
    
    # Collect all traces around boundaries
    traces = []
    abs_traces = []
    time_rel = np.arange(-lag, lag + 1) * dt  # Relative time around boundary
    
    for boundary_idx in boundary_indices:
        start_idx = max(0, boundary_idx - lag)
        end_idx = min(T, boundary_idx + lag + 1)
        
        # Extract window
        window_diff = diff[start_idx:end_idx, :]  # Shape: (window_length, D)
        
        # Pad if necessary (if boundary is near start/end)
        if start_idx == 0:
            # Pad at beginning
            pad_size = lag - boundary_idx
            window_diff = np.pad(window_diff, ((pad_size, 0), (0, 0)), mode='constant', constant_values=np.nan)
        elif end_idx == T:
            # Pad at end
            pad_size = lag + 1 - (T - boundary_idx)
            window_diff = np.pad(window_diff, ((0, pad_size), (0, 0)), mode='constant', constant_values=np.nan)
        
        # Ensure correct length
        if len(window_diff) < 2 * lag + 1:
            continue
            
        traces.append(window_diff)
        abs_traces.append(np.abs(window_diff))
    
    if len(traces) == 0:
        print("   No valid boundary windows to plot")
        return None
    
    traces = np.array(traces)  # Shape: (n_boundaries, window_length, D)
    abs_traces = np.array(abs_traces)  # Shape: (n_boundaries, window_length, D)
    
    # Calculate averages
    mean_diff = np.nanmean(traces, axis=0)  # Shape: (window_length, D)
    mean_abs_diff = np.nanmean(abs_traces, axis=0)  # Shape: (window_length, D)
    
    # Create figure with subplots for each dimension
    n_dims = diff.shape[1]
    fig, axes = plt.subplots(n_dims, 2, figsize=(14, 4 * n_dims))
    if n_dims == 1:
        axes = axes.reshape(1, -1)
    
    for dim_idx in range(n_dims):
        # Left plot: All traces + mean (signed difference)
        ax = axes[dim_idx, 0]
        
        # Plot individual traces with transparency
        for trace in traces:
            ax.plot(time_rel, trace[:, dim_idx], 'gray', alpha=0.2, linewidth=0.5)
        
        # Plot mean difference
        ax.plot(time_rel, mean_diff[:, dim_idx], 'b-', linewidth=2.5, label='Mean difference', zorder=10)
        
        # Vertical line at boundary
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Trial boundary', zorder=9)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3, zorder=1)
        
        ax.set_xlabel('Time relative to boundary (s)')
        ax.set_ylabel(f'Difference (dim {dim_idx})')
        ax.set_title(f'Signed Difference Around Boundaries (dim {dim_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right plot: All absolute traces + mean absolute difference
        ax = axes[dim_idx, 1]
        
        # Plot individual absolute traces with transparency
        for abs_trace in abs_traces:
            ax.plot(time_rel, abs_trace[:, dim_idx], 'gray', alpha=0.2, linewidth=0.5)
        
        # Plot mean absolute difference
        ax.plot(time_rel, mean_abs_diff[:, dim_idx], 'purple', linewidth=2.5, label='Mean |difference|', zorder=10)
        
        # Vertical line at boundary
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Trial boundary', zorder=9)
        
        ax.set_xlabel('Time relative to boundary (s)')
        ax.set_ylabel(f'|Difference| (dim {dim_idx})')
        ax.set_title(f'Absolute Difference Around Boundaries (dim {dim_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"   Boundary-triggered average plot saved to {save_path}")
    
    return fig


def main(num_trials=5, num_epochs=5, save_plots=True):
    """Main test function.
    
    Parameters
    ----------
    num_trials : int, optional
        Number of trials to create, by default 5
    num_epochs : int, optional
        Number of epochs to train, by default 5
    save_plots : bool, optional
        Whether to save comparison plots, by default True
    """
    print("=" * 60)
    print("SIMPL Trial Boundaries Validation Test")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    data_raw = load_datafile('gridcelldata.npz')
    T = len(data_raw['time'])
    print(f"   Data loaded: {T} time steps")
    
    # Create trial boundaries
    print(f"\n2. Creating {num_trials} equal-sized trials...")
    trial_boundaries = create_trial_boundaries(T, num_trials=num_trials)
    print(f"   Trial boundaries: {trial_boundaries}")
    
    # Prepare data (without and with trial boundaries)
    data_no_boundaries = load_and_prepare_demo_data()
    data_with_boundaries = load_and_prepare_demo_data(trial_boundaries=trial_boundaries)
    
    # Create environment
    print("\n3. Creating environment...")
    env = create_environment(data_no_boundaries)
    print("   Environment created")
    
    # Run 1: Without trial boundaries
    print("\n4. Running SIMPL WITHOUT trial boundaries...")
    simpl_model_no_boundaries = create_simpl_model(data_no_boundaries, env)
    run_simpl_training(simpl_model_no_boundaries, num_epochs=num_epochs)
    results_no_boundaries = simpl_model_no_boundaries.results
    print("   Training complete")
    
    # Run 2: With trial boundaries
    print("\n5. Running SIMPL WITH trial boundaries...")
    simpl_model_with_boundaries = create_simpl_model(data_with_boundaries, env)
    run_simpl_training(simpl_model_with_boundaries, num_epochs=num_epochs)
    results_with_boundaries = simpl_model_with_boundaries.results
    print("   Training complete")
    
    # Compare results
    print("\n6. Comparing latent variables...")
    comparison = compare_latent_variables(
        results_no_boundaries, 
        results_with_boundaries, 
        epoch=None  # Auto-detect last training epoch
    )
    
    print(f"\n   Comparison Results:")
    print(f"   Mean absolute difference: {comparison['mean_abs_diff']:.6f} m")
    print(f"   Max absolute difference: {comparison['max_abs_diff']:.6f} m")
    print(f"   Mean difference per dimension: {comparison['mean_diff']}")
    print(f"   Std difference per dimension: {comparison['std_diff']}")
    
    # Check differences at trial boundaries
    print(f"\n7. Checking differences at trial boundaries...")
    boundary_diffs = []
    for boundary in trial_boundaries[1:]:  # Skip first and last
        # Check a small window around each boundary
        window = 5  # Check ±5 time steps around boundary
        start_idx = max(0, boundary - window)
        end_idx = min(T, boundary + window)
        boundary_abs_diff = np.mean(np.abs(comparison['diff'][start_idx:end_idx, :]))
        boundary_diffs.append(boundary_abs_diff)
        print(f"   Boundary at t={boundary}: mean abs diff in window = {boundary_abs_diff:.6f} m")
    
    # Compare boundary regions to non-boundary regions
    boundary_mask = np.zeros(T, dtype=bool)
    window = 5
    for boundary in trial_boundaries[1:]:
        start_idx = max(0, boundary - window)
        end_idx = min(T, boundary + window)
        boundary_mask[start_idx:end_idx] = True
    
    boundary_region_diff = np.mean(np.abs(comparison['diff'][boundary_mask, :]))
    non_boundary_region_diff = np.mean(np.abs(comparison['diff'][~boundary_mask, :]))
    
    print(f"\n   Mean abs diff in boundary regions (±{window} steps): {boundary_region_diff:.6f} m")
    print(f"   Mean abs diff in non-boundary regions: {non_boundary_region_diff:.6f} m")
    print(f"   Ratio (boundary/non-boundary): {boundary_region_diff/non_boundary_region_diff:.3f}")
    
    if boundary_region_diff > non_boundary_region_diff:
        print("   ✓ As expected: Larger differences at trial boundaries!")
    else:
        print("   ⚠ Unexpected: Differences not larger at boundaries")
    
    # Plot comparison
    if save_plots:
        print("\n8. Generating comparison plots...")
        plot_path = Path(__file__).parent / 'trial_boundaries_comparison.png'
        plot_comparison(
            results_no_boundaries,
            results_with_boundaries,
            trial_boundaries,
            comparison,
            save_path=str(plot_path)
        )
        print(f"   Comparison plot saved")
        
        # Plot boundary-triggered average
        print("\n9. Generating boundary-triggered average plot...")
        boundary_plot_path = Path(__file__).parent / 'trial_boundaries_triggered_average.png'
        time = results_no_boundaries.time.values
        plot_boundary_triggered_average(
            comparison,
            trial_boundaries,
            time,
            lag=50,  # 50 time steps before/after each boundary
            save_path=str(boundary_plot_path)
        )
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    
    return {
        'results_no_boundaries': results_no_boundaries,
        'results_with_boundaries': results_with_boundaries,
        'comparison': comparison,
        'trial_boundaries': trial_boundaries,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate SIMPL trial boundaries functionality')
    parser.add_argument('--num_trials', type=int, default=100,
                        help='Number of trials to create (default: 100)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip saving plots')
    
    args = parser.parse_args()
    
    main(
        num_trials=args.num_trials,
        num_epochs=args.num_epochs,
        save_plots=not args.no_plots
    )
