from datetime import datetime
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

from tumortwin.postprocessing.total_cell_count import compute_total_cell_count
from tumortwin.types.base import BasePatientData
from tumortwin.utils import days_since_first


def plot_loss(losses: torch.Tensor, ax: Optional[Axes] = None):
    log_losses = np.log10([loss.detach() for loss in losses])

    # Determine y-axis tick limits
    y_min, y_max = np.ceil(np.min(log_losses) - 1), np.floor(np.max(log_losses) + 1)
    y_ticks = np.arange(y_max, y_min - 1, -1)  # Step of -1 for log scale

    # Create figure
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 2))
    ax.plot(log_losses, 'k')

    # Set y-axis ticks
    ax.set_yticks(y_ticks)
    y_min, y_max = np.ceil(np.min(log_losses)), np.floor(np.max(log_losses))
    y_ticks = np.arange(y_max, y_min - 1, -1)  # Step of -1 for log scale

    plt.xlabel("Optimization iteration")
    plt.ylabel("Loss (log10)")

    # Show the plot
    # plt.show()


def plot_maps_final(
    predicted: torch.Tensor,
    u_target: torch.Tensor,
    u0: torch.Tensor,
    patient_data: BasePatientData,
    t_final: float = 90,
    threshold: float = 0.01,
):
    x_id = np.s_[:]
    y_id = np.s_[:]
    u0_numpy = u0.detach().numpy()
    slice_sums = np.sum(u0_numpy, axis=(0, 1))  # Summing over x and y axes
    slice_id = np.argmax(slice_sums)

    t1_image = patient_data.T1_post_image.array[:, :, slice_id]  # T1 image (grayscale)
    t1_image = (t1_image - np.min(t1_image)) / (
        np.max(t1_image) - np.min(t1_image)
    )  # Normalize to [0,1]
    # Convert grayscale T1 to RGB
    t1_rgb = np.stack([t1_image] * 3, axis=-1)  # Convert grayscale to 3-channel RGB

    initial_cellularity = u0_numpy[x_id, y_id, slice_id]
    initial_cellularity_colored = plt.cm.viridis(initial_cellularity)[
        :, :, :3
    ]  # Apply colormap and remove alpha
    mask = initial_cellularity >= threshold
    blended_initial = t1_rgb.copy()
    blended_initial[mask] = initial_cellularity_colored[
        mask
    ]  # Replace only in high-cellularity areas
    blended_initial = (blended_initial * 255).astype(np.uint8)

    optimized_cellularity = predicted.detach().numpy()[x_id, y_id, slice_id]
    optimized_cellularity_colored = plt.cm.viridis(optimized_cellularity)[
        :, :, :3
    ]  # Apply colormap and remove alpha
    mask = optimized_cellularity >= threshold
    blended_optimized = t1_rgb.copy()
    blended_optimized[mask] = optimized_cellularity_colored[
        mask
    ]  # Replace only in high-cellularity areas
    blended_optimized = (blended_optimized * 255).astype(np.uint8)

    ground_truth_cellularity = u_target.detach().numpy()[x_id, y_id, slice_id]
    ground_truth_cellularity_colored = plt.cm.viridis(ground_truth_cellularity)[
        :, :, :3
    ]  # Apply colormap and remove alpha
    mask = ground_truth_cellularity >= threshold
    blended_ground_truth = t1_rgb.copy()
    blended_ground_truth[mask] = ground_truth_cellularity_colored[
        mask
    ]  # Replace only in high-cellularity areas
    blended_ground_truth = (blended_ground_truth * 255).astype(np.uint8)
    vmax_value = max(
        initial_cellularity.max(),
        optimized_cellularity.max(),
        ground_truth_cellularity.max(),
    )

    # Create figure and subplots
    fig, ax = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)

    # Plot images
    _ = ax[0].imshow(blended_initial, vmin=0, vmax=vmax_value)
    _ = ax[1].imshow(blended_optimized, vmin=0, vmax=vmax_value)
    im3 = ax[2].imshow(blended_ground_truth, vmin=0, vmax=vmax_value)
    ax[0].set_title(
        f"Initial condition: t = {patient_data.visit_days[0]} days",
    )
    ax[1].set_title(
        f"Model tumor slice: t = {t_final} days",
    )
    ax[2].set_title(
        f"Observed tumor slice: t = {t_final} days",
    )

    # Remove axis ticks
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    # Add a single shared colorbar
    cbar_ax = fig.add_axes(
        [0.15, 0.1, 0.7, 0.03]
    )  # Position: [left, bottom, width, height]
    cbar = fig.colorbar(im3, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=16)  # Increase font size
    cbar.ax.set_title("Normalized tumor cell density")

    plt.show()


def plot_measured_TCC(
    measured_cellularity_maps: List[torch.Tensor | np.ndarray],
    timepoints: List[datetime],
    ax: Optional[Axes] = None,
    alpha: float = 1.0,
    carrying_capacity: float = 5062500,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))

    measured_cell_counts = [
        compute_total_cell_count(N, carrying_capacity)/1.0e11
        for N in measured_cellularity_maps
    ]

    ax.plot(
        [days_since_first(t, timepoints[0]) for t in timepoints],
        measured_cell_counts,
        marker="*",
        markersize=15,
        color="#984ea3",
        linestyle="None",
    )
    return ax


def plot_calibration(
    predicted_cellularity_maps: torch.Tensor,
    alpha: float = 1.0,
    carrying_capacity: float = 5062500,
    timepoints: List = None,
    measured_cellularity_maps: List = None,
    patient_data: BasePatientData = None,
):

    predicted_cell_counts = [
        compute_total_cell_count(N, carrying_capacity)
        for N in predicted_cellularity_maps
    ]

    # plt.subplots(figsize=(7, 3.5))
    plt.plot(
        [days_since_first(t, timepoints[0]) for t in timepoints],
        [p.detach()/1.0e11 for p in predicted_cell_counts],
        color="k",
        linewidth=2,
        alpha=alpha,
    )

    measured_cell_counts = [
        compute_total_cell_count(N.array, carrying_capacity)/1.0e11
        for N in measured_cellularity_maps
    ]
    plt.plot(
        patient_data.visit_days,
        measured_cell_counts,
        marker="*",
        markersize=15,
        color="#984ea3",
        linestyle="None",
    )
    plt.xlabel("Days since first image")
    plt.ylabel(r'Total tumor cell count ($\times 10^{11}$)')
    # plt.show()


def plot_calibration_iter(
    sols: Optional[List] = None,
    carrying_capacity: float = 5062500,
    timepoints: List[datetime] = None,
    measured_cellularity_maps: List = None,
    patient_data: BasePatientData = None,
    model: Optional[Callable] = None,
    optimal_parameters: Optional[List[np.ndarray]] = None,
    t_calibration_end: Optional[datetime] = None,
    ax: Optional[Axes] = None,
):
    """Plots multiple histories with varying transparency."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
    if t_calibration_end is None:
        t_calibration_end = timepoints[-1]

    if sols is None:
        assert (
            model is not None
        ), "Need to provide an executable model used by calibration tool."
        assert (
            optimal_parameters is not None
        ), "Need to provide optimal parameters from calibration."
        num_sols = len(optimal_parameters)
        alphas_array = (
            np.logspace(-1, 0, num=num_sols, base=10) * 0.99
        )  # Log-spaced from ~0 to ~0.99
        for i in range(len(optimal_parameters)):
            sol = torch.tensor(model(optimal_parameters[i]))
            alpha = alphas_array[i]
            plot_calibration(
                sol,
                alpha,
                carrying_capacity,
                timepoints,
                measured_cellularity_maps,
                patient_data,
            )
    else:
        num_sols = len(sols)
        alphas_array = (
            np.logspace(-1, 0, num=num_sols, base=10) * 0.99
        )  # Log-spaced from ~0 to ~0.99

        for i, sol in enumerate(sols):
            alpha = alphas_array[i]
            plot_calibration(
                sol,
                alpha,
                carrying_capacity,
                timepoints,
                measured_cellularity_maps,
                patient_data,
            )

    # Get the current y-axis limits
    y_min, y_max = plt.ylim()
    y_min = 0.0  # Set minimum to 0
    tx = [days_since_first(t, timepoints[0]) for t in timepoints]
    t_training = days_since_first(t_calibration_end, timepoints[0])
    # Shade the entire background area for tx <= t_training
    ax = plt.gca()
    ax.fill_between(
        [tx[0], t_training], [y_min, y_min], [y_max, y_max], color="gray", alpha=0.05
    )
    plt.xlim([tx[0], max(tx)])
    plt.ylim([y_min, y_max])
    ax.text(
        tx[0] + 1,
        0.95 * y_max,
        "Calibration",
        color="black",
        ha="left",
        va="top",
    )

    if t_training < tx[-1]:
        ax.text(
            t_training + 1,
            0.95 * y_max,
            "Prediction",
            color="black",
            ha="left",
            va="top",
        )

    plt.xlabel("Days since first image")
    plt.ylabel(r'Total tumor cell count ($\times 10^{11}$)')
    plt.show()
