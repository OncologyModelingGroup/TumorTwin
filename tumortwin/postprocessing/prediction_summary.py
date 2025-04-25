from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

from tumortwin.postprocessing.total_cell_count import compute_total_cell_count
from tumortwin.types.base import BasePatientData
from tumortwin.utils import days_since_first, find_best_slice


def overlay_cellularity_on_t1(
    cellularity: np.ndarray, t1: np.ndarray, threshold: float
):
    normalized_t1 = (t1 - np.min(t1)) / (np.max(t1) - np.min(t1))
    t1_rgb = np.stack(
        [normalized_t1] * 3, axis=-1
    )  # Convert grayscale to 3-channel RGB
    cellularity_colored = plt.cm.viridis(cellularity)[
        :, :, :3
    ]  # Apply colormap and remove alpha
    mask = cellularity >= threshold
    blended_image = t1_rgb.copy()
    blended_image[mask] = cellularity_colored[
        mask
    ]  # Replace only in high-cellularity areas
    blended_image = (blended_image * 255).astype(np.uint8)
    return blended_image


def plot_cellularity_map(
    solution: torch.Tensor,
    patient_data: BasePatientData,
    ax: Optional[Axes] = None,
    time: Optional[float] = None,
    threshold: float = 0.01,
):
    x_id = np.s_[:]
    y_id = np.s_[:]
    slice_id = find_best_slice(solution.detach().numpy())

    cellularity_image = np.rot90(np.pad(solution.detach().numpy()[:, :, slice_id],pad_width=1,mode='constant',constant_values=0),axes=(-1,0))
    t1_image = np.rot90(np.pad(patient_data.T1_post_image.array[:, :, slice_id],pad_width=1,mode='constant',constant_values=0),axes=(-1,0))
    blended_image = overlay_cellularity_on_t1(
        cellularity=cellularity_image, t1=t1_image, threshold=threshold
    )

    # vmax_value = cellularity_image.max()
    vmax_value = 1.0

    # Create figure and subplots
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Plot images
    im = ax.imshow(blended_image, vmin=0, vmax=vmax_value)

    # Titles with Times New Roman font
    if time is not None:
        ax.set_title(
            f"t={time}",
        )

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    return ax, im


def plot_predicted_TCC(
    predicted_cellularity_maps: List[torch.Tensor],
    timepoints: List[datetime],
    ax: Optional[Axes] = None,
    color: str = "k",
    alpha: float = 1.0,
    carrying_capacity: float = 5062500,
):
    predicted_cell_counts = [
        compute_total_cell_count(N, carrying_capacity)
        for N in predicted_cellularity_maps
    ]
    # plt.subplots(figsize=(7, 3.5))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(
        [days_since_first(t, timepoints[0]) for t in timepoints],
        [p.detach()/ 1.0e11 for p in predicted_cell_counts],
        color=color,
        alpha=alpha,
    )

    # ax.set_title("Total tumor cell count ")
    ax.set_xlabel("Days since first image")
    ax.set_ylabel(r'Total tumor cell count ($\times 10^{11}$)')
    return ax
