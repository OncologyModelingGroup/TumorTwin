import matplotlib.pyplot as plt
import numpy as np

from tumortwin.types.base import BasePatientData
from tumortwin.types.imaging import NibabelNifti
from tumortwin.utils import find_best_slice
from matplotlib import transforms
from scipy import ndimage

def plot_imaging_summary(patient_data: BasePatientData):
    """
    Generates a summary figure for a given BasePatientData object.

    Each column represents a visit, and each row represents an imaging modality.

    Args:
        patient_data (BasePatientData): The patient data object to visualize.
    """
    num_cols = len(patient_data.visits)//2
    num_rows = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(0.75*num_cols, 0.9*num_rows))
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(7,2))
    plt.subplots_adjust(wspace=0, hspace=0)
    # Ensure axes is always a 2D array for consistency
    if num_rows == 1:
        axes = np.array([axes])
    if num_cols == 1:
        axes = axes[:, np.newaxis]

    if num_cols == 0:
        print("No visits found in patient data.")
        return
    anatomic_mask = None
    anatomic_image = None
    adc = None
    roi = None

    for attr in dir(patient_data):
        if "mask" in attr and isinstance(
            getattr(patient_data, attr, None), (NibabelNifti, type(None))
        ):
            anatomic_mask = getattr(patient_data, attr, None)

        if "T1" in attr and isinstance(
            getattr(patient_data, attr, None), (NibabelNifti, type(None))
        ):
            anatomic_image = getattr(patient_data, attr, None)

    for col_idx, visit in enumerate(patient_data.visits[::2]):
        for attr in dir(visit):
            if "adc" in attr and isinstance(
                getattr(patient_data, attr, None), (NibabelNifti, type(None))
            ):
                adc = getattr(visit, attr, None)

            if "roi_enhance" in attr and isinstance(
                getattr(patient_data, attr, None), (NibabelNifti, type(None))
            ):
                roi = getattr(visit, attr, None)

        if col_idx == 0:
            best_slice = find_best_slice(roi.array)

        if adc is not None:
            if anatomic_image is not None:
                axes[0, col_idx].imshow(
                    np.rot90(np.pad(anatomic_image.array[:, :, best_slice],pad_width=1,mode='constant',constant_values=0),axes=(1,0),k=3), cmap="gray", aspect="auto"
                )

            axes[1, col_idx].imshow(
                np.rot90(np.pad(adc.array[:, :, best_slice],pad_width=1,mode='constant',constant_values=0),axes=(1,0),k=3), cmap="jet", aspect="auto"
            )

            if anatomic_mask is not None:
                axes[0, col_idx].contour(
                    np.rot90(np.pad(anatomic_mask.array[:, :, best_slice],pad_width=1,mode='constant',constant_values=0),axes=(1,0),k=3), colors="blue"
                )
                axes[1, col_idx].contour(
                    np.rot90(np.pad(anatomic_mask.array[:, :, best_slice],pad_width=1,mode='constant',constant_values=0),axes=(1,0),k=3), colors="blue"
                )

            if roi is not None:
                axes[0, col_idx].contour(np.rot90(np.pad(roi.array[:, :, best_slice],pad_width=1,mode='constant',constant_values=0),axes=(1,0),k=3), colors="red")
                axes[1, col_idx].contour(np.rot90(np.pad(roi.array[:, :, best_slice],pad_width=1,mode='constant',constant_values=0),axes=(1,0),k=3), colors="red")

        axes[0, col_idx].set_title(f"Visit {col_idx*2 + 1}")
        axes[0, 0].set_ylabel("T1")
        axes[1, 0].set_ylabel("ADC")

        for row_idx in range(num_rows):
            axes[row_idx, col_idx].axis("equal")
            axes[row_idx, col_idx].tick_params(
                left=False, labelleft=False, labelbottom=False, bottom=False
            )
            for spine in axes[row_idx, col_idx].spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.show()
