from pathlib import Path
from typing import List, Optional

from typing_extensions import Self

import numpy as np
from pydantic import model_validator

from tumortwin.preprocessing.crop import crop_image_to_bounding_box, get_bounding_box
from tumortwin.types.base import BasePatientData, BaseVisitData
from tumortwin.types.imaging import CropTarget, NibabelNifti


class TNBCVisitData(BaseVisitData):
    """
    Represents data for a single patient visit in the context of triple negative breast cancer (TNBC).

    Attributes:
        adc (Path): Path to the Apparent Diffusion Coefficient (ADC) image file.
        roi_enhance (Path): Path to the file containing the region of interest (ROI) for enhancing tumor regions.

    """

    adc: Path
    roi_enhance: Path

    @property
    def adc_image(self) -> NibabelNifti:
        """
        Loads the ADC image as a NibabelNifti object, optionally cropped to the crop bounding box.

        Returns:
            NibabelNifti: The loaded and optionally cropped ADC image.
        """
        return self._cropped(NibabelNifti.from_file(self.adc))

    @property
    def roi_enhance_image(self) -> NibabelNifti:
        """
        Loads the ROI enhancing image as a NibabelNifti object, optionally cropped to the breast region.

        Returns:
            NibabelNifti: The loaded and optionally cropped ROI enhancing image.
        """
        return self._cropped(NibabelNifti.from_file(self.roi_enhance))


class TNBCPatientData(BasePatientData):
    """
    Represents Triple negative breast cancer  patient data.

    This model contains the patient-specific data, including imaging information,
    treatment history, and metadata. It provides methods to load patient data from
    files and compute derived properties, such as breast-masked images and treatment days.

    Attributes:
        breastmask (Path): Path to the breast mask image file.
        T1_post (Optional[Path]): Path to the post-contrast T1-weighted image file.
        visits (List[TNBCVisitData]): List of patient visits, including imaging and ROI data.
    """

    visits: List[TNBCVisitData]
    breastmask: Path
    T1_post: Optional[Path]

    @property
    def breastmask_image(self) -> NibabelNifti:
        """
        Loads the breast mask image and optionally crops it to the breast bounding box.

        Returns:
            NibabelNifti: The loaded breast mask image.
        """
        img = NibabelNifti.from_file(self.breastmask)
        if self.crop_bounding_box is not None:
            return crop_image_to_bounding_box(img, self.crop_bounding_box)
        else:
            return img

    @property
    def T1_post_image(self) -> Optional[NibabelNifti]:
        """
        Loads the post-contrast T1-weighted image and optionally crops it.

        Returns:
            Optional[NibabelNifti]: The loaded T1-post image, or `None` if unavailable.
        """
        if self.T1_post is None:
            return None
        img = NibabelNifti.from_file(self.T1_post)
        if self.crop_bounding_box is not None:
            return crop_image_to_bounding_box(img, self.crop_bounding_box)
        else:
            return img

    @model_validator(mode="after")
    def _replace_image_dir_variable(self) -> Self:
        """
        Replaces placeholders for the image directory in file paths with the actual image directory.

        This ensures all file paths referencing the image directory (`{$image_dir}`) are updated
        to their correct values.

        Returns:
            TNBCPatientData: The updated instance with replaced paths.
        """

        def replace_image_dir(input: Path) -> Path:
            return Path(str(input).replace("{$image_dir}", str(self.image_dir)))

        if self.image_dir:
            self.breastmask = replace_image_dir(self.breastmask)
            if self.T1_post:
                self.T1_post = replace_image_dir(self.T1_post)
            for visit in self.visits:
                visit.adc = replace_image_dir(visit.adc)
                visit.roi_enhance = replace_image_dir(visit.roi_enhance)
        return self

    @model_validator(mode="after")
    def _populate_crop_bounding_box(self) -> Self:
        """
        Populates the bounding box for cropping if it is not already set.

        The bounding box is derived from the appropriate image according to self.crop_settings and is used to crop all images.

        Returns:
            TNBCPatientData: The updated instance with a populated bounding box.
        """
        if self.crop_bounding_box is not None:
            return self

        if self.crop_settings is None:
            return self

        if self.crop_settings.crop_to == CropTarget.ANATOMY:
            self.crop_bounding_box = get_bounding_box(
                self.breastmask_image.array, padding=self.crop_settings.padding
            )
        elif self.crop_settings.crop_to == CropTarget.ROI_ENHANCE:
            if self.crop_settings.visit_index is None:
                self.crop_bounding_box = get_bounding_box(
                    np.sum([v.roi_enhance_image.array for v in self.visits], axis=0),
                    padding=self.crop_settings.padding,
                )
            else:
                self.crop_bounding_box = get_bounding_box(
                    self.visits[self.crop_settings.visit_index].roi_enhance_image.array,
                    padding=self.crop_settings.padding,
                )
        elif self.crop_settings.crop_to == CropTarget.ROI_NONENHANCE:
            raise ValueError(
                "Unable to crop to Non-enhancing ROI: No ROI-nonenhance is not specified for a TNBC Patient"
            )

        for visit in self.visits:
            visit._crop_bounding_box = self.crop_bounding_box
        return self
