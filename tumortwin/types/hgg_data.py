from pathlib import Path
from typing import List, Optional

from typing_extensions import Self

import numpy as np
from pydantic import model_validator

from tumortwin.preprocessing.crop import get_bounding_box
from tumortwin.types.base import BasePatientData, BaseVisitData
from tumortwin.types.imaging import CropTarget, NibabelNifti


class HGGVisitData(BaseVisitData):
    """
    Represents data for a single patient visit in the context of high-grade glioma (HGG).

    Attributes:
        adc (Path): Path to the Apparent Diffusion Coefficient (ADC) image file.
        roi_enhance (Path): Path to the file containing the region of interest (ROI) for enhancing tumor regions.
        roi_nonenhance (Path): Path to the file containing the ROI for non-enhancing tumor regions.

    """

    adc: Path
    roi_enhance: Path
    roi_nonenhance: Path

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
        Loads the ROI enhancing image as a NibabelNifti object, optionally cropped to the brain region.

        Returns:
            NibabelNifti: The loaded and optionally cropped ROI enhancing image.
        """
        return self._cropped(NibabelNifti.from_file(self.roi_enhance))

    @property
    def roi_nonenhance_image(self) -> NibabelNifti:
        """
        Loads the ROI non-enhancing image as a NibabelNifti object, optionally cropped to the brain region.

        Returns:
            NibabelNifti: The loaded and optionally cropped ROI non-enhancing image.
        """
        return self._cropped(NibabelNifti.from_file(self.roi_nonenhance))


class HGGPatientData(BasePatientData):
    """
    Represents high-grade glioma (HGG) patient data.

    This model contains the patient-specific data, including imaging information,
    treatment history, and metadata. It provides methods to load patient data from
    files and compute derived properties, such as brain-masked images and treatment days.

    Attributes:
        model_config (SettingsConfigDict): Configuration for environment files.
        image_dir (Optional[DirectoryPath]): Path to the directory containing patient images. Hidden in representations.
        patient (str): Patient identifier.
        brainmask (Path): Path to the brain mask image file.
        crop_settings (Optional[CropSettings]): Settings object describing if and how to crop images. Defaults to `None` (no cropping).
        crop_bounding_box (Optional[BoundingBoxIndices]): Bounding box for cropping images, if available.
        T1_pre (Optional[Path]): Path to the pre-contrast T1-weighted image file.
        T1_post (Optional[Path]): Path to the post-contrast T1-weighted image file.
        T2_flair (Optional[Path]): Path to the T2 FLAIR image file.
        visits (List[HGGVisitData]): List of patient visits, including imaging and ROI data.
        radiotherapy (List[RadiotherapyTreatment]): List of radiotherapy treatments.
        chemotherapy (List[ChemotherapyTreatment]): List of chemotherapy treatments.
    """

    visits: List[HGGVisitData]
    brainmask: Path
    T1_pre: Optional[Path]
    T1_post: Optional[Path]
    T2_flair: Optional[Path]

    @property
    def brainmask_image(self) -> NibabelNifti:
        """
        Loads the brain mask image and optionally crops it to the brain bounding box.

        Returns:
            NibabelNifti: The loaded brain mask image.
        """
        return self._cropped(NibabelNifti.from_file(self.brainmask))

    @property
    def T1_pre_image(self) -> Optional[NibabelNifti]:
        """
        Loads the pre-contrast T1-weighted image and optionally crops it.

        Returns:
            Optional[NibabelNifti]: The loaded T1-pre image, or `None` if unavailable.
        """
        if self.T1_pre is None:
            return None
        return self._cropped(NibabelNifti.from_file(self.T1_pre))

    @property
    def T1_post_image(self) -> Optional[NibabelNifti]:
        """
        Loads the post-contrast T1-weighted image and optionally crops it.

        Returns:
            Optional[NibabelNifti]: The loaded T1-post image, or `None` if unavailable.
        """
        if self.T1_post is None:
            return None
        return self._cropped(NibabelNifti.from_file(self.T1_post))

    @property
    def T2_flair_image(self) -> Optional[NibabelNifti]:
        """
        Loads the T2 FLAIR image and optionally crops it.

        Returns:
            Optional[NibabelNifti]: The loaded T2 FLAIR image, or `None` if unavailable.
        """
        if self.T2_flair is None:
            return None
        return self._cropped(NibabelNifti.from_file(self.T2_flair))

    @model_validator(mode="after")
    def _replace_image_dir_variable(self) -> Self:
        """
        Replaces placeholders for the image directory in file paths with the actual image directory.

        This ensures all file paths referencing the image directory (`{$image_dir}`) are updated
        to their correct values.

        Returns:
            HGGPatientData: The updated instance with replaced paths.
        """

        def replace_image_dir(input: Path) -> Path:
            return Path(str(input).replace("{$image_dir}", str(self.image_dir)))

        if self.image_dir:
            self.brainmask = replace_image_dir(self.brainmask)
            if self.T1_pre:
                self.T1_pre = replace_image_dir(self.T1_pre)
            if self.T1_post:
                self.T1_post = replace_image_dir(self.T1_post)
            if self.T2_flair:
                self.T2_flair = replace_image_dir(self.T2_flair)
            for visit in self.visits:
                visit.adc = replace_image_dir(visit.adc)
                visit.roi_enhance = replace_image_dir(visit.roi_enhance)
                visit.roi_nonenhance = replace_image_dir(visit.roi_nonenhance)
        return self

    @model_validator(mode="after")
    def _populate_crop_bounding_box(self) -> Self:
        """
        Populates the bounding box for cropping if it is not already set.

        The bounding box is derived from the appropriate image according to self.crop_settings and is used to crop all images.

        Returns:
            HGGPatientData: The updated instance with a populated bounding box.
        """

        if self.crop_bounding_box is not None:
            return self

        if self.crop_settings is None:
            return self

        if self.crop_settings.crop_to == CropTarget.ANATOMY:
            self.crop_bounding_box = get_bounding_box(
                self.brainmask_image.array, padding=self.crop_settings.padding
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
            if self.crop_settings.visit_index is None:
                self.crop_bounding_box = get_bounding_box(
                    np.sum([v.roi_nonenhance_image.array for v in self.visits], axis=0),
                    padding=self.crop_settings.padding,
                )
            else:
                self.crop_bounding_box = get_bounding_box(
                    self.visits[
                        self.crop_settings.visit_index
                    ].roi_nonenhance_image.array,
                    padding=self.crop_settings.padding,
                )

        for visit in self.visits:
            visit._crop_bounding_box = self.crop_bounding_box
        return self
