import json
from datetime import datetime
from pathlib import Path
from typing import Generic, List, Optional, Text, TypeVar

from typing_extensions import Self

from pydantic import BaseModel, DirectoryPath, Field, FilePath, model_validator
from pydantic_settings import SettingsConfigDict

from tumortwin.preprocessing.crop import BoundingBoxIndices, crop_image_to_bounding_box
from tumortwin.types import (
    ChemotherapyTreatment,
    CropSettings,
    NibabelNifti,
    RadiotherapyTreatment,
)
from tumortwin.utils import days_since_first


class BaseVisitData(BaseModel):
    """
    Represents the minimal data for a single patient visit

    Attributes:
        time (datetime): The timestamp of the patient visit.
        _crop_bounding_box (Optional[BoundingBoxIndices]): Optional bounding box used for cropping images

    """

    time: datetime
    _crop_bounding_box: Optional[BoundingBoxIndices] = None

    def _cropped(self, img: NibabelNifti) -> NibabelNifti:
        return (
            crop_image_to_bounding_box(img, self._crop_bounding_box)
            if self._crop_bounding_box is not None
            else img
        )


SomeVisitData = TypeVar("SomeVisitData", bound="BaseVisitData")


class BasePatientData(BaseModel, Generic[SomeVisitData]):
    """
    Represents a minimal set of patient data.

    This model contains the patient-specific data, including imaging information,
    treatment history, and metadata. It provides methods to load patient data from
    files and compute derived properties, such as masked images and treatment days.

    Attributes:
        model_config (SettingsConfigDict): Configuration for environment files.
        image_dir (Optional[DirectoryPath]): Path to the directory containing patient images. Hidden in representations.
        patient (str): Patient identifier.

        crop_settings (Optional[CropSettings]): Settings object describing if and how to crop images. Defaults to `None` (no cropping).
        crop_bounding_box (Optional[BoundingBoxIndices]): Bounding box for cropping images, if available.

        visits (List[type[BaseVisitData]]): List of patient visits, which including imaging and ROI data.
        radiotherapy (List[RadiotherapyTreatment]): List of radiotherapy treatments.
        chemotherapy (List[ChemotherapyTreatment]): List of chemotherapy treatments.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    image_dir: Optional[DirectoryPath] = Field(default=None, repr=False)

    patient: Text

    crop_settings: Optional[CropSettings] = None
    crop_bounding_box: Optional[BoundingBoxIndices] = None

    visits: List[SomeVisitData]
    radiotherapy: List[RadiotherapyTreatment] = Field(default=[])
    chemotherapy: List[ChemotherapyTreatment] = Field(default=[])

    def _cropped(self, img: NibabelNifti) -> NibabelNifti:
        return (
            crop_image_to_bounding_box(img, self.crop_bounding_box)
            if self.crop_bounding_box is not None
            else img
        )

    @classmethod
    def from_file(
        cls,
        patient_info_path: FilePath,
        image_dir: Optional[Path] = None,
        crop_settings: Optional[CropSettings] = None,
    ) -> Self:
        """
        Creates an instance of BasePatientData (or a subclass) from a JSON file.

        Args:
            patient_info_path (FilePath): Path to the JSON file containing patient information.
            image_dir (Optional[Path]): Directory containing patient images.
            crop_settings (Optional[CropSettings]): Settings object specifying how to crop image data.

        Returns:
            BasePatientData: The initialized patient data model.
        """
        with open(patient_info_path) as f:
            input_json = json.loads(f.read())
        if image_dir is not None:
            input_json["image_dir"] = image_dir
        if crop_settings is not None:
            input_json["crop_settings"] = crop_settings.model_dump()
        return cls.model_validate(input_json)

    @property
    def radiotherapy_days(self) -> List[float]:
        """
        Computes the days of radiotherapy treatments relative to the first visit.

        Returns:
            List[float]: A list of time differences (in days) between each radiotherapy treatment
            and the first visit.
        """
        if self.radiotherapy is None:
            return []
        return [
            days_since_first(r.time, self.visits[0].time) for r in self.radiotherapy
        ]

    @property
    def chemotherapy_days(self) -> List[float]:
        """
        Computes the days of chemotherapy treatments relative to the first visit.

        Returns:
            List[float]: A list of time differences (in days) between each chemotherapy treatment
            and the first visit.
        """
        if self.chemotherapy is None:
            return []
        return [
            days_since_first(r.time, self.visits[0].time) for r in self.chemotherapy
        ]

    @property
    def visit_days(self) -> List[float]:
        """
        Computes the days of patient visits relative to the first visit.

        Returns:
            List[float]: A list of time differences (in days) between each visit
            and the first visit.
        """
        return [days_since_first(r.time, self.visits[0].time) for r in self.visits]

    @model_validator(mode="after")
    def _sort_dates(self) -> Self:
        """
        Sorts visits, radiotherapy treatments, and chemotherapy treatments by their timestamps.

        This ensures that all date-based data is chronologically ordered for consistency.

        Returns:
            HGGPatientData: The updated instance with sorted dates.
        """
        self.visits = sorted(self.visits, key=lambda v: v.time)
        self.radiotherapy = (
            sorted(self.radiotherapy, key=lambda v: v.time) if self.radiotherapy else []
        )
        self.chemotherapy = (
            sorted(self.chemotherapy, key=lambda v: v.time) if self.chemotherapy else []
        )
        return self
