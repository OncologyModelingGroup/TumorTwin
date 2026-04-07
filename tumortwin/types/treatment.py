from datetime import datetime
from enum import Enum
from typing import Dict, List, TypeAlias

from typing_extensions import Self

import numpy as np
import torch
from pydantic import BaseModel, model_validator

TreatmentTime: TypeAlias = datetime
"""
Alias for how the data type we choose to represent the time of a treatment.
"""


class RadiotherapyUnit(str, Enum):
    """
    Enumeration of possible units for radiotherapy doses.

    Attributes:
        Gy (str): Gray, the SI unit of absorbed radiation dose.
    """

    Gy = "Gy"


class ChemotherapyUnit(str, Enum):
    """
    Enumeration of possible units for chemotherapy doses.

    Attributes:
        mg (str): Milligrams, a unit of mass.
    """

    mg = "mg"


class RadiotherapyTreatment(BaseModel):
    """
    Represents a single radiotherapy treatment session.

    Attributes:
        time (TreatmentTime): The time of the treatment.
        dose (float): The dose delivered during the treatment.
        units (RadiotherapyUnit): The unit of the dose (e.g., Gy).
    """

    time: TreatmentTime
    dose: float
    units: RadiotherapyUnit


RadiotherapyProtocol: TypeAlias = Dict[TreatmentTime, float]
"""
Alias for a radiotherapy protocol, mapping treatment times to dose values.
"""


class RadiotherapySpecification(BaseModel):
    """
    Defines a radiotherapy protocol, including dose schedule and biological parameters.

    Attributes:
        alpha (float): Tissue-specific radiosensitivity parameter (α).
        alpha_beta_ratio (float): Tissue-specific α/β ratio.
        times (List[TreatmentTime]): Times of the treatments in the protocol.
        doses (List[float]): Doses corresponding to each treatment time.
        protocol(RadiotherapyProtocol): The complete mapping of treatment times to radiotherapy doses.
    """

    alpha: float
    alpha_beta_ratio: float
    times: List[TreatmentTime]
    doses: List[float]

    @property
    def protocol(self) -> RadiotherapyProtocol:
        return {t: d for t, d in zip(self.times, self.doses)}

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        """
        Validates that the `times` and `doses` lists have the same length.

        Raises:
            AssertionError: If the lengths of `times` and `doses` differ.

        Returns:
            Self: The validated instance.
        """
        assert len(self.times) == len(
            self.doses
        ), f"Length of radiotherapy times ('{len(self.times)}') does not match length of radiotherapy doses ('{len(self.doses)}')"
        return self


class ChemotherapyTreatment(BaseModel):
    """
    Represents a single chemotherapy treatment session.

    Attributes:
        time (TreatmentTime): The time of the treatment.
        dose (float): The dose delivered during the treatment.
        units (ChemotherapyUnit): The unit of the dose (e.g., mg).
    """

    time: TreatmentTime
    dose: float
    units: ChemotherapyUnit


ChemotherapyProtocol: TypeAlias = Dict[TreatmentTime, float]
"""
Alias for a chemotherapy protocol, mapping treatment times to dose values.
"""


class ChemotherapySpecification(BaseModel):
    """
    Defines a chemotherapy treatment specification, including dose schedule and pharmacokinetic parameters.

    Attributes:
        sensitivity (float): The sensitivity parameter of the drug.
        decay_rate (float): The decay rate of the drug in the body.
        times (List[datetime]): Times of the treatments in the protocol.
        doses (List[float]): Doses corresponding to each treatment time.
        protocol(ChemotherapyProtocol): The complete mapping of treatment times to chemotherapy doses.
    """

    class Config:
        arbitrary_types_allowed = True

    sensitivity: float | torch.Tensor
    decay_rate: float
    times: List[datetime]
    doses: List[float]

    @property
    def protocol(self) -> ChemotherapyProtocol:
        return {t: d for t, d in zip(self.times, self.doses)}

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        """
        Validates that the `times` and `doses` lists have the same length.

        Raises:
            AssertionError: If the lengths of `times` and `doses` differ.

        Returns:
            Self: The validated instance.
        """
        assert len(self.times) == len(
            self.doses
        ), f"Length of chemotherapy times ('{len(self.times)}') does not match length of chemotherapy doses ('{len(self.doses)}')"
        return self

    @model_validator(mode="after")
    def normalize_doses(self) -> Self:
        max_dose = np.max(self.doses)
        self.doses = [d / max_dose for d in self.doses]
        return self
