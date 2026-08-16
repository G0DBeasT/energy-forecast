"""
Base Data Loader Interface.

Provides an abstract base class for grid demand and weather datasets, ensuring
uniform schema, validation, and transparent data provenance tracking.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from src.config import DATE_COL, TARGET_COL, TEMP_COL, HUMIDITY_COL


class BaseDataLoader(ABC):
    """Abstract Base Class for power grid data sources."""

    @property
    @abstractmethod
    def data_provenance(self) -> str:
        """
        Data provenance label:
        - 'synthetic_simulated': Physically simulated grid demand data.
        - 'real_recorded': Actual historical utility/transmission meter data.
        - 'benchmark_hybrid': Real weather paired with modeled physical demand response.
        """
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load or generate the raw dataset.

        Returns:
            pd.DataFrame: DataFrame indexed or containing datetime, demand_mw, temp_c, relative_humidity.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata describing the dataset origin, sample frequency, and notes.

        Returns:
            dict: Metadata dictionary.
        """
        pass

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the loaded DataFrame contains required columns and datetime index.

        Args:
            df: DataFrame to validate.

        Returns:
            bool: True if valid.

        Raises:
            ValueError: If schema requirements are violated.
        """
        if not isinstance(df.index, pd.DatetimeIndex) and DATE_COL not in df.columns:
            raise ValueError(f"Dataset must either have a DatetimeIndex or contain '{DATE_COL}' column.")

        if TARGET_COL not in df.columns:
            raise ValueError(f"Dataset is missing target column '{TARGET_COL}'.")

        return True
