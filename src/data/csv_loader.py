"""
CSV Grid Data Loader for Real-World Recorded Data.

Allows the user or enterprise deployment to supply empirical power grid data
(e.g., PJM, ERCOT, ENTSO-E, POSOCO/MERIT India) from external CSV files.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
from src.config import DATE_COL, TARGET_COL, TEMP_COL, HUMIDITY_COL
from src.data.base import BaseDataLoader


class CSVGridLoader(BaseDataLoader):
    """
    Loader for empirical, recorded grid demand datasets stored as CSV.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        date_col: str = DATE_COL,
        demand_col: str = TARGET_COL,
        temp_col: Optional[str] = TEMP_COL,
        humidity_col: Optional[str] = HUMIDITY_COL,
        dataset_name: str = "custom_recorded_grid",
    ):
        self.filepath = Path(filepath)
        self.date_col = date_col
        self.demand_col = demand_col
        self.temp_col = temp_col
        self.humidity_col = humidity_col
        self.dataset_name = dataset_name

    @property
    def data_provenance(self) -> str:
        return "real_recorded"

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": self.data_provenance,
            "filepath": str(self.filepath),
            "dataset_name": self.dataset_name,
            "date_col": self.date_col,
            "demand_col": self.demand_col,
            "temp_col": self.temp_col,
            "humidity_col": self.humidity_col,
            "provenance_note": f"Real recorded grid data loaded from {self.filepath.name}",
        }

    def load(self) -> pd.DataFrame:
        """
        Load and standardize external CSV into standard schema.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Input data file does not exist: {self.filepath}")

        df = pd.read_csv(self.filepath)

        if self.date_col not in df.columns:
            raise KeyError(f"Date column '{self.date_col}' not found in CSV columns: {list(df.columns)}")

        if self.demand_col not in df.columns:
            raise KeyError(f"Demand column '{self.demand_col}' not found in CSV columns: {list(df.columns)}")

        # Rename to standard names
        rename_map = {
            self.demand_col: TARGET_COL,
        }
        if self.temp_col and self.temp_col in df.columns and self.temp_col != TEMP_COL:
            rename_map[self.temp_col] = TEMP_COL
        if self.humidity_col and self.humidity_col in df.columns and self.humidity_col != HUMIDITY_COL:
            rename_map[self.humidity_col] = HUMIDITY_COL

        df = df.rename(columns=rename_map)
        df[DATE_COL] = pd.to_datetime(df[self.date_col])
        df = df.set_index(DATE_COL).sort_index()

        self.validate_schema(df)
        return df
