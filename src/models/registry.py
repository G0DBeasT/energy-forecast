"""
Model Registry & Artifact Persistence.

Provides robust serialization, metadata packaging, and loading of trained models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pickle
import pandas as pd
from src.config import DATA_PROC, MODELS_DIR


class ModelRegistry:
    """Manages serialization and metadata tracking for trained forecasting models."""

    @staticmethod
    def save_model(
        model: Any,
        feature_cols: List[str],
        metrics: Dict[str, Any],
        model_name: str = "lightgbm_tuned",
        provenance: str = "synthetic_simulated",
        destination_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save model artifact, feature list, and metadata package.
        """
        dest = destination_dir or DATA_PROC
        dest.mkdir(parents=True, exist_ok=True)

        model_pkl_path = dest / "best_model.pkl"
        with open(model_pkl_path, "wb") as f:
            pickle.dump(model, f)

        metadata = {
            "model_name": model_name,
            "model_class": type(model).__name__,
            "data_provenance": provenance,
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "metrics": metrics,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        meta_json_path = dest / "model_metadata.json"
        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return model_pkl_path

    @staticmethod
    def load_model(
        model_path: Optional[Path] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load serialized model and its associated metadata.
        """
        path = model_path or (DATA_PROC / "best_model.pkl")
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            model = pickle.load(f)

        meta_path = path.parent / "model_metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        return model, metadata
