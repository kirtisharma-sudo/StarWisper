import numpy as np
from ..models.schemas import FeatureVector
from ..processing import feature_extraction
from ..utils.logging import get_logger

logger = get_logger(__name__)

class FeatureExtractor:
    @staticmethod
    def extract(time: np.ndarray, flux: np.ndarray, transit_params: dict) -> FeatureVector:
        features = feature_extraction(time, flux, transit_params)
        return FeatureVector(**features)

    @staticmethod
    def load_features(result_path: str) -> FeatureVector:
        data = np.load(result_path, allow_pickle=True).item()
        return FeatureVector(**data)
