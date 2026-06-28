import numpy as np
import xgboost as xgb
import joblib
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import logging
import os

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def feature_importances(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        pass

class XGBoostModel(BaseModel):
    def __init__(self, model: xgb.XGBClassifier, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importances(self) -> Optional[np.ndarray]:
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        names_path = path.replace('.joblib', '_features.joblib')
        joblib.dump(self.feature_names, names_path)

    @classmethod
    def load(cls, path: str) -> 'XGBoostModel':
        model = joblib.load(path)
        names_path = path.replace('.joblib', '_features.joblib')
        if os.path.exists(names_path):
            feature_names = joblib.load(names_path)
        else:
            feature_names = [
                'period', 'duration', 'depth', 'epoch', 'snr', 'chi2',
                'bls_power', 'transit_signal_positive', 'even_odd_ratio',
                'secondary_eclipse_depth', 'blend_probability', 'contamination'
            ]
        return cls(model, feature_names)

class DummyModel(BaseModel):
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.thresholds = {'depth': 0.01, 'snr': 10.0}

    def predict(self, X: np.ndarray) -> np.ndarray:
        depth_idx = self._get_feature_index('depth')
        snr_idx = self._get_feature_index('snr')
        preds = np.zeros(X.shape[0], dtype=int)
        if depth_idx is not None and snr_idx is not None:
            preds = ((X[:, depth_idx] > self.thresholds['depth']) &
                     (X[:, snr_idx] > self.thresholds['snr'])).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        proba = np.zeros((X.shape[0], 2))
        proba[:, preds] = 0.9
        proba[:, 1 - preds] = 0.1
        return proba

    def feature_importances(self) -> Optional[np.ndarray]:
        return None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'thresholds': self.thresholds, 'feature_names': self.feature_names}, path)

    @classmethod
    def load(cls, path: str) -> 'DummyModel':
        data = joblib.load(path)
        model = cls(data['feature_names'])
        model.thresholds = data['thresholds']
        return model

    def _get_feature_index(self, name: str) -> Optional[int]:
        try:
            return self.feature_names.index(name)
        except ValueError:
            return None
