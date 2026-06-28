import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
import os

from .models import XGBoostModel, DummyModel, BaseModel
from .features import extract_all_features, feature_vector_to_array

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, model_type: str = 'xgboost', **model_kwargs):
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.model: Optional[BaseModel] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        self.feature_names = feature_names
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        if self.model_type.lower() == 'xgboost':
            model_obj = XGBoostModel.from_params(**self.model_kwargs)
            model_obj.model.fit(X_train_scaled, y_train)
            self.model = model_obj
        else:
            logger.warning(f"Unknown model type '{self.model_type}'. Using DummyModel.")
            self.model = DummyModel(feature_names)
        y_pred = self.model.predict(X_test_scaled)
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = y_pred
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
        }
        try:
            cv_scores = cross_val_score(self.model.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            metrics['cv_roc_auc_mean'] = cv_scores.mean()
            metrics['cv_roc_auc_std'] = cv_scores.std()
        except:
            pass
        return metrics

    def save(self, model_path: str, scaler_path: Optional[str] = None):
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        self.model.save(model_path)
        if self.scaler is not None and scaler_path:
            joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, model_path: str, scaler_path: Optional[str] = None) -> 'TrainingPipeline':
        try:
            model = XGBoostModel.load(model_path)
            model_type = 'xgboost'
        except:
            model = DummyModel.load(model_path)
            model_type = 'dummy'
        pipeline = cls(model_type=model_type)
        pipeline.model = model
        if scaler_path and os.path.exists(scaler_path):
            pipeline.scaler = joblib.load(scaler_path)
        pipeline.feature_names = model.feature_names
        return pipeline

class InferencePipeline:
    def __init__(self, model: BaseModel, scaler: Optional[StandardScaler] = None):
        self.model = model
        self.scaler = scaler
        self.feature_names = model.feature_names

    def predict_from_lightcurve(self, time: np.ndarray, flux: np.ndarray, transit_mask: Optional[np.ndarray] = None, period: Optional[float] = None, duration: Optional[float] = None, epoch: Optional[float] = None) -> Dict[str, Any]:
        features_dict = extract_all_features(time, flux, transit_mask, period, duration, epoch)
        X = feature_vector_to_array(features_dict, self.feature_names)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X.reshape(1, -1))
        else:
            X_scaled = X.reshape(1, -1)
        pred = int(self.model.predict(X_scaled)[0])
        proba = self.model.predict_proba(X_scaled)[0].tolist()
        return {
            'prediction': pred,
            'probability': proba[pred],
            'probabilities': proba,
            'features': features_dict,
        }

    @classmethod
    def from_paths(cls, model_path: str, scaler_path: Optional[str] = None) -> 'InferencePipeline':
        try:
            model = XGBoostModel.load(model_path)
        except:
            model = DummyModel.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
        return cls(model, scaler)
