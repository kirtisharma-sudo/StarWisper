import joblib
import os
from .models import XGBoostModel, DummyModel, BaseModel
from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

class ModelManager:
    _model: BaseModel = None
    _feature_names = None

    @classmethod
    def load_models(cls):
        model_path = settings.MODEL_PATH
        if os.path.exists(model_path):
            try:
                cls._model = XGBoostModel.load(model_path)
                logger.info("Loaded XGBoost model.")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost: {e}. Trying DummyModel.")
                cls._model = DummyModel.load(model_path)
        else:
            logger.warning("Model not found, creating DummyModel.")
            feature_names = [
                "period", "duration", "depth", "epoch", "snr", "chi2",
                "bls_power", "transit_signal_positive", "even_odd_ratio",
                "secondary_eclipse_depth", "blend_probability", "contamination"
            ]
            cls._model = DummyModel(feature_names)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            cls._model.save(model_path)
        cls._feature_names = cls._model.feature_names

    @classmethod
    def get_model(cls) -> BaseModel:
        if cls._model is None:
            cls.load_models()
        return cls._model

    @classmethod
    def get_feature_names(cls):
        if cls._feature_names is None:
            cls.load_models()
        return cls._feature_names
