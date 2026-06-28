import json
import numpy as np
from pathlib import Path
from ..models.schemas import ClassificationResult, FeatureVector
from ..config import settings
from ..utils.logging import get_logger
from ..ml.pipeline import InferencePipeline
from ..ml.explain import explain_prediction

logger = get_logger(__name__)

class ClassifierService:
    _pipeline = None

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            model_path = settings.MODEL_PATH
            scaler_path = settings.SCALER_PATH
            if not Path(model_path).exists():
                logger.warning("Model not found. Creating dummy pipeline.")
                from ..ml.models import DummyModel
                from sklearn.preprocessing import StandardScaler
                feature_names = [
                    "period", "duration", "depth", "epoch", "snr", "chi2",
                    "bls_power", "transit_signal_positive", "even_odd_ratio",
                    "secondary_eclipse_depth", "blend_probability", "contamination"
                ]
                model = DummyModel(feature_names)
                scaler = StandardScaler()
                scaler.fit(np.random.randn(100, 12))
                cls._pipeline = InferencePipeline(model, scaler)
            else:
                cls._pipeline = InferencePipeline.from_paths(model_path, scaler_path)
        return cls._pipeline

    @classmethod
    def classify(cls, features: FeatureVector) -> ClassificationResult:
        pipeline = cls.get_pipeline()
        feature_names = pipeline.feature_names
        X = np.array([[
            features.period, features.duration, features.depth, features.epoch,
            features.snr, features.chi2, features.bls_power,
            features.transit_signal_positive, features.even_odd_ratio,
            features.secondary_eclipse_depth, features.blend_probability,
            features.contamination
        ]])
        if pipeline.scaler is not None:
            X_scaled = pipeline.scaler.transform(X)
        else:
            X_scaled = X
        pred = int(pipeline.model.predict(X_scaled)[0])
        proba = pipeline.model.predict_proba(X_scaled)[0]
        label = "CANDIDATE" if pred == 1 else "FALSE_POSITIVE"
        probability = float(proba[pred])
        confidence = float(proba.max())

        importances = pipeline.model.feature_importances()
        if importances is not None:
            importance_dict = {feature_names[i]: float(importances[i]) for i in range(len(importances))}
        else:
            importance_dict = {}

        features_dict = features.dict()
        try:
            shap_result = explain_prediction(
                pipeline.model, X_scaled, feature_names,
                features_dict=features_dict, prediction=pred
            )
            interpretation = shap_result['reasoning']
        except Exception as e:
            logger.warning(f"Explainability failed: {e}")
            interpretation = "Candidate" if pred == 1 else "False positive"

        return ClassificationResult(
            predicted_label=label,
            probability=probability,
            confidence=confidence,
            feature_importance=importance_dict,
            interpretation=interpretation
        )

    @staticmethod
    def load_classification(result_path: str) -> ClassificationResult:
        with open(result_path, 'r') as f:
            data = json.load(f)
        return ClassificationResult(**data)
