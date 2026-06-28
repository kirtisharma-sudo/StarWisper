from ..services.classifier import ClassifierService
from ..services.lightcurve_processor import LightCurveProcessor

def get_classifier() -> ClassifierService:
    return ClassifierService()

def get_processor() -> LightCurveProcessor:
    return LightCurveProcessor()
