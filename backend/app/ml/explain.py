import numpy as np
import shap
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ShapExplainer:
    def __init__(self, model: Any, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer: Optional[shap.Explainer] = None

    def fit(self, X: np.ndarray):
        if self.background_data is None:
            n_samples = min(len(X), 100)
            self.background_data = X[:n_samples] if len(X) > n_samples else X
        self.explainer = shap.TreeExplainer(self.model.model, self.background_data)
        return self

    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        contributions = {}
        for i, name in enumerate(self.feature_names):
            contributions[name] = float(shap_values[0, i]) if shap_values.ndim > 1 else float(shap_values[i])
        return {
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            'base_value': float(base_value),
            'feature_contributions': contributions,
        }

def generate_reasoning(features: Dict[str, float], prediction: int, shap_contributions: Optional[Dict[str, float]] = None, threshold_depth: float = 0.01, threshold_snr: float = 10.0) -> str:
    reasons = []
    if prediction == 1:
        reasons.append("The signal is classified as a CANDIDATE.")
        depth = features.get('depth', 0.0)
        snr_val = features.get('snr', 0.0)
        bls_power = features.get('bls_power', 0.0)
        even_odd = features.get('even_odd_ratio', 1.0)
        sec_eclipse = features.get('secondary_eclipse_depth', 0.0)
        blend = features.get('blend_probability', 0.0)
        if depth > threshold_depth:
            reasons.append(f"Transit depth of {depth*1e6:.1f} ppm is significant.")
        else:
            reasons.append(f"Transit depth ({depth*1e6:.1f} ppm) is marginal.")
        if snr_val > threshold_snr:
            reasons.append(f"Signal‑to‑noise ratio of {snr_val:.1f} is high.")
        else:
            reasons.append(f"SNR of {snr_val:.1f} is moderate.")
        if bls_power > 0.5:
            reasons.append(f"BLS power of {bls_power:.2f} indicates a strong periodic signal.")
        if even_odd < 0.8 or even_odd > 1.2:
            reasons.append(f"Even‑odd depth ratio of {even_odd:.2f} shows inconsistency.")
        if sec_eclipse > 0.005:
            reasons.append(f"Secondary eclipse depth of {sec_eclipse*1e6:.1f} ppm suggests a false positive.")
        if blend > 0.2:
            reasons.append(f"High blend probability ({blend:.2f}) indicates possible false positive.")
        else:
            reasons.append("Low blend probability supports a genuine planet.")
        if shap_contributions:
            sorted_features = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:3]
            reasons.append("Top contributing features: " + ", ".join([f"{f} ({v:.3f})" for f, v in top_features]))
    else:
        reasons.append("The signal is classified as a FALSE POSITIVE.")
        depth = features.get('depth', 0.0)
        snr_val = features.get('snr', 0.0)
        even_odd = features.get('even_odd_ratio', 1.0)
        sec_eclipse = features.get('secondary_eclipse_depth', 0.0)
        blend = features.get('blend_probability', 0.0)
        contamination = features.get('contamination', 0.0)
        if depth < threshold_depth:
            reasons.append(f"Transit depth of {depth*1e6:.1f} ppm is too shallow.")
        if snr_val < threshold_snr:
            reasons.append(f"Low SNR ({snr_val:.1f}) suggests noise.")
        if even_odd < 0.7 or even_odd > 1.3:
            reasons.append(f"Inconsistent even‑odd depths ({even_odd:.2f}) indicate false positive.")
        if sec_eclipse > 0.005:
            reasons.append(f"Secondary eclipse depth of {sec_eclipse*1e6:.1f} ppm suggests blended eclipsing binary.")
        if blend > 0.2:
            reasons.append(f"High blend probability ({blend:.2f}) points to a false positive.")
        if contamination > 0.1:
            reasons.append(f"Contamination of {contamination:.2f} indicates nearby source.")
        if shap_contributions:
            sorted_features = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:3]
            reasons.append("Top contributing features: " + ", ".join([f"{f} ({v:.3f})" for f, v in top_features]))
    return "\n".join(reasons)

def explain_prediction(model: Any, X: np.ndarray, feature_names: List[str], background_data: Optional[np.ndarray] = None, features_dict: Optional[Dict[str, float]] = None, prediction: Optional[int] = None) -> Dict[str, Any]:
    if hasattr(model, 'model') and hasattr(model.model, 'get_booster'):
        explainer = ShapExplainer(model, feature_names, background_data)
        explainer.fit(background_data if background_data is not None else X)
        shap_result = explainer.explain(X)
    else:
        shap_result = {'shap_values': [], 'base_value': 0.0, 'feature_contributions': {}}
    if prediction is None:
        pred = int(model.predict(X)[0])
    else:
        pred = prediction
    if features_dict is None:
        features_dict = {feature_names[i]: float(X[0, i]) for i in range(len(feature_names))}
    reasoning = generate_reasoning(features_dict, pred, shap_result.get('feature_contributions', {}))
    contribs = shap_result.get('feature_contributions', {})
    top_features = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    return {
        'shap_explanation': shap_result,
        'reasoning': reasoning,
        'top_features': top_features,
        'prediction': pred,
    }
