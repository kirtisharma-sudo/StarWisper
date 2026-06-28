# Optional script to seed a dummy model and catalog
import os
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Create dummy model
model = RandomForestClassifier(n_estimators=10)
X = np.random.randn(200, 12)
y = np.random.choice([0,1], size=200)
model.fit(X, y)
os.makedirs("/app/backend/app/ml/artifacts", exist_ok=True)
joblib.dump(model, "/app/backend/app/ml/artifacts/xgboost_model.joblib")
joblib.dump([
    "period", "duration", "depth", "epoch", "snr", "chi2",
    "bls_power", "transit_signal_positive", "even_odd_ratio",
    "secondary_eclipse_depth", "blend_probability", "contamination"
], "/app/backend/app/ml/artifacts/feature_names.joblib")

# Create catalog
df = pd.DataFrame({'tic_id': [123456789, 987654321], 'label': ['CANDIDATE', 'FALSE_POSITIVE']})
os.makedirs("/data", exist_ok=True)
df.to_csv("/data/catalog.csv", index=False)
