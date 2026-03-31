import pandas as pd
from sklearn.ensemble import IsolationForest

# ===============================
# Load labeled ADS-B data
# ===============================
DATA_FILE = "../data/opensky_labeled.csv"
df = pd.read_csv(DATA_FILE)

# ===============================
# Select features for ML
# ===============================
features = [
    "latitude",
    "longitude",
    "altitude_ft",
    "speed_kmh",
    "vertical_rate"
]

X = df[features]

# ===============================
# Isolation Forest Model
# ===============================
model = IsolationForest(
    n_estimators=200,
    contamination=0.07,
    random_state=42
)

df["ml_anomaly"] = model.fit_predict(X)
df["ml_anomaly"] = df["ml_anomaly"].map({1: 0, -1: 1})

# ===============================
# Save results
# ===============================
OUTPUT_FILE = "../data/opensky_ml_results.csv"
df.to_csv(OUTPUT_FILE, index=False)

print("Isolation Forest completed")
print("ML anomalies detected:", df["ml_anomaly"].sum())
