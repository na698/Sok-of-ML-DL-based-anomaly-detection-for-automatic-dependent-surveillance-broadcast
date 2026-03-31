import pandas as pd

# ===============================
# Load cleaned data
# ===============================
DATA_FILE = "data/opensky_cleaned.csv"
df = pd.read_csv(DATA_FILE)

# ===============================
# Rule-based anomaly detection
# ===============================

# Speed anomaly (km/h)
df["speed_anomaly"] = (df["speed_kmh"] < 100) | (df["speed_kmh"] > 1000)

# Altitude anomaly (feet)
df["altitude_anomaly"] = (df["altitude_ft"] < 500) | (df["altitude_ft"] > 45000)

# Vertical rate anomaly (m/s)
df["vertical_rate_anomaly"] = df["vertical_rate"].abs() > 50

# Position anomaly
df["position_anomaly"] = (
    (df["latitude"].abs() > 90) |
    (df["longitude"].abs() > 180)
)

# ===============================
# FINAL rule-based anomaly label
# ===============================
df["rule_anomaly"] = (
    df["speed_anomaly"] |
    df["altitude_anomaly"] |
    df["vertical_rate_anomaly"] |
    df["position_anomaly"]
)

# ===============================
# Save
# ===============================
OUTPUT_FILE = "data/opensky_labeled.csv"
df.to_csv(OUTPUT_FILE, index=False)

print("Rule-based anomaly labeling completed")
print("Total records:", len(df))
print("Rule-based anomalies:", df["rule_anomaly"].sum())
print(f"Saved: {OUTPUT_FILE}")

