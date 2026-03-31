import pandas as pd
import os
import glob

# ===============================
# Project paths (SAFE)
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ===============================
# Load latest OpenSky snapshot
# ===============================
files = sorted(glob.glob(os.path.join(DATA_DIR, "opensky_snapshot_*.csv")))

if not files:
    raise FileNotFoundError("No OpenSky snapshot found in data folder")

latest_file = files[-1]
print(f"Loading file: {latest_file}")

df = pd.read_csv(latest_file)

# ===============================
# Keep REQUIRED columns
# ===============================
required_columns = [
    "icao24",
    "callsign",
    "country",
    "latitude",
    "longitude",
    "baro_altitude",
    "velocity",
    "heading",
    "vertical_rate",
    "time_position",
    "on_ground"
]

df = df[required_columns]

# ===============================
# Keep only airborne aircraft
# ===============================
df = df[df["on_ground"] == False]

# ===============================
# Drop missing critical values
# ===============================
df = df.dropna(subset=[
    "latitude",
    "longitude",
    "baro_altitude",
    "velocity",
    "vertical_rate",
    "time_position"
])

# ===============================
# Feature engineering
# ===============================
df["altitude_ft"] = df["baro_altitude"] * 3.28084
df["speed_kmh"] = df["velocity"] * 3.6
df["timestamp"] = pd.to_datetime(df["time_position"], errors="coerce")


# ===============================
# Save cleaned dataset
# ===============================
output_file = os.path.join(DATA_DIR, "opensky_cleaned.csv")
df.to_csv(output_file, index=False)

print("Cleaned dataset saved:", output_file)
print("Final records:", len(df))

