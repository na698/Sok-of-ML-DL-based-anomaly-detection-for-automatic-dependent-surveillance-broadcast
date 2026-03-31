import requests
import pandas as pd
from datetime import datetime
import os

# ===============================
# OpenSky API Configuration
# ===============================
OPENSKY_URL = "https://opensky-network.org/api/states/all"

# Create data folder if not exists
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

os.makedirs(DATA_DIR, exist_ok=True)

# ===============================
# Fetch OpenSky Data
# ===============================
def fetch_opensky_data():
    print("Fetching data from OpenSky...")

    response = requests.get(OPENSKY_URL)

    if response.status_code != 200:
        raise Exception("Failed to fetch data from OpenSky")

    data = response.json()
    states = data.get("states", [])

    columns = [
        "icao24",
        "callsign",
        "country",
        "time_position",
        "last_contact",
        "longitude",
        "latitude",
        "baro_altitude",
        "on_ground",
        "velocity",
        "heading",
        "vertical_rate"
    ]

    records = []
    for s in states:
        records.append([
            s[0],   # icao24
            s[1],   # callsign
            s[2],   # origin country
            s[3],   # time_position
            s[4],   # last_contact
            s[5],   # longitude
            s[6],   # latitude
            s[7],   # barometric altitude
            s[8],   # on_ground
            s[9],   # velocity (m/s)
            s[10],  # heading
            s[11]   # vertical_rate
        ])

    df = pd.DataFrame(records, columns=columns)

    # Convert UNIX time to readable UTC time
    df["time_position"] = pd.to_datetime(df["time_position"], unit="s")
    df["last_contact"] = pd.to_datetime(df["last_contact"], unit="s")

    # Save snapshot
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = f"{DATA_DIR}/opensky_snapshot_{timestamp}.csv"
    df.to_csv(file_path, index=False)

    print(f"Saved snapshot: {file_path}")
    print(f"Total aircraft records: {len(df)}")

    return df


if __name__ == "__main__":
    fetch_opensky_data()
