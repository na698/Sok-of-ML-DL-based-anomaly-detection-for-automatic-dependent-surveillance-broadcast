# =========================================================
# SoK: ML & DL Based Anomaly Detection for ADS-B
# FINAL STABLE VERSION (2026 Compatible)
# =========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import numpy as np
import geopandas as gpd
import sqlite3
from datetime import datetime
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    layout="wide",
    page_title="SoK: ML & DL Based Anomaly Detection for ADS-B"
)

st_autorefresh(interval=10000, key="refresh")


# =========================================================
# OPENSKY API CONFIG
# =========================================================

USERNAME = "nasreen"
PASSWORD = "@Nasreen_2026"


# =========================================================
# FETCH LIVE DATA
# =========================================================

@st.cache_data(ttl=10)
def fetch_live_data():
    url = "https://opensky-network.org/api/states/all"

    try:
        r = requests.get(url, auth=(USERNAME, PASSWORD), timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
    except:
        return pd.DataFrame()

    columns = [
        "icao24", "callsign", "origin_country", "time_position", "last_contact",
        "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
        "heading", "vertical_rate", "sensors", "geo_altitude",
        "squawk", "spi", "position_source"
    ]

    df = pd.DataFrame(data["states"], columns=columns)
    df = df.rename(columns={"origin_country": "country"})

    return df
df = fetch_live_data()

if df.empty or "callsign" not in df.columns:
    import os

file_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "opensky_snapshot_20260128_201219.csv"
)

df = pd.read_csv(file_path)

# ==============================
# DATA CLEANING
# ==============================

# handle different column names
if "lat" in df.columns:
    df = df.rename(columns={"lat": "latitude"})
if "long" in df.columns:
    df = df.rename(columns={"long": "longitude"})

if "latitude" in df.columns and "longitude" in df.columns:
    df = df.dropna(subset=["latitude", "longitude"])

# handle velocity
if "speed" in df.columns:
    df = df.rename(columns={"speed": "velocity"})

if "velocity" in df.columns:
    df["velocity"] = pd.to_numeric(df["velocity"], errors="coerce")

# handle altitude
if "baro_altitude" in df.columns:
    df["baro_altitude"] = pd.to_numeric(df["baro_altitude"], errors="coerce")

# safe drop
if "velocity" in df.columns and "baro_altitude" in df.columns:
    df = df.dropna(subset=["velocity", "baro_altitude"])

if "velocity" in df.columns:
    df["velocity_kmh"] = df["velocity"] * 3.6

df["timestamp"] = datetime.utcnow()

if "country" in df.columns:
    df["country"] = df["country"].astype(str)


# ==============================
# ML MODEL
# ==============================

if "velocity_kmh" in df.columns and "baro_altitude" in df.columns:
    iso = IsolationForest(contamination=0.02, random_state=42)

    df["anomaly_flag"] = iso.fit_predict(df[["velocity_kmh", "baro_altitude"]])

    df["anomaly"] = df["anomaly_flag"].apply(
        lambda x: "Anomaly" if x == -1 else "Normal"
    )
else:
    df["anomaly"] = "Unknown"


# =========================================================
# SQLITE HISTORICAL TRACKING
# =========================================================

def init_db():
    conn = sqlite3.connect("anomaly_history.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_log (
            timestamp TEXT,
            total_anomalies INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_snapshot(df):
    conn = sqlite3.connect("anomaly_history.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO anomaly_log VALUES (?, ?)",
        (
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            int((df["anomaly"]=="Anomaly").sum())
        )
    )
    conn.commit()
    conn.close()

init_db()
log_snapshot(df)


# =========================================================
# FIGURES
# =========================================================

# 🌍 MAP
fig_map = px.scatter_mapbox(
    df,
    lat="latitude",
    lon="longitude",
    color="anomaly",
    hover_name="callsign",
    hover_data=["country", "velocity_kmh", "baro_altitude"],
    zoom=1,
    height=600
)

fig_map.update_traces(marker=dict(size=6, opacity=0.7))
fig_map.update_layout(
    mapbox_style="carto-darkmatter",
    margin=dict(l=0, r=0, t=0, b=0)
)


# 📊 ANOMALY PIE
anom_counts = df["anomaly"].value_counts().reset_index()
anom_counts.columns = ["Type", "Count"]

fig_anom_pie = px.pie(
    anom_counts,
    names="Type",
    values="Count",
    hole=0.4,
    template="plotly_dark"
)


# 🛫 ALTITUDE PIE
df["altitude_band"] = pd.cut(
    df["baro_altitude"],
    bins=[-1000,1000,5000,10000,20000,40000],
    labels=["0-1km","1-5km","5-10km","10-20km","20km+"]
)

alt_counts = df["altitude_band"].value_counts().reset_index()
alt_counts.columns = ["Band", "Count"]

fig_alt_pie = px.pie(
    alt_counts,
    names="Band",
    values="Count",
    hole=0.4,
    template="plotly_dark"
)


# 🌎 COUNTRY SPEED
country_speed = df.groupby("country")["velocity_kmh"].mean().reset_index()

fig_country_speed = px.bar(
    country_speed.sort_values("velocity_kmh", ascending=False).head(20),
    x="country",
    y="velocity_kmh",
    template="plotly_dark"
)


# ⏳ TIME SERIES
df["minute"] = df["timestamp"].dt.strftime("%H:%M")

time_anom = df.groupby(["minute","anomaly"]).size().reset_index(name="count")

fig_time = px.line(
    time_anom,
    x="minute",
    y="count",
    color="anomaly",
    markers=True,
    template="plotly_dark"
)


# 🛩 AIRCRAFT TYPE
df["aircraft_type"] = df["icao24"].str[:2]

type_counts = df["aircraft_type"].value_counts().reset_index()
type_counts.columns = ["Type","Count"]

fig_type = px.bar(
    type_counts.head(20),
    x="Type",
    y="Count",
    template="plotly_dark"
)


# 🔥 DENSITY HEATMAP
anom_df = df[df["anomaly"]=="Anomaly"]

fig_heat = px.density_mapbox(
    anom_df,
    lat="latitude",
    lon="longitude",
    radius=5,
    zoom=1,
    mapbox_style="carto-darkmatter"
)


# 🌍 TRUE COUNTRY POLYGON HEATMAP
try:
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )

    country_anom = (
        df[df["anomaly"]=="Anomaly"]
        .groupby("country")
        .size()
        .reset_index(name="anomaly_count")
    )

    world = world.merge(
        country_anom,
        how="left",
        left_on="ADMIN",
        right_on="country"
    )

    world["anomaly_count"] = world["anomaly_count"].fillna(0)

    fig_country_polygon = px.choropleth(
        world,
        geojson=world.geometry,
        locations=world.index,
        color="anomaly_count",
        projection="natural earth",
        template="plotly_dark"
    )

    fig_country_polygon.update_geos(fitbounds="locations", visible=False)

except:
    fig_country_polygon = None


# 📈 HISTORICAL TREND (FIXED)

conn = sqlite3.connect("anomaly_history.db")
history_df = pd.read_sql("SELECT * FROM anomaly_log", conn)
conn.close()

history_df["timestamp"] = history_df["timestamp"].astype(str)
history_df["total_anomalies"] = pd.to_numeric(
    history_df["total_anomalies"], errors="coerce"
)

fig_history = px.line(
    history_df,
    x="timestamp",
    y="total_anomalies",
    markers=True,
    template="plotly_dark"
)


# =========================================================
# DASHBOARD UI
# =========================================================

st.title("📡 SoK: ML & DL Based Anomaly Detection for ADS-B")

col1, col2, col3, col4 = st.columns(4)
col1.metric("✈ Aircraft", len(df))
col2.metric("🚨 Anomalies", (df["anomaly"]=="Anomaly").sum())
col3.metric("🌍 Countries", df["country"].nunique())
col4.metric("⏱ Last Update", datetime.utcnow().strftime("%H:%M:%S UTC"))

st.divider()


colA, colB = st.columns([2,1])

with colA:
    st.subheader("🌍 Global Aircraft Map")
    st.plotly_chart(fig_map, width="stretch", key="map")

with colB:
    st.subheader("📊 Anomaly Distribution")
    st.plotly_chart(fig_anom_pie, width="stretch", key="pie")

st.divider()

colC, colD = st.columns(2)

with colC:
    st.subheader("🛫 Altitude Distribution")
    st.plotly_chart(fig_alt_pie, width="stretch", key="alt")

with colD:
    st.subheader("🌎 Country-wise Avg Speed")
    st.plotly_chart(fig_country_speed, width="stretch", key="speed")

st.divider()

st.subheader("⏳ Time-Series Anomaly Tracking")
st.plotly_chart(fig_time, width="stretch", key="time")

st.divider()

colE, colF = st.columns(2)

with colE:
    st.subheader("🛩 Aircraft Type Distribution")
    st.plotly_chart(fig_type, width="stretch", key="type")

with colF:
    st.subheader("🔥 Anomaly Heatmap")
    st.plotly_chart(fig_heat, width="stretch", key="heat")

st.divider()

if fig_country_polygon:
    st.subheader("🌍 Country-Level Anomaly Heatmap")
    st.plotly_chart(fig_country_polygon, width="stretch", key="polygon")

st.divider()

st.subheader("📈 Historical Anomaly Trend")
st.plotly_chart(fig_history, width="stretch", key="history")

st.divider()

st.subheader("📄 Live Aircraft Data")
st.dataframe(df.head(500), use_container_width=True)