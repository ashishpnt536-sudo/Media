
import streamlit as st
import pandas as pd
import numpy as np
from utils.kpi import add_basic_kpis, add_blended_kpis, compute_pacing
from utils.anomaly import add_baselines, classify_anomalies, revenue_at_risk

st.set_page_config(layout="wide")
st.title("🏠 Unified Campaign Intelligence Hub")

@st.cache_data
def load_sample():
    files = ["data/Google_India_Mobile.csv","data/Meta_US_Desktop.csv","data/Programmatic_EU_Mixed.csv"]
    dfs = [pd.read_csv(f, parse_dates=["ts"]) for f in files]
    return pd.concat(dfs, ignore_index=True)

df = st.session_state.get("df")
if df is None:
    df = load_sample()
    df = add_basic_kpis(df)
    df = add_blended_kpis(df)
    df["Pacing"] = compute_pacing(df)
    df = add_baselines(df, value_cols=["CTR","CPC","CVR","CPA","QCI","EI","Pacing","spend","revenue"], group_cols=("campaign_id",), ts_col="ts")
    # default: show without persistence on Hub
    df = classify_anomalies(df, sensitivity=st.session_state.get("sensitivity","Balanced"), require_persistence=False)
    df["revenue_at_risk"] = revenue_at_risk(df)
    st.session_state["df"] = df

with st.expander("Alert settings", expanded=False):
    require_persistence = st.checkbox("Require persistence (2 intervals)", value=False)

cols = st.columns(4)
with cols[0]:
    channel = st.multiselect("Channel", options=sorted(df["channel"].unique()), default=list(sorted(df["channel"].unique())))
with cols[1]:
    geo = st.multiselect("Geo", options=sorted(df["geo"].unique()), default=list(sorted(df["geo"].unique())))
with cols[2]:
    device = st.multiselect("Device", options=sorted(df["device"].unique()), default=list(sorted(df["device"].unique())))
with cols[3]:
    kpi = st.selectbox("Primary KPI", ["EI","QCI","CPA","CVR","CTR","CPC","Pacing"], index=0)

mask = df["channel"].isin(channel) & df["geo"].isin(geo) & df["device"].isin(device)
view = df.loc[mask].copy()

# If user wants persistence, recompute alerts flag
if require_persistence:
    view = classify_anomalies(view, sensitivity=st.session_state.get("sensitivity","Balanced"), require_persistence=True)

latest = view.sort_values("ts").groupby("campaign_id").tail(1).copy()
latest["severity"] = latest["row_severity"].astype(int)
latest["impact_hr_num"] = latest["revenue_at_risk"].fillna(0)

# If persistence filter is on, show only rows with any *_alert True
if require_persistence:
    alert_cols = [c for c in latest.columns if c.endswith("_alert")]
    if alert_cols:
        latest["any_alert"] = latest[alert_cols].any(axis=1)
        latest = latest[latest["any_alert"]]

latest = latest.rename(columns={f"{kpi}_pct_dev": "pct_dev"})
feed_cols = ["campaign_id","channel","geo","device", kpi, f"baseline_median_{kpi}", "pct_dev", "severity","impact_hr_num"]
latest_display = latest[feed_cols].sort_values(["severity","impact_hr_num"], ascending=[True,False]).copy()
latest_display["pct_dev"] = (latest_display["pct_dev"]*100).round(1).astype(str) + "%"
latest_display = latest_display.rename(columns={"impact_hr_num":"impact_hr"})
latest_display["impact_hr"] = latest_display["impact_hr"].round(0).apply(lambda x: f"₹{x:,.0f}")

st.markdown("### 🔔 Real-time Anomaly Feed")
st.dataframe(latest_display, use_container_width=True)
st.caption("Severity: P1=1 (critical) … P4=4 (info). Impact/hr is monetary: max(0, Baseline Revenue − Actual Revenue) per interval.")
