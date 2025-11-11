
import streamlit as st
import pandas as pd
import numpy as np
from utils.kpi import add_basic_kpis, add_blended_kpis, compute_pacing
from utils.anomaly import add_baselines, classify_anomalies, revenue_at_risk

import streamlit as st
import pandas as pd
import numpy as np
from utils.kpi import add_basic_kpis, add_blended_kpis, compute_pacing
from utils.anomaly import add_baselines, classify_anomalies, revenue_at_risk

st.set_page_config(
    page_title="📊 Dashboard – Media.net Anomaly MVP",
    page_icon="📊",
    layout="wide"
)

# --- Force the sidebar label for the root page to "📊 Dashboard"
st.markdown("""
<style>
/* Change first nav item (root page) label */
section[data-testid="stSidebarNav"] li:first-child a span {
  visibility: hidden;
}
section[data-testid="stSidebarNav"] li:first-child a:after {
  content: "📊 Dashboard";
  visibility: visible;
  display: inline-block;
  margin-left: -1.2rem; /* nudge to align with icon */
}
</style>
""", unsafe_allow_html=True)


st.title("📊 Dashboard")
st.caption("Overview of campaign performance and anomalies")


st.markdown("""
Use the sidebar to **load data** (bundled sample or upload). Then explore:
- **🏠 Hub**: Alert feed with severities (P1–P4)
- **🔍 Alert Detail**: Active alerts list + per-alert drilldown
- **⚙️ Baseline Lab**: Context-aware thresholds
""")

with st.sidebar:
    st.header("Data")
    sample = st.toggle("Use bundled sample data", value=True)
    uploaded = st.file_uploader("Or upload CSVs (ts,campaign_id,channel,geo,device,impressions,clicks,spend,conversions,revenue,planned_spend)", type=["csv"], accept_multiple_files=True)
    sensitivity = st.selectbox("Sensitivity", ["Aggressive","Balanced","Conservative"], index=1)
    st.session_state["sensitivity"] = sensitivity

def load_data(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["ts"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_sample_files():
    files = [
        "data/Google_India_Mobile.csv",
        "data/Meta_US_Desktop.csv",
        "data/Programmatic_EU_Mixed.csv",
    ]
    dfs = [pd.read_csv(p, parse_dates=["ts"]) for p in files]
    return pd.concat(dfs, ignore_index=True)

def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts").reset_index(drop=True)
    df = add_basic_kpis(df)
    df = add_blended_kpis(df)
    df["Pacing"] = compute_pacing(df)
    dfb = add_baselines(df, value_cols=["CTR","CPC","CVR","CPA","QCI","EI","Pacing","spend","revenue"], group_cols=("campaign_id",), ts_col="ts")
    dfc = classify_anomalies(dfb, sensitivity=st.session_state.get("sensitivity","Balanced"), require_persistence=False)
    dfc["revenue_at_risk"] = revenue_at_risk(dfc)
    st.session_state["df"] = dfc
    return dfc

# Preload
if sample:
    try:
        df = load_sample_files()
        prep(df)
        st.success("Sample data loaded.")
    except Exception as e:
        st.error(f"Failed to load sample data: {e}")
elif uploaded:
    try:
        df = load_data(uploaded)
        prep(df)
        st.success("Uploaded data loaded.")
    except Exception as e:
        st.error(f"Failed to load uploaded data: {e}")
else:
    st.info("Toggle sample data or upload CSVs to begin.")

st.markdown("---")
st.subheader("Project Structure")
st.code("""
pages/
  1_🏠_Hub.py           # Unified anomaly feed
  2_🔍_Alert_Detail.py  # Active alerts list + recommendations
  3_⚙️_Baseline_Lab.py  # Contextual baselines and ESI
utils/
  kpi.py, anomaly.py    # KPI math and anomaly logic
data/
  *.csv                 # Sample datasets
""", language="text")

# --- 📊 Campaign Overview Dashboard ---
import plotly.express as px
import numpy as np
import pandas as pd

st.markdown("## 📊 Overview")

df = st.session_state.get("df")

if df is None or df.empty:
    st.info("Load or generate data first to view the campaign dashboard.")
else:
    # Aggregate metrics
    df["date"] = df["ts"].dt.date
    summary = (
        df.groupby("date")[["clicks","conversions","spend"]]
        .sum()
        .reset_index()
    )

    clicks = int(df["clicks"].sum())
    convs = int(df["conversions"].sum())
    cost = float(df["spend"].sum())
    cpa = cost / convs if convs > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clicks", f"{clicks:,}")
    c2.metric("Conversions", f"{convs:,}")
    c3.metric("Cost / Conv.", f"₹{cpa:,.0f}")
    c4.metric("Total Spend", f"₹{cost:,.0f}")

    # Trendline
    st.markdown("### Clicks & Conversions Trend")
    fig = px.line(summary, x="date", y=["clicks","conversions"], markers=True, title="")
    st.plotly_chart(fig, use_container_width=True)

    # Campaign summary
    camp = (
        df.groupby("campaign_id")
          .agg(clicks=("clicks","sum"),
               conversions=("conversions","sum"),
               spend=("spend","sum"),
               ctr=("CTR","mean"),
               cvr=("CVR","mean"))
          .reset_index()
    )
    camp["CTR"] = (camp["ctr"]*100).round(2).astype(str) + "%"
    camp["CVR"] = (camp["cvr"]*100).round(2).astype(str) + "%"
    camp["Spend"] = camp["spend"].apply(lambda x: f"₹{x:,.0f}")

    st.markdown("### Campaign Performance Summary")
    st.dataframe(camp[["campaign_id","clicks","conversions","Spend","CTR","CVR"]],
                 use_container_width=True)
