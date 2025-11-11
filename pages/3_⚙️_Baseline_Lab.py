
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("⚙️ Baseline & Context Lab")

df = st.session_state.get("df")
if df is None:
    st.warning("Load data on the Hub page first.")
    st.stop()

campaign = st.selectbox("Campaign", sorted(df["campaign_id"].unique()))
metric = st.selectbox("Metric", ["CTR","CVR","CPC","CPA","QCI","EI","Pacing"], index=0)

sub = df[df["campaign_id"]==campaign].sort_values("ts").copy()
sub["median"] = sub[f"baseline_median_{metric}"]
sub["band"] = sub[f"baseline_band_{metric}"]

import plotly.express as px
fig = px.line(sub, x="ts", y=metric, title=f"{metric} with Baseline & MAD Bands")
fig.add_scatter(x=sub["ts"], y=sub["median"], mode="lines", name="Baseline Median")
fig.add_scatter(x=sub["ts"], y=sub["median"]+sub["band"], mode="lines", name="+MAD Band", line=dict(dash="dot"))
fig.add_scatter(x=sub["ts"], y=sub["median"]-sub["band"], mode="lines", name="-MAD Band", line=dict(dash="dot"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Sensitivity Modes (demo)")
st.write("Current sensitivity:", st.session_state.get("sensitivity","Balanced"))
st.caption("Aggressive lowers thresholds (more alerts); Conservative raises thresholds (fewer alerts).")
