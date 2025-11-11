
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🔍 Alerts – Active List & Detail")

df = st.session_state.get("df")
if df is None:
    st.warning("Load data on the Hub page first.")
    st.stop()

# ---- Change #2: Active alerts list with severity, RCA, recommendation ----
# Build latest snapshot per campaign and pick the worst KPI
snap = df.sort_values("ts").groupby("campaign_id").tail(1).copy()

# Identify the KPI with highest severity & largest pct deviation
kpis = ["EI","QCI","CPA","CVR","CTR","CPC","Pacing"]
sev_cols = {k: f"{k}_sev" for k in kpis if f"{k}_sev" in snap.columns}
pct_cols = {k: f"{k}_pct_dev" for k in kpis if f"{k}_pct_dev" in snap.columns}

def pick_primary(row):
    # rank by severity (1 highest), then by pct_dev
    best = None
    best_key = None
    for k in sev_cols.keys():
        sev = row.get(sev_cols[k], 4)
        pct = row.get(pct_cols[k], 0.0)
        score = (sev, -pct)  # lower sev is better (P1), higher pct_dev preferred
        if best is None or score < best:
            best = score
            best_key = k
    return best_key

snap["primary_kpi"] = snap.apply(pick_primary, axis=1)
snap["severity"] = snap["row_severity"].astype(int)

# RCA & Recommendations heuristics
def rca_and_reco(row):
    k = row["primary_kpi"]
    cur = row.get(k, np.nan)
    base = row.get(f"baseline_median_{k}", np.nan)

    if k in ("CTR","QCI"):
        rca = "Creative fatigue or audience drift"
        reco = "Rotate creatives, refresh audiences, test new hooks"
    elif k == "CVR":
        rca = "Landing/pixel issues or mismatch"
        reco = "QA journey, fix tags, align offer & audience"
    elif k in ("CPC","CPA"):
        # decide direction from current vs baseline
        rca = "Auction pressure or bid strategy drift"
        reco = "Reduce bids, refine placements, shift to stable ad sets"
    elif k in ("EI",):
        rca = "Efficiency drop (cost ↑ / revenue ↓)"
        reco = "Reallocate 15–25% budget to high-EI segments"
    else:
        rca = "Pacing / planning variance"
        reco = "Rebalance daily caps to plan; pause unstable groups"
    return rca, reco

snap[["RCA","Recommendation"]] = snap.apply(lambda r: pd.Series(rca_and_reco(r)), axis=1)

# Pretty columns
def pct_fmt(x): 
    try: return f"{x*100:.1f}%"
    except: return "-"

def money_fmt(x): 
    try: return f"₹{x:,.0f}"
    except: return "-"

# Get the pct_dev column name for each row’s primary_kpi, then pick that value per row
snap["pct_dev"] = snap.apply(
    lambda r: r.get(f"{r['primary_kpi']}_pct_dev", np.nan), axis=1
)

snap["pct_dev"] = snap["pct_dev"].apply(pct_fmt)
snap["impact_hr"] = snap["revenue_at_risk"].fillna(0).round(0).apply(money_fmt)

active_cols = ["severity","campaign_id","primary_kpi","pct_dev","impact_hr","RCA","Recommendation"]
active = snap[active_cols].sort_values(["severity","impact_hr"], ascending=[True,False])

st.markdown("### 🚨 Active Alerts (P1/P2/P3 before details)")
st.dataframe(active, use_container_width=True)

st.markdown("---")
# ---- Per-alert drilldown (kept from earlier) ----
campaign = st.selectbox("Inspect campaign", sorted(df["campaign_id"].unique()))
primary_kpi = st.selectbox("Metric", ["EI","QCI","CPA","CVR","CTR","CPC","Pacing"], index=0)

sub = df[df["campaign_id"]==campaign].sort_values("ts").copy()
sub["median"] = sub[f"baseline_median_{primary_kpi}"]
sub["band"] = sub[f"baseline_band_{primary_kpi}"]

fig = px.line(sub, x="ts", y=primary_kpi, title=f"{primary_kpi} vs Baseline")
fig.add_scatter(x=sub["ts"], y=sub["median"], mode="lines", name="Baseline Median")
fig.add_scatter(x=sub["ts"], y=sub["median"]+sub["band"], mode="lines", name="+MAD Band", line=dict(dash="dot"))
fig.add_scatter(x=sub["ts"], y=sub["median"]-sub["band"], mode="lines", name="-MAD Band", line=dict(dash="dot"))
st.plotly_chart(fig, use_container_width=True)

# Minimal metrics
cols = st.columns(3)
with cols[0]:
    st.metric("Current", f"{sub[primary_kpi].iloc[-1]:.4f}")
with cols[1]:
    st.metric("Baseline", f"{sub['median'].iloc[-1]:.4f}")
with cols[2]:
    sev = int(sub["row_severity"].iloc[-1])
    st.metric("Severity (P1–P4)", f"P{sev}")
