
import numpy as np
import pandas as pd

SENSITIVITY = {
    "Aggressive": {"pct": 0.15, "z": 2.3},
    "Balanced":   {"pct": 0.25, "z": 3.0},
    "Conservative":{"pct": 0.35, "z": 3.7},
}

# Metric floors to avoid tiny baselines skewing pct deviation
_PCT_FLOOR = {
    "CTR": 0.005,   # 0.5%
    "CVR": 0.005,   # 0.5%
    "CPC": 0.05,    # currency units
    "CPA": 0.50,
    "QCI": 0.0001,
    "EI":  0.05,
    "Pacing": 0.05,
}

def hour_of_week(ts: pd.Series) -> pd.Series:
    dow = ts.dt.dayofweek
    hour = ts.dt.hour
    return dow*24 + hour

def robust_baseline(x: pd.Series) -> tuple[float, float]:
    med = x.median()
    mad = (x - med).abs().median()
    return med, 1.4826*mad + 1e-9

def add_baselines(df: pd.DataFrame, value_cols, group_cols=("campaign_id",), ts_col="ts"):
    df = df.copy()
    df["_how"] = hour_of_week(df[ts_col])
    for col in value_cols:
        bmed = f"baseline_median_{col}"
        bband = f"baseline_band_{col}"
        df[bmed] = np.nan
        df[bband] = np.nan
        for _, idx in df.groupby(list(group_cols)+["_how"]).groups.items():
            s = df.loc[idx, col]
            med, band = robust_baseline(s)
            df.loc[idx, bmed] = med
            df.loc[idx, bband] = band
    return df

def severity_from_pct(pct_change: np.ndarray) -> np.ndarray:
    sev = np.full_like(pct_change, fill_value=4, dtype=int)
    sev[pct_change >= 0.45] = 1
    sev[(pct_change >= 0.30) & (pct_change < 0.45)] = 2
    sev[(pct_change >= 0.20) & (pct_change < 0.30)] = 3
    return sev

def classify_anomalies(df: pd.DataFrame, sensitivity="Balanced", require_persistence=False) -> pd.DataFrame:
    df = df.copy()
    cfg = SENSITIVITY.get(sensitivity, SENSITIVITY["Balanced"])

    for col in ["CTR","CPC","CVR","CPA","QCI","EI","Pacing"]:
        med = df.get(f"baseline_median_{col}")
        cur = df.get(col)
        if med is None or cur is None:
            continue

        floor = _PCT_FLOOR.get(col, 1e-6)
        denom = np.maximum(np.abs(med), floor)

        with np.errstate(divide='ignore', invalid='ignore'):
            pct = np.abs(cur - med) / denom

        df[f"{col}_pct_dev"] = pct
        df[f"{col}_sev"] = severity_from_pct(pct)

        if col in ("CTR","CVR","EI","QCI"):
            df[f"{col}_is_anom"] = (cur < med) & (pct >= cfg["pct"])
        elif col in ("CPC","CPA","Pacing"):
            df[f"{col}_is_anom"] = (cur > med) & (pct >= cfg["pct"])
        else:
            df[f"{col}_is_anom"] = pct >= cfg["pct"]

    df = df.sort_values(["campaign_id","ts"]).reset_index(drop=True)
    for col in ["CTR","CPC","CVR","CPA","QCI","EI","Pacing"]:
        flag = df[f"{col}_is_anom"].astype(int)
        roll = flag.groupby(df["campaign_id"]).rolling(2).sum().reset_index(level=0, drop=True)
        df[f"{col}_persist"] = (roll >= 2)
        df[f"{col}_alert"] = df[f"{col}_is_anom"] & (df[f"{col}_persist"] if require_persistence else True)

    k_cols = [f"{k}_sev" for k in ["EI","QCI","CPA","CVR","CTR","CPC","Pacing"] if f"{k}_sev" in df.columns]
    df["row_severity"] = np.nanmax(np.vstack([df[c].fillna(4).values for c in k_cols]), axis=0)
    return df

def revenue_at_risk(df: pd.DataFrame) -> pd.Series:
    """
    Monetary Revenue at Risk per interval:
    Prefer: (baseline_median_revenue - revenue).clip(lower=0)
    Fallback: (baseline_median_EI - EI).clip(lower=0) * spend
    """
    idx = df.index
    cur_rev = df.get("revenue")
    base_rev = df.get("baseline_median_revenue")
    if cur_rev is not None and base_rev is not None:
        return (base_rev - cur_rev).clip(lower=0)

    med_ei = df.get("baseline_median_EI")
    cur_ei = df.get("EI")
    spend = df.get("spend")
    if med_ei is None or cur_ei is None or spend is None:
        return pd.Series([np.nan] * len(df), index=idx)
    with np.errstate(invalid='ignore'):
        return ((med_ei - cur_ei).clip(lower=0)) * spend
