
import numpy as np
import pandas as pd

def add_basic_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["impressions"] = df["impressions"].clip(lower=0)
    df["clicks"] = df["clicks"].clip(lower=0)
    df["spend"] = df["spend"].clip(lower=0.0)
    df["conversions"] = df["conversions"].clip(lower=0)
    df["revenue"] = df.get("revenue", pd.Series([np.nan]*len(df)))
    df["CTR"] = np.where(df["impressions"]>0, df["clicks"]/df["impressions"], 0.0)
    df["CPC"] = np.where(df["clicks"]>0, df["spend"]/df["clicks"], np.nan)
    df["CVR"] = np.where(df["clicks"]>0, df["conversions"]/df["clicks"], 0.0)
    df["CPA"] = np.where(df["conversions"]>0, df["spend"]/df["conversions"], np.nan)
    df["ROAS"] = np.where(df["spend"]>0, df["revenue"]/df["spend"], np.nan)
    return df

def add_blended_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ei = df["ROAS"]
    fallback = np.where(df["CPA"]>0, 1.0/df["CPA"], np.nan)
    df["EI"] = np.where(np.isfinite(ei), ei, fallback)
    df["QCI"] = df["CTR"] * df["CVR"]
    return df

def compute_pacing(df: pd.DataFrame, plan_col: str = "planned_spend") -> pd.Series:
    planned = df.get(plan_col)
    if planned is None or planned.isna().all():
        return pd.Series([np.nan]*len(df), index=df.index)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (df["spend"] - planned) / planned
        out[~np.isfinite(out)] = np.nan
        return out
