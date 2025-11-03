"""
Market Analytics Dashboard — EIA Daily + Forward Curve (no options)
-------------------------------------------------------------------
- EIA (Daily): 3y history + strip builder (1M/3M/Seasonal/Calendar/Custom)
- Forward Curve: CL/BZ/NG via free Yahoo futures contracts + forward strip builder

Setup on Streamlit Cloud:
1) App Settings → Secrets → TOML:
   EIA_API_KEY="your_real_key_here"
2) requirements.txt should include: streamlit, yfinance, plotly, pandas, numpy, requests, python-dateutil
"""

import os
from datetime import date
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Market Analytics Dashboard", page_icon="📈", layout="wide")

# ---------------- CONSTANTS ----------------
MONTH_CODES = ["F","G","H","J","K","M","N","Q","U","V","X","Z"]  # Jan..Dec
TODAY = date.today()
THREE_YEARS_AGO = TODAY - relativedelta(years=3)

# EIA Daily series IDs (spot/daily)
EIA_SERIES = {
    "WTI (Cushing) — Daily": "PET.RWTC.D",
    "Brent — Daily": "PET.RBRTE.D",
    "Henry Hub — Daily": "NG.RNGWHHD.D",
}

# Desk seasons
SEASONS = {
    "Summer (Apr–Oct)": [4,5,6,7,8,9,10],
    "Winter (Nov–Mar)": [11,12,1,2,3],
}

# ---------------- HELPERS: EIA ----------------
def parse_eia_period(p: str) -> pd.Timestamp:
    p = str(p)
    if len(p) == 4 and p.isdigit():
        return pd.Timestamp(p) + pd.offsets.YearEnd(0)
    if len(p) == 6 and p.isdigit():
        return pd.Timestamp(p[:4] + "-" + p[4:]) + pd.offsets.MonthEnd(0)
    if len(p) == 8 and p.isdigit():
        return pd.to_datetime(p, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(p, errors="coerce")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_eia_series_daily(series_id: str, api_key: Optional[str], start: date, end: date) -> pd.DataFrame:
    if not series_id:
        return pd.DataFrame()
    url = "https://api.eia.gov/series/"
    params = {"series_id": series_id}
    if api_key:
        params["api_key"] = api_key
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    j = r.json()
    if "series" not in j or not j["series"]:
        return pd.DataFrame()
    data = j["series"][0].get("data", [])
    df = pd.DataFrame(data, columns=["Period", "Value"])
    df["Date"] = df["Period"].apply(parse_eia_period)
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]
    return df[["Date", "Value"]].reset_index(drop=True)

# ---------------- HELPERS: Forward Curve ----------------
def mm_to_num(code: str) -> int:
    return MONTH_CODES.index(code) + 1

def build_contract_list(root: str, months: int, start_date: date) -> List[Tuple[str, str]]:
    out = []
    y = start_date.year
    m = start_date.month
    for i in range(months):
        mm = (m - 1 + i) % 12
        yy = y + (m - 1 + i) // 12
        code = MONTH_CODES[mm]
        yy2 = yy % 100
        contract = f"{root}{code}{yy2:02d}"
        label = f"{yy}-{mm_to_num(code):02d}"
        out.append((contract, label))
    return out

def try_yahoo_contract(symbol_base: str) -> pd.Series:
    candidates = [symbol_base, f"{symbol_base}.NYM"]  # NYMEX suffix often works for CL/NG
    for t in candidates:
        try:
            df = yf.download(t, period="14d", interval="1d", progress=False)
            if not df.empty and "Close" in df:
                s = df["Close"].dropna()
                if not s.empty:
                    s.name = t
                    return s
        except Exception:
            continue
    return pd.Series(dtype=float)

def get_forward_curve(root: str, months_ahead: int, ref_date: date) -> pd.DataFrame:
    rows = []
    for contract, label in build_contract_list(root, months_ahead, ref_date):
        s = try_yahoo_contract(contract)
        if not s.empty:
            last_val = float(s.iloc[-1])
            rows.append({"Contract": contract, "Delivery": label, "Price": last_val})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Delivery_dt"] = pd.to_datetime(df["Delivery"] + "-01", errors="coerce")
        df = df.dropna(subset=["Delivery_dt"]).sort_values("Delivery_dt").reset_index(drop=True)
    return df

def avg_strip_from_curve(curve_df: pd.DataFrame, deliveries: List[str]) -> Optional[float]:
    if curve_df.empty:
        return None
    sub = curve_df[curve_df["Delivery"].isin(deliveries)]
    if sub.empty:
        return None
    return float(sub["Price"].mean())

# ---------------- SIDEBAR / NAV ----------------
st.sidebar.header("Navigation")
tab_choice = st.sidebar.radio(
    "Choose a module:",
    options=["EIA (Daily)", "Forward Curve"],
    index=1,  # default to Forward Curve
)

# API key handling (for EIA)
EIA_KEY = st.secrets.get("EIA_API_KEY", os.getenv("EIA_API_KEY", ""))
with st.sidebar.expander("API Keys", expanded=False):
    st.caption("EIA key is read from Secrets (TOML) or environment.")
    if EIA_KEY:
        st.success("EIA key loaded.")
    else:
        st.warning('Add in Cloud Secrets as TOML: EIA_API_KEY="your_key_here"')

# ====================== EIA DAILY ======================
if tab_choice == "EIA (Daily)":
    st.title("EIA Daily — 3-Year History & Strips")
    st.caption("Source: U.S. EIA Open Data API")

    c1, c2 = st.columns([2,1])
    with c1:
        series_mode = st.radio(
            "Series",
            ["WTI (Cushing) — Daily","Brent — Daily","Henry Hub — Daily","Custom (enter EIA series ID)"],
            horizontal=True
        )
    with c2:
        default_window = st.checkbox("Use last 3 years", value=True)

    if series_mode == "Custom (enter EIA series ID)":
        custom_series = st.text_input("EIA series ID", value="", placeholder="e.g., PET.RWTC.D")
        series_id = custom_series.strip()
        series_label = series_id or "Custom"
    else:
        series_id = EIA_SERIES[series_mode]
        series_label = series_mode

    start_dt, end_dt = (THREE_YEARS_AGO, TODAY) if default_window else st.date_input(
        "Date range (daily)", (THREE_YEARS_AGO, TODAY)
    )

  with st.spinner("Fetching EIA daily data…"):
    df_eia, eia_info = fetch_eia_series_daily(series_id, EIA_KEY, start_dt, end_dt)
has_data = not df_eia.empty

with st.expander("EIA Diagnostics", expanded=False):
    st.write({
        "api_key_present": eia_info.get("api_key_present"),
        "request_status": eia_info.get("status"),
        "ok": eia_info.get("ok"),
        "reason": eia_info.get("reason"),
        "request_url": eia_info.get("url"),
        "raw_error": eia_info.get("raw_error")[:500] if eia_info.get("raw_error") else ""
    })

   if st.button("Run EIA connectivity test (WTI daily)"):
    test_series = "PET.RWTC.D"  # WTI Cushing spot daily
    test_df, test_info = fetch_eia_series_daily(test_series, EIA_KEY, THREE_YEARS_AGO, TODAY)
    st.write("Test result:", {
        "ok": test_info.get("ok"),
        "status": test_info.get("status"),
        "reason": test_info.get("reason"),
        "rows": len(test_df),
    })
    if not test_df.empty:
        st.write(test_df.tail(5))

    st.subheader(f"{series_label}")
    st.caption(f"Window: {start_dt} → {end_dt}  |  Points: {len(df_eia)}")

    if has_data:
        fig = px.line(df_eia, x="Date", y="Value", labels={"Value":"Price","Date":"Date"})
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Data (daily)", expanded=False):
            st.dataframe(df_eia, use_container_width=True)
            st.download_button(
                "Download CSV",
                df_eia.to_csv(index=False).encode("utf-8"),
                file_name=f"EIA_{series_id.replace('.','_')}_{start_dt}_{end_dt}.csv",
                mime="text/csv",
            )
    else:
        st.warning("No EIA data returned. Check key/series/range.")

# =================== FORWARD CURVE (with Strip Builder) ===================
else:
    st.title("Forward Curve")
    st.caption("Source: Yahoo individual futures contracts (best-effort, free)")

    commodity = st.selectbox(
        "Commodity",
        options=["WTI Crude (CL)", "Brent Crude (BZ)", "Henry Hub (NG)"],
        index=0
    )
    months_ahead = st.slider("Months ahead", min_value=3, max_value=24, value=12)
    root = {"WTI Crude (CL)":"CL", "Brent Crude (BZ)":"BZ", "Henry Hub (NG)":"NG"}[commodity]

    with st.spinner("Building curve…"):
        curve = get_forward_curve(root, months_ahead, TODAY)

    if curve.empty:
        st.warning("No contract data returned. Try fewer months or a different commodity.")
        st.stop()

    # Curve chart & table
    fig = px.line(
        curve, x="Delivery", y="Price",
        markers=True,
        labels={"Delivery":"Delivery Month (YYYY-MM)", "Price":"Price"},
        title=None
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Curve data", expanded=False):
        st.dataframe(curve[["Contract","Delivery","Price"]], use_container_width=True)
        st.download_button(
            "Download Curve CSV",
            curve[["Contract","Delivery","Price"]].to_csv(index=False).encode("utf-8"),
            file_name=f"{root}_forward_curve.csv",
            mime="text/csv",
        )

    # ----------------- Forward Curve Strip Builder -----------------
    st.markdown("---")
    st.subheader("Forward Curve — Strip Builder")

    deliveries_all = curve["Delivery"].tolist()
    deliveries_dt = curve[["Delivery","Delivery_dt"]].drop_duplicates().set_index("Delivery")["Delivery_dt"].to_dict()
    years_available = sorted({d.year for d in deliveries_dt.values()})

    strip_type = st.radio(
        "Strip type",
        ["1 Month", "3 Month", "Seasonal", "Calendar Year", "Custom Month Range"],
        horizontal=True
    )

    selected_deliveries: List[str] = []

    if strip_type == "1 Month":
        selected_deliveries = [deliveries_all[0]]

    elif strip_type == "3 Month":
        selected_deliveries = deliveries_all[:3]

    elif strip_type == "Seasonal":
        base_year = years_available[-1] if years_available else TODAY.year
        def is_summer(dt: date, y: int) -> bool:
            return (dt.year == y) and (dt.month in [4,5,6,7,8,9,10])
        def is_winter(dt: date, y: int) -> bool:
            return ((dt.year == y-1 and dt.month in [11,12]) or (dt.year == y and dt.month in [1,2,3]))
        season = st.radio("Season", ["Summer (Apr–Oct)","Winter (Nov–Mar)"], horizontal=True)
        sel = []
        for lab, dttm in deliveries_dt.items():
            d = dttm.date()
            if (season.startswith("Summer") and is_summer(d, base_year)) or \
               (season.startswith("Winter") and is_winter(d, base_year)):
                sel.append(lab)
        selected_deliveries = [lab for lab in deliveries_all if lab in set(sel)]

    elif strip_type == "Calendar Year":
        yr = st.selectbox("Year", years_available, index=len(years_available)-1 if years_available else 0)
        selected_deliveries = [lab for lab, dttm in deliveries_dt.items() if dttm.year == yr]
        selected_deliveries = [lab for lab in deliveries_all if lab in set(selected_deliveries)]

    else:  # Custom Month Range
        c1, c2 = st.columns(2)
        with c1:
            start_lab = st.selectbox("Start delivery", deliveries_all, index=0)
        with c2:
            end_lab = st.selectbox("End delivery", deliveries_all, index=min(2, len(deliveries_all)-1))
        i0, i1 = deliveries_all.index(start_lab), deliveries_all.index(end_lab)
        if i1 < i0:
            i0, i1 = i1, i0
        selected_deliveries = deliveries_all[i0:i1+1]

    if selected_deliveries:
        avg = avg_strip_from_curve(curve, selected_deliveries)
        included = curve[curve["Delivery"].isin(selected_deliveries)][["Contract","Delivery","Price"]]
        st.success(
            f"Strip: {selected_deliveries[0]} → {selected_deliveries[-1]} • "
            f"Months: {len(selected_deliveries)} • **Average: {avg:,.4f}**"
        )
        st.dataframe(included, use_container_width=True)
        st.download_button(
            "Download Strip CSV",
            included.to_csv(index=False).encode("utf-8"),
            file_name=f"{root}_forward_strip_{strip_type.replace(' ','_')}.csv",
            mime="text/csv",
        )
    else:
        st.info("Choose a strip to compute its average across curve months.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© Market Analytics Dashboard — Informational only. Sources: EIA Open Data, Yahoo Finance.")
