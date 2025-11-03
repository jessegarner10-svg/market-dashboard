"""
Market Analytics Dashboard — EIA Daily + Forward Curve
------------------------------------------------------
- Professional UI (no Yahoo tab)
- EIA Daily (3y history) with strip builder (1M/3M, Seasonal, Calendar, Custom)
- Forward Curve (WTI/Brent/Henry Hub) via free Yahoo futures contract tickers

Setup (Streamlit Cloud):
1) In app Settings → Secrets, add:
   EIA_API_KEY = your_real_key_here
2) requirements.txt already covers deps.

Note: Data provided as-is for informational use.
"""

import os
from datetime import date, timedelta, datetime
from typing import List, Dict, Tuple, Optional

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

# EIA series IDs (Daily):
EIA_SERIES = {
    "WTI (Cushing) — Daily": "PET.RWTC.D",       # WTI Spot, Daily
    "Brent — Daily": "PET.RBRTE.D",              # Brent Spot, Daily
    "Henry Hub — Daily": "NG.RNGWHHD.D",         # Henry Hub Spot, Daily
}

# Seasons (generic energy desk conventions)
SEASONS = {
    "Summer (Apr–Oct)": [4,5,6,7,8,9,10],
    "Winter (Nov–Mar)": [11,12,1,2,3],
}

# ---------------- HELPERS ----------------
def parse_eia_period(p: str) -> pd.Timestamp:
    """
    Parse EIA period strings: YYYY, YYYYMM, YYYYMMDD, or ISO-like.
    """
    p = str(p)
    if len(p) == 4 and p.isdigit():
        # Yearly -> year-end
        return pd.Timestamp(p) + pd.offsets.YearEnd(0)
    if len(p) == 6 and p.isdigit():
        # Monthly -> month-end
        return pd.Timestamp(p[:4] + "-" + p[4:]) + pd.offsets.MonthEnd(0)
    if len(p) == 8 and p.isdigit():
        # Daily: YYYYMMDD
        return pd.to_datetime(p, format="%Y%m%d", errors="coerce")
    # Fallback
    return pd.to_datetime(p, errors="coerce")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_eia_series_daily(series_id: str, api_key: Optional[str], start: date, end: date) -> pd.DataFrame:
    """
    Fetch EIA series and clip to last 3y daily (or the given window).
    Returns DataFrame columns: Date, Value.
    """
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
    df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    return df[["Date","Value"]].reset_index(drop=True)

def mm_to_num(code: str) -> int:
    return MONTH_CODES.index(code) + 1

def build_contract_list(root: str, months: int, start_date: date) -> List[Tuple[str, str]]:
    """
    Build (contract_code, pretty_label) pairs, e.g. ('CLZ24', '2024-12').
    Starts at current month; increase range to look further out the curve.
    """
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
    """
    Try multiple Yahoo variants for a contract:
    e.g., CLZ24, CLZ24.NYM (returns last Close series if found)
    """
    candidates = [symbol_base, f"{symbol_base}.NYM"]
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
        df = df.sort_values("Delivery")
    return df

def strip_average(df_daily: pd.DataFrame, start_dt: date, end_dt: date) -> Optional[float]:
    """
    Average of daily 'Value' between start_dt and end_dt (inclusive).
    """
    if df_daily.empty:
        return None
    mask = (df_daily["Date"].dt.date >= start_dt) & (df_daily["Date"].dt.date <= end_dt)
    sub = df_daily.loc[mask, "Value"].dropna()
    if sub.empty:
        return None
    return float(sub.mean())

def strip_label(start_dt: date, end_dt: date) -> str:
    return f"{start_dt.isoformat()} → {end_dt.isoformat()}"

# ---------------- SIDEBAR ----------------
st.sidebar.header("Navigation")
tab_choice = st.sidebar.radio(
    "Choose a module:",
    options=["EIA (Daily)", "Forward Curve"],
    index=0,
)

# API key handling
EIA_KEY = st.secrets.get("EIA_API_KEY", os.getenv("EIA_API_KEY", ""))
with st.sidebar.expander("API Keys", expanded=False):
    st.caption("EIA key is read from **Secrets** (preferred) or environment.")
    if EIA_KEY:
        st.success("EIA key loaded.")
    else:
        st.warning("No EIA key found. Add it in Settings → Secrets as `EIA_API_KEY`.")

# ---------------- EIA DAILY ----------------
if tab_choice == "EIA (Daily)":
    st.title("EIA Daily — 3-Year History & Strips")
    st.caption("Source: U.S. EIA Open Data API")
    c1, c2 = st.columns([2,1])
    with c1:
        series_mode = st.radio("Series", ["WTI (Cushing) — Daily","Brent — Daily","Henry Hub — Daily","Custom (enter EIA series ID)"], horizontal=True)
    with c2:
        # Window is fixed to last 3 years by default (but allow override)
        default_window = st.checkbox("Use last 3 years", value=True)
    if series_mode == "Custom (enter EIA series ID)":
        custom_series = st.text_input("EIA series ID", value="", placeholder="e.g., PET.RWTC.D")
        series_id = custom_series.strip()
        series_label = series_id or "Custom"
    else:
        series_id = EIA_SERIES[series_mode]
        series_label = series_mode

    if default_window:
        start_dt, end_dt = THREE_YEARS_AGO, TODAY
    else:
        start_dt, end_dt = st.date_input("Date range (daily)", (THREE_YEARS_AGO, TODAY))

    # Fetch
    with st.spinner("Fetching EIA daily data…"):
        df_eia = fetch_eia_series_daily(series_id, EIA_KEY, start_dt, end_dt)

    if df_eia.empty:
        st.error("No data returned. Check EIA key, series ID, or date window.")
        st.stop()

    st.subheader(f"{series_label}")
    st.caption(f"Window: {start_dt} → {end_dt}  |  Points: {len(df_eia)}")

    # Chart
    fig = px.line(df_eia, x="Date", y="Value", title=None, labels={"Value":"Price","Date":"Date"})
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Table + download
    with st.expander("Data (daily)", expanded=False):
        st.dataframe(df_eia, use_container_width=True)
        st.download_button(
            "Download CSV",
            df_eia.to_csv(index=False).encode("utf-8"),
            file_name=f"EIA_{series_id.replace('.','_')}_{start_dt}_{end_dt}.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.subheader("Strip Builder")

    # Strip controls
    strip_type = st.radio(
        "Strip type",
        ["1 Month", "3 Month", "Seasonal", "Calendar Year", "Custom Range"],
        horizontal=True
    )

    # Helper: get month span from anchor
    latest_date = df_eia["Date"].max().date()
    def end_of_month(d: date) -> date:
        first_next = (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()
        return first_next

    if strip_type == "1 Month":
        anchor = st.date_input("Anchor month", value=date(latest_date.year, latest_date.month, 1))
        start_strip = date(anchor.year, anchor.month, 1)
        end_strip = end_of_month(start_strip)

    elif strip_type == "3 Month":
        anchor = st.date_input("Anchor (first month of strip)", value=date(latest_date.year, latest_date.month, 1))
        # 3 consecutive months
        end_m = anchor + relativedelta(months=2)
        start_strip = date(anchor.year, anchor.month, 1)
        end_strip = end_of_month(date(end_m.year, end_m.month, 1))

    elif strip_type == "Seasonal":
        season_name = st.selectbox("Season", list(SEASONS.keys()), index=0)
        year = st.number_input("Contract year", min_value=2000, max_value=2100, value=latest_date.year)
        months = SEASONS[season_name]
        # Build start/end from months list (note: Winter spans years)
        months_sorted = sorted(months, key=lambda m: (m<months[0], m))  # keep order roughly as defined
        first_m = months_sorted[0]
        last_m = months_sorted[-1]
        if season_name.startswith("Winter"):
            # Nov–Mar spans two years: Nov–Dec of (year-1) + Jan–Mar of (year)
            start_strip = date(year-1, 11, 1)
            end_strip = end_of_month(date(year, 3, 1))
        else:
            # Summer Apr–Oct, single year
            start_strip = date(year, 4, 1)
            end_strip = end_of_month(date(year, 10, 1))

    elif strip_type == "Calendar Year":
        cal_year = st.number_input("Year", min_value=2000, max_value=2100, value=latest_date.year)
        start_strip = date(cal_year, 1, 1)
        end_strip = date(cal_year, 12, 31)

    else:  # Custom Range
        start_strip, end_strip = st.date_input("Custom strip window", (latest_date.replace(day=1), latest_date))

    # Compute average
    avg = strip_average(df_eia, start_strip, end_strip)
    if avg is None:
        st.warning("No data points in the selected strip window.")
    else:
        st.success(f"Strip: {strip_label(start_strip, end_strip)}  •  **Average: {avg:,.4f}**")
        # Show monthly breakdown table for clarity
        # Group daily into months intersecting the strip
        df_in = df_eia[(df_eia["Date"].dt.date >= start_strip) & (df_eia["Date"].dt.date <= end_strip)].copy()
        if not df_in.empty:
            df_in["YearMonth"] = df_in["Date"].dt.to_period("M")
            monthly = df_in.groupby("YearMonth")["Value"].mean().reset_index()
            monthly["YearMonth"] = monthly["YearMonth"].astype(str)
            st.caption("Monthly averages within strip")
            st.dataframe(monthly, use_container_width=True)

# ---------------- FORWARD CURVE ----------------
else:
    st.title("Forward Curve (Free Contracts)")
    st.caption("Source: Yahoo individual futures contracts (best-effort)")

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
    else:
        fig = px.line(
            curve, x="Delivery", y="Price",
            markers=True,
            labels={"Delivery":"Delivery Month (YYYY-MM)", "Price":"Price"},
            title=None
        )
        fig.update_traces(mode="lines+markers")
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(curve, use_container_width=True)
        st.download_button(
            "Download Curve CSV",
            curve.to_csv(index=False).encode("utf-8"),
            file_name=f"{root}_forward_curve.csv",
            mime="text/csv",
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© Market Analytics Dashboard — For informational use only. Data sources: EIA Open Data, Yahoo Finance.")
