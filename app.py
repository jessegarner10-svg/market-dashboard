"""
Market Analytics Dashboard — EIA Daily + Forward Curve (with Forward Strip Builder)
-----------------------------------------------------------------------------------
- EIA (Daily): 3-year history + daily strip builder (unchanged)
- Forward Curve: CL/BZ/NG via free Yahoo futures tickers
- NEW: Forward Curve Strip Builder (1M, 3M, Seasonal, Calendar Year, Custom range)
  -> averages across selected contract months from the current curve snapshot

Setup on Streamlit Cloud:
1) App menu → Settings → Secrets → add:
   EIA_API_KEY = your_real_key_here
2) requirements.txt already covers deps.

Note: Data provided as-is for informational use only.
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

# EIA Daily series IDs (spot/daily)
EIA_SERIES = {
    "WTI (Cushing) — Daily": "PET.RWTC.D",     # WTI Spot, Daily
    "Brent — Daily": "PET.RBRTE.D",            # Brent Spot, Daily
    "Henry Hub — Daily": "NG.RNGWHHD.D",       # Henry Hub Spot, Daily
}

# Desk seasons (generic)
SEASONS = {
    "Summer (Apr–Oct)": [4,5,6,7,8,9,10],
    "Winter (Nov–Mar)": [11,12,1,2,3],
}

# ---------------- HELPERS: EIA ----------------
def parse_eia_period(p: str) -> pd.Timestamp:
    """
    Parse EIA period strings: YYYY, YYYYMM, YYYYMMDD, or ISO date.
    """
    p = str(p)
    if len(p) == 4 and p.isdigit():
        return pd.Timestamp(p) + pd.offsets.YearEnd(0)      # yearly -> year-end
    if len(p) == 6 and p.isdigit():
        return pd.Timestamp(p[:4] + "-" + p[4:]) + pd.offsets.MonthEnd(0)  # monthly -> month-end
    if len(p) == 8 and p.isdigit():
        return pd.to_datetime(p, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(p, errors="coerce")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_eia_series_daily(series_id: str, api_key: Optional[str], start: date, end: date) -> pd.DataFrame:
    """
    Fetch EIA series and clip to [start, end]. Returns DataFrame: Date, Value (daily if available).
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
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]
    return df[["Date", "Value"]].reset_index(drop=True)

# ---------------- HELPERS: Forward Curve ----------------
def mm_to_num(code: str) -> int:
    return MONTH_CODES.index(code) + 1

def build_contract_list(root: str, months: int, start_date: date) -> List[Tuple[str, str]]:
    """
    Build (contract_code, delivery_label) pairs, e.g. ('CLZ24','2024-12') for next `months` months.
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
    Try multiple Yahoo variants for a futures contract and return last Close series if found.
    """
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
    """
    Arithmetic average of contract prices for the given list of 'YYYY-MM' delivery labels.
    Only labels present in curve_df are included.
    """
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
    index=1,  # default to Forward Curve since that's where strips live now
)

# API key handling
EIA_KEY = st.secrets.get("EIA_API_KEY", os.getenv("EIA_API_KEY", ""))
with st.sidebar.expander("API Keys", expanded=False):
    st.caption("EIA key is read from **Secrets** (preferred) or environment.")
    if EIA_KEY:
        st.success("EIA key loaded.")
    else:
        st.warning("No EIA key found. Add it in Settings → Secrets as `EIA_API_KEY`.")

# ====================== EIA DAILY (unchanged) ======================
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
        df_eia = fetch_eia_series_daily(series_id, EIA_KEY, start_dt, end_dt)
    has_data = not df_eia.empty

    st.subheader(f"{series_label}")
    st.caption(f"Window: {start_dt} → {end_dt}  |  Points: {len(df_eia)}")

    if has_data:
        fig = px.line(df_eia, x="Date", y="Value", title=None, labels={"Value":"Price","Date":"Date"})
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
        st.warning("No EIA data returned. Check the key (Settings → Secrets), the series ID, or narrow the window.")

    # (Daily strip builder remains here if you want to use daily averages too.)

# =================== FORWARD CURVE (with Strip Builder) ===================
else:
    st.title("Forward Curve")
    st.caption("Source: Yahoo individual futures contracts (best-effort, free)")

    # Curve fetch controls
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

    # Helpers for delivery lists
    deliveries_all = curve["Delivery"].tolist()
    deliveries_dt = curve[["Delivery","Delivery_dt"]].drop_duplicates().set_index("Delivery")["Delivery_dt"].to_dict()
    years_available = sorted({d.year for d in deliveries_dt.values()})

    strip_type = st.radio(
        "Strip type",
        ["1 Month", "3 Month", "Seasonal", "Calendar Year", "Custom Month Range"],
        horizontal=True
    )

    selected_deliveries: List[str] = []

    # 1) 1 Month (choose delivery from those present)
    if strip_type == "1 Month":
        sel = st.selectbox("Delivery month", deliveries_all, index=0)
        selected_deliveries = [sel]

    # 2) 3 Month (consecutive months starting from an anchor delivery)
    elif strip_type == "3 Month":
        anchor = st.selectbox("Anchor delivery", deliveries_all, index=0)
        # take anchor + next two deliveries in the curve list
        try:
            i = deliveries_all.index(anchor)
            selected_deliveries = deliveries_all[i:i+3]
        except ValueError:
            selected_deliveries = []

    # 3) Seasonal (Apr–Oct or Nov–Mar), for a chosen contract year
    elif strip_type == "Seasonal":
        season_name = st.selectbox("Season", list(SEASONS.keys()), index=0)
        # choose the "season year" based on available years
        season_year = st.selectbox("Season year", years_available, index=len(years_available)-1)
        months = SEASONS[season_name]

        def is_in_summer(dt: date, base_year: int) -> bool:
            return (dt.year == base_year) and (dt.month in [4,5,6,7,8,9,10])

        def is_in_winter(dt: date, base_year: int) -> bool:
            # Winter: Nov (base_year-1), Dec (base_year-1), Jan–Mar (base_year)
            return ((dt.year == base_year - 1 and dt.month in [11,12]) or
                    (dt.year == base_year and dt.month in [1,2,3]))

        for lab, dttm in deliveries_dt.items():
            d = dttm.date()
            if season_name.startswith("Summer"):
                if is_in_summer(d, season_year):
                    selected_deliveries.append(lab)
            else:
                if is_in_winter(d, season_year):
                    selected_deliveries.append(lab)

        selected_deliveries = [lab for lab in deliveries_all if lab in set(selected_deliveries)]

    # 4) Calendar Year (all deliveries within a year)
    elif strip_type == "Calendar Year":
        yr = st.selectbox("Year", years_available, index=len(years_available)-1)
        selected_deliveries = [lab for lab, dttm in deliveries_dt.items() if dttm.year == yr]
        selected_deliveries = [lab for lab in deliveries_all if lab in set(selected_deliveries)]

    # 5) Custom Month Range (pick start and end delivery from the curve)
    else:
        c1, c2 = st.columns(2)
        with c1:
            start_lab = st.selectbox("Start delivery", deliveries_all, index=0)
        with c2:
            end_lab = st.selectbox("End delivery", deliveries_all, index=min(2, len(deliveries_all)-1))
        try:
            i_start = deliveries_all.index(start_lab)
            i_end = deliveries_all.index(end_lab)
            if i_end < i_start:
                i_start, i_end = i_end, i_start
            selected_deliveries = deliveries_all[i_start:i_end+1]
        except ValueError:
            selected_deliveries = []

    # Compute forward strip average
    if not selected_deliveries:
        st.warning("No matching deliveries found for this strip selection.")
    else:
        avg = avg_strip_from_curve(curve, selected_deliveries)
        if avg is None:
            st.warning("Selected deliveries not present in the current curve.")
        else:
            st.success(
                f"Strip ({strip_type}): {selected_deliveries[0]} → {selected_deliveries[-1]} "
                f"• Included: {len(selected_deliveries)} months • **Average: {avg:,.4f}**"
            )

            # Show included contracts
            included = curve[curve["Delivery"].isin(selected_deliveries)][["Contract","Delivery","Price"]]
            st.dataframe(included, use_container_width=True)
            st.download_button(
                "Download Strip CSV",
                included.to_csv(index=False).encode("utf-8"),
                file_name=f"{root}_strip_{strip_type.replace(' ','_')}.csv",
                mime="text/csv",
            )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© Market Analytics Dashboard — For informational use only. Data sources: EIA Open Data, Yahoo Finance.")
