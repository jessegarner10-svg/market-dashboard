"""
Market Analytics Dashboard — EIA Daily + Forward Curve (with EIA diagnostics)
-----------------------------------------------------------------------------
- EIA (Daily): 3y history + diagnostics + connectivity test + strip builder (1M/3M/Seasonal/Calendar/Custom)
- Forward Curve: CL/BZ/NG via free Yahoo futures contract tickers + forward strip builder

Setup on Streamlit Cloud:
1) App Settings → Secrets → TOML:
   EIA_API_KEY="your_real_key_here"
2) requirements.txt should include: streamlit, yfinance, plotly, pandas, numpy, requests, python-dateutil
"""

import os
from datetime import date
from typing import List, Tuple, Optional, Dict

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
def _parse_eia_period(p: str) -> pd.Timestamp:
    """Parse EIA period strings: YYYY, YYYYMM, YYYYMMDD, or ISO-like."""
    p = str(p)
    if len(p) == 8 and p.isdigit():   # daily
        return pd.to_datetime(p, format="%Y%m%d", errors="coerce")
    if len(p) == 6 and p.isdigit():   # monthly
        return pd.to_datetime(p[:4] + "-" + p[4:] + "-01", errors="coerce") + pd.offsets.MonthEnd(0)
    if len(p) == 4 and p.isdigit():   # yearly
        return pd.to_datetime(p + "-12-31", errors="coerce")
    return pd.to_datetime(p, errors="coerce")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_eia_series_daily(series_id: str, api_key: Optional[str], start: date, end: date):
    """
    Fetch EIA series via v1 endpoint and return (df, info).
    df: DataFrame with columns [Date, Value] (or empty)
    info: diagnostics dict {ok, status, url, reason, api_key_present, raw_error}
    """
    info: Dict[str, object] = {
        "ok": False, "status": None, "url": "", "reason": "",
        "api_key_present": bool(api_key), "raw_error": ""
    }
    if not series_id:
        info["reason"] = "No series_id provided."
        return pd.DataFrame(), info

    url = "https://api.eia.gov/series/"
    params = {"series_id": series_id}
    if api_key:
        params["api_key"] = api_key

    info["url"] = f"{url}?series_id={series_id}" + ("&api_key=***" if api_key else "")

    try:
        r = requests.get(url, params=params, timeout=30)
        info["status"] = r.status_code
        if r.status_code != 200:
            info["reason"] = f"HTTP {r.status_code}"
            try:
                info["raw_error"] = str(r.json())[:500]
            except Exception:
                info["raw_error"] = r.text[:500]
            return pd.DataFrame(), info

        j = r.json()
        if "error" in j:
            info["reason"] = "API error"
            info["raw_error"] = str(j["error"])[:500]
            return pd.DataFrame(), info

        if "series" not in j or not j["series"]:
            info["reason"] = "No 'series' in response"
            info["raw_error"] = str(j)[:500]
            return pd.DataFrame(), info

        data = j["series"][0].get("data", [])
        if not data:
            info["reason"] = "Empty 'data' array"
            return pd.DataFrame(), info

        df = pd.DataFrame(data, columns=["Period", "Value"])
        df["Date"] = df["Period"].apply(_parse_eia_period)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Date", "Value"]).sort_values("Date")

        # Clip to requested window
        df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)].reset_index(drop=True)

        info["ok"] = True
        return df[["Date", "Value"]], info

    except Exception as e:
        info["reason"] = f"Exception: {e}"
        return pd.DataFrame(), info

# ---------------- HELPERS: Forward Curve ----------------
def mm_to_num(code: str) -> int:
    return MONTH_CODES.index(code) + 1

def build_contract_list(root: str, months: int, start_date: date) -> List[Tuple[str, str]]:
    """Return (contract_code, delivery_label 'YYYY-MM') for next `months` months."""
    out: List[Tuple[str, str]] = []
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
    """Try a couple of Yahoo suffixes for a futures contract; return last Close if found."""
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
    index=0,
)

# API key handling (for EIA)
EIA_KEY = st.secrets.get("EIA_API_KEY", os.getenv("EIA_API_KEY", ""))
with st.sidebar.expander("API Keys", expanded=False):
    st.caption("EIA key is read from Secrets (TOML) or environment.")
    if EIA_KEY:
        st.success("EIA key loaded.")
    else:
        st.warning('Add in Cloud Secrets as TOML: EIA_API_KEY="your_key_here"')

# ====================== EIA (Daily) ======================
if tab_choice == "EIA (Daily)":
    st.title("EIA Daily — 3-Year History & Strips")
    st.caption("Source: U.S. EIA Open Data API")

    c1, c2 = st.columns([2,1])
    with c1:
        series_mode = st.radio(
            "Series",
            ["WTI (Cushing) — Daily", "Brent — Daily", "Henry Hub — Daily", "Custom (enter EIA series ID)"],
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

    # Fetch with diagnostics
    with st.spinner("Fetching EIA daily data…"):
        df_eia, eia_info = fetch_eia_series_daily(series_id, EIA_KEY, start_dt, end_dt)
    has_data = not df_eia.empty

    # Diagnostics panel
    with st.expander("EIA Diagnostics", expanded=False):
        st.write({
            "api_key_present": eia_info.get("api_key_present"),
            "request_status": eia_info.get("status"),
            "ok": eia_info.get("ok"),
            "reason": eia_info.get("reason"),
            "request_url": eia_info.get("url"),
            "raw_error": (eia_info.get("raw_error") or "")[:500]
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
        fig = px.line(df_eia, x="Date", y="Value", labels={"Value": "Price", "Date": "Date"})
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
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
        st.warning("No EIA data returned. Check key/series/date window — see Diagnostics above.")

    # -------- Strip Builder (daily averages over chosen window) --------
    st.markdown("---")
    st.subheader("Strip Builder (Daily averages)")

    # Helper: end-of-month
    def end_of_month(d: date) -> date:
        return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()

    strip_type = st.radio(
        "Strip type",
        ["1 Month", "3 Month", "Seasonal", "Calendar Year", "Custom Range"],
        horizontal=True
    )

    latest_date = df_eia["Date"].max().date() if has_data else TODAY

    if strip_type == "1 Month":
        anchor = st.date_input("Anchor month", value=date(latest_date.year, latest_date.month, 1), key="strip1m_eia")
        start_strip = date(anchor.year, anchor.month, 1)
        end_strip = end_of_month(start_strip)

    elif strip_type == "3 Month":
        anchor = st.date_input("Anchor (first month of strip)", value=date(latest_date.year, latest_date.month, 1), key="strip3m_eia")
        end_mo = anchor + relativedelta(months=2)
        start_strip = date(anchor.year, anchor.month, 1)
        end_strip = end_of_month(date(end_mo.year, end_mo.month, 1))

    elif strip_type == "Seasonal":
        season_name = st.radio("Season", ["Summer (Apr–Oct)", "Winter (Nov–Mar)"], horizontal=True)
        year = st.number_input("Contract year", min_value=2000, max_value=2100, value=latest_date.year)
        if season_name.startswith("Winter"):
            start_strip = date(year-1, 11, 1)
            end_strip = end_of_month(date(year, 3, 1))
        else:
            start_strip = date(year, 4, 1)
            end_strip = end_of_month(date(year, 10, 1))

    elif strip_type == "Calendar Year":
        cal_year = st.number_input("Year", min_value=2000, max_value=2100, value=latest_date.year)
        start_strip = date(cal_year, 1, 1)
        end_strip = date(cal_year, 12, 31)

    else:
        default_start = latest_date.replace(day=1)
        start_strip, end_strip = st.date_input("Custom strip window", (default_start, latest_date), key="stripcustom_eia")

    # Compute strip average (only if data exists)
    if has_data:
        mask = (df_eia["Date"].dt.date >= start_strip) & (df_eia["Date"].dt.date <= end_strip)
        sub = df_eia.loc[mask, "Value"].dropna()
        if sub.empty:
            st.info("No daily points inside the selected strip window.")
        else:
            avg = float(sub.mean())
            st.success(f"Strip: {start_strip.isoformat()} → {end_strip.isoformat()}  •  **Average: {avg:,.4f}**")

            # Monthly breakdown
            df_in = df_eia.loc[mask].copy()
            df_in["YearMonth"] = df_in["Date"].dt.to_period("M").astype(str)
            monthly = df_in.groupby("YearMonth")["Value"].mean().reset_index()
            st.caption("Monthly averages within strip")
            st.dataframe(monthly, use_container_width=True)
    else:
        st.info("Strip controls are ready. Add a valid EIA key/series, then click Rerun to compute averages.")

# =================== FORWARD CURVE ===================
else:
    st.title("Forward Curve")
    st.caption("Source: Yahoo individual futures contracts (best-effort, free)")

    commodity = st.selectbox(
        "Commodity",
        options=["WTI Crude (CL)", "Brent Crude (BZ)", "Henry Hub (NG)"],
        index=0
    )
    months_ahead = st.slider("Months ahead", min_value=3, max_value=24, value=12)
    root = {"WTI Crude (CL)": "CL", "Brent Crude (BZ)": "BZ", "Henry Hub (NG)": "NG"}[commodity]

    with st.spinner("Building curve…"):
        curve = get_forward_curve(root, months_ahead, TODAY)

    if curve.empty:
        st.warning("No contract data returned. Try fewer months or a different commodity.")
        st.stop()

    # Curve chart & table
    fig = px.line(
        curve, x="Delivery", y="Price",
        markers=True,
        labels={"Delivery": "Delivery Month (YYYY-MM)", "Price": "Price"},
        title=None
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Curve data", expanded=False):
        st.dataframe(curve[["Contract", "Delivery", "Price"]], use_container_width=True)
        st.download_button(
            "Download Curve CSV",
            curve[["Contract", "Delivery", "Price"]].to_csv(index=False).encode("utf-8"),
            file_name=f"{root}_forward_curve.csv",
            mime="text/csv",
        )

    # -------- Forward Curve Strip Builder (curve-average) --------
    st.markdown("---")
    st.subheader("Forward Curve — Strip Builder")

    deliveries_all = curve["Delivery"].tolist()
    deliveries_dt = curve[["Delivery", "Delivery_dt"]].drop_duplicates().set_index("Delivery")["Delivery_dt"].to_dict()
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
        season = st.radio("Season", ["Summer (Apr–Oct)", "Winter (Nov–Mar)"], horizontal=True)
        sel = []
        for lab, dttm in deliveries_dt.items():
            d = dttm.date()
            if (season.startswith("Summer") and is_summer(d, base_year)) or \
               (season.startswith("Winter") and is_winter(d, base_year)):
                sel.append(lab)
        selected_deliveries = [lab for lab in deliveries_all if lab in set(sel)]

    elif strip_type == "Calendar Year":
        if years_available:
            yr = st.selectbox("Year", years_available, index=len(years_available)-1)
            selected_deliveries = [lab for lab, dttm in deliveries_dt.items() if dttm.year == yr]
            selected_deliveries = [lab for lab in deliveries_all if lab in set(selected_deliveries)]
        else:
            selected_deliveries = []

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
        included = curve[curve["Delivery"].isin(selected_deliveries)][["Contract", "Delivery", "Price"]]
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
