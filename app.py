"""
Live Market Data Dashboard (Streamlit) — with Forward Curve tab
---------------------------------------------------------------
Sources: Yahoo Finance (free), EIA (official API).

Quickstart:
1) Save as app.py
2) Streamlit Cloud will use requirements.txt to build
3) Optional: add EIA key in Streamlit Secrets as EIA_API_KEY
"""

import os
from datetime import date, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Live Market Dashboard", page_icon="📈", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Data Source")
source = st.sidebar.radio("Choose source:", ["Yahoo Finance", "EIA", "Forward Curve"], index=0)

today = date.today()
default_start = today - relativedelta(years=1)

# Date input for non-curve tabs
if source in ("Yahoo Finance", "EIA"):
    start_date, end_date = st.sidebar.date_input("Date range", (default_start, today))

# --- Yahoo settings ---
if source == "Yahoo Finance":
    tickers = st.sidebar.text_input(
        "Enter tickers (comma-separated):",
        value="CL=F, NG=F, ^GSPC, ^VIX, BTC-USD",
        help="Examples: CL=F (WTI), NG=F (NatGas), ^GSPC (S&P 500), BTC-USD (Bitcoin)",
    )
    interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

# --- EIA settings ---
elif source == "EIA":
    eia_series = st.sidebar.text_input(
        "EIA Series ID:",
        value="PET.RWTC.D",
        help="Example: PET.RWTC.D (Cushing WTI Spot Price, Daily)",
    )
    api_key = st.secrets.get("EIA_API_KEY", os.getenv("EIA_API_KEY", ""))
    st.sidebar.caption("EIA API Key: " + ("✔️ loaded" if api_key else "⚠️ not set"))

# --- Forward Curve settings ---
else:
    st.sidebar.subheader("Forward Curve Settings")
    commodity = st.sidebar.selectbox(
        "Commodity",
        options=["WTI Crude (CL)", "Brent Crude (BZ)", "Henry Hub (NG)"],
        index=0,
        help="Uses Yahoo Finance individual contract tickers."
    )
    months_ahead = st.sidebar.slider("Months ahead", min_value=3, max_value=24, value=12)
    st.sidebar.caption("Tip: increase months ahead for a longer curve.")

# ---------------- FETCH FUNCTIONS ----------------
@st.cache_data(ttl=300)
def fetch_yahoo(symbols: List[str], start: date, end: date, interval="1d") -> Dict[str, pd.DataFrame]:
    data = {}
    for s in symbols:
        try:
            df = yf.download(s, start=start, end=end + timedelta(days=1), interval=interval, progress=False)
            if not df.empty:
                df["Symbol"] = s
                data[s] = df
        except Exception as e:
            st.warning(f"Failed for {s}: {e}")
    return data

@st.cache_data(ttl=300)
def fetch_eia_series(series_id: str, api_key: str, start: date, end: date) -> pd.DataFrame:
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
    data = j["series"][0]["data"]
    df = pd.DataFrame(data, columns=["Period", "Value"])

    # Parse Period that may be YYYYMM, YYYY, or YYYY-MM-DD
    def parse_period(p: str) -> pd.Timestamp:
        p = str(p)
        if len(p) == 4:
            return pd.Timestamp(p) + pd.offsets.YearEnd(0)
        if len(p) == 6:
            return pd.Timestamp(p[:4] + "-" + p[4:]) + pd.offsets.MonthEnd(0)
        return pd.to_datetime(p, errors="coerce")

    df["Date"] = df["Period"].apply(parse_period)
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]
    return df[["Date", "Value"]]

# ---------------- FORWARD CURVE HELPERS ----------------
MONTH_CODES = ["F","G","H","J","K","M","N","Q","U","V","X","Z"]  # Jan..Dec

def build_contract_list(root: str, months: int, start_date: date) -> List[Tuple[str, str]]:
    """
    Build a list of (contract_code, pretty_label) like ('CLZ24', '2024-12') for the next `months`.
    """
    out = []
    y = start_date.year
    m = start_date.month
    # start from current month; shift to next if you prefer next calendar month:
    for i in range(months):
        mm = (m - 1 + i) % 12
        yy = y + (m - 1 + i) // 12
        code = MONTH_CODES[mm]
        yy2 = yy % 100  # 2-digit year
        contract = f"{root}{code}{yy2:02d}"
        label = f"{yy}-{mm_to_num(code):02d}"
        out.append((contract, label))
    return out

def mm_to_num(code: str) -> int:
    return MONTH_CODES.index(code) + 1

def try_yahoo_contract(symbol_base: str) -> pd.Series:
    """
    Try multiple Yahoo variants for a contract:
    e.g., CLZ24, CLZ24.NYM  (keep first that has data)
    Returns a Series with the last available Close price; empty if none.
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

def get_curve(root: str, months_ahead: int, ref_date: date) -> pd.DataFrame:
    rows = []
    for contract, label in build_contract_list(root, months_ahead, ref_date):
        s = try_yahoo_contract(contract)
        if not s.empty:
            last_val = float(s.iloc[-1])
            rows.append({"Contract": contract, "Label": label, "Price": last_val})
    df = pd.DataFrame(rows)
    if not df.empty:
        # sort by Label (YYYY-MM)
        df = df.sort_values("Label")
    return df

# ---------------- LOAD & RENDER ----------------
st.title("📊 Live Market Data Dashboard")

if source == "Yahoo Finance":
    st.caption(f"Source: Yahoo Finance | Date range: {start_date} → {end_date}")
    data = fetch_yahoo([t.strip() for t in tickers.split(",") if t.strip()], start_date, end_date, interval)
    if not data:
        st.error("No Yahoo data returned.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Chart", "Table", "Compare"])

    with tab1:
        symbol = st.selectbox("Choose symbol", list(data.keys()))
        df = data[symbol]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{symbol} Close"))
        fig.update_layout(title=f"{symbol} Price", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sym = st.selectbox("Table view symbol", list(data.keys()), key="table")
        st.dataframe(data[sym])
        st.download_button(
            "Download CSV",
            data[sym].to_csv().encode("utf-8"),
            file_name=f"{sym}.csv",
            mime="text/csv",
        )

    with tab3:
        selected = st.multiselect("Compare tickers", list(data.keys()), default=list(data.keys())[:2])
        if selected:
            dfs = [data[s]["Close"].rename(s) for s in selected]
            combined = pd.concat(dfs, axis=1).dropna()
            norm = (combined / combined.iloc[0] - 1) * 100
            fig2 = px.line(norm, labels={"value": "% Change", "index": "Date", "variable": "Ticker"})
            st.plotly_chart(fig2, use_container_width=True)

elif source == "EIA":
    st.caption(f"Source: EIA | Date range: {start_date} → {end_date}")
    df_eia = fetch_eia_series(eia_series, api_key, start_date, end_date)
    if df_eia.empty:
        st.error("No EIA data returned.")
        st.stop()
    st.subheader(f"EIA Series: {eia_series}")
    st.line_chart(df_eia.set_index("Date")["Value"])
    st.download_button(
        "Download CSV",
        df_eia.to_csv().encode("utf-8"),
        file_name=f"{eia_series}.csv",
        mime="text/csv",
    )

else:  # Forward Curve
    st.caption("Source: Yahoo Finance individual futures contracts (free)")
    # Map display -> root
    root = {"WTI Crude (CL)": "CL", "Brent Crude (BZ)": "BZ", "Henry Hub (NG)": "NG"}[commodity]
    st.subheader(f"Forward Curve — {commodity}")
    with st.spinner("Fetching contract months..."):
        df_curve = get_curve(root=root, months_ahead=months_ahead, ref_date=today)

    if df_curve.empty:
        st.warning(
            "No contract data found yet. Try reducing months ahead, or switch commodity. "
            "If this persists, Yahoo may not list some contract symbols; we’ll expand the symbol search in the next iteration."
        )
    else:
        # Chart
        fig = px.line(
            df_curve,
            x="Label",
            y="Price",
            markers=True,
            title=f"{commodity} — Term Structure",
            labels={"Label": "Delivery Month (YYYY-MM)", "Price": "Price"},
        )
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)

        # Table + download
        st.dataframe(df_curve, use_container_width=True)
        st.download_button(
            "Download Curve CSV",
            df_curve.to_csv(index=False).encode("utf-8"),
            file_name=f"{root}_forward_curve.csv",
            mime="text/csv",
        )

# ---------------- ABOUT ----------------
st.markdown("---")
st.markdown(
    """
    ### ℹ️ About this App
    - **Yahoo Finance:** Free, unofficial data via `yfinance`. Individual contracts are queried like `CLZ24` or `CLZ24.NYM`; whichever returns data is used.
    - **EIA:** Official U.S. Energy Information Administration Open Data API (set `EIA_API_KEY` in Streamlit Secrets).
    - Example EIA series:
        - `PET.RWTC.D` → Cushing, OK WTI Spot Price (Daily)
        - `NG.RNGWHHD.D` → Henry Hub Natural Gas Spot Price (Daily)
    ---
    **Note:** For educational/informational use only.
    """
)
