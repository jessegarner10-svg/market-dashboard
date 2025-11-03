"""
Market Analytics Dashboard — EIA Historicals (v2) + EIA Projections + Forward Curve
-----------------------------------------------------------------------------------
Tabs:
- EIA — Historical Prices (v2 /seriesid/{id})    : daily/monthly historical charts + diagnostics
- EIA — Projections (v2 URL or CSV upload)       : paste an EIA v2 API URL OR upload EIA CSV (e.g., STEO/AEO)
- Forward Curve                                   : CL/BZ/NG via Yahoo futures + strip builder

Setup on Streamlit Cloud:
1) App Settings → Secrets → TOML:
   EIA_API_KEY="your_real_key_here"
2) requirements.txt:
   streamlit, yfinance, plotly, pandas, numpy, requests, python-dateutil
"""

import os
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from datetime import date
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Market Analytics Dashboard", page_icon="📈", layout="wide")

# ---------------- CONSTANTS ----------------
MONTH_CODES = ["F","G","H","J","K","M","N","Q","U","V","X","Z"]  # Jan..Dec
TODAY = date.today()
THREE_YEARS_AGO = TODAY - relativedelta(years=3)

# EIA v1-style series IDs (work with v2/seriesid/)
EIA_SERIES = {
    "WTI (Cushing) — Daily": "PET.RWTC.D",
    "Brent — Daily": "PET.RBRTE.D",
    "Henry Hub — Daily": "NG.RNGWHHD.D",
}

# ---------------- HELPERS: EIA v2 (historicals) ----------------
def _parse_eia_period_v2(p: str) -> pd.Timestamp:
    """EIA v2 'period' values: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD'."""
    p = str(p)
    if len(p) == 10:  # YYYY-MM-DD
        return pd.to_datetime(p, format="%Y-%m-%d", errors="coerce")
    if len(p) == 7:   # YYYY-MM (normalize to month-end for nicer charts)
        return pd.to_datetime(p + "-01", errors="coerce") + pd.offsets.MonthEnd(0)
    if len(p) == 4:   # YYYY
        return pd.to_datetime(p + "-12-31", errors="coerce")
    return pd.to_datetime(p, errors="coerce")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_eia_series_daily(series_id: str, api_key: Optional[str], start: date, end: date):
    """
    Fetch EIA historical series via v2 backward-compat:
      https://api.eia.gov/v2/seriesid/{series_id}?api_key=...&start=...&end=...
    Returns (df, info):
      - df: columns [Date, Value] or empty
      - info: diagnostics dict
    """
    info: Dict[str, object] = {
        "ok": False, "status": None, "url": "", "reason": "",
        "api_key_present": bool(api_key), "raw_error": ""
    }
    if not series_id:
        info["reason"] = "No series_id provided."
        return pd.DataFrame(), info

    base = f"https://api.eia.gov/v2/seriesid/{series_id}"
    params = {"start": start.isoformat(), "end": end.isoformat()}
    if api_key:
        params["api_key"] = api_key

    info["url"] = base + f"?start={params['start']}&end={params['end']}" + ("&api_key=***" if api_key else "")

    try:
        r = requests.get(base, params=params, timeout=30)
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

        resp = j.get("response", {})
        data = resp.get("data", [])
        if not isinstance(data, list) or len(data) == 0:
            info["reason"] = "Empty data"
            info["raw_error"] = str(j)[:500]
            return pd.DataFrame(), info

        df = pd.DataFrame(data)

        # Choose a numeric value column
        val_col = None
        for cand in ["value", "Value", "VALUE"]:
            if cand in df.columns:
                val_col = cand
                break
        if val_col is None:
            numeric_cols = [c for c in df.columns if c.lower() != "period" and pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                val_col = numeric_cols[0]
            else:
                info["reason"] = "No numeric value column found"
                info["raw_error"] = str(df.columns.tolist())
                return pd.DataFrame(), info

        if "period" not in df.columns:
            info["reason"] = "Missing 'period' in response"
            info["raw_error"] = str(df.columns.tolist())
            return pd.DataFrame(), info

        df["Date"] = df["period"].apply(_parse_eia_period_v2)
        df["Value"] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=["Date", "Value"]).sort_values("Date").reset_index(drop=True)

        # Local clip (the API also filters, but this is a guard)
        df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]
        info["ok"] = not df.empty
        if df.empty:
            info["reason"] = "No rows after filtering (check date window/series frequency)."
        return df[["Date", "Value"]], info

    except Exception as e:
        info["reason"] = f"Exception: {e}"
        return pd.DataFrame(), info

# ---------------- HELPERS: EIA v2 generic (Projections tab) ----------------
def ensure_api_key_in_url(url: str, api_key: Optional[str]) -> str:
    """
    If the URL is to api.eia.gov and lacks an api_key parameter, append the provided key.
    """
    try:
        u = urlparse(url)
        if "api.eia.gov" not in u.netloc.lower() or not api_key:
            return url  # not EIA, or we don't have a key to add
        qs = parse_qs(u.query)
        if "api_key" in qs and len(qs["api_key"]) > 0 and qs["api_key"][0]:
            return url  # already present
        qs["api_key"] = [api_key]
        new_qs = urlencode({k: v[0] if isinstance(v, list) else v for k, v in qs.items()})
        return urlunparse((u.scheme, u.netloc, u.path, u.params, new_qs, u.fragment))
    except Exception:
        return url

@st.cache_data(ttl=300, show_spinner=False)
def fetch_eia_v2_json_url(url: str) -> Dict[str, object]:
    """
    Fetch arbitrary EIA v2 JSON (e.g., /v2/steo/data?...).
    Returns dict with keys: ok, status, reason, raw (json), df (DataFrame or None)
    Tries to build a DataFrame with columns [Date, Value] by detecting 'period' and a numeric column.
    """
    out = {"ok": False, "status": None, "reason": "", "raw": None, "df": None}
    try:
        r = requests.get(url, timeout=30)
        out["status"] = r.status_code
        if r.status_code != 200:
            out["reason"] = f"HTTP {r.status_code}"
            try:
                out["raw"] = r.json()
            except Exception:
                out["raw"] = r.text[:500]
            return out
        j = r.json()
        out["raw"] = j

        # Try v2 standard shape
        data = None
        if isinstance(j, dict):
            resp = j.get("response")
            if isinstance(resp, dict) and "data" in resp and isinstance(resp["data"], list):
                data = resp["data"]
            elif "data" in j and isinstance(j["data"], list):
                data = j["data"]

        if not data:
            out["reason"] = "No data array found (expected response.data or data)."
            return out

        df = pd.DataFrame(data)
        if "period" not in df.columns:
            # Try common alternatives (rare)
            cand = [c for c in df.columns if c.lower() in ("period", "date", "periodname")]
            if not cand:
                out["reason"] = "Missing period column."
                return out
            df["period"] = df[cand[0]]

        # Choose a numeric value column (skip non-numeric and 'period')
        value_col = None
        numeric_cols = [c for c in df.columns if c.lower() != "period" and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            value_col = numeric_cols[0]
        else:
            # Try to coerce any candidate except 'period'
            for c in df.columns:
                if c.lower() == "period":
                    continue
                maybe = pd.to_numeric(df[c], errors="coerce")
                if maybe.notna().sum() > 0:
                    df[c] = maybe
                    value_col = c
                    break
        if not value_col:
            out["reason"] = "No numeric series column found."
            return out

        # Parse period and clean
        df["Date"] = df["period"].apply(_parse_eia_period_v2)
        df["Value"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["Date", "Value"]).sort_values("Date").reset_index(drop=True)
        out["df"] = df[["Date","Value"]]
        out["ok"] = not out["df"].empty
        if not out["ok"]:
            out["reason"] = "No rows after parsing."
        return out

    except Exception as e:
        out["reason"] = f"Exception: {e}"
        return out

# ---------------- HELPERS: Forward Curve ----------------
def mm_to_num(code: str) -> int:
    return MONTH_CODES.index(code) + 1

def build_contract_list(root: str, months: int, start_date: date) -> List[Tuple[str, str]]:
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
    options=["EIA (Daily)", "EIA Projections", "Forward Curve"],
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

# ====================== EIA (Daily) — HISTORICALS ======================
if tab_choice == "EIA (Daily)":
    st.title("EIA — Historical Prices (v2)")
    st.caption("Source: U.S. EIA Open Data API v2 (backward-compat seriesid)")

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
        series_id = st.text_input("EIA series ID (v1-style is fine here)", value="", placeholder="e.g., PET.RWTC.D").strip()
        series_label = series_id or "Custom"
    else:
        series_id = EIA_SERIES[series_mode]
        series_label = series_mode

    start_dt, end_dt = (THREE_YEARS_AGO, TODAY) if default_window else st.date_input(
        "Date range", (THREE_YEARS_AGO, TODAY)
    )

    with st.spinner("Fetching EIA data (v2)…"):
        df_eia, eia_info = fetch_eia_series_daily(series_id, EIA_KEY, start_dt, end_dt)
    has_data = not df_eia.empty

    with st.expander("EIA Diagnostics", expanded=False):
        st.write({
            "api_key_present": eia_info.get("api_key_present"),
            "request_status": eia_info.get("status"),
            "ok": eia_info.get("ok"),
            "reason": eia_info.get("reason"),
            "request_url": eia_info.get("url"),
            "raw_error": (eia_info.get("raw_error") or "")[:500]
        })

    st.subheader(f"{series_label}")
    st.caption(f"Window: {start_dt} → {end_dt}  |  Points: {len(df_eia)}")

    if has_data:
        fig = px.line(df_eia, x="Date", y="Value", labels={"Value": "Price", "Date": "Date"})
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Data", expanded=False):
            st.dataframe(df_eia, use_container_width=True)
            st.download_button(
                "Download CSV",
                df_eia.to_csv(index=False).encode("utf-8"),
                file_name=f"EIA_{series_id.replace('.','_')}_{start_dt}_{end_dt}.csv",
                mime="text/csv",
            )
    else:
        st.warning("No EIA rows returned. Check date window/series frequency — see Diagnostics above.")

# ====================== EIA PROJECTIONS (NEW) ======================
elif tab_choice == "EIA Projections":
    st.title("EIA — Projections (STEO/AEO via v2)")
    st.caption("Two easy ways: paste an **EIA v2 API URL** (from the EIA API Query Builder) or **upload a CSV** exported from EIA.")

    mode = st.radio("Choose input method", ["Paste EIA v2 API URL", "Upload EIA CSV"], horizontal=True)

    if mode == "Paste EIA v2 API URL":
        st.write("Tip: Build URLs at **api.eia.gov** → API → Query Builder, then paste the full URL here.")
        url_in = st.text_input(
            "EIA v2 URL",
            value="",
            placeholder="e.g., https://api.eia.gov/v2/steo/data?frequency=monthly&data[0]=value&facets[series][]=WTISPLC&start=2020-01&end=2027-12"
        ).strip()
        if url_in:
            safe_url = ensure_api_key_in_url(url_in, EIA_KEY)
            with st.spinner("Fetching projections…"):
                res = fetch_eia_v2_json_url(safe_url)
            with st.expander("Request Diagnostics", expanded=False):
                st.write({"requested_url": safe_url, "status": res.get("status"), "ok": res.get("ok"), "reason": res.get("reason")})
            if res.get("ok") and isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty:
                dfp = res["df"]
                st.subheader("Projection Series")
                fig = px.line(dfp, x="Date", y="Value", labels={"Value": "Value", "Date": "Date"})
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Data", expanded=False):
                    st.dataframe(dfp, use_container_width=True)
                    st.download_button(
                        "Download CSV",
                        dfp.to_csv(index=False).encode("utf-8"),
                        file_name="EIA_projections_from_url.csv",
                        mime="text/csv",
                    )
            else:
                st.info("No rows parsed. Check URL parameters (frequency/start/end/series) or try CSV upload.")

    else:  # Upload CSV
        upl = st.file_uploader("Upload EIA CSV (e.g., STEO/AEO export)", type=["csv"])
        if upl:
            try:
                df_raw = pd.read_csv(upl)
                # Try to find 'period' column
                period_col = None
                for cand in df_raw.columns:
                    if str(cand).lower() in ("period","date","periodname"):
                        period_col = cand
                        break
                if not period_col:
                    st.error("Could not find a 'period' or 'date' column in the CSV.")
                else:
                    # Choose a numeric column (first numeric not 'period')
                    value_col = None
                    for c in df_raw.columns:
                        if c == period_col:
                            continue
                        if pd.api.types.is_numeric_dtype(df_raw[c]):
                            value_col = c
                            break
                    if not value_col:
                        # Try coercion
                        for c in df_raw.columns:
                            if c == period_col:
                                continue
                            tmp = pd.to_numeric(df_raw[c], errors="coerce")
                            if tmp.notna().sum() > 0:
                                df_raw[c] = tmp
                                value_col = c
                                break
                    if not value_col:
                        st.error("No numeric value column found in the CSV.")
                    else:
                        df_raw["Date"] = df_raw[period_col].apply(_parse_eia_period_v2)
                        df_raw["Value"] = pd.to_numeric(df_raw[value_col], errors="coerce")
                        dfp = df_raw.dropna(subset=["Date","Value"]).sort_values("Date")
                        if dfp.empty:
                            st.info("No rows after parsing. Check the CSV columns and data.")
                        else:
                            st.subheader("Projection Series")
                            fig = px.line(dfp, x="Date", y="Value", labels={"Value":"Value","Date":"Date"})
                            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                            with st.expander("Data", expanded=False):
                                st.dataframe(dfp[["Date","Value"]], use_container_width=True)
                                st.download_button(
                                    "Download CSV",
                                    dfp[["Date","Value"]].to_csv(index=False).encode("utf-8"),
                                    file_name="EIA_projections_from_csv.csv",
                                    mime="text/csv",
                                )
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")

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
    else:
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
st.caption("© Market Analytics Dashboard — Informational only. Sources: EIA Open Data v2, Yahoo Finance.")
