"""
Market Analytics Dashboard — EIA Daily + Forward Curve + Options (Collars)
---------------------------------------------------------------------------
- EIA (Daily): 3y history
- Forward Curve: CL/BZ/NG via free Yahoo futures contracts + forward strip builder
- Options (Collars):
    A) Upload settlements CSV (CME/ICE) to compute costless producer collars
    B) NEW: Model from Curve (Black-76) — compute costless collars from the curve using chosen vols

Setup on Streamlit Cloud:
1) App Settings → Secrets → TOML:
   EIA_API_KEY="your_real_key_here"
2) requirements.txt is already sufficient.

NOTE: Modeled options use assumed vol/time; for trading decisions, use licensed options data.
"""

import os
from io import StringIO
from math import log, sqrt, exp, erf
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

# Desk seasons
SEASONS = {
    "Summer (Apr–Oct)": [4,5,6,7,8,9,10],
    "Winter (Nov–Mar)": [11,12,1,2,3],
}

# ---------------- HELPERS: generic ----------------
def n_cdf(x: float) -> float:
    # Standard normal CDF using erf
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black76_call(F: float, K: float, vol: float, T: float, r: float=0.0) -> float:
    # F: futures (forward) price, K: strike, vol in decimal, T in years, r risk-free
    if F <= 0 or K <= 0 or vol <= 0 or T <= 0:
        # intrinsic approx (limit cases)
        return max(0.0, exp(-r*T)*(F - K))
    d1 = (log(F/K) + 0.5*vol*vol*T) / (vol*sqrt(T))
    d2 = d1 - vol*sqrt(T)
    return exp(-r*T) * (F * n_cdf(d1) - K * n_cdf(d2))

def black76_put(F: float, K: float, vol: float, T: float, r: float=0.0) -> float:
    if F <= 0 or K <= 0 or vol <= 0 or T <= 0:
        return max(0.0, exp(-r*T)*(K - F))
    d1 = (log(F/K) + 0.5*vol*vol*T) / (vol*sqrt(T))
    d2 = d1 - vol*sqrt(T)
    return exp(-r*T) * (K * n_cdf(-d2) - F * n_cdf(-d1))

def yearfrac_to(month_label: str, as_of: date) -> float:
    # month_label = "YYYY-MM" -> T to the 1st of that month
    try:
        dt = pd.to_datetime(month_label + "-01")
        days = max((dt.date() - as_of).days, 1)
        return days / 365.0
    except Exception:
        return 30/365.0

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

# ---------------- HELPERS: Options (Collars) — uploads ----------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: "".join(ch.lower() for ch in c if ch.isalnum()) for c in df.columns}
    return df.rename(columns=mapping)

CANDIDATE_COLS = {
    "expiry": ["expiry","expiration","maturity","contractmonth","contract","month","delivery","expdate","exp","settlementdate"],
    "strike": ["strike","strikeprice","k"],
    "call_settle": ["callsettle","callsettlement","calllast","callclose","callprice","csettle","settlecall"],
    "put_settle":  ["putsettle","putsettlement","putlast","putclose","putprice","psettle","settleput"],
    "underlying":  ["underlying","futures","future","futuressettle","futuresettlement","underlyingprice","underlyinglast","settle","futuresettl"],
}
def find_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in df.columns:
            return k
    return None

def autodetect_option_columns(raw: pd.DataFrame):
    df = normalize_cols(raw.copy())
    cols = {
        "expiry": find_col(df, CANDIDATE_COLS["expiry"]),
        "strike": find_col(df, CANDIDATE_COLS["strike"]),
        "call_settle": find_col(df, CANDIDATE_COLS["call_settle"]),
        "put_settle": find_col(df, CANDIDATE_COLS["put_settle"]),
        "underlying": find_col(df, CANDIDATE_COLS["underlying"]),
    }
    return cols, df

def parse_expiry_as_period(s: pd.Series) -> pd.Series:
    def _parse(x):
        try:
            d = pd.to_datetime(x, errors="coerce")
            if pd.isna(d):
                raise Exception
            return f"{d.year}-{d.month:02d}"
        except Exception:
            xs = str(x)
            if len(xs)==6 and xs.isdigit():
                return f"{xs[:4]}-{xs[4:]}"
            if len(xs)==7 and xs[4]=="-":
                return xs
            return np.nan
    return s.apply(_parse)

def estimate_atm(df_exp: pd.DataFrame, strike_col: str, call_col: str, put_col: str) -> Optional[float]:
    sub = df_exp[[strike_col, call_col, put_col]].dropna()
    if sub.empty:
        return None
    diff = (sub[call_col] - sub[put_col]).abs()
    i = diff.idxmin()
    try:
        return float(sub.loc[i, strike_col])
    except Exception:
        return None

def collar_candidates_for_expiry(
    df_exp: pd.DataFrame,
    strike_col: str,
    call_col: str,
    put_col: str,
    atm: Optional[float],
    otm_only: bool,
    min_width: float,
    tie_break: str = "closest_to_atm",
) -> Optional[Dict[str, float]]:
    df = df_exp[[strike_col, call_col, put_col]].dropna().copy()
    if df.empty:
        return None
    if otm_only:
        if atm is None:
            return None
        df_puts = df[df[strike_col] <= atm]
        df_calls = df[df[strike_col] >= atm]
    else:
        df_puts = df.copy()
        df_calls = df.copy()
    if df_puts.empty or df_calls.empty:
        return None

    df_puts  = df_puts.rename(columns={strike_col:"Kp", put_col:"P"}).loc[:, ["Kp","P"]]
    df_calls = df_calls.rename(columns={strike_col:"Kc", call_col:"C"}).loc[:, ["Kc","C"]]

    cand = df_puts.merge(df_calls, how="cross")
    cand["width"] = cand["Kc"] - cand["Kp"]
    cand = cand[cand["width"] >= float(min_width)]
    if cand.empty:
        return None
    cand["net"] = cand["C"] - cand["P"]
    cand["absnet"] = cand["net"].abs()

    if tie_break == "max_width":
        cand = cand.sort_values(["absnet","width"], ascending=[True, False])
    else:
        if atm is not None:
            cand["mid_from_atm"] = ((cand["Kc"] + cand["Kp"])/2 - atm).abs()
            cand = cand.sort_values(["absnet","mid_from_atm"])
        else:
            cand = cand.sort_values(["absnet"])

    best = cand.iloc[0]
    return {
        "K_put": float(best["Kp"]),
        "Put_settle": float(best["P"]),
        "K_call": float(best["Kc"]),
        "Call_settle": float(best["C"]),
        "Width": float(best["width"]),
        "Net_premium": float(best["net"]),
        "ATM": float(atm) if atm is not None else np.nan,
    }

# ---------------- SIDEBAR / NAV ----------------
st.sidebar.header("Navigation")
tab_choice = st.sidebar.radio(
    "Choose a module:",
    options=["EIA (Daily)", "Forward Curve", "Options (Collars)"],
    index=1,
)

# API key handling (for EIA)
EIA_KEY = st.secrets.get("EIA_API_KEY", os.getenv("EIA_API_KEY", ""))
with st.sidebar.expander("API Keys", expanded=False):
    st.caption("EIA key is read from Secrets (TOML) or environment.")
    if EIA_KEY:
        st.success("EIA key loaded.")
    else:
        st.warning("No EIA key found. Add TOML: EIA_API_KEY=\"...\"")

# ====================== EIA DAILY ======================
if tab_choice == "EIA (Daily)":
    st.title("EIA Daily — 3-Year History")
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
        st.warning("No EIA data returned. Check key/series/range.")

# =================== FORWARD CURVE ===================
elif tab_choice == "Forward Curve":
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

    # Simple curve-average strip (unchanged)
    st.markdown("---")
    st.subheader("Forward Curve — Strip Builder (Curve Avg)")
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
        yr = st.selectbox("Year", years_available, index=len(years_available)-1)
        selected_deliveries = [lab for lab, dttm in deliveries_dt.items() if dttm.year == yr]
        selected_deliveries = [lab for lab in deliveries_all if lab in set(selected_deliveries)]
    else:
        c1, c2 = st.columns(2)
        with c1:
            start_lab = st.selectbox("Start delivery", deliveries_all, index=0)
        with c2:
            end_lab = st.selectbox("End delivery", deliveries_all, index=min(2, len(deliveries_all)-1))
        i0, i1 = deliveries_all.index(start_lab), deliveries_all.index(end_lab)
        if i1 < i0: i0, i1 = i1, i0
        selected_deliveries = deliveries_all[i0:i1+1]

    if selected_deliveries:
        avg = avg_strip_from_curve(curve, selected_deliveries)
        included = curve[curve["Delivery"].isin(selected_deliveries)][["Contract","Delivery","Price"]]
        st.success(f"Strip: {selected_deliveries[0]} → {selected_deliveries[-1]} • Months: {len(selected_deliveries)} • **Average: {avg:,.4f}**")
        st.dataframe(included, use_container_width=True)
        st.download_button(
            "Download Strip CSV",
            included.to_csv(index=False).encode("utf-8"),
            file_name=f"{root}_forward_strip_{strip_type.replace(' ','_')}.csv",
            mime="text/csv",
        )
    else:
        st.info("Choose a strip to compute its average across curve months.")

# =================== OPTIONS (COLLARS) ===================
else:
    st.title("Options (Collars)")
    st.caption("Compute costless producer collars either from uploaded option settlements, or model from the live forward curve using Black-76.")

    tabA, tabB = st.tabs(["Upload Settlements (CME/ICE)", "Model from Curve (Black-76)"])

    # ---------- A) Upload settlements (existing workflow) ----------
    with tabA:
        upl = st.file_uploader("Upload CSV (option settles)", type=["csv"], key="upl_csv")
        if not upl:
            st.info("Upload a CSV with at least: expiry, strike, call_settle, put_settle. Underlying is optional.")
        else:
            raw = pd.read_csv(upl)
            autodet, df_norm = autodetect_option_columns(raw)

            st.markdown("**Column mapping (auto-detected; override if needed):**")
            cols = {}
            for key, candidates in CANDIDATE_COLS.items():
                detected = autodet.get(key)
                options = ["<none>"] + [c for c in df_norm.columns if c in candidates]
                idx = options.index(detected) if detected in options else 0
                sel = st.selectbox(key.replace("_"," ").title(), options=options, index=idx, key=f"map_{key}")
                cols[key] = None if sel == "<none>" else sel

            need = ["expiry","strike","call_settle","put_settle"]
            if any(cols[k] is None for k in need):
                st.error("Map all required fields: expiry, strike, call_settle, put_settle.")
            else:
                df = df_norm[[cols["expiry"], cols["strike"], cols["call_settle"], cols["put_settle"]] + ([cols["underlying"]] if cols["underlying"] else [])].copy()
                df.columns = ["expiry","strike","call_settle","put_settle"] + (["underlying"] if cols["underlying"] else [])
                df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
                df["call_settle"] = pd.to_numeric(df["call_settle"], errors="coerce")
                df["put_settle"]  = pd.to_numeric(df["put_settle"], errors="coerce")
                if "underlying" in df.columns:
                    df["underlying"] = pd.to_numeric(df["underlying"], errors="coerce")
                df["expiry_label"] = parse_expiry_as_period(df["expiry"])
                df = df.dropna(subset=["strike","call_settle","put_settle","expiry_label"]).copy()

                expiries = sorted(df["expiry_label"].unique().tolist())
                if not expiries:
                    st.error("Could not parse any expiries.")
                else:
                    st.markdown("### Controls")
                    c1,c2,c3 = st.columns(3)
                    with c1:
                        otm_only = st.checkbox("OTM only (put ≤ ATM, call ≥ ATM)", value=True)
                    with c2:
                        min_width = st.number_input("Min width (K_call − K_put)", value=0.0, min_value=0.0, step=0.5)
                    with c3:
                        tie_break = st.selectbox("Tie-break", ["closest_to_atm","max_width"], index=0)

                    # ATM
                    atm_auto = {e: estimate_atm(df[df["expiry_label"]==e], "strike","call_settle","put_settle") for e in expiries}
                    with st.expander("ATM overrides (optional)"):
                        atm_eff = {}
                        for e in expiries:
                            default_val = atm_auto[e] if atm_auto[e] is not None else np.nan
                            val = st.number_input(f"ATM for {e}", value=float(default_val) if not np.isnan(default_val) else 0.0, step=0.5, key=f"atm_{e}")
                            atm_eff[e] = (val if (not np.isnan(default_val) or val!=0.0) else None)

                    # Single or strip
                    mode = st.radio("Compute", ["Single expiry","Strip"], horizontal=True)
                    results_rows = []
                    if mode == "Single expiry":
                        e = st.selectbox("Expiry", expiries, index=0, key="exp_upl")
                        subset = df[df["expiry_label"]==e]
                        res = collar_candidates_for_expiry(subset, "strike","call_settle","put_settle",
                                                           atm=atm_eff.get(e), otm_only=otm_only, min_width=min_width, tie_break=tie_break)
                        if res:
                            results_rows.append({"Expiry": e, **res})
                        else:
                            st.warning("No valid collar pair with current constraints.")
                    else:
                        labels = expiries
                        c1, c2 = st.columns(2)
                        with c1:
                            start_lab = st.selectbox("Start expiry", labels, index=0, key="start_upl")
                        with c2:
                            end_lab = st.selectbox("End expiry", labels, index=min(2, len(labels)-1), key="end_upl")
                        i0, i1 = labels.index(start_lab), labels.index(end_lab)
                        if i1 < i0: i0, i1 = i1, i0
                        selected = labels[i0:i1+1]
                        for e in selected:
                            subset = df[df["expiry_label"]==e]
                            res = collar_candidates_for_expiry(subset, "strike","call_settle","put_settle",
                                                               atm=atm_eff.get(e), otm_only=otm_only, min_width=min_width, tie_break=tie_break)
                            if res:
                                results_rows.append({"Expiry": e, **res})
                    if results_rows:
                        out = pd.DataFrame(results_rows)
                        if len(out) > 1:
                            st.success(
                                f"Strip summary — Months: {len(out)} • "
                                f"Avg Floor: {out['K_put'].mean():,.2f} • "
                                f"Avg Ceiling: {out['K_call'].mean():,.2f} • "
                                f"Avg Width: {out['Width'].mean():,.2f} • "
                                f"Avg |Net|: {out['Net_premium'].abs().mean():,.4f}"
                            )
                        st.dataframe(out, use_container_width=True)
                        st.download_button(
                            "Download Collars CSV",
                            out.to_csv(index=False).encode("utf-8"),
                            file_name="collars_results.csv",
                            mime="text/csv",
                        )

    # ---------- B) Model from Curve (Black-76) ----------
    with tabB:
        st.markdown("Use the live forward curve as the **forward F** and choose a volatility/term to **model costless collars** (Black-76).")

        # Choose commodity & build curve (re-use code here for UX)
        commodity = st.selectbox(
            "Commodity",
            options=["WTI Crude (CL)", "Brent Crude (BZ)", "Henry Hub (NG)"],
            index=0, key="opt_curve_comm"
        )
        months_ahead = st.slider("Months ahead (from curve)", min_value=3, max_value=24, value=12, key="opt_curve_months")
        root = {"WTI Crude (CL)":"CL", "Brent Crude (BZ)":"BZ", "Henry Hub (NG)":"NG"}[commodity]
        with st.spinner("Building curve…"):
            curve = get_forward_curve(root, months_ahead, TODAY)

        if curve.empty:
            st.warning("No curve months returned; try fewer months.")
            st.stop()

        # Vol / rate controls
        defaults = {"WTI Crude (CL)":0.35, "Brent Crude (BZ)":0.30, "Henry Hub (NG)":0.45}
        vol_const = st.number_input("Vol (annual, %) — constant for all months", value=int(defaults[commodity]*100), min_value=1, max_value=300, step=1)/100.0
        allow_custom_vol = st.checkbox("Customize vol per month", value=False)
        r_percent = st.number_input("Risk-free rate (annual, %)", value=0.0, step=0.25)/100.0

        # Tick size (strike step) by commodity
        tick_default = {"WTI Crude (CL)":0.01, "Brent Crude (BZ)":0.01, "Henry Hub (NG)":0.001}.get(commodity, 0.01)
        strike_step = st.number_input("Strike step", value=float(tick_default), min_value=0.0001, step=float(tick_default))

        # Constraints
        c1, c2 = st.columns(2)
        with c1:
            otm_only = st.checkbox("OTM only (Put ≤ F, Call ≥ F)", value=True)
            min_width = st.number_input("Min width (K_call − K_put)", value=0.0, min_value=0.0, step=float(strike_step))
        with c2:
            search_span_pct = st.number_input("Search band around F (±%)", value=30, min_value=5, max_value=200, step=5)/100.0
            tie_break = st.selectbox("Tie-break", ["closest_to_atm","max_width"], index=0)

        # Build per-month vols
        curve = curve[["Delivery","Price","Delivery_dt"]].copy()
        curve["T"] = curve["Delivery"].apply(lambda m: yearfrac_to(m, TODAY))
        if allow_custom_vol:
            vols = []
            st.caption("Custom vol per month (%):")
            for m in curve["Delivery"]:
                v = st.number_input(f"{m}", value=int(vol_const*100), step=1, key=f"vol_{m}")/100.0
                vols.append(v)
            curve["vol"] = vols
        else:
            curve["vol"] = vol_const

        # Solve costless collar per month
        results = []
        for _, row in curve.iterrows():
            F = float(row["Price"])
            T = max(float(row["T"]), 1.0/365.0)
            vol = max(float(row["vol"]), 0.0001)

            # define strike search grids
            span = search_span_pct
            k_min_put  = max(0.01, F*(1.0 - span))
            k_max_put  = F  # OTM put cap at F if OTM-only
            k_min_call = F  # OTM call floor at F if OTM-only
            k_max_call = F*(1.0 + span)

            if not otm_only:
                k_max_put  = F*(1.0 + span)
                k_min_call = max(0.01, F*(1.0 - span))

            # discretize
            Ks_put  = np.arange(k_min_put,  k_max_put + 1e-9,  strike_step)
            Ks_call = np.arange(k_min_call, k_max_call + 1e-9,  strike_step)
            if len(Ks_put)==0 or len(Ks_call)==0:
                continue

            # brute-force search for costless
            best = None
            best_key = None
            for Kp in Ks_put:
                P = black76_put(F, Kp, vol, T, r_percent)
                # enforce min width by starting call grid above Kp + min_width
                start_call = max(Ks_call[0], Kp + float(min_width))
                # find index to start
                j0 = int(max(0, np.searchsorted(Ks_call, start_call)))
                for Kc in Ks_call[j0:]:
                    C = black76_call(F, Kc, vol, T, r_percent)
                    net = C - P  # producer: short call, long put; costless -> ~0
                    absnet = abs(net)
                    width = Kc - Kp
                    if best is None:
                        # penalty for tie-break
                        if tie_break == "closest_to_atm":
                            mid_from_atm = abs(((Kc+Kp)/2.0) - F)
                            best_key = (absnet, mid_from_atm)
                        else:
                            best_key = (absnet, -width)  # prefer wider
                        best = (Kp,P,Kc,C,net,width,best_key)
                    else:
                        if tie_break == "closest_to_atm":
                            key = (absnet, abs(((Kc+Kp)/2.0) - F))
                        else:
                            key = (absnet, -width)
                        if key < best_key:
                            best = (Kp,P,Kc,C,net,width,key)
                            best_key = key

            if best is None:
                continue
            Kp,P,Kc,C,net,width,_ = best
            results.append({
                "Delivery": row["Delivery"],
                "F": F, "T(yrs)": T, "Vol": vol,
                "K_put": Kp, "Put_model": P,
                "K_call": Kc, "Call_model": C,
                "Width": width, "Net(C-P)": net
            })

        if not results:
            st.warning("No costless pairs found with current settings. Loosen the search band or reduce min width.")
        else:
            out = pd.DataFrame(results)
            st.success(
                f"Computed collars for {len(out)} month(s). "
                f"Avg Floor: {out['K_put'].mean():,.3f} • Avg Ceiling: {out['K_call'].mean():,.3f} • "
                f"Avg Width: {out['Width'].mean():,.3f} • Avg |Net|: {out['Net(C-P)'].abs().mean():,.5f}"
            )
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download Modeled Collars CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name=f"{root}_modeled_collars_black76.csv",
                mime="text/csv",
            )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© Market Analytics Dashboard — Informational only. Sources: EIA Open Data, Yahoo Finance; modeled options via Black-76.")
