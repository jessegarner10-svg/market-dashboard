"""
Market Analytics Dashboard — EIA Daily + Forward Curve + Options (Collars)
---------------------------------------------------------------------------
- EIA (Daily): 3y history + (daily strip builder remains available if needed)
- Forward Curve: CL/BZ/NG via free Yahoo futures contracts + forward strip builder
- NEW: Options (Collars): Upload exchange option settlements CSV, compute costless producer collars
  * per-expiry or across strips (1M/3M/Seasonal/Calendar/Custom)
  * OTM-only toggle, minimum width, ATM auto/manual, download results

Setup on Streamlit Cloud:
1) App menu → (your workspace) → this app → Settings → Secrets → add (TOML):
   EIA_API_KEY="your_real_key_here"
2) requirements.txt is already sufficient.

NOTE: Data is for informational use only. For production, connect licensed feeds (CME/ICE/Nasdaq Data Link).
"""

import os
from io import StringIO
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

# ---------------- HELPERS: EIA ----------------
def parse_eia_period(p: str) -> pd.Timestamp:
    """Parse EIA period strings: YYYY, YYYYMM, YYYYMMDD, or ISO date."""
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
    """Fetch EIA series and clip to [start, end]. Returns DataFrame: Date, Value (daily if available)."""
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
    """Build (contract_code, delivery_label) pairs, e.g. ('CLZ24','2024-12') for next `months` months."""
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
    """Try multiple Yahoo variants for a futures contract and return last Close series if found."""
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
    """Arithmetic average of contract prices for given 'YYYY-MM' labels present in curve_df."""
    if curve_df.empty:
        return None
    sub = curve_df[curve_df["Delivery"].isin(deliveries)]
    if sub.empty:
        return None
    return float(sub["Price"].mean())

# ---------------- HELPERS: Options (Collars) ----------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lower/strip non-alnum for robust column matching."""
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

def autodetect_option_columns(raw: pd.DataFrame) -> Dict[str, Optional[str]]:
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
    """Try to coerce expiry to a monthly period label YYYY-MM."""
    def _parse(x):
        # Try datetime
        try:
            d = pd.to_datetime(x, errors="coerce")
            if pd.isna(d):
                raise Exception
            return f"{d.year}-{d.month:02d}"
        except Exception:
            # Try strings like 2025M01, 202501, CLZ25 not supported here
            xs = str(x)
            # 202501
            if len(xs)==6 and xs.isdigit():
                return f"{xs[:4]}-{xs[4:]}"
            # 2025-01
            if len(xs)==7 and xs[4]=="-":
                return xs
            return np.nan
    return s.apply(_parse)

def estimate_atm(df_exp: pd.DataFrame, strike_col: str, call_col: str, put_col: str) -> Optional[float]:
    """Estimate ATM strike by minimizing |call_settle - put_settle| across strikes."""
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
    tie_break: str = "closest_to_atm",  # or "max_width"
) -> Optional[Dict[str, float]]:
    """
    Find a near costless collar per expiry:
    - choose (K_put, K_call) s.t. |CallSettle - PutSettle| minimized
    - apply OTM-only filter and min width (K_call - K_put >= min_width)
    - tie-break by closest_to_atm or max_width
    Returns dict with strikes, settles, net, width, atm.
    """
    df = df_exp[[strike_col, call_col, put_col]].dropna().copy()
    if df.empty:
        return None

    # OTM filters need ATM
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

    # Cartesian join (efficient enough for typical strike counts)
    df_puts  = df_puts.rename(columns={strike_col:"Kp", put_col:"P"}).loc[:, ["Kp","P"]]
    df_calls = df_calls.rename(columns={strike_col:"Kc", call_col:"C"}).loc[:, ["Kc","C"]]
    cand = df_puts.merge(df_calls, how="cross")

    # width & costless metrics
    cand["width"] = cand["Kc"] - cand["Kp"]
    cand = cand[cand["width"] >= float(min_width)]
    if cand.empty:
        return None

    cand["net"] = cand["C"] - cand["P"]  # producer sells call, buys put; costless -> net ~ 0
    cand["absnet"] = cand["net"].abs()

    # Rank by absnet, then tie-break
    if tie_break == "max_width":
        cand = cand.sort_values(["absnet","width"], ascending=[True, False])
    else:
        # closest_to_atm: prefer pairs whose mid-strike near ATM
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

def build_strip_selection(delivery_labels: List[str], deliveries_dt: Dict[str, pd.Timestamp],
                          strip_type: str, years_available: List[int]) -> List[str]:
    """Return a subset of delivery labels for the chosen strip type on the curve page."""
    # This is reused conceptually for month lists in options too (by expiry labels).
    selected = []

    if strip_type == "1 Month":
        return [delivery_labels[0]]

    if strip_type == "3 Month":
        return delivery_labels[:3]

    if strip_type == "Seasonal":
        # choose latest available year
        base_year = years_available[-1] if years_available else TODAY.year
        def is_summer(dt: date, y: int) -> bool:
            return (dt.year == y) and (dt.month in [4,5,6,7,8,9,10])
        def is_winter(dt: date, y: int) -> bool:
            return ((dt.year == y-1 and dt.month in [11,12]) or (dt.year == y and dt.month in [1,2,3]))
        # default Summer
        for lab, dttm in deliveries_dt.items():
            d = dttm.date()
            if is_summer(d, base_year):
                selected.append(lab)
        # keep curve order
        return [lab for lab in delivery_labels if lab in set(selected)]

    if strip_type == "Calendar Year":
        yr = years_available[-1] if years_available else TODAY.year
        selected = [lab for lab, dttm in deliveries_dt.items() if dttm.year == yr]
        return [lab for lab in delivery_labels if lab in set(selected)]

    # Custom handled in UI; default to all
    return delivery_labels

# ---------------- SIDEBAR / NAV ----------------
st.sidebar.header("Navigation")
tab_choice = st.sidebar.radio(
    "Choose a module:",
    options=["EIA (Daily)", "Forward Curve", "Options (Collars)"],
    index=1,  # default to Forward Curve
)

# API key handling (for EIA)
EIA_KEY = st.secrets.get("EIA_API_KEY", os.getenv("EIA_API_KEY", ""))
with st.sidebar.expander("API Keys", expanded=False):
    st.caption("EIA key is read from your Cloud 'Secrets' (TOML) or environment.")
    if EIA_KEY:
        st.success("EIA key loaded.")
    else:
        st.warning("No EIA key found. Add it in Settings → Secrets as TOML: EIA_API_KEY=\"...\"")

# ====================== EIA DAILY (unchanged) ======================
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

# =================== FORWARD CURVE (with existing strip builder) ===================
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

    # Simple forward strip (curve-average) still available here
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
    if strip_type in ["1 Month", "3 Month", "Seasonal", "Calendar Year"]:
        selected_deliveries = build_strip_selection(deliveries_all, deliveries_dt, strip_type, years_available)
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

# =================== OPTIONS (COLLARS) — NEW ===================
else:
    st.title("Options (Collars) — Costless Producer Collars")
    st.caption("Upload exchange option settlements (CSV). The app finds put/call strike pairs with near-zero net premium per expiry, with OTM-only and min-width constraints.")

    # ---- Upload & autodetect ----
    upl = st.file_uploader("Upload CSV (CME/ICE option settles)", type=["csv"])
    if not upl:
        st.info("Upload a CSV with at least: expiry, strike, call_settle, put_settle. Underlying/futures settle optional.")
        st.stop()

    raw = pd.read_csv(upl)
    autodet, df_norm = autodetect_option_columns(raw)

    st.markdown("**Column mapping (auto-detected; override if needed):**")
    cols = {}
    for key, candidates in CANDIDATE_COLS.items():
        detected = autodet[0].get(key) if isinstance(autodet, tuple) else autodet.get(key)
        # candidates available in normalized df
        options = [c for c in df_norm.columns if c in candidates]
        sel = st.selectbox(
            key.replace("_"," ").title(),
            options=["<none>"] + options,
            index=(options.index(detected) + 1) if (detected in options) else 0
        )
        cols[key] = None if sel == "<none>" else sel

    # Rebuild a working frame with normalized columns
    need = ["expiry","strike","call_settle","put_settle"]
    if any(cols[k] is None for k in need):
        st.error("Missing a required column mapping (expiry/strike/call_settle/put_settle). Please map all required fields.")
        st.stop()

    df = df_norm[[cols["expiry"], cols["strike"], cols["call_settle"], cols["put_settle"]] + ([cols["underlying"]] if cols["underlying"] else [])].copy()
    df.columns = ["expiry","strike","call_settle","put_settle"] + (["underlying"] if cols["underlying"] else [])

    # Clean types
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["call_settle"] = pd.to_numeric(df["call_settle"], errors="coerce")
    df["put_settle"]  = pd.to_numeric(df["put_settle"], errors="coerce")
    if "underlying" in df.columns:
        df["underlying"] = pd.to_numeric(df["underlying"], errors="coerce")

    # Parse expiry label as YYYY-MM for grouping
    expiry_label = parse_expiry_as_period(df["expiry"])
    df["expiry_label"] = expiry_label
    df = df.dropna(subset=["strike","call_settle","put_settle","expiry_label"]).copy()

    expiries = sorted(df["expiry_label"].unique().tolist())
    if not expiries:
        st.error("Could not parse any expiries. Check the 'expiry' column mapping.")
        st.stop()

    st.markdown("### Controls")
    c1,c2,c3 = st.columns(3)
    with c1:
        otm_only = st.checkbox("OTM only (put ≤ ATM, call ≥ ATM)", value=True)
    with c2:
        min_width = st.number_input("Min width (K_call − K_put)", value=0.0, min_value=0.0, step=0.5)
    with c3:
        tie_break = st.selectbox("Tie-break", ["closest_to_atm","max_width"], index=0)

    st.markdown("### Select Expiries / Strips")
    mode = st.radio("Mode", ["Single expiry","Strip (1M/3M/Seasonal/Calendar/Custom)"], horizontal=True)

    # Build ATM dictionary (auto-estimate per expiry)
    atm_auto: Dict[str, Optional[float]] = {}
    for e in expiries:
        atm_auto[e] = estimate_atm(df[df["expiry_label"]==e], "strike", "call_settle", "put_settle")

    # Allow manual ATM overrides
    with st.expander("ATM overrides (optional)"):
        manual = {}
        for e in expiries:
            default_val = atm_auto[e] if atm_auto[e] is not None else np.nan
            val = st.number_input(f"ATM for {e}", value=float(default_val) if not np.isnan(default_val) else 0.0, step=0.5, key=f"atm_{e}")
            manual[e] = None if (np.isnan(default_val) and val==0.0) else val
    # Effective ATM: manual if provided else auto
    atm_eff = {e: (manual[e] if manual[e] is not None else atm_auto[e]) for e in expiries}

    results_rows = []

    if mode == "Single expiry":
        exp_sel = st.selectbox("Expiry", expiries, index=0)
        subset = df[df["expiry_label"]==exp_sel]
        res = collar_candidates_for_expiry(
            subset, "strike","call_settle","put_settle",
            atm=atm_eff.get(exp_sel), otm_only=otm_only, min_width=min_width, tie_break=tie_break
        )
        if res is None:
            st.warning("No valid collar pair found with current constraints for this expiry.")
        else:
            row = {"Expiry": exp_sel, **res}
            results_rows.append(row)

    else:
        # Strip selection across expiry labels
        exp_dt = pd.to_datetime(pd.Series(expiries).astype(str) + "-01")
        deliveries_dt = {e: exp_dt.iloc[i] for i,e in enumerate(expiries)}
        years_available = sorted({d.year for d in deliveries_dt.values()})

        strip_type = st.selectbox("Strip type", ["1 Month","3 Month","Seasonal","Calendar Year","Custom Month Range"], index=1)

        if strip_type in ["1 Month","3 Month","Seasonal","Calendar Year"]:
            # Reuse builders (semantics by expiry-month)
            labels_in_order = expiries
            selected = []
            if strip_type == "1 Month":
                selected = [labels_in_order[0]]
            elif strip_type == "3 Month":
                selected = labels_in_order[:3]
            elif strip_type == "Seasonal":
                # choose latest year by default
                base_year = years_available[-1] if years_available else TODAY.year
                def is_summer(dt: date, y: int) -> bool:
                    return (dt.year == y) and (dt.month in [4,5,6,7,8,9,10])
                def is_winter(dt: date, y: int) -> bool:
                    return ((dt.year == y-1 and dt.month in [11,12]) or (dt.year == y and dt.month in [1,2,3]))
                season = st.radio("Season", ["Summer (Apr–Oct)","Winter (Nov–Mar)"], horizontal=True)
                for e, dttm in deliveries_dt.items():
                    d = dttm.date()
                    if (season.startswith("Summer") and is_summer(d, base_year)) or \
                       (season.startswith("Winter") and is_winter(d, base_year)):
                        selected.append(e)
                selected = [e for e in labels_in_order if e in set(selected)]
            else:
                yr = st.selectbox("Calendar year", years_available, index=len(years_available)-1)
                selected = [e for e,dttm in deliveries_dt.items() if dttm.year == yr]
                selected = [e for e in labels_in_order if e in set(selected)]
        else:
            # Custom Month Range
            labels_in_order = expiries
            c1, c2 = st.columns(2)
            with c1:
                start_lab = st.selectbox("Start expiry", labels_in_order, index=0)
            with c2:
                end_lab = st.selectbox("End expiry", labels_in_order, index=min(2, len(labels_in_order)-1))
            i0, i1 = labels_in_order.index(start_lab), labels_in_order.index(end_lab)
            if i1 < i0: i0, i1 = i1, i0
            selected = labels_in_order[i0:i1+1]

        if not selected:
            st.warning("No expiries matched this strip selection.")
        else:
            for e in selected:
                subset = df[df["expiry_label"]==e]
                res = collar_candidates_for_expiry(
                    subset, "strike","call_settle","put_settle",
                    atm=atm_eff.get(e), otm_only=otm_only, min_width=min_width, tie_break=tie_break
                )
                if res:
                    row = {"Expiry": e, **res}
                    results_rows.append(row)

    if results_rows:
        out = pd.DataFrame(results_rows)
        # Summary for strips
        if len(out) > 1:
            avg_floor = out["K_put"].mean()
            avg_ceiling = out["K_call"].mean()
            avg_width = out["Width"].mean()
            avg_absnet = out["Net_premium"].abs().mean()
            st.success(
                f"Strip summary — Months: {len(out)} • "
                f"Avg Floor (K_put): {avg_floor:,.2f} • Avg Ceiling (K_call): {avg_ceiling:,.2f} • "
                f"Avg Width: {avg_width:,.2f} • Avg |Net|: {avg_absnet:,.4f}"
            )
        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download Collars CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="collars_results.csv",
            mime="text/csv",
        )
    else:
        st.info("No collar results to show yet — adjust constraints or mapping, or try another expiry/strip.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© Market Analytics Dashboard — Informational only. Sources: EIA Open Data, Yahoo Finance, CME/ICE uploads.")
