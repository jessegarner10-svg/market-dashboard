import os
import io
import math
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import requests
import streamlit as st

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="Market Dashboard", page_icon="📊", layout="wide")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data(show_spinner=False)
def _get_secret_key(name: str) -> str:
    # Streamlit Cloud: st.secrets; local: env var fallback
    return st.secrets.get(name, os.environ.get(name, ""))

def _api_key_ok() -> bool:
    return bool(_get_secret_key("EIA_API_KEY"))

def _errbox(msg: str):
    st.error(msg, icon="⚠️")

# ----------------------------
# EIA API v2 helper
# Uses the official v2 "seriesid" backward-compat route so we can pass v1-style IDs
# Docs: EIA API v2 + seriesid path
# ----------------------------
EIA_V2_BASE = "https://api.eia.gov/v2"

@st.cache_data(show_spinner=False)
def eia_seriesid(series_id: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch a series via API v2 "seriesid" route using legacy v1 series_id.
    Returns tidy DataFrame with columns: period (datetime), value (float).
    """
    key = _get_secret_key("EIA_API_KEY")
    if not key:
        raise RuntimeError("Missing EIA_API_KEY")

    url = f"{EIA_V2_BASE}/seriesid/{series_id}"
    params = {"api_key": key}
    if start: params["start"] = start
    if end:   params["end"] = end

    r = requests.get(url, params=params, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # Try to surface EIA's helpful error payload, if present
        try:
            j = r.json()
            raise RuntimeError(f"EIA error: {j}") from e
        except Exception:
            raise

    j = r.json()
    # Expected shape: { "response": {"data":[{"period":"2024-10-31","value":...}, ...]}}
    data = j.get("response", {}).get("data", [])
    if not data:
        return pd.DataFrame(columns=["period", "value"])

    df = pd.DataFrame(data)
    # Normalize column names
    if "period" not in df.columns:  # some series return 'date' or similar, but seriesid should have 'period'
        # Best-effort fallback
        if "date" in df.columns:
            df.rename(columns={"date":"period"}, inplace=True)
        else:
            raise RuntimeError("Unexpected payload: no 'period' in EIA response")
    if "value" not in df.columns:
        # Value column can be named 'value' in seriesid route. If not, try 'data'
        val_col = next((c for c in df.columns if c.lower() == "value"), None)
        if val_col:
            df.rename(columns={val_col: "value"}, inplace=True)
        else:
            # try last numericish column
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                df.rename(columns={num_cols[0]: "value"}, inplace=True)
            else:
                raise RuntimeError("Unexpected payload: no numeric value column found")

    # Coerce types
    # Period may be YYYY-MM-DD for daily series
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("period")
    return df[["period", "value"]]

def last_n_years_dates(n_years=3):
    end = date.today()
    start = end - timedelta(days=int(365.25*n_years))
    return start, end

# ----------------------------
# Simple in-app projection model
# - Resample to monthly average
# - Decompose a linear trend (OLS) + simple month-of-year seasonality
# - Project M months ahead with uncertainty bands based on recent RMSE
# No external libs beyond numpy/pandas.
# ----------------------------
def monthly_averages_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    m = df.set_index("period")["value"].resample("MS").mean().dropna()
    return m.to_frame("value")

def _design_matrix(t_idx, month_idx):
    # X = [1, t, month dummies (11 dummies; month=1..12 with one dropped)]
    T = len(t_idx)
    X = np.ones((T, 1+1+11))
    X[:,1] = t_idx
    for i, m in enumerate(month_idx):
        # month m in 1..12; use 12 as the baseline (no column)
        if m < 12:
            X[i, 1+1+(m-1)] = 1.0
    return X

def fit_trend_seasonal(y: np.ndarray, dates: pd.DatetimeIndex):
    # t index starting at 0
    t = np.arange(len(y), dtype=float)
    months = dates.month.values
    X = _design_matrix(t, months)
    # OLS: beta = (X'X)^(-1)X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0] if hasattr(np.linalg, "lstsq") else np.linalg.solve(XtX, Xty)
    y_hat = X @ beta
    resid = y - y_hat
    rmse = float(np.sqrt(np.mean(resid**2))) if len(resid) else 0.0
    return beta, rmse

def forecast_trend_seasonal(beta, rmse, last_date: pd.Timestamp, horizon_months: int):
    # Build future dates (month start)
    fut_dates = pd.date_range((last_date + pd.offsets.MonthBegin(1)).normalize(), periods=horizon_months, freq="MS")
    # t continues
    start_t = 0
    # We need to know what t would be for each future month:
    # We'll reconstruct length from beta dimension
    # But easier: caller can pass last observed length. We'll just append range.
    # We'll return a function that evaluates X*beta.

    return fut_dates, rmse

def predict_from_beta(beta, hist_len: int, fut_dates: pd.DatetimeIndex):
    preds = []
    for i, d in enumerate(fut_dates):
        t = hist_len + i  # continue time index
        month = d.month
        # Build row X_t
        x = np.zeros(1+1+11)
        x[0] = 1.0
        x[1] = float(t)
        if month < 12:
            x[1+1+(month-1)] = 1.0
        pred = float(x @ beta)
        preds.append(pred)
    return np.array(preds)

def project_monthly(df_monthly: pd.DataFrame, horizon: int = 18, conf_z: float = 1.64):
    y = df_monthly["value"].values.astype(float)
    dates = df_monthly.index
    if len(y) < 18:
        raise RuntimeError("Not enough monthly data to project (need at least ~18 months).")
    beta, rmse = fit_trend_seasonal(y, dates)
    fut_dates = pd.date_range(dates[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    yhat = predict_from_beta(beta, len(y), fut_dates)
    # Simple i.i.d. uncertainty band
    se = rmse
    lower = yhat - conf_z*se
    upper = yhat + conf_z*se
    proj = pd.DataFrame({"value": yhat, "lower": lower, "upper": upper}, index=fut_dates)
    proj.index.name = "period"
    return proj

# ----------------------------
# Yahoo Finance (forward curves) helper
# ----------------------------
@st.cache_data(show_spinner=False)
def yf_curve(symbol: str, days: int = 1):
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance not available. Add it to requirements.txt") from e

    tkr = yf.Ticker(symbol)
    # Use last close
    hist = tkr.history(period=f"{days}d", interval="1d")
    if hist.empty:
        return pd.DataFrame()
    last_close = float(hist["Close"].iloc[-1])
    # Also get futures chain if available
    try:
        fut = tkr.futures  # yfinance experimental
    except Exception:
        fut = []

    return pd.DataFrame({"Contract": ["Spot/Last"], "Price": [last_close]})

# ----------------------------
# UI
# ----------------------------
st.title("📊 Market Dashboard")

tabs = st.tabs(["Forward Curves", "EIA Historicals", "Projections"])

# ----------------------------
# Tab 1: Forward Curves (simple spot proxy for now)
# ----------------------------
with tabs[0]:
    st.subheader("Forward Curves (quick view)")
    col1, col2 = st.columns([1,1])
    with col1:
        symbol = st.selectbox(
            "Instrument",
            ["CL=F  (WTI Crude)", "BZ=F  (Brent Crude)", "NG=F  (Henry Hub)"],
            index=0
        )
        ticker = symbol.split()[0]
        df_curve = yf_curve(ticker, days=10)
        if df_curve.empty:
            _errbox("No data from Yahoo Finance right now.")
        else:
            st.dataframe(df_curve, use_container_width=True)

    with col2:
        st.markdown(
            "Tip: this tab is a quick glance. For **history** and **projections**, use the tabs on the left."
        )

# ----------------------------
# Tab 2: EIA Historicals (daily, last 3 years)
# ----------------------------
with tabs[1]:
    st.subheader("EIA Historicals (Daily, last 3 years)")
    if not _api_key_ok():
        _errbox("Add your EIA_API_KEY in Streamlit secrets to enable EIA data.")
        st.stop()

    eia_map = {
        "WTI Cushing, OK spot (US$/bbl) — daily": "PET.RWTC.D",
        "Brent Europe spot (US$/bbl) — daily": "PET.RBRTE.D",
        "Henry Hub spot (US$/MMBtu) — daily": "NG.RNGWHHD.D",
    }
    choice = st.selectbox("Series", list(eia_map.keys()), index=0)

    start_dt, end_dt = last_n_years_dates(3)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    with st.spinner("Fetching EIA daily data…"):
        try:
            df = eia_seriesid(eia_map[choice], start=start_str, end=end_str)
        except Exception as e:
            _errbox(f"Failed to fetch EIA data.\n\n{e}")
            df = pd.DataFrame(columns=["period","value"])

    if df.empty:
        _errbox("No data returned for that selection and window.")
    else:
        st.line_chart(df.set_index("period")["value"])
        st.caption(f"Source: EIA Open Data API v2 (seriesid route). Last {len(df)} daily observations.")

        csv = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"{choice.replace(' ','_')}_daily_last3y.csv",
            mime="text/csv",
        )

# ----------------------------
# Tab 3: Projections (built in-app)
# ----------------------------
with tabs[2]:
    st.subheader("Projections — built in the app (no pasting)")

    if not _api_key_ok():
        _errbox("Add your EIA_API_KEY in Streamlit secrets to enable projections.")
        st.stop()

    proj_col, opt_col = st.columns([1,1])

    with proj_col:
        base_series = st.selectbox(
            "Pick base series to forecast",
            [
                "WTI Cushing, OK spot (US$/bbl) — daily",
                "Brent Europe spot (US$/bbl) — daily",
                "Henry Hub spot (US$/MMBtu) — daily",
            ],
            index=0
        )
        horizon = st.slider("Forecast horizon (months)", min_value=3, max_value=24, value=18, step=1)

        start_dt, end_dt = last_n_years_dates(5)  # use 5 years for stronger signal
        start_str, end_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

        with st.spinner("Fetching history & generating projection…"):
            try:
                hist = eia_seriesid(eia_map[base_series], start=start_str, end=end_str)
            except Exception as e:
                _errbox(f"Failed to fetch EIA data.\n\n{e}")
                hist = pd.DataFrame(columns=["period","value"])

            if hist.empty or len(hist) < 100:
                _errbox("Not enough history to produce a projection.")
            else:
                monthly = monthly_averages_from_daily(hist)
                try:
                    proj = project_monthly(monthly, horizon=horizon, conf_z=1.64)
                    # Combine for plotting
                    joined = pd.concat(
                        [monthly.rename(columns={"value":"history"}), proj.rename(columns={"value":"projection"})],
                        axis=0,
                        join="outer"
                    )

                    st.line_chart(joined[["history","projection"]])

                    # Show bands table (projection only)
                    st.dataframe(
                        proj.reset_index().rename(columns={"period":"Month", "value":"Projection", "lower":"Lower (≈90%)", "upper":"Upper (≈90%)"}),
                        use_container_width=True,
                        height=300
                    )

                    # Download
                    out = proj.reset_index()
                    out_csv = out.to_csv(index=False).encode()
                    st.download_button(
                        "Download projection CSV",
                        data=out_csv,
                        file_name=f"{base_series.split(' — ')[0].replace(' ','_')}_projection.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    _errbox(f"Projection failed: {e}")

    with opt_col:
        st.markdown("**Methodology**")
        st.write(
            "We convert daily prices to **monthly averages**, fit a simple **linear trend + month-of-year seasonality** "
            "model (ordinary least squares), and project forward. The gray band reflects recent RMSE (≈90% band). "
            "This is lightweight and fast—no external ML dependencies."
        )
        st.caption("Source data via EIA API v2 (seriesid route).")
