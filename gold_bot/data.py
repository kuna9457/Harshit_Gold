"""Data ingestion with an on-disk cache. Every loader returns raw, un-lagged data on its native dates."""
import io
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OVERRIDES = ROOT / "data" / "overrides"
CACHE.mkdir(parents=True, exist_ok=True)
OVERRIDES.mkdir(parents=True, exist_ok=True)

MAX_AGE_HOURS = 12
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}


def _fresh(path, max_age_hours):
    return path.exists() and (max_age_hours is None or time.time() - path.stat().st_mtime < max_age_hours * 3600)


def _cached(name, fetch, refresh=False, max_age_hours=MAX_AGE_HOURS):
    """Return fetch() cached as CSV. A failed download falls back to the last good copy."""
    path = CACHE / f"{name}.csv"
    if not refresh and _fresh(path, max_age_hours):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    try:
        df = fetch()
        if df is None or len(df) == 0:
            raise ValueError("empty response")
        df.to_csv(path)
        return df
    except Exception:
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        raise


# ---------------------------------------------------------
# Market data (Yahoo Finance)
# ---------------------------------------------------------
def _yahoo(symbol):
    df = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=False)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df[~df.index.duplicated(keep="last")].dropna(subset=["Close"])


def yahoo(symbol, refresh=False):
    name = "yf_" + symbol.replace("=", "_").replace("^", "").replace(".", "_").replace("-", "_")
    return _cached(name, lambda: _yahoo(symbol), refresh)


def load_gold(refresh=False):
    df = yahoo("GC=F", refresh).copy()
    df = df[df["Close"] > 0]
    df["Open"] = df["Open"].where(df["Open"] > 0, df["Close"].shift(1)).fillna(df["Close"])
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].where(lambda x: x > 0).min(axis=1)
    df["Volume"] = df["Volume"].replace(0, np.nan).ffill()
    return df


def _money_flow(df):
    """Daily money-flow volume and dollar volume (Chaikin). Sums of these give net flow over a window."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    dv = df["Close"] * df["Volume"]
    return pd.DataFrame({"mfv": (clv * dv).fillna(0.0), "dv": dv})


# ---------------------------------------------------------
# Official macro data
# ---------------------------------------------------------
def _treasury_real_year(year):
    url = (f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
           f"{year}/all?type=daily_treasury_real_yield_curve&field_tdr_date_value={year}&page&_format=csv")
    r = requests.get(url, headers=_UA, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df.set_index("Date")[["10 YR"]].rename(columns={"10 YR": "value"}).sort_index()


def load_real_yield(refresh=False):
    this_year = date.today().year
    parts = []
    for y in range(2003, this_year + 1):
        # Past years never change, so they are cached for good.
        age = MAX_AGE_HOURS if y == this_year else None
        try:
            parts.append(_cached(f"treasury_real_{y}", lambda y=y: _treasury_real_year(y), refresh and y == this_year, age))
        except Exception:
            continue
    if not parts:
        raise RuntimeError("Treasury real yield download failed")
    return pd.concat(parts)["value"].dropna().sort_index()


def _nyfed(kind, rate):
    url = (f"https://markets.newyorkfed.org/api/rates/{kind}/{rate}/search.csv"
           f"?startDate=2000-01-01&endDate={date.today():%Y-%m-%d}")
    r = requests.get(url, headers=_UA, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Effective Date"], format="%m/%d/%Y")
    return df.set_index("Date")[["Rate (%)"]].rename(columns={"Rate (%)": "value"}).sort_index()


def _bls_cpi():
    rows = []
    this_year = date.today().year
    for start in range(1999, this_year + 1, 10):
        body = {"seriesid": ["CUSR0000SA0"], "startyear": str(start), "endyear": str(min(start + 9, this_year))}
        js = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/", json=body, headers=_UA, timeout=60).json()
        if js.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS: {js.get('message')}")
        for d in js["Results"]["series"][0]["data"]:
            if d["period"].startswith("M") and d["period"] != "M13":
                rows.append((pd.Timestamp(int(d["year"]), int(d["period"][1:]), 1), pd.to_numeric(d["value"], errors="coerce")))
    df = pd.DataFrame(rows, columns=["Date", "value"]).dropna().drop_duplicates("Date")
    if df["Date"].min().year > 2000:
        raise RuntimeError("BLS returned a truncated CPI history")
    return df.set_index("Date").sort_index()


def _override(name):
    """Optional user file data/overrides/<name>.csv with columns date,value."""
    path = OVERRIDES / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                     index=pd.to_datetime(df["date"]), name="value").dropna().sort_index()


# ---------------------------------------------------------
# Everything at once
# ---------------------------------------------------------
def load_all(refresh=False):
    """Returns (gold OHLCV, {input key: raw Series or money-flow DataFrame}, {input key: status text})."""
    gold = load_gold(refresh)
    raw, status = {"gold": gold}, {"gold": "ok"}

    def add(key, loader):
        try:
            raw[key] = loader()
            status[key] = "ok"
        except Exception as e:
            status[key] = f"FAILED: {type(e).__name__}: {e}"

    add("dxy", lambda: yahoo("DX-Y.NYB", refresh)["Close"])
    add("usdinr", lambda: yahoo("USDINR=X", refresh)["Close"])
    add("tbill", lambda: yahoo("^IRX", refresh)["Close"])
    add("real_yield", lambda: load_real_yield(refresh))
    add("effr", lambda: _cached("nyfed_effr", lambda: _nyfed("unsecured", "effr"), refresh)["value"])
    add("sofr", lambda: _cached("nyfed_sofr", lambda: _nyfed("secured", "sofr"), refresh)["value"])
    # CPI changes once a month and the BLS open API allows 25 calls a day, so refresh it weekly at most.
    add("cpi", lambda: _cached("bls_cpi", _bls_cpi, False, 24 * 7)["value"])

    def etf_flow(name, symbols):
        user = _override(name)
        if user is not None:
            status[name] = "ok (your file in data/overrides)"
            return user
        return sum(_money_flow(yahoo(s, refresh)) for s in symbols).dropna()

    for name, symbols in (("gold_etf", ["GLD"]), ("bond_flow", ["TLT", "IEF"])):
        try:
            status.setdefault(name, "ok")
            raw[name] = etf_flow(name, symbols)
        except Exception as e:
            status[name] = f"FAILED: {type(e).__name__}: {e}"
    return gold, raw, status
