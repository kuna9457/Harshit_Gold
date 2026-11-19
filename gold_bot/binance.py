"""Binance public market data for gold: historical klines (cached on disk) and the live price.

No account or API key is needed. Times are UTC, and each bar is indexed by its OPEN time.
Only closed bars are returned, so a decision never uses a bar that is still forming.
"""
import time
from pathlib import Path

import pandas as pd
import requests

from .data import ROOT

STORE = ROOT / "data" / "binance"
STORE.mkdir(parents=True, exist_ok=True)

MARKETS = {
    "PAXGUSDT": dict(label="PAXGUSDT spot (tokenised gold, 1 token = 1 oz)", base="https://api.binance.com",
                     klines="/api/v3/klines", ticker="/api/v3/ticker/price", book="/api/v3/ticker/bookTicker",
                     first="2020-08-28"),
    "XAUUSDT": dict(label="XAUUSDT perpetual futures (gold, Binance USDT-M)", base="https://fapi.binance.com",
                    klines="/fapi/v1/klines", ticker="/fapi/v1/ticker/price", book="/fapi/v1/ticker/bookTicker",
                    first="2025-12-11"),
}
BASE_INTERVAL = "15m"  # every timeframe is built from these bars, so they all agree with each other
_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
_COLS = ["open_time", "Open", "High", "Low", "Close", "Volume", "close_time", "quote_volume", "trades",
         "taker_base", "taker_quote", "ignore"]


def _get(url, params, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 451:
                raise RuntimeError("Binance refuses requests from this location (HTTP 451). "
                                   "US-hosted servers are blocked; run it locally or on a non-US server.")
            if r.status_code in (418, 429):  # rate limited: wait as told
                time.sleep(int(r.headers.get("Retry-After", 5)))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(1 + k)
    raise RuntimeError("Binance request failed")


def _download(symbol, interval, start_ms, end_ms=None, progress=None):
    m, step = MARKETS[symbol], _MS[interval]
    rows, cur = [], start_ms
    end_ms = end_ms or int(time.time() * 1000)
    total = max(end_ms - start_ms, 1)
    while cur < end_ms:
        batch = _get(m["base"] + m["klines"], dict(symbol=symbol, interval=interval, startTime=cur, limit=1000))
        if not batch:
            break
        rows += batch
        cur = batch[-1][0] + step
        if progress:
            progress(min(1.0, (cur - start_ms) / total))
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "close_time"])
    df = pd.DataFrame(rows, columns=_COLS)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["Open", "High", "Low", "Close", "Volume", "close_time"]].astype(float)
    return df[~df.index.duplicated(keep="last")]


_MEM = {}  # path -> frame already read, so a live refresh does not re-read the whole file


def klines(symbol="PAXGUSDT", interval=BASE_INTERVAL, update=True, progress=None):
    """All closed bars from the symbol's first listing to now. The first call downloads everything
    (a few minutes for PAXGUSDT 15m); later calls only fetch and append the bars added since."""
    path = STORE / f"{symbol}_{interval}.csv"
    old = _MEM.get(path)
    if old is None and path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True)
    df = old
    if update or old is None:
        start = int(pd.Timestamp(MARKETS[symbol]["first"]).timestamp() * 1000) if old is None or old.empty             else int(old.index[-1].timestamp() * 1000) + _MS[interval]
        try:
            new = _download(symbol, interval, start, progress=progress)
            new = new[new["close_time"] < time.time() * 1000]  # only closed bars are stored
            if old is not None:
                new = new[new.index > old.index[-1]]
            if len(new):
                new.to_csv(path, mode="a" if old is not None else "w", header=old is None)
                df = new if old is None else pd.concat([old, new])
        except Exception:
            if old is None:
                raise
    _MEM[path] = df
    return df[["Open", "High", "Low", "Close", "Volume"]]


def resample(df, rule):
    """15m bars -> 1h / 4h / 1d (UTC days). Bars stay indexed by their open time."""
    if rule == BASE_INTERVAL:
        return df
    pd_rule = {"1h": "1h", "4h": "4h", "1d": "1D"}[rule]
    out = df.resample(pd_rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    out = out.dropna(subset=["Close"])
    # The last bar is only complete when every 15m bar inside it has closed.
    step = pd.Timedelta(minutes=minutes(rule))
    last_needed = df.index[-1] + pd.Timedelta(minutes=minutes(BASE_INTERVAL))
    if len(out) and out.index[-1] + step > last_needed:
        out = out.iloc[:-1]
    return out


def minutes(interval):
    return _MS[interval] // 60_000


def live_price(symbol="PAXGUSDT"):
    """Last trade, best bid and best ask right now."""
    m = MARKETS[symbol]
    last = _get(m["base"] + m["ticker"], dict(symbol=symbol))
    book = _get(m["base"] + m["book"], dict(symbol=symbol))
    return dict(price=float(last["price"]), bid=float(book["bidPrice"]), ask=float(book["askPrice"]),
                time=pd.Timestamp.now(tz="UTC").tz_localize(None))
