"""Live paper trading on the Binance gold feed.

A session is just: strategy settings + market + start time, saved in data/live/session.json.
Every refresh pulls the newest closed bars, rebuilds the multi-horizon signal and runs the SAME backtest engine
from the start time to now. The open position is the one the engine is still holding at the last bar.
Nothing is sent to an exchange.
"""
import json
from datetime import datetime, timezone

import pandas as pd

from . import binance, mtf, ticks
from .config import DEFAULT_PARAMS, TIMEFRAMES
from .data import ROOT

LIVE = ROOT / "data" / "live"
STRATS = ROOT / "data" / "strategies"
LIVE.mkdir(parents=True, exist_ok=True)
STRATS.mkdir(parents=True, exist_ok=True)
SESSION = LIVE / "session.json"
LOG = LIVE / "events.csv"


# ---------- saved strategies ----------
def save_strategy(name, params, note=""):
    name = "".join(ch for ch in name if ch.isalnum() or ch in " -_").strip() or "strategy"
    clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items() if k in DEFAULT_PARAMS}
    (STRATS / f"{name}.json").write_text(json.dumps(dict(name=name, note=note, params=clean,
                                                         saved=datetime.now(timezone.utc).isoformat(timespec="seconds")), indent=1))
    return name


def list_strategies():
    return sorted(p.stem for p in STRATS.glob("*.json"))


def load_strategy(name):
    d = json.loads((STRATS / f"{name}.json").read_text())
    return {**DEFAULT_PARAMS, **{k: v for k, v in d["params"].items() if k in DEFAULT_PARAMS}}


# ---------- session ----------
def start_session(strategy_name, params, symbol, costs, capital):
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    s = dict(strategy=strategy_name, symbol=symbol, params={k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
             costs=costs, capital=capital, started=now.isoformat(timespec="seconds"))
    SESSION.write_text(json.dumps(s, indent=1))
    return s


def get_session():
    return json.loads(SESSION.read_text()) if SESSION.exists() else None


def stop_session():
    s = get_session()
    if s:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        SESSION.rename(LIVE / f"session_stopped_{stamp}.json")
    return s


_BUILD = {}  # the closed-bar signal build, reused until the next 15m bar closes


def _closed(session, p, gold, raw, hmm, warmup_days):
    """Closed bars and the multi-horizon signals built from them. The build takes seconds, so it is redone only when
    a new 15m bar has closed (or the session / settings changed), not on every tick refresh."""
    symbol, started = session["symbol"], pd.Timestamp(session["started"])
    key = (symbol, session["started"], json.dumps(session["params"], sort_keys=True, default=str), id(hmm))
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    c = _BUILD.get("c")
    if c and c["key"] == key and now < c["until"]:
        return c
    bars = binance.klines(symbol, update=True)
    bars = bars[bars.index >= started - pd.Timedelta(days=warmup_days)]
    feat, base_tf = mtf.build(bars, gold, raw, p, hmm)
    # the next 15m bar to close finishes 30 minutes after the last closed bar opened; rebuild the moment it does.
    # (if Binance is a moment late with it, retry in 2 seconds; with_forming refuses to draw over the gap meanwhile)
    until = max(bars.index[-1] + pd.Timedelta(minutes=30), now + pd.Timedelta(seconds=2))
    _BUILD["c"] = c = dict(key=key, bars=bars, feat=feat, base_tf=base_tf, until=until)
    return c


def with_forming(feat, bars15, base_tf, stream):
    """Closed features plus one row for the trading-timeframe bar still in progress (open / high / low / last price so far).

    The engine then fills an order decided on the last closed bar at the REAL open of the new bar, and checks stops and
    targets on its high / low as ticks arrive. The new row has no score, so it can never open a trade on a half-built bar."""
    step = pd.Timedelta(minutes=TIMEFRAMES[base_tf]["minutes"])
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    ws = now.floor(step)
    if ws != feat.index[-1] + step:
        # Either the bar is already in feat, or a closed bar is missing between the last one and now (the cache is a moment
        # stale). Drawing the new bar over that gap would make the engine see a price jump and fire stops / targets at the
        # new open, so wait for the closed bar instead.
        return feat
    parts = bars15[bars15.index >= ws]
    f15 = stream.forming() if stream is not None else None
    if f15 is not None and f15["start"] >= ws:
        o = parts["Open"].iloc[0] if len(parts) else f15["open"]
        hi = max(parts["High"].max() if len(parts) else -1e18, f15["high"])
        lo = min(parts["Low"].min() if len(parts) else 1e18, f15["low"])
        cl, vol = f15["close"], (parts["Volume"].sum() if len(parts) else 0.0) + f15["volume"]
    elif len(parts):
        o, hi, lo, cl, vol = parts["Open"].iloc[0], parts["High"].max(), parts["Low"].min(), parts["Close"].iloc[-1], parts["Volume"].sum()
    else:
        return feat
    row = feat.iloc[[-1]].copy()
    row.index = pd.DatetimeIndex([ws])
    row[["Open", "High", "Low", "Close", "Volume"]] = [o, hi, lo, cl, vol]
    row["score"] = float("nan")
    return pd.concat([feat, row])


def snapshot(session, gold, raw, hmm=None, warmup_days=400):
    """Everything the Live tab shows, recalculated from the session start to the live price.
    Signals come from closed bars only; the bar in progress (built from ticks) is added for fills, stops and targets."""
    p = {**DEFAULT_PARAMS, **session["params"]}
    symbol, started = session["symbol"], pd.Timestamp(session["started"])
    c = _closed(session, p, gold, raw, hmm, warmup_days)
    feat, base_tf = c["feat"], c["base_tf"]
    stream = ticks.get_stream(symbol)
    live_feat = with_forming(feat, c["bars"], base_tf, stream if stream.fresh() else None)
    step = pd.Timedelta(minutes=TIMEFRAMES[base_tf]["minutes"])
    out = dict(base_tf=base_tf, feat=feat, feat_live=live_feat, last_bar=feat.index[-1], next_close=feat.index[-1] + 2 * step,
               started=started, symbol=symbol, stream=stream)
    if stream.fresh():
        out["price"] = dict(price=stream.mid, bid=stream.bid, ask=stream.ask, time=pd.Timestamp.now(tz="UTC").tz_localize(None), source="websocket")
    else:
        try:
            out["price"] = {**binance.live_price(symbol), "source": "REST"}
        except Exception as e:
            out["price"] = dict(price=float(live_feat["Close"].iloc[-1]), bid=float("nan"), ask=float("nan"), time=None,
                                error=str(e), source="last bar")

    live_bars = live_feat[live_feat.index >= started]
    if len(live_bars) < 2:
        out.update(res=None, position=None, closed=pd.DataFrame(), equity=None)
        return out
    res = mtf.run(None, None, None, p, session["costs"], session["capital"], start=live_bars.index[0], feat=live_feat,
                  base_tf=base_tf, with_gross=False, min_bars=2)
    tr = res["trades"]
    position, closed = None, tr
    if not tr.empty and tr["exit_reason"].iloc[-1] == "End of test":
        position, closed = tr.iloc[-1].to_dict(), tr.iloc[:-1]
        px = out["price"]["price"]
        sign = 1 if position["side"] == "Long" else -1
        position["live_price"] = px
        position["unrealised_pnl"] = sign * position["qty"] * (px - position["entry_price"])
    out.update(res=res, position=position, closed=closed, equity=res["curves"]["Strategy (net of charges)"])
    return out


def decision_text(feat, p):
    """What the bot will do at the next bar's open, from the last closed bar."""
    row = feat.iloc[-1]
    s = row["score"]
    if pd.isna(s):
        return "No signal yet (warming up)"
    if s >= p["entry_threshold"] and row.get("vol_ok_long", True):
        return f"Long setup (score {s:+.2f} >= {p['entry_threshold']:.2f})"
    if p["allow_short"] and s <= -p["entry_threshold"] and row.get("vol_ok_short", True):
        return f"Short setup (score {s:+.2f})"
    return f"No new entry (score {s:+.2f}; entry needs {p['entry_threshold']:+.2f})"


def log_events(snap):
    """Append new trades (entries and exits) to data/live/events.csv; returns the rows added."""
    res = snap.get("res")
    if res is None or res["trades"].empty:
        return []
    seen = set(pd.read_csv(LOG)["key"]) if LOG.exists() else set()
    rows = []
    for _, t in res["trades"].iterrows():
        k_in = f"{snap['symbol']}|in|{t['entry_date']}"
        if k_in not in seen:
            rows.append(dict(key=k_in, time=t["entry_date"], event=f"Open {t['side']}", price=t["entry_price"], qty=t["qty"], pnl=None))
        if t["exit_reason"] != "End of test":
            k_out = f"{snap['symbol']}|out|{t['exit_date']}|{t['entry_date']}"
            if k_out not in seen:
                rows.append(dict(key=k_out, time=t["exit_date"], event=f"Close {t['side']} ({t['exit_reason']})",
                                 price=t["exit_price"], qty=t["qty"], pnl=t["net_pnl"]))
    if rows:
        pd.DataFrame(rows).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    return rows
