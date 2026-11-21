"""Tick-by-tick Binance feed over websocket (trades + best bid/ask), kept in a background thread.

One stream per symbol lives for as long as the Python process does, so Streamlit reruns just read from it.
- Quotes (bookTicker) arrive many times a second and give the live price (mid of best bid / best ask).
- Trades (aggTrade) arrive less often, especially on spot gold; they are shown as a tape and counted into volume.
- The 15m bar that is still forming is built here from the ticks (seeded from REST so its open is the true open).
  live.py appends it to the closed bars so stops and fills see the live bar, not only finished ones.
Reconnects by itself. Nothing is ordered; this is a read-only market data feed.
"""
import json
import threading
import time
from collections import deque

import pandas as pd
import websocket

from . import binance
from .data import ROOT

TICK_DIR = ROOT / "data" / "ticks"
URLS = {
    "PAXGUSDT": "wss://stream.binance.com:9443/stream?streams=paxgusdt@aggTrade/paxgusdt@bookTicker",
    "XAUUSDT": "wss://fstream.binance.com/stream?streams=xauusdt@aggTrade/xauusdt@bookTicker",
}
BAR_SECONDS = 900  # the forming bar is a 15m bar, the base interval of every timeframe


def _ts(t):
    return pd.Timestamp(t, unit="s")  # naive UTC, like every other time in the app


class TickStream:
    def __init__(self, symbol):
        self.symbol = symbol
        self.trades = deque(maxlen=5000)   # (time, price, qty, "buy"/"sell" = the aggressor)
        self.quotes = deque(maxlen=40000)  # (time, bid, ask), only stored when the quote changes
        self.bid = self.ask = self.last = self.last_qty = float("nan")
        self.last_side = ""
        self.last_msg = 0.0
        self.connected = False
        self.reconnects = 0
        self.n_trades = self.n_quotes = 0
        self.started = time.time()
        self.error = ""
        self.record = False
        self._buf, self._last_flush, self._last_rec_quote = [], time.time(), 0.0
        self._bar = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"ticks-{symbol}")
        self._thread.start()

    # ---------- connection ----------
    def _run(self):
        while not self._stop:
            try:
                ws = websocket.WebSocketApp(URLS[self.symbol], on_open=self._on_open, on_message=self._on_message,
                                            on_error=self._on_error, on_close=self._on_close)
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:  # keep the thread alive whatever happens
                self.error = f"{type(e).__name__}: {e}"
            self.connected = False
            if self._stop:
                break
            self.reconnects += 1
            time.sleep(min(30, 1 + self.reconnects))

    def _on_open(self, ws):
        self.connected, self.error = True, ""

    def _on_close(self, ws, *a):
        self.connected = False

    def _on_error(self, ws, err):
        self.error = str(err)[:200]

    def stop(self):
        self._stop = True

    # ---------- messages ----------
    def _on_message(self, ws, raw):
        try:
            m = json.loads(raw)
            kind, d, now = m["stream"].split("@")[1], m["data"], time.time()
            self.last_msg = now
            if kind == "bookTicker":
                bid, ask = float(d["b"]), float(d["a"])
                if bid == self.bid and ask == self.ask:
                    return
                self.bid, self.ask = bid, ask
                self.quotes.append((now, bid, ask))
                self.n_quotes += 1
                self._on_price(now, (bid + ask) / 2, 0.0)
                if self.record and now - self._last_rec_quote >= 1.0:  # quotes are sampled once a second on disk
                    self._last_rec_quote = now
                    self._buf.append(f"{now:.3f},quote,{bid},{ask},,")
            elif kind == "aggTrade":
                price, qty = float(d["p"]), float(d["q"])
                side = "sell" if d["m"] else "buy"  # buyer is the maker => the aggressor sold
                t = d["T"] / 1000
                self.last, self.last_qty, self.last_side = price, qty, side
                self.trades.append((t, price, qty, side))
                self.n_trades += 1
                self._on_price(t, price, qty)
                if self.record:
                    self._buf.append(f"{t:.3f},trade,{price},{qty},{side},")
            if self.record and self._buf and time.time() - self._last_flush > 5:
                self._flush()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def _flush(self):
        TICK_DIR.mkdir(parents=True, exist_ok=True)
        path = TICK_DIR / f"{self.symbol}_{time.strftime('%Y%m%d', time.gmtime())}.csv"
        new = not path.exists()
        with open(path, "a") as f:
            if new:
                f.write("time,type,price_or_bid,qty_or_ask,side,\n")
            f.write("\n".join(self._buf) + "\n")
        self._buf, self._last_flush = [], time.time()

    # ---------- the forming 15m bar ----------
    def _on_price(self, t, price, qty):
        start = int(t // BAR_SECONDS * BAR_SECONDS)
        with self._lock:
            b = self._bar
            if b is None or b["start"] != start:
                b = self._seed(start, price)
            b["high"], b["low"], b["close"] = max(b["high"], price), min(b["low"], price), price
            b["volume"] += qty
            self._bar = b

    def _seed(self, start, price):
        """Open / high / low / volume so far from REST, so a feed that started mid-bar still has the true open."""
        bar = dict(start=start, open=price, high=price, low=price, close=price, volume=0.0)
        try:
            m = binance.MARKETS[self.symbol]
            rows = binance._get(m["base"] + m["klines"], dict(symbol=self.symbol, interval="15m", startTime=start * 1000, limit=1), tries=1)
            if rows and rows[0][0] == start * 1000:
                o, h, l, c, v = (float(x) for x in rows[0][1:6])
                bar.update(open=o, high=max(h, price), low=min(l, price), volume=v)
        except Exception:
            pass
        return bar

    def forming(self):
        """The 15m bar in progress as a dict (start as a Timestamp), or None if the feed has nothing for the current window."""
        with self._lock:
            b = dict(self._bar) if self._bar else None
        if b is None or b["start"] != int(time.time() // BAR_SECONDS * BAR_SECONDS):
            return None
        b["start"] = _ts(b["start"])
        return b

    # ---------- read side ----------
    @property
    def mid(self):
        return (self.bid + self.ask) / 2 if self.bid == self.bid and self.ask == self.ask else float("nan")

    def age(self):
        """Seconds since the last message (inf if none yet)."""
        return time.time() - self.last_msg if self.last_msg else float("inf")

    def fresh(self, within=10):
        return self.connected and self.age() < within and self.mid == self.mid

    def rate(self, seconds=10):
        """Quotes and trades per second over the last `seconds`."""
        cut = time.time() - seconds
        q = sum(1 for t, *_ in reversed(self.quotes) if t >= cut)
        n = sum(1 for t, *_ in reversed(self.trades) if t >= cut)
        return (q + n) / seconds

    def recent_mid(self, seconds=300, max_points=600):
        cut = time.time() - seconds
        rows = [(t, (b + a) / 2) for t, b, a in list(self.quotes) if t >= cut]
        if len(rows) > max_points:
            rows = rows[::len(rows) // max_points + 1]
        return pd.DataFrame(rows, columns=["time", "mid"]).assign(time=lambda d: pd.to_datetime(d["time"], unit="s"))

    def tape(self, n=10):
        rows = list(self.trades)[-n:][::-1]
        return pd.DataFrame([(_ts(t), p, q, s) for t, p, q, s in rows], columns=["Time (UTC)", "Price", "Size (oz)", "Aggressor"])

    def status(self):
        return dict(connected=self.connected, age=self.age(), reconnects=self.reconnects, quotes=self.n_quotes, trades=self.n_trades,
                    rate=self.rate(), error=self.error, uptime=time.time() - self.started)


_STREAMS, _REG = {}, threading.Lock()


def get_stream(symbol):
    """The running stream for a symbol (started on first use)."""
    with _REG:
        s = _STREAMS.get(symbol)
        if s is None or not s._thread.is_alive():
            if symbol.startswith("MT5:"):
                from . import mt5
                s = _STREAMS[symbol] = mt5.MT5Stream(symbol)
            else:
                s = _STREAMS[symbol] = TickStream(symbol)
        return s
