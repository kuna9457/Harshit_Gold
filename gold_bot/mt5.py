"""MetaTrader 5 as the XAUUSD data source, and a way to draw the bot's trades on an MT5 chart.

Needs the MT5 desktop terminal installed on the SAME Windows PC (the MetaTrader5 Python package talks to it), so this
cannot run on Render. Read-only: no order is ever sent. Passwords are kept in memory only and never written to disk.

- connect(): log in to a broker account (or use whatever account the terminal is already logged in to)
- klines():  15m history (cached in data/mt5/), converted from broker server time to UTC, same shape as binance.klines
- MT5Stream: tick by tick quotes, same interface as ticks.TickStream, so the Live tab works unchanged
- write_chart_file() + indicator GoldBotSignals.mq5: arrows, entry-to-exit lines, stop and target drawn on the MT5 chart
"""
import json
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd

from .data import ROOT

try:
    import MetaTrader5 as _mt5
except ImportError:  # the app still runs without it
    _mt5 = None

STORE = ROOT / "data" / "mt5"
STORE.mkdir(parents=True, exist_ok=True)
PREFIX = "MT5:"
CONN_FILE = STORE / "connection.json"  # login, server and terminal path only; never the password
SIGNAL_FILE = "gold_bot_signals.csv"
INDICATOR = "GoldBotSignals"
LOCK = threading.RLock()  # the MT5 package is process-wide, so every call goes through this
STATE = dict(connected=False, error="", symbol=None, account=None)
BAR_SECONDS = 900


def available():
    return _mt5 is not None


def is_mt5(symbol):
    return str(symbol).startswith(PREFIX)


def name_of(symbol):
    return symbol[len(PREFIX):] if is_mt5(symbol) else symbol


# ---------------------------------------------------------------- connection
def saved_connection():
    try:
        return json.loads(CONN_FILE.read_text())
    except Exception:
        return {}


def connect(login=0, password="", server="", path=""):
    """Start / attach to the terminal and log in. With no login the terminal's current account is used."""
    if _mt5 is None:
        STATE.update(connected=False, error="The MetaTrader5 Python package is not installed (pip install MetaTrader5).")
        return False
    with LOCK:
        _mt5.shutdown()
        kw = {}
        if path:
            kw["path"] = path
        if login:
            kw.update(login=int(login), password=password, server=server)
        if not _mt5.initialize(**kw):
            code, msg = _mt5.last_error()
            STATE.update(connected=False, account=None,
                         error=f"MT5 refused the connection ({code}: {msg}). Check the terminal is installed, "
                               "and that the login, password and server are right.")
            return False
        acc = _mt5.account_info()
        if acc is None:
            STATE.update(connected=False, account=None, error="Terminal started but no account is logged in. Enter login, password and server.")
            return False
        STATE.update(connected=True, error="", account=acc._asdict())
    try:
        CONN_FILE.write_text(json.dumps(dict(login=int(acc.login), server=acc.server, path=path)))
    except Exception:
        pass
    return True


def disconnect():
    with LOCK:
        if _mt5 is not None:
            _mt5.shutdown()
        STATE.update(connected=False, account=None, symbol=None)
    _OFFSET.clear()


def info():
    """Account, terminal and server-time details for the dashboard."""
    with LOCK:
        t, a = _mt5.terminal_info(), _mt5.account_info()
        if t is None or a is None:
            STATE.update(connected=False)
            return None
        return dict(account=a._asdict(), terminal=t._asdict(),
                    demo=a.trade_mode == _mt5.ACCOUNT_TRADE_MODE_DEMO)


def find_symbols():
    """Gold symbols this broker offers (names differ: XAUUSD, GOLD, XAUUSDm, XAUUSD.a ...)."""
    with LOCK:
        syms = _mt5.symbols_get() or []
    out = [s.name for s in syms if s.name.upper().startswith(("XAUUSD", "GOLD")) and "EUR" not in s.name.upper()
           and "GBP" not in s.name.upper() and "AUD" not in s.name.upper()]
    return sorted(out, key=lambda n: (len(n), n))


def select_symbol(name):
    with LOCK:
        if not _mt5.symbol_select(name, True):
            raise RuntimeError(f"Cannot show {name} in Market Watch: {_mt5.last_error()}")
        STATE["symbol"] = name
        s = _mt5.symbol_info(name)
        return s._asdict() if s else {}


# ---------------------------------------------------------------- server time <-> UTC
_OFFSET = {}


def _us_dst(ts):
    """Brokers on 'EET with US daylight saving' (most MT5 brokers) are UTC+3 while the US is on summer time, else UTC+2."""
    y = ts.year
    mar = pd.Timestamp(y, 3, 1)
    start = mar + pd.Timedelta(days=(6 - mar.dayofweek) % 7 + 7)       # second Sunday of March
    nov = pd.Timestamp(y, 11, 1)
    end = nov + pd.Timedelta(days=(6 - nov.dayofweek) % 7)             # first Sunday of November
    return start <= ts < end


def server_offset_hours(symbol):
    """Hours the broker's clock is ahead of UTC right now. A live tick carries the broker's clock, so the difference to the
    PC's UTC clock is the offset. A stale tick (market closed) does not line up with a half hour, so then the last good value
    is kept, or the usual 'EET with US daylight saving' (UTC+2 / UTC+3) is assumed."""
    if "now" in _OFFSET and time.time() - _OFFSET["at"] < 600:
        return _OFFSET["now"]
    with LOCK:
        tick = _mt5.symbol_info_tick(symbol)
    if tick is not None and tick.time:
        diff = (tick.time - time.time()) / 3600
        half = round(diff * 2) / 2
        if abs(diff) < 14 and abs(diff - half) < 0.03:
            _OFFSET.update(now=half, at=time.time())
            return half
    if "now" in _OFFSET:
        return _OFFSET["now"]
    return 3 if _us_dst(pd.Timestamp.now(tz="UTC").tz_localize(None)) else 2


def to_utc(index, symbol):
    """Broker server times -> naive UTC, following US daylight saving when the broker does (UTC+2 / UTC+3)."""
    now_h = server_offset_hours(symbol)
    if now_h in (2, 3):
        off = pd.Series([3 if _us_dst(t) else 2 for t in index], index=index)
        return index - pd.to_timedelta(off.values, unit="h")
    return index - pd.Timedelta(hours=now_h)


def to_server(ts_utc, symbol):
    now_h = server_offset_hours(symbol)
    if now_h in (2, 3):
        return ts_utc + pd.Timedelta(hours=3 if _us_dst(ts_utc) else 2)
    return ts_utc + pd.Timedelta(hours=now_h)


# ---------------------------------------------------------------- history
_MEM = {}


def _fetch(symbol, start_utc, end_utc):
    """15m bars between two UTC times (naive), as a UTC-indexed frame (tick volume)."""
    frames, cur = [], pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    # copy_rates_range takes datetimes and reads them as server time, so shift by the offset
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=180), end)
        a, b = (to_server(cur, symbol) + pd.Timedelta(hours=1)).to_pydatetime(), (to_server(nxt, symbol) + pd.Timedelta(hours=1)).to_pydatetime()
        with LOCK:
            r = _mt5.copy_rates_range(symbol, _mt5.TIMEFRAME_M15, a, b)
        if r is not None and len(r):
            frames.append(pd.DataFrame(r))
        cur = nxt
    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.concat(frames)
    df.index = pd.to_datetime(df["time"], unit="s")   # server clock written as if it were UTC
    df = df.rename(columns=dict(open="Open", high="High", low="Low", close="Close", tick_volume="Volume"))
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = to_utc(df.index, symbol)
    return df


def _closed(df):
    return df[df.index + pd.Timedelta(minutes=15) <= pd.Timestamp.now(tz="UTC").tz_localize(None)]


def klines(symbol, update=True, years=10, progress=None):
    """Closed 15m bars for 'MT5:<name>' from the broker, cached on disk (UTC). First call pulls up to `years` of history."""
    name = name_of(symbol)
    path = STORE / f"{name}_15m.csv"
    old = _MEM.get(path)
    if old is None and path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True)
    df = old
    if (update or old is None) and STATE["connected"]:
        select_symbol(name)
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        start = now - pd.Timedelta(days=int(365.25 * years)) if old is None or old.empty else old.index[-1] - pd.Timedelta(hours=6)
        new = _closed(_fetch(name, start, now + pd.Timedelta(days=1)))
        if progress:
            progress(1.0)
        if len(new):
            df = new if old is None or old.empty else pd.concat([old[old.index < new.index[0]], new])
            df.to_csv(path)
    if df is None or df.empty:
        raise RuntimeError("No MT5 history yet. Connect to MT5 in the MT5 tab and download history first.")
    _MEM[path] = df
    return df[["Open", "High", "Low", "Close", "Volume"]]


def cached_symbols():
    return sorted(p.name[:-8] for p in STORE.glob("*_15m.csv"))


def live_price(symbol):
    with LOCK:
        t = _mt5.symbol_info_tick(name_of(symbol))
    if t is None:
        raise RuntimeError(f"No quote for {name_of(symbol)}: {_mt5.last_error()}")
    return dict(price=(t.bid + t.ask) / 2, bid=t.bid, ask=t.ask, time=pd.Timestamp.now(tz="UTC").tz_localize(None))


# ---------------------------------------------------------------- tick stream (same interface as ticks.TickStream)
class MT5Stream:
    """Polls the terminal ~20 times a second and keeps every quote change. The terminal itself receives ticks as they happen."""

    def __init__(self, symbol):
        self.symbol, self.name = symbol, name_of(symbol)
        self.trades = deque(maxlen=5000)
        self.quotes = deque(maxlen=40000)
        self.bid = self.ask = self.last = self.last_qty = float("nan")
        self.last_side = ""
        self.last_msg = 0.0
        self.connected = False
        self.reconnects = 0
        self.n_trades = self.n_quotes = 0
        self.started = time.time()
        self.error = ""
        self.record = False
        self._bar, self._last_msc = None, 0
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"mt5-{self.name}")
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                with LOCK:
                    t = _mt5.symbol_info_tick(self.name) if STATE["connected"] else None
                if t is None:
                    self.connected = False
                    self.error = "MT5 not connected" if not STATE["connected"] else str(_mt5.last_error())
                    time.sleep(1)
                    continue
                self.connected, self.error = True, ""
                if t.time_msc != self._last_msc and t.bid > 0 and t.ask > 0:
                    self._last_msc = t.time_msc
                    now = time.time()
                    self.last_msg = now
                    if (t.bid, t.ask) != (self.bid, self.ask):
                        self.bid, self.ask = t.bid, t.ask
                        self.quotes.append((now, t.bid, t.ask))
                        self.n_quotes += 1
                        self._on_price(now, (t.bid + t.ask) / 2)
                elif t.time_msc == self._last_msc and self.last_msg and time.time() - self.last_msg > 0:
                    pass
                time.sleep(0.05)
            except Exception as e:
                self.connected, self.error = False, f"{type(e).__name__}: {e}"
                self.reconnects += 1
                time.sleep(2)

    def _on_price(self, t, price):
        start = int(t // BAR_SECONDS * BAR_SECONDS)
        with self._lock:
            b = self._bar
            if b is None or b["start"] != start:
                b = self._seed(start, price)
            b["high"], b["low"], b["close"] = max(b["high"], price), min(b["low"], price), price
            b["volume"] += 1  # tick volume, like the history
            self._bar = b

    def _seed(self, start, price):
        bar = dict(start=start, open=price, high=price, low=price, close=price, volume=0.0)
        try:
            with LOCK:
                r = _mt5.copy_rates_from_pos(self.name, _mt5.TIMEFRAME_M15, 0, 1)
            if r is not None and len(r):
                row = r[0]
                srv = pd.Timestamp(int(row["time"]), unit="s")
                if abs((to_utc(pd.DatetimeIndex([srv]), self.name)[0] - pd.Timestamp(start, unit="s")).total_seconds()) < 1:
                    bar.update(open=float(row["open"]), high=max(float(row["high"]), price), low=min(float(row["low"]), price),
                               volume=float(row["tick_volume"]))
        except Exception:
            pass
        return bar

    def forming(self):
        with self._lock:
            b = dict(self._bar) if self._bar else None
        if b is None or b["start"] != int(time.time() // BAR_SECONDS * BAR_SECONDS):
            return None
        b["start"] = pd.Timestamp(b["start"], unit="s")
        return b

    @property
    def mid(self):
        return (self.bid + self.ask) / 2 if self.bid == self.bid and self.ask == self.ask else float("nan")

    def age(self):
        return time.time() - self.last_msg if self.last_msg else float("inf")

    def fresh(self, within=15):
        # a closed market sends no ticks; the last quote is then not a live price
        return self.connected and self.age() < within and self.mid == self.mid

    def rate(self, seconds=10):
        cut = time.time() - seconds
        return sum(1 for t, *_ in reversed(self.quotes) if t >= cut) / seconds

    def recent_mid(self, seconds=300, max_points=600):
        cut = time.time() - seconds
        rows = [(t, (b + a) / 2) for t, b, a in list(self.quotes) if t >= cut]
        if len(rows) > max_points:
            rows = rows[::len(rows) // max_points + 1]
        return pd.DataFrame(rows, columns=["time", "mid"]).assign(time=lambda d: pd.to_datetime(d["time"], unit="s"))

    def tape(self, n=10):
        return pd.DataFrame(columns=["Time (UTC)", "Price", "Size (oz)", "Aggressor"])

    def status(self):
        return dict(connected=self.connected, age=self.age(), reconnects=self.reconnects, quotes=self.n_quotes, trades=0,
                    rate=self.rate(), error=self.error, uptime=time.time() - self.started)

    def stop(self):
        self._stop = True


# ---------------------------------------------------------------- drawing on the MT5 chart
def common_files_dir():
    with LOCK:
        t = _mt5.terminal_info()
    return Path(t.commondata_path) / "Files"


def chart_rows(symbol, trades, position=None, stop=None, target=None, signals=None, label=""):
    """Rows for the chart file. Times are written in BROKER SERVER time because that is what MT5 charts use.
    Kinds: ENTRY, EXIT, LINE (entry -> exit), SL, TP (open position), SIG (strategy signal, no trade)."""
    rows = []

    def add(kind, t, price, side, text, t2="", p2="", win=""):
        rows.append(dict(kind=kind, time=to_server(pd.Timestamp(t), symbol).strftime("%Y.%m.%d %H:%M"), price=f"{float(price):.5f}",
                         side=side, text=str(text).replace(",", ";"), time2=(to_server(pd.Timestamp(t2), symbol).strftime("%Y.%m.%d %H:%M") if t2 != "" else ""),
                         price2=("" if p2 == "" else f"{float(p2):.5f}"), win=win))

    for _, t in trades.iterrows():
        side = "BUY" if t["side"] == "Long" else "SELL"
        open_ = t["exit_reason"] == "End of test"
        add("ENTRY", t["entry_date"], t["entry_price"], side, f"{side} {t['qty']:.2f} @ {t['entry_price']:.2f}")
        if open_:
            continue
        win = 1 if t["net_pnl"] > 0 else 0
        add("EXIT", t["exit_date"], t["exit_price"], side, f"Exit {t['exit_reason']} {t['net_pnl']:+,.0f}", win=win)
        add("LINE", t["entry_date"], t["entry_price"], side, "", t["exit_date"], t["exit_price"], win)
    if position is not None:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        side = "BUY" if position["side"] == "Long" else "SELL"
        add("LINE", position["entry_date"], position["entry_price"], side, "open", now, position.get("live_price", position["entry_price"]), 2)
        if stop is not None:
            add("SL", position["entry_date"], stop, side, f"Stop {stop:.2f}", now + pd.Timedelta(hours=12), stop)
        if target is not None:
            add("TP", position["entry_date"], target, side, f"Target {target:.2f}", now + pd.Timedelta(hours=12), target)
    for t, price, side, text in (signals or []):
        add("SIG", t, price, side, text)
    return rows


def write_chart_file(symbol, rows, title=""):
    """Writes data for the indicator, in MT5's common Files folder (all terminals and charts can read it)."""
    d = common_files_dir()
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["kind", "time", "price", "side", "text", "time2", "price2", "win"])
    path = d / SIGNAL_FILE
    # first line names the symbol the rows belong to, so an indicator on another chart ignores them
    tmp = d / (SIGNAL_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(f"#symbol,{name_of(symbol)},{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M:%S},{title}\n")
        df.to_csv(f, index=False, lineterminator="\n")
    tmp.replace(path)
    return path, len(df)


def install_indicator():
    """Copies GoldBotSignals.mq5 into the terminal and compiles it with MetaEditor. Returns (ok, message)."""
    src = ROOT / "mt5" / f"{INDICATOR}.mq5"
    with LOCK:
        t = _mt5.terminal_info()
    ind_dir = Path(t.data_path) / "MQL5" / "Indicators"
    ind_dir.mkdir(parents=True, exist_ok=True)
    dst = ind_dir / f"{INDICATOR}.mq5"
    shutil.copyfile(src, dst)
    editor = Path(t.path) / "MetaEditor64.exe"
    if not editor.exists():
        return False, f"Copied to {dst}, but MetaEditor was not found at {editor}. Open it, open the file and press F7 to compile."
    log = ind_dir / f"{INDICATOR}.log"
    subprocess.run([str(editor), f"/compile:{dst}", f"/log:{log}"], capture_output=True, timeout=120)
    ex5 = ind_dir / f"{INDICATOR}.ex5"
    if ex5.exists() and ex5.stat().st_mtime >= dst.stat().st_mtime - 5:
        return True, f"Installed and compiled: {ex5}"
    msg = ""
    try:
        msg = log.read_text(encoding="utf-16", errors="ignore")[-400:]
    except Exception:
        pass
    return False, f"Copied to {dst} but compiling failed. Open it in MetaEditor and press F7. {msg}"
