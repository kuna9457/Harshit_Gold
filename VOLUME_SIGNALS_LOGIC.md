# Volume Signals: Logic Reference for Integration

This file covers two volume signals from the Market Anomaly Lab, written so they can be moved into another strategy (for example the gold strategy):

1. **#1 RVOL Z-Score Spike**: volume is far above normal, so a big player is likely active.
2. **#7 Volume Dry-Up (VDU) Pocket**: volume falls to almost nothing during a tight sideways range, so sellers may be exhausted.

Source code in this project:
- Signals: `anomalies/volume.py` (`RvolSpike`, `VolumeDryUp`)
- Helpers: `anomalies/base.py` (`prior_mean_std`, `zscore_vs_past`, `atr`, `build_events`)
- Backtest engine: `core/engine.py` (`run_study`, `_strategy`)

Everything works on **daily OHLCV bars**. The signal is known at the **close** of the signal day.

---

## 0. Input data contract

A pandas DataFrame indexed by date (sorted ascending, one row per trading day) with these columns:

| column | meaning |
|---|---|
| `open`, `high`, `low`, `close` | adjusted prices (the lab uses dividend/split-adjusted prices) |
| `volume` | traded volume for the day (shares/contracts/lots) |

Rows with missing or zero volume should be dropped or set to NaN before you compute anything. A zero-volume day breaks both the z-score and the percentile.

---

## 1. Shared helpers

### 1.1 Rolling mean/std of the previous N days (no look-ahead)
```
prev        = volume.shift(1)                 # today is excluded
mean_N      = prev.rolling(N, min_periods=max(5, int(0.8*N))).mean()
std_N       = prev.rolling(N, min_periods=max(5, int(0.8*N))).std()   # sample std (ddof=1)
```
- **Today is excluded** from its own baseline. A spike can't inflate its own average.
- At least 80% of the window must be present (`min_periods`), otherwise NaN and no signal.

### 1.2 Z-score vs the past
```
z = (volume - mean_N) / std_N        # std == 0 -> NaN (no signal)
```

### 1.3 ATR (simple average, not Wilder)
```
prev_close = close.shift(1)
TR  = max(high - low, |high - prev_close|, |low - prev_close|)
ATR = TR.rolling(n, min_periods=n).mean()
```
Note: this is a **simple moving average** of true range, not Wilder's smoothing. If the gold code uses Wilder ATR, the VDU numbers will differ slightly.

### 1.4 Cooldown (event de-duplication)
Once a signal fires on bar `i`, any further signals are ignored until bar `i + cooldown + 1`.
```
keep = []; last = -inf
for i in indices_where(mask):
    if i - last > cooldown:
        keep.append(i); last = i
```
This stops one long episode from being counted as many trades.

---

## 2. RVOL Z-Score Spike (#1)

### Idea
Volume 3 to 5 standard deviations above its 20-day average shows institutional urgency. Price tends to **keep moving in the direction of the spike day's candle**.

### Parameters (defaults)
| param | default | range | meaning |
|---|---|---|---|
| `window` | 20 | 5–250 | lookback for the volume mean/std |
| `z_threshold` | 3.0 | 1–10 | fire when z ≥ this (research says 3 to 5) |
| `direction` | `day_direction` | `day_direction` / `long` / `neutral` | how to set the trade side |
| `hold_days` | 5 | 1–250 | holding period in trading days |
| `cooldown` | 5 | 0–250 | days to ignore new signals after one fires |

### Logic
```
mean_N, std_N = prior mean/std of volume over `window`   (section 1.1)
z             = (volume - mean_N) / std_N
day_ret       = close / close.shift(1) - 1               # close-to-close

SIGNAL        = z >= z_threshold

side:
  day_direction -> sign(day_ret)    (+1 up day = long, -1 down day = short, 0 if flat)
  long          -> +1 always
  neutral       -> 0 (observe only)

apply cooldown -> events
```

### Extra columns saved per event (for analysis)
- `volume_z` = z
- `volume_x_avg` = volume / mean_N (plain RVOL multiple)
- `day_return_%` = day_ret × 100

### Things to watch
- Direction comes from the **close-to-close** return, not the candle body (close vs open). A gap-up day that closes below its open still counts as "up" if it closed above yesterday's close.
- The research uses intraday time-of-day volume buckets and excludes scheduled news. The daily version can't do either, so results days, budget days, and similar events are included.

---

## 3. Volume Dry-Up (VDU) Pocket (#7)

### Idea
Volume drops below the 5th percentile of the last 50 days **while** price is in a tight consolidation. This suggests sellers are exhausted, and a breakout (expected upward) tends to follow.

### Parameters (defaults)
| param | default | range | meaning |
|---|---|---|---|
| `vol_window` | 50 | 10–250 | lookback for the volume percentile |
| `percentile` | 5.0 | 1–30 | volume must be below this percentile |
| `base_days` | 10 | 3–60 | length of the consolidation window |
| `base_atr_mult` | 3.0 | 0.5–10 | max consolidation range, in ATRs |
| `direction` | `long` | `long` / `neutral` | expected breakout direction |
| `hold_days` | 10 | 1–250 | holding period |
| `cooldown` | 10 | 0–250 | days to ignore new signals after one fires |

### Logic
```
# 1. Volume dry-up threshold, from the PREVIOUS vol_window days (today excluded)
q = volume.shift(1).rolling(vol_window, min_periods=int(0.8*vol_window))
                   .quantile(percentile / 100)          # pandas linear interpolation
dry = volume < q

# 2. Tight base: the range of the last base_days bars (TODAY INCLUDED)
base_range = high.rolling(base_days).max() - low.rolling(base_days).min()
A          = ATR(20)                                     # today included, simple mean
tight      = base_range <= base_atr_mult * A

SIGNAL = dry AND tight

side: long -> +1, neutral -> 0
apply cooldown -> events
```

### Extra columns saved per event
- `volume_vs_threshold` = volume / q (below 1 means dried up; smaller means drier)
- `base_range_atr` = base_range / ATR(20)

### Things to watch
- The percentile window **excludes** today, but the base range and ATR **include** today. That's fine because everything is known at today's close. Don't enter before the close on the same bar.
- Because the ATR includes the low-range dry-up day itself, it is slightly smaller than a "previous 20 days" ATR. This makes the tight test a bit stricter.
- There is no breakout confirmation. The signal fires on the quiet day itself, not when price breaks out. If the gold strategy needs confirmation, add a rule such as `close > high.rolling(base_days).max().shift(1)` within the next K days.
- The lab also reports a **move-size ratio**: average |move| after events divided by average |move| on normal days. On the stocks tested, VDU predicted **smaller** later moves (0.8× on INFY, significant), not bigger breakouts.

---

## 4. How the lab backtests a signal (to replicate the results)

From `core/engine.py`:

| item | rule |
|---|---|
| Entry price | `close` mode: signal-day close (default). `next_open` mode: next day's open (more realistic) |
| Exit | close of bar `i + hold_days` |
| Trade return | `dir × (close[i+H] / entry_px − 1) − cost` |
| Direction | `dir = -1 if side < 0 else +1`. **Note: `side = 0` (neutral / flat day) is traded as LONG in the strategy curve** |
| Cost | `cost_bps` (default 10 bps), charged once per trade at exit |
| Overlap | a new event is skipped if it falls before the previous trade's exit (`i < last_exit`) |
| Horizons reported | 1, 3, 5, 10, 20, 60 days, plus `hold_days` |

### Statistics per horizon
- **Avg % / Median % / Win %**: signed return after events (`dir × r`)
- **Normal-day avg %**: the same horizon's average return across *all* days in range (the baseline)
- **Edge %**: `mean(dir × (r − baseline_mean))`
- **t-stat / p-value**: one-sample t-test of the edge against 0
- **Move size ratio**: `mean(|r_event|) / mean(|r_all_days|)`, with its own p-value

### Past results (defaults, daily Yahoo data, 2000–2026)
| symbol | signal | events | avg move (signal dir) | normal | p-value |
|---|---|---|---|---|---|
| RELIANCE | RVOL spike, 5d | 171 | +0.14% | +0.03% | 0.77 |
| INFY | RVOL spike, 5d | 192 | +0.19% | +0.01% | 0.66 |
| SUZLON | RVOL spike, 5d (next open) | 163 | −0.46% | −0.04% | 0.57 |
| NIFTY | RVOL spike, 5d | 76 | −0.17% | −0.03% | 0.60 |
| RELIANCE | VDU, 10d | 156 | +0.28% | +0.87% | 0.24 |
| INFY | VDU, 10d | 156 | +0.49% | +0.67% | 0.69 (moves 0.8× normal, significant) |
| SUZLON | VDU, 10d (next open) | 112 | +0.82% | +0.19% | 0.61 |
| NIFTY | VDU, 10d | 63 | +0.19% | +0.45% | 0.44 |

On their own, neither signal showed a statistically meaningful edge on these Indian stocks. They are more promising as **filters/confirmations** inside another strategy than as standalone entries.

---

## 5. Drop-in Python (self-contained, pandas + numpy only)

```python
import numpy as np
import pandas as pd


# ---------- helpers ----------
def prior_mean_std(s: pd.Series, window: int):
    """Mean/std of the PREVIOUS `window` values (today excluded)."""
    mp = max(5, int(window * 0.8))
    prev = s.shift(1)
    return prev.rolling(window, min_periods=mp).mean(), prev.rolling(window, min_periods=mp).std()


def atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Simple-average ATR (not Wilder)."""
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def apply_cooldown(mask: pd.Series, cooldown: int) -> pd.Series:
    """Keep only the first True in each cooldown window."""
    m = mask.fillna(False).astype(bool).values
    out = np.zeros(len(m), dtype=bool)
    last = -10**9
    for i in np.flatnonzero(m):
        if i - last > cooldown:
            out[i] = True
            last = i
    return pd.Series(out, index=mask.index)


# ---------- #1 RVOL z-score spike ----------
def rvol_spike(df: pd.DataFrame, window: int = 20, z_threshold: float = 3.0,
               direction: str = "day_direction", cooldown: int = 5) -> pd.DataFrame:
    """Returns df with columns: rvol_z, rvol_x, rvol_signal (bool), rvol_side (+1/-1/0)."""
    v = df["volume"]
    m, sd = prior_mean_std(v, window)
    z = (v - m) / sd.replace(0, np.nan)
    day_ret = df["close"].pct_change()
    if direction == "day_direction":
        side = np.sign(day_ret).fillna(0)
    elif direction == "long":
        side = pd.Series(1, index=df.index)
    else:
        side = pd.Series(0, index=df.index)
    sig = apply_cooldown(z >= z_threshold, cooldown)
    return pd.DataFrame({
        "rvol_z": z,
        "rvol_x": v / m,
        "rvol_signal": sig,
        "rvol_side": side.where(sig, 0).astype(int),
    }, index=df.index)


# ---------- #7 Volume dry-up pocket ----------
def volume_dryup(df: pd.DataFrame, vol_window: int = 50, percentile: float = 5.0,
                 base_days: int = 10, base_atr_mult: float = 3.0,
                 direction: str = "long", cooldown: int = 10) -> pd.DataFrame:
    """Returns df with columns: vdu_ratio, vdu_base_atr, vdu_signal (bool), vdu_side."""
    v = df["volume"]
    q = v.shift(1).rolling(vol_window, min_periods=int(vol_window * 0.8)).quantile(percentile / 100)
    base_range = df["high"].rolling(base_days).max() - df["low"].rolling(base_days).min()
    a = atr(df, 20)
    tight = base_range <= base_atr_mult * a
    sig = apply_cooldown((v < q) & tight, cooldown)
    side_val = 1 if direction == "long" else 0
    return pd.DataFrame({
        "vdu_ratio": v / q,                 # < 1 means dried up
        "vdu_base_atr": base_range / a,     # consolidation width in ATRs
        "vdu_signal": sig,
        "vdu_side": np.where(sig, side_val, 0),
    }, index=df.index)


# ---------- simple backtest, same rules as the lab ----------
def backtest(df: pd.DataFrame, signal: pd.Series, side: pd.Series, hold_days: int,
             entry: str = "close", cost_bps: float = 10.0) -> pd.DataFrame:
    close = df["close"].values
    entry_px = (df["close"] if entry == "close" else df["open"].shift(-1)).values
    cost = cost_bps / 1e4
    trades, last_exit = [], -1
    for i in np.flatnonzero(signal.values):
        if i < last_exit:
            continue
        x = i + hold_days
        if x >= len(close) or not np.isfinite(entry_px[i]):
            continue
        d = -1 if side.iloc[i] < 0 else 1          # lab treats side 0 as long
        trades.append({"signal_date": df.index[i], "exit_date": df.index[x],
                       "side": d, "return_%": (d * (close[x] / entry_px[i] - 1) - cost) * 100})
        last_exit = x
    return pd.DataFrame(trades)
```

### Quick sanity test
```python
r = rvol_spike(df); v = volume_dryup(df)
print(r["rvol_signal"].sum(), v["vdu_signal"].sum())       # event counts
t = backtest(df, r["rvol_signal"], r["rvol_side"], hold_days=5)
print(t["return_%"].describe(), (t["return_%"] > 0).mean())
```
On RELIANCE.NS 2000–2026 with defaults, expect about **171 RVOL events** and **156 VDU events** (same daily Yahoo data). If your counts are far off, check the data rather than the logic.

---

## 6. Integrating into a gold strategy

### 6.1 Which volume series to use
| gold instrument | volume usable? | notes |
|---|---|---|
| MCX GOLD / GOLDM futures | yes | **Roll effect**: volume collapses in the expiring contract and jumps in the next one. Use a continuous series with volume summed across contracts (or the most active contract's volume), otherwise you get fake spikes and fake dry-ups on every roll. |
| COMEX GC futures | yes | same roll issue; use back-adjusted continuous data with combined volume |
| GOLDBEES / gold ETFs (NSE), GLD | yes | cleanest volume; no rolls. Watch for low-liquidity days in Indian gold ETFs |
| Spot XAUUSD (forex/CFD) | **no real volume** | only broker tick volume. RVOL on tick volume is a rough proxy; VDU percentiles on tick volume are unreliable |

### 6.2 Ways to plug in
1. **As a filter (recommended first)**: only take the gold strategy's existing long entries when `vdu_signal` occurred within the last K bars (a quiet base before entry), or skip entries on `rvol_signal` days whose side is against your trade.
2. **As a confirmation**: require `rvol_z >= 2` (looser than 3) on the breakout bar of the gold strategy. High volume confirms the breakout.
3. **As a standalone entry**: use `rvol_side` / `vdu_side` directly. The tests above suggest this alone has no edge, so treat it as a baseline to beat.

### 6.3 Gold-specific settings to try
- Gold trends more smoothly than single stocks. Try RVOL `z_threshold` 2.0–2.5 (3.0 may give too few events).
- For VDU, try `percentile` 10 and `base_atr_mult` 2.5. Gold consolidations are often tighter than equities.
- Avoid known event days if you can: US CPI, FOMC, NFP, and in India budget day and import-duty changes. These create volume spikes that aren't "institutional urgency".

### 6.4 Test checklist
- [ ] No look-ahead: signals use only data up to the signal bar's close; enter at that close or the next open.
- [ ] Compare against a **baseline** (average return over the same holding period on all days), not against zero.
- [ ] Report event count, average/median signed return, win rate, edge vs baseline, and t-test p-value.
- [ ] Run both `close` and `next_open` entries. The edge should survive next-open entry.
- [ ] Include costs (10 bps default here; use real MCX/ETF costs for gold).
- [ ] Check the year-by-year table so one year doesn't explain the whole result.
- [ ] With the filter on vs off, compare the gold strategy's CAGR, Sharpe, max drawdown, and trade count.
