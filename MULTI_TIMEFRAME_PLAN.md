# Multi-timeframe gold bot: plan

**Goal:** the bot looks at gold on several timeframes at once (for example Daily, 4h, 1h, 15m). Slower timeframes
decide the direction; faster ones decide when to enter. You pick the timeframes and every part of the strategy with
tick boxes, train (tune) it, and compare the results against the daily-only bot.

---

## 1. How the multi-horizon idea works

| Timeframe | Job | What it uses |
|---|---|---|
| Daily | **Direction** (bias) | Macro factors (real yield, DXY, Fed, CPI, flows) + daily technicals |
| 4h / 1h | **Setup** (confirmation) | Z-score, ATR expansion, RVOL volume spike |
| 15m | **Entry timing** | Z-score / breakout on the entry bar, volume spike, tighter stop |

Macro data changes once a day or once a month, so it can't create intraday signals. Faster timeframes only add
entry timing and tighter stops.

Two ways to combine timeframes (selectable):

* **Top-down filter (recommended):** trade on the fastest ticked timeframe, but only in the direction every
  slower ticked timeframe agrees with.
* **Weighted blend:** each timeframe gives a score from -1 to +1 and you give each one a weight (0-10), the same
  way the factors work now.

## 2. Tick boxes (everything optional)

One "Include in backtest" group in the sidebar:

* **Timeframes:** 15m, 1h, 4h, Daily (pick any set; the fastest one is the trading timeframe)
* **Factors:** the 8 current factors
* **Volume filters:** RVOL spike, VDU
* **Regime filters:** rule-based regime, HMM gate. These can be switched off completely for runs that don't need them.

## 3. Data: the main constraint

| Source | Timeframes | History | Volume | Cost |
|---|---|---|---|---|
| Yahoo GC=F (current) | 1h | last ~2 years only | futures volume | free |
| Yahoo GC=F | 15m | last 60 days only | futures volume | free |
| **Dukascopy XAUUSD (recommended)** | 1-minute, resampled to any timeframe | 2003 onward | tick volume only | free |
| Broker MT5 export (your broker) | M1 to D1 | broker-dependent, often 5-10 years | tick volume | free |
| Futures data vendor (e.g. Databento) | 1-minute GC | 2010 onward | real contract volume | paid |

* Yahoo intraday history is too short to train on.
* Plan: download Dukascopy 1-minute data once, store it locally, and build every timeframe from that one file so
  they stay consistent.
* Spot gold has no real volume, only tick counts. RVOL on tick volume is a rough proxy, and VDU on tick volume is
  unreliable. To trust the volume filters intraday you need futures data (paid) or GLD intraday data.

## 4. Build phases

**Phase A: intraday data layer**
Download, cache and clean 1-minute XAUUSD data. Resample to 15m, 1h, 4h and Daily. Handle the daily trading
break, weekends and holidays. Show a data-status table in the app.

**Phase B: engine that works on any timeframe**
* Lookbacks set in bars per timeframe.
* Sharpe and CAGR calculated from bars per year, not 252 days.
* Overnight financing charged by hours held.
* Optional "flat by end of session" and "no trading in the hour around CPI / FOMC / NFP".

**Phase C: features per timeframe, with no look-ahead**
* Technical and volume signals computed on each timeframe.
* A slower bar is only used after it has **closed**: at 10:15 the bot sees yesterday's daily bar, not today's
  unfinished one.
* Macro values are applied from their real release time (for example CPI at 8:30 New York time).

**Phase D: multi-horizon combiner**
Top-down filter and weighted-blend modes, timeframe tick boxes and weights, plus a column in the trade log showing
which timeframes agreed on each entry.

**Phase E: training (walk-forward)**
* Tune parameters on a past window, test on the next unseen window, roll forward, and only report the unseen results.
* Options: tune per timeframe or jointly. The HMM can be trained on any ticked timeframe.
* Report "does each timeframe earn its place": the same run with each timeframe removed one at a time.

**Phase F: UI and reports**
* Timeframe picker and training settings.
* Side-by-side table: daily-only vs multi-timeframe, net of charges.
* Results by timeframe, by hour of day and by year.

## 5. Risks to keep in mind

* **Charges:** faster timeframes mean more trades, and spread plus commission can wipe out the edge. Every
  comparison is shown net of charges.
* **Overfitting:** more timeframes and parameters make it easier to fit noise. Only walk-forward results count.
* **Stop fills:** the current engine checks stops on the bar's high and low. On 15m bars that is more accurate than
  on daily bars, but gaps over news and weekends still happen.
* **Speed:** 20 years of 15m bars is about 500k bars. Sweeps need a faster loop (numba) or caching.

## 6. Live section (paper trading on the Binance feed)

**Goal:** pick a strategy (the current sidebar settings or a saved one), start a session, and watch it trade the
live gold market bar by bar, using exactly the same code as the backtest.

| Item | Choice |
|---|---|
| Live feed | Binance public market data (no account or API key needed) |
| Markets | **PAXGUSDT** spot (tokenised gold, 1 token = 1 oz) or **XAUUSDT** perpetual futures (gold, Binance futures) |
| Bars | 15m bars from Binance, resampled to 1h / 4h / 1d. Only **closed** bars are used for decisions. |
| Daily layer | Macro factors + daily technicals from the existing daily data (Yahoo / Treasury / NY Fed / BLS) |
| Orders | **Paper only.** Signal on a bar's close, filled at the next bar's open, stop and target checked on each bar's high/low (same as the backtest) |
| State | `data/live/session.json` (strategy, market, start time). The result is recalculated from the start time on every refresh, so a restart or a closed browser loses nothing |
| Running | The Live tab refreshes itself every N seconds while open. `python live_runner.py` does the same in a terminal and logs every new signal and trade |

Why the same code matters: if the live session and the backtest disagree, one of them is wrong. Here the live
session *is* the backtest run on the bars since the start time, so the two can be compared directly.

**Later (not built now), only after paper results look right:** real orders through the Binance API on testnet
first, then live, with a hard daily loss limit and a kill switch.

## 7. Data used (decided)

* **Training / backtest history:** Binance PAXGUSDT 15m klines (Aug 2020 onward, real exchange volume), cached in
  `data/binance/`. XAUUSDT perpetual history is also available but only from Dec 2025.
* **Daily layer:** the existing daily data (gold futures from 2000 + macro), so the macro score has full warm-up.
* A daily bar is only used from 00:00 UTC on the next day; a 4h / 1h bar only after it has closed.

## 8. Build order

1. Engine works on any bar length (annualising, financing by hours). **Done**
2. Binance data layer: klines download, incremental cache, resampling, live ticker. **Done**
3. Multi-horizon features and combiner (top-down / weighted), timeframe ticks. **Done**
4. Walk-forward training on intraday data. **Done**
5. Saved strategies + Live (paper) tab + `live_runner.py`. **Done**
6. Later: event blackout (CPI / FOMC / NFP), end-of-day flat option, faster loop for big sweeps, real orders.

## 9. Defaults chosen (change any of them in the sidebar)

1. Data: Binance PAXGUSDT for intraday, existing daily data for the macro layer.
2. Timeframes: 15m, 1h, 4h, Daily, all ticked.
3. Combining: top-down filter (weighted blend selectable).
4. Trades may run overnight (gold on Binance trades 24/7).
5. Regime filter and HMM gate: tick boxes in "Include in backtest", off by default.
