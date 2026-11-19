# Gold (XAUUSD) macro strategy backtester: user guide

A Streamlit app that backtests a gold strategy built from 8 macro and technical factors, with a safety layer that
reads the market type (regime) before every trade.

```
pip install -r requirements.txt
streamlit run app.py
```

The phase-by-phase plan and the results of each phase are in [REGIME_PLAN.md](REGIME_PLAN.md).

---

## 1. How the bot makes a decision

```
raw data ──► 8 factor signals (-1..+1) ──► final score (-1..+1) ──► entry? ──► Phase 1 rule gate ──► Phase 2b HMM gate ──► quantity
                                                                                (market type)          (HMM state)
```

1. **Factors.** Each factor turns one input into a signal between -1 (bearish gold) and +1 (bullish gold).
2. **Final score** = weighted average of the signals, using the 0-10 scores in the sidebar.
3. **Entry.** Long when the score is at or above the entry threshold (default 0.20). Exit when it drops to the exit
   threshold (default 0), or at the stop loss / take profit.
4. **Gates (optional).** The rule filter and the HMM gate can block a trade or shrink its size. They never open a trade.
5. **Execution.** Decided on the daily close, filled at the next day's open, with spread, slippage, commission and
   overnight financing.

### Inputs

| Input | Source | Used for |
|---|---|---|
| Gold OHLCV (COMEX GC=F) | Yahoo | Price, ATR, volume, regime, HMM |
| US 10Y real yield | US Treasury | Real yield factor |
| Dollar index (DXY) | Yahoo | DXY factor |
| USD/INR | Yahoo | Rupee factor |
| EFFR / SOFR / 13-week T-bill | NY Fed / Yahoo | Fed and repo rate factor |
| US CPI | BLS | Inflation factor (used about 45 days after the month starts) |
| GLD price x volume | Yahoo | Gold ETF flow (proxy) |
| TLT + IEF price x volume | Yahoo | Bond flow (proxy) |

Every input is only used after its publication delay, so the backtest never sees a number before the market did.

---

## 2. Phase 1: rule-based regime filter

**Sidebar → Regime filter (Phase 1, rule based) → Use regime filter.** Nothing is trained; it is a fixed set of rules.

### What it calculates on every bar

| Column | Meaning |
|---|---|
| `regime` | **Uptrend**: close above the 200-day average and the average rising over 20 days. **Downtrend**: the opposite. **Mixed**: anything else. **Not enough history**: the first 200 days. |
| `rg_hivol` | True when 20-day volatility is above the 80th percentile of the last 3 years |
| `rg_confirm_up` / `rg_confirm_dn` | Price has turned: close above (below) the 20-day EMA and a higher low (lower high) over the last 5 days |
| `regime_label` | `regime` plus "+ high vol" when volatile, shown in the tables |

### What it does to a trade

| Situation | Size multiplier (default) |
|---|---|
| Trading with the trend (long in an uptrend) | 1.0 |
| Mixed market | 0.5, and the score must reach 0.40 |
| Against the trend (buying a bounce in a downtrend) | blocked, unless price confirms; then 0.25 |
| High volatility | x 0.5 on top |
| Price veto (optional, off) | blocks a long when the technical factor is negative |

Size multiplier 0 = trade blocked. 0.5 = half the normal quantity.

---

## 3. Phase 2a: training the HMM

**Regime lab tab → Phase 2a → Train HMM.** About 20-30 seconds with the defaults.

### What an HMM is

A Hidden Markov Model assumes the market is always in one of a few hidden "states" (for example calm, normal,
volatile), each with its own typical return and volatility, and that it tends to stay in a state for a while before
switching. Training finds those states from price behaviour alone; it does not look at macro data or at the
strategy's trades.

### How training works (and why the results are honest)

1. The first model is trained on the first N years (default 5, so states start in 2005).
2. It is used for the next 252 trading days (one year), then retrained on all data up to that point. This repeats to
   today (about 21 refits). This is called **walk-forward** training.
3. Each day's state is read with the **forward filter**: probability of each state given data up to that day only.
4. Inputs are standardised using the training data only.
5. After each refit, states are sorted by volatility so **S0 is always the calmest** and the last state the most
   volatile. Without this, the numbers would swap between refits.

### Training settings

| Setting | Default | What it changes |
|---|---|---|
| Model inputs | Daily return, 20-day return, 20-day volatility | What the HMM looks at. Volatility inputs give calm/volatile states; return inputs give up/down states. |
| Number of states | 3 | 2 = calm / volatile. 3 = calm / normal / volatile. 4 adds "active". More states fit more noise. |
| First training period | 5 years | More years = more stable first model, later start of the test. |
| Refit every | 252 days | Shorter = adapts faster, more noise, slower training. |
| Training window | Expanding | Expanding uses all history so far. Rolling uses only the last N years (forgets old markets). |
| Random restarts | 2 | Each fit starts from a few random points and keeps the best. More = more reliable, slower. |
| HMM-only trading rule | Expected return > 0 | Only for the HMM-only backtest (see below). |

### HMM input features

| Feature | Formula | Captures |
|---|---|---|
| Daily return | log(close / previous close) | Day-to-day moves |
| 5-day return | log(close / close 5 days ago) | Short swings |
| 20-day return | log(close / close 20 days ago) | One-month trend |
| 20-day volatility | log of the std of daily returns over 20 days | How wild the market is |
| Daily range | log(today's high-low range / its 20-day average) | Intraday stress |

### What training outputs

| Output | Meaning |
|---|---|
| `probs` | Probability of each state on each day (they add up to 1). Shown as the stacked area chart. |
| `state` | The most likely state on each day. Shown as the coloured gold chart. |
| `exp_ret` | The model's expected next-day return: tomorrow's state probabilities x each state's average return. |
| Refit log | For every refit: training dates, log likelihood (fit quality; only comparable between fits on similar data), each state's volatility, annualised return and "stay probability" (chance of being in the same state tomorrow; 0.98 means it lasts about 50 days on average). |
| State table | Out of sample: share of days in each state, gold's return and volatility the day after, % up days, average stay. A useful state has clearly different numbers from the others. |

### HMM-only backtest

The HMM used as a strategy by itself: fully invested (1x) while the rule says "on", flat otherwise (or short if
ticked), no stop or target, same charges. Measures in R are blank for it because it has no stop.

Rules: **expected return > 0**, **avoid the most volatile state**, or **only the calmest state**.

---

## 4. Phase 2b: combining the HMM with the strategy

**Sidebar → HMM gate (Phase 2b) → Use HMM gate.** Needs a trained HMM (Phase 2a) in the same session.

| Setting | Default | Meaning |
|---|---|---|
| How the state is read | probability | **probability**: multiplier = sum of (state probability x state multiplier), changes smoothly. **most_likely**: the most likely state's multiplier only. |
| Size in calmest state | 1.0 | |
| Size in middle state(s) | 1.0 | |
| Size in most volatile state | 0.5 | 0 = no new trades in that state |
| Block longs when expected return < 0 | off | Extra veto from the HMM's expected return |

Final quantity = normal quantity x rule multiplier x HMM multiplier. Before the first HMM training period ends the
HMM gate has no effect, so compare runs on the HMM period (the Regime lab does this for you).

**Use rules + HMM in the sidebar** (bottom of the Regime lab) turns both gates on, so the Backtest and Parameter sweep
tabs use the combined setup.

---

## 5. Reading the results

### Comparison table rows

| Row | Meaning | Better is |
|---|---|---|
| CAGR % | Yearly growth rate, after charges | Higher |
| Max drawdown % | Largest fall from a peak | Closer to 0 |
| Sharpe | Return per unit of volatility, above the risk-free rate | Higher |
| Calmar | CAGR / max drawdown | Higher |
| Profit factor | Total won / total lost | Higher (above 1.5 is good) |
| Win rate % | Share of trades that made more than 0.1R | Higher |
| Losing trades | Trades that lost more than 0.1R | Lower |
| Avg loss (R) | Average losing trade, in multiples of the risk taken | Closer to 0 |
| Stop overruns (< -1.1R) | Losses that went past the stop (gaps, costs) | Lower |
| R lost beyond the stop | Extra R lost by those overruns | Closer to 0 |
| Longest losing streak | Most losses in a row | Lower |
| Worst trade (R) | Single worst trade | Closer to 0 |
| Sum of R | Total of all trades in R | Higher |
| Time in market % | Share of days with a position | |
| Total charges | All costs paid | Lower |

**R** = profit or loss divided by the money risked on that trade (entry to stop). -1R = a normal stop.
R is measured against each trade's own (possibly reduced) size, so a gate that only shrinks size does not change
R or trade counts; it shows up in drawdown, P&L and profit factor instead.

### Group tables (by market type / HMM state)

Each row is the trades opened in that market type or state, measured on the signal bar. A group with a low win rate
and negative "Sum of R" is a place the gate should block or shrink. If a gate's groups look alike, it is not
separating good trades from bad ones.

### "Does it hold up?" (Phase 2b)

The same four runs on each half of the period. Keep a setting only if it helps in both halves; one that helps in one
half only is probably fitted to that half.

---

## 6. Results so far (default settings, CFD charges, 2005-2026)

| | Original | Rules | HMM gate | Rules + HMM |
|---|---|---|---|---|
| CAGR | 5.57% | 6.03% | 5.39% | **6.12%** |
| Max drawdown | -19.6% | -13.9% | -16.5% | **-12.2%** |
| Calmar | 0.28 | 0.43 | 0.33 | **0.50** |
| Profit factor | 1.47 | 1.92 | 1.57 | **2.18** |
| HMM only (switch) | CAGR 3.48%, max drawdown -53.4% | | | |

* 1st half (2005-2016): both gates lower CAGR (7.0% to about 5.7%) but cut drawdown and raise Calmar.
* 2nd half (2016-2026): both gates improve every number.
* The HMM alone is a poor strategy, but its Volatile state finds weak trades (+0.01R a trade against about +0.6R
  in the other states), which is why it helps as a gate.

---

## 7. Suggested workflow

1. **Backtest tab:** run the original strategy and look at "trades by market type".
2. **Regime lab → Phase 1:** run the rule comparison and tune the regime filter in the sidebar.
3. **Regime lab → Phase 2a:** train the HMM; check that the states differ in the state table and that the
   Volatile state's trades are worse in "Does the HMM separate...".
4. **Regime lab → Phase 2b:** compare the four ways and check both halves.
5. If the combined setup holds up, press **Use rules + HMM in the sidebar** and sweep its multipliers in the Parameter
   sweep tab. Check the best result on a Test period it was not tuned on.

## 8. Limits to keep in mind

* Daily bars only. Intraday bounces need intraday data (Phase 7 in the plan).
* The trained HMM lives in the browser session; retrain after reloading the page.
* Gold ETF and bond flows are price x volume proxies, not real fund flows.
* Past results do not promise future ones; prefer settings that work across both halves over the single best run.

## 9. Files

| File | What it holds |
|---|---|
| `app.py` | The Streamlit front end (tabs, charts, tables) |
| `gold_bot/config.py` | Inputs, factors, every sidebar parameter, cost presets |
| `gold_bot/data.py` | Downloads and caches the data in `data/cache` |
| `gold_bot/features.py` | Factor signals, final score, Phase 1 regime columns, Phase 2b HMM columns |
| `gold_bot/regime_hmm.py` | Phase 2a HMM: features, walk-forward training, forward filter, HMM-only signal, state summary |
| `gold_bot/engine.py` | The backtest: fills, stops, sizing, both gates, charges, metrics, sweeps |
| `REGIME_PLAN.md` | Phase plan and results |

## Volume filter (RVOL z-score spike + volume dry-up)

Logic: `VOLUME_SIGNALS_LOGIC.md`. Code: `gold_bot/features.py` (`add_volume`), gate in `gold_bot/engine.py`.
Sidebar: tick boxes in **Include in backtest**, settings in **Volume filter**. Backtest tab: **Run volume filter comparison**.

* Volume series: GLD ETF volume by default (no futures roll effect, from Nov 2004); GC=F volume optional.
* **RVOL spike**: volume z-score vs the previous 20 days >= 2.5. `block_against` (default) skips a long if a spike day in the
  last 3 bars closed down (a short if it closed up). `confirm` instead requires a spike in the trade's direction.
* **VDU**: volume below the 10th percentile of the previous 50 days inside a tight base (10-day range <= 2.5 x ATR 20).
  Longs need a dry-up within the last 20 bars.
* The filter only removes entries; exits are unchanged. Before volume history exists it does nothing.

Backtest, 2005 onwards, CFD charges, all other settings default:

| | CAGR | Max DD | Profit factor | Trades |
|---|---|---|---|---|
| No volume filter | 5.61% | -19.6% | 1.48 | 197 |
| RVOL block_against | 5.58% | -18.9% | 1.54 | 175 |
| VDU only | 1.46% | -21.0% | 1.46 | 61 |
| No volume filter, regime filter ON | 6.05% | -14.3% | 1.94 | 173 |
| RVOL block_against, regime filter ON | 6.11% | -13.8% | 2.10 | 152 |

RVOL `block_against` gives fewer, better trades in both halves of the period at about the same CAGR. VDU and RVOL `confirm`
remove most trades and cut CAGR to 1-2%.

## Multi-timeframe bot and Live (paper) trading on Binance

Plan and design: `MULTI_TIMEFRAME_PLAN.md`. Code: `gold_bot/binance.py` (data + live price), `gold_bot/mtf.py`
(timeframes, combiner, walk-forward training), `gold_bot/live.py` (paper sessions, saved strategies), `live_runner.py`.

* **Data:** Binance 15m klines, cached in `data/binance/` (first download ~5 minutes). PAXGUSDT spot from Aug 2020,
  XAUUSDT perpetual from Dec 2025. 1h / 4h / 1d are built from the 15m bars. The daily layer uses the existing daily data.
* **Tick boxes** (sidebar, "Include in backtest"): timeframes 15m / 1h / 4h / Daily, factors, volume filters, regime
  rules, HMM gate. The fastest ticked timeframe is the one the bot trades on.
* **No look-ahead:** a 1h / 4h bar is used only after it closes; a daily bar dated D from 00:00 UTC on D+1.
* **Multi-timeframe tab:** backtest, "does each timeframe add value", walk-forward training, save as a strategy.
* **Live (paper) tab:** pick a strategy and market, press Start. Every refresh re-runs the same engine from the start
  time to the last closed bar. `python live_runner.py` does the same in a terminal and logs to `data/live/events.csv`.
  Nothing is sent to Binance. Binance blocks US-hosted servers (HTTP 451), so deploy outside the US.

PAXGUSDT, 2021-09 to 2026-11, Binance spot charges (0.1% a side), default settings. Buy and hold: 16.1% a year.

| Ticked timeframes | CAGR net | CAGR gross | Max DD | Trades |
|---|---|---|---|---|
| 15m + 1h + 4h + 1d (all) | -73.9% | -7.4% | -99.9% | 1341 |
| 1h + 4h + 1d | -28.0% | 3.7% | -81.9% | 388 |
| 4h + 1d (VDU off) | -3.5% | 19.3% | -38.0% | 248 |
| 1d only (VDU off) | 6.8% | 9.2% | -14.2% | 53 |

The default intraday signal (z-score momentum) loses before charges on 15m and 1h; flipped (mean reversion) it makes a
small gross profit on 15m that charges remove. 4h + 1d beats buy and hold before charges, but 0.1% spot fees wipe it out.

## Live tick-by-tick feed

Code: `gold_bot/ticks.py` (websocket), `gold_bot/live.py` (`with_forming`). Read-only market data, no API key.

* Streams best bid / ask (`bookTicker`) and trades (`aggTrade`) for PAXGUSDT (spot) or XAUUSDT (perpetual) in a background
  thread; it reconnects by itself. The live price is the mid of bid and ask. Spot gold trades only every few seconds, so
  the trade tape is sparse while quotes update many times a second.
* The 15m bar in progress is built from the ticks (open seeded from REST) and added as the last row of the trading
  timeframe. Orders decided on the last closed bar fill at the real open of the new bar, and stops / targets are checked
  on the live high / low. The new row has no score, so a half-built bar never opens a trade.
* Signals are rebuilt only when a new 15m bar closes, so the tick panel refreshes every second and the strategy panel
  every 2-120 seconds. If the websocket is down it falls back to the REST price and closed bars only.
* "Record ticks to disk" (Live tab, or `python live_runner.py --record`) saves trades and 1-second quote snapshots to
  `data/ticks/<symbol>_<date>.csv`.
