# Regime and risk layer for the gold bot: phase plan

**Goal:** stop the bot from buying volatile bounces in falling or choppy markets, and lose less often and by less
when it is wrong. The strategy still decides *what* and *when*; the regime layer decides *whether, and how big*.

## Why it is needed (findings from the original strategy, default settings, daily bars, CFD charges)

| Market type on the signal bar | Trades | Win rate | Sum of R |
|---|---|---|---|
| Uptrend, normal volatility | 143 | 53% | **+73.9R** |
| Uptrend, high volatility | 53 | 30% | -2.5R |
| Mixed / choppy | 26 | 19% | **-12.6R** |
| Downtrend (incl. high vol) | 21 | 48% | +8.4R |

* Almost all profit comes from uptrends with normal volatility.
* Macro factors (Fed, bonds, real yield) can outvote price (the technical factor is 8 of 52 weight points), so the
  score turns bullish during liquidity sell-offs such as 2008 and March 2020.
* After two losses in a row the next trade averages 0.10R, against 0.32R otherwise.

---

## Phase 1: rule-based regime filter (DONE)

Where: `gold_bot/features.py` (`add_regime`, `regime_gate`), `gold_bot/engine.py`, sidebar group
**Regime filter (Phase 1, rule based)**, tab **Regime lab**.

Market type on each bar (data up to that close only):

| Label | Rule |
|---|---|
| Uptrend | Close above the 200-day average and the average higher than 20 days ago |
| Downtrend | Close below the 200-day average and the average lower than 20 days ago |
| Mixed | Anything else |
| High volatility | 20-day volatility above the 80th percentile of the last 3 years |
| Price confirms up | Close above the 20-day EMA and the last 5-day low above the 5-day low before it |

Entry gate (size multiplier, 0 = blocked):

| Situation | Multiplier (default) |
|---|---|
| Trading with the trend | 1.0 |
| Mixed market | 0.5, and the score must be at least 0.40 |
| Against the trend (e.g. buying a bounce in a downtrend) | blocked unless price confirms, then 0.25 |
| High volatility | x 0.5 on top of the above |
| Optional price veto | blocks a long when the technical factor is negative (off by default) |

Result, full history:

| | Original | With rule filter |
|---|---|---|
| CAGR | 4.71% | 5.36% |
| Max drawdown | -20.2% | -15.6% |
| Sharpe | 0.11 | 0.20 |
| Profit factor | 1.42 | 1.79 |
| Losing trades | 138 | 117 |
| Longest losing streak | 8 | 6 |

## Phase 2a: HMM alone (DONE)

Where: `gold_bot/regime_hmm.py`, tab **Regime lab**, button **Train HMM**.

* Gaussian HMM (`hmmlearn`), 2-4 states, inputs chosen from daily return, 5-day return, 20-day return,
  20-day volatility and daily range.
* Walk-forward: first fit on the first N years (default 5), then refitted every 252 trading days on past data only
  (expanding or rolling window). Inputs are standardised with training data only.
* Each day's state comes from the **forward filter** (data up to that day). Smoothed probabilities, which use later
  bars, are only used inside the training window.
* States are sorted by volatility after every fit, so S0 is always the calmest (no label switching).
* HMM-only rules: long when expected next-day return > 0 / long unless in the most volatile state / long only in
  the calmest state. Fully invested while on, no stop, same charges.

Result, 2005-2026 (same period for all):

| | Original | With rule filter | HMM only | Buy and hold |
|---|---|---|---|---|
| CAGR | 5.57% | 6.03% | 3.48% | 10.80% |
| Max drawdown | -19.6% | **-13.9%** | -53.4% | -44.4% |
| Sharpe | 0.21 | **0.30** | 0.02 | 0.43 |
| Calmar | 0.28 | **0.43** | 0.07 | 0.24 |

Original strategy trades grouped by HMM state: Calm +0.42R, Normal +0.40R, **Volatile +0.08R** (55% stopped out).

**Reading:** HMM alone is not a trading strategy (deep drawdowns, late switches). As a label it does separate bad
trades (the Volatile state), much like the rule-based high-volatility flag. Phase 2b tests whether it adds anything
on top of the rules.

---

## Phase 2b: combine HMM with the strategy (DONE)

Where: `features.add_hmm`, the HMM gate in `engine._simulate`, sidebar group **HMM gate (Phase 2b)**, Regime lab
section **Phase 2b**.

* Entry size x HMM multiplier (calmest 1.0, middle 1.0, most volatile 0.5), read either probability-weighted or from
  the most likely state. Optional veto: no longs when the HMM expected return is below 0.
* Four runs compared on the same period: original / rules / HMM gate / rules + HMM, plus each half of the period.

Result (2005-2026):

| | Original | Rules | HMM gate | Rules + HMM |
|---|---|---|---|---|
| CAGR | 5.57% | 6.03% | 5.39% | **6.12%** |
| Max drawdown | -19.6% | -13.9% | -16.5% | **-12.2%** |
| Calmar | 0.28 | 0.43 | 0.33 | **0.50** |
| Profit factor | 1.47 | 1.92 | 1.57 | **2.18** |

| Calmar by half | Original | Rules | HMM gate | Rules + HMM |
|---|---|---|---|---|
| 2005-2016 | 0.36 | 0.41 | 0.39 | **0.46** |
| 2016-2026 | 0.27 | 0.64 | 0.28 | **0.75** |

Volatile-state multiplier 0 / 0.25 / 0.5 / 0.75 gives Calmar 0.52 / 0.51 / 0.50 / 0.48: the result does not hinge on
the exact value. In 2005-2016 both gates cost CAGR (7.0% to about 5.7%) in exchange for lower drawdown.

**Reading:** rules + HMM is the best of the four on risk-adjusted numbers in both halves. The HMM gate alone adds
little; on top of the rules it trims drawdown further.

## Phase 3: sizing by volatility and losing streaks

* Volatility targeting: quantity x (target volatility / current volatility), capped at 1.
* Losing-streak brake: after 2 losses in a row, half risk until the next win.
* Daily / weekly loss limit (for intraday): stop trading for the day after -2R.

## Phase 4: trade management

* Move the stop to breakeven at +1R.
* Time stop: exit if the trade has not reached +0.5R within N bars.
* Stop overruns (losses worse than -1.1R): avoid entries just before CPI, FOMC and payrolls; size on the expected
  worst fill rather than the stop price.

## Phase 5: crisis detector

* Needs new data: VIX, S&P 500.
* Crisis = VIX above 30 or up more than 20% in a day, or gold and S&P falling together for 3+ days, or gold more
  than 3 x ATR below its 20-day high. No new trades; tighten stops on open trades.

## Phase 6: meta-labeling (optional)

* A second model (gradient boosting) predicts the chance each signal hits its target before its stop, from regime,
  volatility, drawdown, score and losing streak. That chance sets the size.

## Phase 7: intraday

* The issue was seen intraday, and the backtest is daily. Load 15m or 1h XAUUSD from the broker, keep the regime on
  daily or 4h bars and take entries on intraday bars.

---

## How to judge every phase

Compare against the previous phase on the same period and charges, in the **Regime lab** tab:
losing trades, average loss (R), stop overruns, longest losing streak, max drawdown, Calmar, profit factor and CAGR.
Use round-number thresholds and check them on a period they were not tuned on (Test period slider), so the
filter is not just fitted to the past.
