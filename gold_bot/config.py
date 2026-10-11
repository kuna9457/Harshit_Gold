"""Single source of truth for the gold bot: data inputs, factors, tweakable parameters, cost presets.

The Streamlit front end builds its tables and widgets from the registries in this file, so
adding a factor or a parameter here is enough for it to show up in the UI and in the sweeps.
"""

# ---------------------------------------------------------
# 1. DATA CATEGORIES (how often the raw number actually changes)
# ---------------------------------------------------------
TICK = "Tick by tick"
DAILY = "Daily (end of day)"
MONTHLY = "Monthly (lagged release)"
CATEGORY_ORDER = [TICK, DAILY, MONTHLY]

# ---------------------------------------------------------
# 2. RAW DATA INPUTS
# ---------------------------------------------------------
# lag_days = business days between the date a value refers to and the date the market can see it.
# The backtest only ever uses a value after that delay (no lookahead).
DATA_INPUTS = {
    "gold": dict(
        label="Gold OHLCV (COMEX front month, GC=F)", category=TICK, source="Yahoo Finance",
        native="Tick / streaming", lag_days=0, proxy=True,
        note="Daily bars are used for the backtest. Free tick history for XAUUSD does not exist for "
             "20+ years, so the long run uses COMEX futures daily OHLCV as the XAUUSD price.",
    ),
    "real_yield": dict(
        label="US 10-Year Real Yield (TIPS par real curve)", category=TICK, source="US Treasury",
        native="Tick in the market, official fix once a day", lag_days=0, proxy=False,
        note="Official daily 10Y real yield. This replaces ^TNX, which is the NOMINAL 10Y yield.",
    ),
    "dxy": dict(
        label="US Dollar Index (DXY)", category=TICK, source="Yahoo Finance (DX-Y.NYB)",
        native="Tick / streaming", lag_days=0, proxy=False, note="",
    ),
    "usdinr": dict(
        label="USD/INR spot", category=TICK, source="Yahoo Finance (USDINR=X)",
        native="Tick / streaming", lag_days=0, proxy=False, note="",
    ),
    "effr": dict(
        label="Fed Funds Effective Rate (EFFR)", category=DAILY, source="New York Fed",
        native="Daily; policy target moves only at FOMC meetings (8 a year)", lag_days=1, proxy=False,
        note="Published the next morning, so it is used with a 1 day delay.",
    ),
    "sofr": dict(
        label="Repo rate (SOFR)", category=DAILY, source="New York Fed",
        native="Daily", lag_days=1, proxy=False,
        note="Starts April 2018. Before that the engine falls back to EFFR.",
    ),
    "tbill": dict(
        label="13-week T-bill yield (^IRX)", category=TICK, source="Yahoo Finance",
        native="Tick / streaming", lag_days=0, proxy=True,
        note="Market's short rate. Moves ahead of the Fed, so it is a faster proxy for policy.",
    ),
    "cpi": dict(
        label="US CPI, all items, seasonally adjusted", category=MONTHLY, source="BLS",
        native="Monthly, released about 2 weeks after month end", lag_days=None, proxy=False,
        note="Delay is set by the 'CPI publication lag' parameter (calendar days after the month starts).",
    ),
    "gold_etf": dict(
        label="Gold ETF activity and flow (GLD)", category=DAILY, source="Yahoo Finance (GLD OHLCV)",
        native="Daily (holdings are published once a day)", lag_days=0, proxy=True,
        note="PROXY: money flow from GLD price x volume. SPDR no longer serves the holdings history "
             "file. Drop real tonnage in data/overrides/gold_etf.csv (date,value) to replace it.",
    ),
    "bond_flow": dict(
        label="US bond inflow / outflow (TLT + IEF)", category=DAILY, source="Yahoo Finance (TLT, IEF OHLCV)",
        native="Official fund flows are weekly (ICI) and monthly (TIC)", lag_days=0, proxy=True,
        note="PROXY: money flow into Treasury ETFs from price x volume. Drop real flow data in "
             "data/overrides/bond_flow.csv (date,value) to replace it.",
    ),
}

# ---------------------------------------------------------
# 3. FACTORS (the Strategy column of the sheet). Each one gets a 0-10 score.
# ---------------------------------------------------------
# direction: +1 = factor up is bullish gold, -1 = factor up is bearish gold.
FACTORS = {
    "tech": dict(
        label="ATR + Volume + Z-score", inputs=["gold"], category=TICK, direction=+1, weight=8, window=60,
        logic="Direction from the z-score of gold's close. Full strength only when ATR is expanding "
              "and volume is above its baseline; each missing confirmation removes a third.",
    ),
    "real_yield": dict(
        label="US 10-Year Real Yield", inputs=["real_yield"], category=TICK, direction=-1, weight=8, window=60,
        logic="Normalised level of the 10Y real yield. Rising real yield = bearish gold.",
    ),
    "fed": dict(
        label="US Macro & Fed Repo / Funds Rate", inputs=["effr", "sofr", "tbill"], category=DAILY,
        direction=-1, weight=8, window=252,
        logic="Normalised level of the policy / repo rate. Hikes = bearish gold, cuts = bullish.",
    ),
    "dxy": dict(
        label="US Dollar Index (DXY)", inputs=["dxy"], category=TICK, direction=-1, weight=8, window=60,
        logic="Normalised DXY level. Strong dollar = bearish gold.",
    ),
    "usdinr": dict(
        label="USD/INR", inputs=["usdinr"], category=TICK, direction=+1, weight=2, window=60,
        logic="Normalised USD/INR level. Weak rupee lifts local (MCX) gold.",
    ),
    "cpi": dict(
        label="US CPI (Inflation)", inputs=["cpi"], category=MONTHLY, direction=+1, weight=8, window=252,
        logic="Year-on-year inflation rate, normalised. Rising inflation = bullish gold.",
    ),
    "bond_flow": dict(
        label="USA bond inflow / outflow", inputs=["bond_flow"], category=DAILY, direction=+1, weight=5, window=60,
        logic="Money flow into Treasuries over the flow lookback. Inflow pushes yields down = bullish gold.",
    ),
    "gold_etf": dict(
        label="Gold ETF activity and flow", inputs=["gold_etf"], category=DAILY, direction=+1, weight=5, window=60,
        logic="Money flow into GLD over the flow lookback. Inflow = investment demand = bullish gold.",
    ),
}

# ---------------------------------------------------------
# 4. TWEAKABLE PARAMETERS
# ---------------------------------------------------------
# kind: int / float / choice / bool.  sweep=True means it can be varied in the sweep tab.
def _p(group, label, kind, default, lo=None, hi=None, step=None, choices=None, sweep=True, help=""):
    return dict(group=group, label=label, kind=kind, default=default, lo=lo, hi=hi, step=step,
                choices=choices, sweep=sweep and kind in ("int", "float"), help=help)

G_WEIGHT, G_DIR, G_FEAT, G_TECH, G_MACRO = "Factor scores (0-10)", "Factor direction", "Feature engineering", "ATR / Volume / Z-score", "Macro and flow inputs"
G_SIGNAL, G_RISK, G_SIZE = "Entry and exit", "Risk management", "Position sizing"
GROUP_ORDER = [G_WEIGHT, G_DIR, G_FEAT, G_TECH, G_MACRO, G_SIGNAL, G_RISK, G_SIZE]

PARAMS = {}
for _k, _f in FACTORS.items():
    PARAMS[f"w_{_k}"] = _p(G_WEIGHT, _f["label"], "int", _f["weight"], 0, 10, 1,
                           help="0 switches the factor off. Scores are relative: final score = sum(score x signal) / sum(score).")
for _k, _f in FACTORS.items():
    PARAMS[f"dir_{_k}"] = _p(G_DIR, _f["label"], "choice", _f["direction"], choices=[1, -1],
                             help="+1: factor rising is bullish gold. -1: factor rising is bearish gold.")
for _k, _f in FACTORS.items():
    PARAMS[f"win_{_k}"] = _p(G_FEAT, f"Lookback: {_f['label']}", "int", _f["window"], 10, 756, 5,
                             help="Rolling window (trading days) used to normalise this factor.")

PARAMS.update({
    "norm_method": _p(G_FEAT, "Normalisation method", "choice", "zscore", choices=["zscore", "logmean", "minmax", "iqr"],
                      help="zscore: (x-mean)/std. logmean: log(x/mean)/std of logs. minmax: position in the rolling "
                           "range. iqr: (x-median)/IQR. All are rescaled so that about +/-2 is an extreme reading."),
    "signal_mode": _p(G_FEAT, "Signal shape", "choice", "continuous", choices=["continuous", "binary"],
                      help="continuous: signal grows with the reading, between -1 and +1. binary: +1 / -1 only (sheet 2 style)."),
    "z_clip": _p(G_FEAT, "Reading that counts as a full +/-1 signal", "float", 2.0, 0.5, 4.0, 0.25,
                 help="Continuous mode only. A normalised reading of this size maps to a signal of 1."),
    "buffer": _p(G_FEAT, "Neutral buffer", "float", 0.0, 0.0, 1.5, 0.05,
                 help="Readings smaller than this (in normalised units) give a signal of 0."),

    "atr_period": _p(G_TECH, "ATR period (days)", "int", 14, 5, 60, 1),
    "atr_baseline": _p(G_TECH, "ATR baseline (days)", "int", 30, 10, 120, 5,
                       help="ATR is 'expanding' when it is above its own average over this many days."),
    "vol_window": _p(G_TECH, "Volume baseline (days)", "int", 20, 5, 120, 5),
    "vol_mult": _p(G_TECH, "Volume expansion multiple", "float", 1.2, 0.8, 3.0, 0.1,
                   help="Volume confirms when it is above this multiple of its baseline."),

    "fed_series": _p(G_MACRO, "Fed / repo series", "choice", "effr", choices=["effr", "sofr", "tbill"],
                     help="effr: Fed Funds effective. sofr: repo rate (2018+, EFFR before). tbill: 13-week bill, market-led."),
    "cpi_lag_days": _p(G_MACRO, "CPI publication lag (calendar days)", "int", 45, 30, 75, 5,
                       help="A month's CPI is usable this many days after the month starts. Real release is around day 41-46."),
    "flow_window": _p(G_MACRO, "Flow lookback (days)", "int", 20, 5, 120, 5,
                      help="Window over which ETF and bond money flow is accumulated."),

    "entry_threshold": _p(G_SIGNAL, "Entry: final score at or above", "float", 0.20, 0.0, 0.9, 0.05,
                          help="Final score runs from -1 to +1. Go long when it reaches this level."),
    "exit_threshold": _p(G_SIGNAL, "Exit: final score at or below", "float", 0.0, -0.9, 0.9, 0.05),
    "allow_short": _p(G_SIGNAL, "Allow short trades", "bool", False,
                      help="Short when the final score is at or below minus the entry threshold."),
    "cooldown_bars": _p(G_SIGNAL, "Bars to wait after a stop or target", "int", 1, 0, 20, 1),

    "use_sl": _p(G_RISK, "Use stop loss", "bool", True),
    "sl_atr_mult": _p(G_RISK, "Stop loss = N x ATR", "float", 1.5, 0.5, 6.0, 0.25),
    "use_tp": _p(G_RISK, "Use take profit", "bool", True),
    "tp_rr": _p(G_RISK, "Take profit = N x stop distance", "float", 2.0, 0.5, 10.0, 0.25),

    "sizing_mode": _p(G_SIZE, "Sizing method", "choice", "risk", choices=["risk", "allocation"],
                      help="risk: Qty = risk per trade / stop distance. allocation: fixed share of equity (1.0 = same exposure as buy and hold)."),
    "risk_pct": _p(G_SIZE, "Risk per trade (% of equity)", "float", 2.0, 0.1, 10.0, 0.1),
    "alloc_pct": _p(G_SIZE, "Allocation (% of equity)", "float", 100.0, 10.0, 500.0, 10.0),
    "max_leverage": _p(G_SIZE, "Maximum leverage (x equity)", "float", 2.0, 0.5, 20.0, 0.5,
                       help="Caps position notional. Risk sizing with a tight stop can ask for more than this."),
    "scale_by_score": _p(G_SIZE, "Smaller quantity on weaker score", "bool", False,
                         help="Scales quantity by |final score| / 0.5, capped at 1. Sheet 2: 'Buy (less Qty)'."),
})

DEFAULT_PARAMS = {k: v["default"] for k, v in PARAMS.items()}

# ---------------------------------------------------------
# 5. CHARGES
# ---------------------------------------------------------
COST_FIELDS = {
    "spread": ("Bid-ask spread (price units, full spread)", 0.0, 5.0, 0.01),
    "slippage_bps": ("Slippage per side (bps of price)", 0.0, 20.0, 0.1),
    "commission_per_lot": ("Commission per lot per side", 0.0, 50.0, 0.5),
    "lot_size": ("Lot size (oz)", 1.0, 1000.0, 1.0),
    "pct_buy": ("Charges on buy side (% of turnover)", 0.0, 0.5, 0.0005),
    "pct_sell": ("Charges on sell side (% of turnover)", 0.0, 0.5, 0.0005),
    "fixed_per_order": ("Fixed charge per order", 0.0, 50.0, 0.05),
    "financing_long_pct": ("Overnight financing, long (% a year)", 0.0, 15.0, 0.25),
    "financing_short_pct": ("Overnight financing, short (% a year)", -10.0, 15.0, 0.25),
}

_ZERO = dict(spread=0.0, slippage_bps=0.0, commission_per_lot=0.0, lot_size=100.0, pct_buy=0.0,
             pct_sell=0.0, fixed_per_order=0.0, financing_long_pct=0.0, financing_short_pct=0.0)

# These are typical retail levels, not quotes. Check them against the broker's contract note.
COST_PRESETS = {
    "XAUUSD CFD, raw spread + commission": {**_ZERO, "spread": 0.15, "slippage_bps": 0.5, "commission_per_lot": 3.5,
                                             "financing_long_pct": 6.0, "financing_short_pct": -2.0},
    "XAUUSD CFD, spread only": {**_ZERO, "spread": 0.35, "slippage_bps": 0.5,
                                "financing_long_pct": 6.0, "financing_short_pct": -2.0},
    # MCX: brokerage ~Rs 20 an order, CTT 0.01% on sell, exchange + SEBI + stamp duty + 18% GST on brokerage.
    "MCX gold futures (India), approximate": {**_ZERO, "spread": 0.10, "slippage_bps": 1.0, "pct_buy": 0.0050,
                                              "pct_sell": 0.0130, "fixed_per_order": 0.28},
    "No charges (gross)": dict(_ZERO),
}
DEFAULT_COST_PRESET = "XAUUSD CFD, raw spread + commission"
