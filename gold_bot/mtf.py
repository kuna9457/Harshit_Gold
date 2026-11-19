"""Multi-horizon gold bot on Binance bars.

Each ticked timeframe gives a signal from -1 to +1:
  15m / 1h / 4h  technical signal on that timeframe's bars (z-score of close, confirmed by ATR and volume expansion)
  1d             the daily strategy's final score (macro factors + daily technicals, from the daily data)
The signals are moved onto the fastest ticked timeframe (the trading timeframe) using only bars that had CLOSED
by then, and combined top-down or weighted. The result goes through the same engine as the daily backtest.
"""
import numpy as np
import pandas as pd

from . import binance, engine
from .config import DEFAULT_PARAMS, TIMEFRAMES
from .features import add_hmm, add_regime, add_volume, build_features, normalise, to_signal

MIN_PER_YEAR = 365 * 24 * 60  # Binance gold trades around the clock, every day


def ticked(p):
    """Ticked timeframes, fastest first."""
    tfs = [k for k in TIMEFRAMES if p.get(f"inc_tf_{k}", True)]
    return sorted(tfs, key=lambda k: TIMEFRAMES[k]["minutes"])


def bars_per_year(tf):
    return MIN_PER_YEAR / TIMEFRAMES[tf]["minutes"]


def _atr(df, n):
    pc = df["Close"].shift(1)
    tr = np.maximum(df["High"] - df["Low"], np.maximum((df["High"] - pc).abs(), (df["Low"] - pc).abs()))
    return tr.rolling(int(n)).mean()


def tech_signal(df, p, tf=None):
    """Same recipe as the daily technical factor, measured in bars of this timeframe.
    With the 'reversion' style the sign is flipped: a stretched price is expected to snap back."""
    z = normalise(df["Close"], int(p["mtf_win"]), p["norm_method"])
    atr = _atr(df, p["atr_period"])
    atr_up = (atr > atr.rolling(int(p["atr_baseline"])).mean()).astype(float)
    vol_up = (df["Volume"] > p["vol_mult"] * df["Volume"].rolling(int(p["vol_window"])).mean()).astype(float)
    shape = dict(mode=p["signal_mode"], z_clip=p["z_clip"], buffer=p["buffer"])
    flip = -1 if p.get(f"mtf_style_{tf}") == "reversion" else 1
    return flip * p["dir_tech"] * to_signal(z, **shape) * (1 + atr_up + vol_up) / 3


def on_base(values, known_at, base_decide_at):
    """Value of a slower series on each trading bar: the last value whose `known_at` time is at or before the
    moment the trading bar closes (`base_decide_at`). No forward look: an unfinished slower bar is never used."""
    order = np.argsort(known_at.values, kind="stable")
    kt, vals = known_at.values[order], np.asarray(values)[order]
    pos = np.searchsorted(kt, base_decide_at.values, side="right") - 1
    out = np.where(pos >= 0, vals[np.clip(pos, 0, None)], None)
    try:
        return out.astype(float)
    except (TypeError, ValueError):
        return out


def daily_layer(gold, raw, p, hmm=None):
    """Daily score from the daily data. A daily bar dated D is used from 00:00 UTC on D+1 (gold futures close
    around 21:00-22:00 UTC). Today's unfinished daily bar is dropped."""
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    feat = build_features(gold[gold.index < today], raw, p)
    if hmm is not None:
        feat = add_hmm(feat, hmm, p)
    return feat


def build(bars15, gold, raw, params=None, hmm=None):
    """Trading-timeframe frame with OHLCV, ATR, sig_<tf> for every ticked timeframe and the combined `score`,
    plus the regime and volume columns the engine reads. Returns (frame, trading timeframe)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    tfs = ticked(p)
    if not tfs:
        raise ValueError("No timeframe is ticked. Tick at least one in 'Include in backtest'.")
    base_tf = tfs[0]
    base = binance.resample(bars15, base_tf).copy()
    step = pd.Timedelta(minutes=TIMEFRAMES[base_tf]["minutes"])
    decide_at = pd.Series(base.index + step, index=base.index)  # a trading bar is acted on when it closes
    base["ATR"] = _atr(base, p["atr_period"])

    daily = None
    for tf in tfs:
        if tf == "1d":
            daily = daily_layer(gold, raw, p, hmm)
            known = pd.Series(daily.index + pd.Timedelta(days=1), index=daily.index)
            base["sig_1d"] = on_base(daily["score"], known, decide_at)
        elif tf == base_tf:
            base[f"sig_{tf}"] = tech_signal(base, p, tf)
        else:
            b = binance.resample(bars15, tf)
            known = pd.Series(b.index + pd.Timedelta(minutes=TIMEFRAMES[tf]["minutes"]), index=b.index)
            base[f"sig_{tf}"] = on_base(tech_signal(b, p, tf), known, decide_at)

    sig = base[[f"sig_{tf}" for tf in tfs]]
    if p["mtf_mode"] == "weighted":
        w = np.array([float(p[f"w_tf_{tf}"]) for tf in tfs])
        if w.sum() <= 0:
            raise ValueError("Every ticked timeframe has a weight of 0.")
        num = (sig.fillna(0) * w).sum(axis=1)
        den = (sig.notna() * w).sum(axis=1)
        score = (num / den.where(den > 0)).where(sig[f"sig_{base_tf}"].notna())
    else:
        lead = sig[f"sig_{base_tf}"]
        agree = pd.Series(True, index=base.index)
        for tf in tfs[1:]:
            s = sig[f"sig_{tf}"]
            ok = (np.sign(s) == np.sign(lead)) & (s.abs() >= p["mtf_agree_min"])
            agree &= ok | s.isna()  # a timeframe with no history yet does not block
        score = lead.where(agree, 0.0).where(lead.notna())
    base["score"] = score
    base["tf_agree"] = (np.sign(sig).nunique(axis=1) == 1) & sig.notna().all(axis=1)

    base = add_regime(base, p)  # rule-based regime in bars of the trading timeframe (only used if ticked)
    base = add_volume(base, {}, {**p, "vol_source": "own"})  # the exchange's own volume on the trading bars
    if hmm is not None and daily is not None and "hmm_mult" in daily:
        known = pd.Series(daily.index + pd.Timedelta(days=1), index=daily.index)
        for col in ("hmm_mult", "hmm_exp_ret", "hmm_state"):
            base[col] = on_base(daily[col], known, decide_at)
    return base, base_tf


def run(bars15, gold, raw, params=None, costs=None, capital=100_000, start=None, end=None, rf=0.04, hmm=None,
        feat=None, base_tf=None, **kw):
    """Backtest the multi-horizon bot. Same fills, stops, sizing and charges as the daily engine."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if feat is None:
        feat, base_tf = build(bars15, gold, raw, p, hmm)
    p_run = {**p, "use_hmm_gate": p["use_hmm_gate"] and "hmm_mult" in feat}
    res = engine.run_backtest(None, None, p_run, costs, capital, start, end, rf, feat=feat,
                              ppy=bars_per_year(base_tf), **kw)
    res["base_tf"] = base_tf
    return res


# ---------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------
def _random_sets(ranges, n, seed):
    rng = np.random.default_rng(seed)
    pools = {}
    for k, (lo, hi, step, kind) in ranges.items():
        vals = np.arange(lo, hi + step / 2, step)
        pools[k] = [int(round(v)) if kind == "int" else float(round(v, 6)) for v in vals]
    return [{k: pool[rng.integers(len(pool))] for k, pool in pools.items()} for _ in range(n)]


SCORERS = {
    "Calmar": lambda m, x: m["Calmar"],
    "Sharpe": lambda m, x: m["Sharpe"],
    "CAGR %": lambda m, x: m["CAGR %"],
    "Profit factor": lambda m, x: min(x["Profit factor"], 10.0),
}


def walk_forward(bars15, gold, raw, params, ranges, n_sets=30, train_days=365, test_days=90, score_by="Calmar",
                 min_trades=10, costs=None, capital=100_000, rf=0.04, hmm=None, seed=0, progress=None):
    """Tune on a past window, test on the next unseen window, roll forward.

    For every fold the parameter set with the best `score_by` on the training window (with at least `min_trades`
    trades) is run on the test window. Only the test windows are joined into the out-of-sample result.
    The sidebar settings (no tuning) are run on the same test windows for comparison.
    Indicators only look back, so features are built once per parameter set on the full history and then cut.
    """
    base_p = {**DEFAULT_PARAMS, **params}
    sets = [{}] + _random_sets(ranges, n_sets, seed)  # set 0 = the sidebar settings as they are
    feat0, base_tf = build(bars15, gold, raw, base_p, hmm)
    first = feat0["score"].first_valid_index()
    if first is None:
        raise ValueError("No signal: not enough history for the lookbacks.")
    t0, t_end = first + pd.Timedelta(days=train_days), feat0.index[-1]
    folds = []
    while t0 + pd.Timedelta(days=min(test_days, 7)) < t_end:
        folds.append((t0 - pd.Timedelta(days=train_days), t0, min(t0 + pd.Timedelta(days=test_days), t_end)))
        t0 += pd.Timedelta(days=test_days)
    if not folds:
        raise ValueError(f"Not enough history: need more than {train_days} days after the warm-up.")

    kw = dict(costs=costs, capital=capital, rf=rf, with_gross=False, min_bars=10)
    # the training windows of every fold, plus the latest window ending today (its winner is the one to use live)
    windows = [(a, b) for a, b, _ in folds] + [(t_end - pd.Timedelta(days=train_days), t_end + pd.Timedelta(seconds=1))]
    train_score = np.full((len(sets), len(windows)), -np.inf)
    total = len(sets) + len(folds)
    for i, ov in enumerate(sets):
        try:
            feat, tf = (feat0, base_tf) if not ov else build(bars15, gold, raw, {**base_p, **ov}, hmm)
            for j, (a, b) in enumerate(windows):
                try:
                    r = run(None, None, None, {**base_p, **ov}, start=a, end=b - pd.Timedelta(seconds=1),
                            feat=feat, base_tf=tf, **kw)
                    m, x = r["metrics"]["Strategy (net of charges)"], r["extra"]
                    if x["Trades"] >= min_trades:
                        train_score[i, j] = SCORERS[score_by](m, x)
                except ValueError:
                    pass
        except ValueError:
            pass
        if progress:
            progress((i + 1) / total)

    rows, oos, base_oos, trades = [], [], [], []
    feat_cache = {0: (feat0, base_tf)}
    for j, (a, b, c) in enumerate(folds):
        best = int(np.argmax(train_score[:, j])) if np.isfinite(train_score[:, j]).any() else 0
        if best not in feat_cache:
            feat_cache[best] = build(bars15, gold, raw, {**base_p, **sets[best]}, hmm)
        for idx, store in ((best, oos), (0, base_oos)):
            feat, tf = feat_cache[idx]
            try:
                r = run(None, None, None, {**base_p, **sets[idx]}, start=b, end=c, feat=feat, base_tf=tf, **kw)
            except ValueError:
                continue
            store.append(r["curves"]["Strategy (net of charges)"].pct_change().fillna(0.0))
            if store is oos:
                bh = r["curves"]["Buy and hold gold"]
                m, x = r["metrics"]["Strategy (net of charges)"], r["extra"]
                rows.append({"Train from": a.date(), "Test from": b.date(), "Test to": c.date(),
                             f"Train {score_by}": train_score[best, j], **(sets[best] or {"(settings)": "sidebar"}),
                             "Test CAGR %": m["CAGR %"], "Test max DD %": m["Max drawdown %"], "Test trades": x["Trades"],
                             "Test profit factor": x["Profit factor"], "Gold in test %": (bh.iloc[-1] / bh.iloc[0] - 1) * 100})
                if not r["trades"].empty:
                    trades.append(r["trades"].assign(fold=j + 1))
        if progress:
            progress((len(sets) + j + 1) / total)

    def join(parts):
        ret = pd.concat(parts)
        ret = ret[~ret.index.duplicated(keep="first")]
        return capital * (1 + ret).cumprod()

    eq_tuned, eq_base = join(oos), join(base_oos)
    gold_px = feat0["Close"].reindex(eq_tuned.index)
    curves = pd.DataFrame({"Walk-forward (tuned, out of sample)": eq_tuned, "Sidebar settings (no tuning)": eq_base.reindex(eq_tuned.index),
                           "Buy and hold gold": capital * gold_px / gold_px.iloc[0]})
    ppy = bars_per_year(base_tf)
    table = pd.DataFrame({n: engine.metrics(curves[n].dropna(), rf, ppy) for n in curves})
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    last = train_score[:, -1]
    latest = sets[int(np.argmax(last))] if np.isfinite(last).any() else {}
    return dict(folds=pd.DataFrame(rows), curves=curves, metrics=table, trades=all_trades, base_tf=base_tf,
                latest_params=dict(latest), latest_window=windows[-1], latest_score=float(last.max()),
                trade_stats=engine.trade_stats(all_trades))
