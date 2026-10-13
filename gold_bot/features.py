"""Feature engineering: raw inputs -> normalised readings -> factor signals (-1..+1) -> weighted final score."""
import numpy as np
import pandas as pd

from .config import DATA_INPUTS, FACTORS

_EPS = 1e-9


def align(series, index, lag=0):
    """Put a series on the gold calendar, carrying the last known value forward, then delay it by `lag` bars."""
    s = series[~series.index.duplicated(keep="last")]
    return s.reindex(index.union(s.index)).ffill().reindex(index).shift(lag or 0)


def normalise(x, window, method="zscore", floor=_EPS):
    """Rolling normalisation. Every method is scaled so roughly +/-2 is an extreme reading."""
    r = x.rolling(window, min_periods=window)
    if method == "logmean" and (x.dropna() > 0).all():
        lx = np.log(x)
        lr = lx.rolling(window, min_periods=window)
        return (lx - lr.mean()) / np.maximum(lr.std(), floor / r.mean().abs().clip(lower=_EPS))
    if method == "minmax":
        lo, hi = r.min(), r.max()
        return ((x - lo) / np.maximum(hi - lo, floor * 4) - 0.5) * 4
    if method == "iqr":
        iqr = r.quantile(0.75) - r.quantile(0.25)
        return (x - r.median()) / np.maximum(iqr, floor * 1.35) * 1.35
    # zscore, and logmean on series that can go negative (real yield, money flow)
    return (x - r.mean()) / np.maximum(r.std(), floor)


def to_signal(z, mode="continuous", z_clip=2.0, buffer=0.0):
    z = z.where(z.abs() >= buffer, 0.0).where(z.notna())
    if mode == "binary":
        return np.sign(z)
    return (z / z_clip).clip(-1, 1)


def _flow(raw, key, window):
    """Net flow over the lookback. ETF proxy -> Chaikin money flow. User file -> change in holdings / summed flow."""
    if isinstance(raw, pd.DataFrame):
        return raw["mfv"].rolling(window).sum() / raw["dv"].rolling(window).sum().replace(0, np.nan)
    if key == "gold_etf":
        return raw.pct_change(window) * 100
    return raw.rolling(window).sum()


def factor_inputs(raw, index, p):
    """The one raw number each factor reads, on the gold calendar and delayed by its publication lag."""
    lag = lambda k: DATA_INPUTS[k]["lag_days"]
    out = {}
    if "real_yield" in raw:
        out["real_yield"] = align(raw["real_yield"], index, lag("real_yield"))
    if "dxy" in raw:
        out["dxy"] = align(raw["dxy"], index, lag("dxy"))
    if "usdinr" in raw:
        out["usdinr"] = align(raw["usdinr"], index, lag("usdinr"))

    fed_key = p["fed_series"]
    if fed_key == "sofr" and "sofr" in raw and "effr" in raw:
        out["fed"] = align(raw["sofr"].combine_first(raw["effr"]), index, lag("sofr"))
    elif fed_key in raw:
        out["fed"] = align(raw[fed_key], index, lag(fed_key))
    elif "effr" in raw:
        out["fed"] = align(raw["effr"], index, lag("effr"))

    if "cpi" in raw:
        monthly = raw["cpi"].asfreq("MS")
        yoy = (monthly.pct_change(12, fill_method=None) * 100).dropna()
        yoy.index = yoy.index + pd.Timedelta(days=int(p["cpi_lag_days"]))
        out["cpi"] = align(yoy, index)

    for key in ("gold_etf", "bond_flow"):
        if key in raw:
            out[key] = align(_flow(raw[key], key, int(p["flow_window"])), index, lag(key))
    return out


RATE_LIKE = ("real_yield", "fed", "cpi")  # already in percent: use point changes, not % changes


def pearson(x, y):
    """Pearson r = cov(x, y) / (std(x) * std(y)) on the rows where both exist. Returns (r, n)."""
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    if n < 3:
        return np.nan, n
    xd, yd = d.iloc[:, 0] - d.iloc[:, 0].mean(), d.iloc[:, 1] - d.iloc[:, 1].mean()
    den = np.sqrt((xd ** 2).sum() * (yd ** 2).sum())
    return (float((xd * yd).sum() / den) if den > 0 else np.nan), n


def correlation_table(gold, raw, p, start=None, end=None, basis="change", horizon=1):
    """Pearson correlation of every factor's input with gold's close.
    basis 'level': price levels. basis 'change': `horizon`-day changes (avoids the spurious
    correlation two trending series show on levels). Uses the same lagged inputs as the backtest."""
    close = gold["Close"]
    inputs = factor_inputs(raw, close.index, p)
    rows = []
    for key in FACTORS:
        if key == "tech" or key not in inputs:
            continue
        x, g = inputs[key], close
        if basis == "change":
            x = x.diff(horizon) if key in RATE_LIKE else x.pct_change(horizon, fill_method=None)
            g = close.pct_change(horizon)
        x, g = x.replace([np.inf, -np.inf], np.nan), g
        if start is not None:
            x, g = x.loc[str(start):str(end)], g.loc[str(start):str(end)]
        r, n = pearson(g, x)
        rows.append(dict(key=key, r=r, n=n))
    return pd.DataFrame(rows)


def strength(r):
    if pd.isna(r):
        return "n/a"
    a = abs(r)
    return "Very strong" if a >= 0.8 else "Strong" if a >= 0.6 else "Moderate" if a >= 0.4 else "Weak" if a >= 0.2 else "Negligible"


def build_features(gold, raw, p):
    """Returns the gold frame plus ATR, raw_<f>, z_<f>, sig_<f> for every factor and the final `score` (-1..+1)."""
    df = gold[["Open", "High", "Low", "Close", "Volume"]].copy()
    prev = df["Close"].shift(1)
    tr = np.maximum(df["High"] - df["Low"], np.maximum((df["High"] - prev).abs(), (df["Low"] - prev).abs()))
    df["ATR"] = tr.rolling(int(p["atr_period"])).mean()
    shape = dict(mode=p["signal_mode"], z_clip=p["z_clip"], buffer=p["buffer"])

    # Technical factor: z-score gives the direction, ATR and volume expansion confirm it.
    z = normalise(df["Close"], int(p["win_tech"]), p["norm_method"])
    atr_up = (df["ATR"] > df["ATR"].rolling(int(p["atr_baseline"])).mean()).astype(float)
    vol_up = (df["Volume"] > p["vol_mult"] * df["Volume"].rolling(int(p["vol_window"])).mean()).astype(float)
    df["raw_tech"], df["z_tech"] = df["Close"], z
    df["atr_expanding"], df["vol_expanding"] = atr_up, vol_up
    df["sig_tech"] = p["dir_tech"] * to_signal(z, **shape) * (1 + atr_up + vol_up) / 3

    inputs = factor_inputs(raw, df.index, p)
    for key in FACTORS:
        if key == "tech":
            continue
        if key not in inputs:
            df[f"raw_{key}"] = df[f"z_{key}"] = df[f"sig_{key}"] = np.nan
            continue
        x = inputs[key]
        # A policy rate sits flat for years; the floor stops 1bp of noise reading as a 2-sigma move.
        floor = 0.05 if key == "fed" else _EPS
        z = normalise(x, int(p[f"win_{key}"]), p["norm_method"], floor)
        df[f"raw_{key}"], df[f"z_{key}"] = x, z
        df[f"sig_{key}"] = p[f"dir_{key}"] * to_signal(z, **shape)

    # Final score: score-weighted average of the signals. A factor with no data yet is left out, not counted as 0.
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for key in FACTORS:
        w, s = float(p[f"w_{key}"]), df[f"sig_{key}"]
        num += w * s.fillna(0.0)
        den += w * s.notna()
    df["score"] = (num / den.where(den > 0)).where(df["sig_tech"].notna())
    return df
