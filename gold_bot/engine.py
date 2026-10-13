"""Event-driven daily backtest: next-open execution, intrabar stop / target, sizing, and all charges."""
import itertools

import numpy as np
import pandas as pd

from .config import COST_PRESETS, DEFAULT_PARAMS
from .features import build_features

ZERO_COSTS = COST_PRESETS["No charges (gross)"]
FULL_SIZE_SCORE = 0.5  # with scale_by_score on, a |score| of this or more trades full quantity


def _simulate(feat, p, c, capital):
    idx = feat.index
    O, H, L, C = (feat[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    ATR, S = feat["ATR"].to_numpy(float), feat["score"].to_numpy(float)
    gap_days = np.r_[0, np.diff(idx.values).astype("timedelta64[D]").astype(int)]
    n = len(idx)

    entry_th, exit_th = p["entry_threshold"], p["exit_threshold"]
    use_sl, use_tp, allow_short = p["use_sl"], p["use_tp"], p["allow_short"]
    half_spread, slip = c["spread"] / 2, c["slippage_bps"] / 1e4

    def fill(px, side):  # side +1 = buy, -1 = sell; you always trade at the worse price
        return px + side * (half_spread + px * slip)

    def charges(px, q, side):
        pct = c["pct_buy"] if side > 0 else c["pct_sell"]
        return px * q * pct / 100 + q / c["lot_size"] * c["commission_per_lot"] + c["fixed_per_order"]

    cash, qty, cooldown = float(capital), 0.0, 0
    entry_px = sl = tp = risk_amt = 0.0
    entry_i, entry_cost, fin_cost, total_cost = 0, 0.0, 0.0, 0.0
    want_exit, want_dir, want_score = False, 0, 0.0
    equity, notional, trades = np.empty(n), np.zeros(n), []
    equity[0] = cash

    def close(i, px, reason):
        nonlocal cash, qty, total_cost
        side = -1 if qty > 0 else 1
        fpx = fill(px, side)
        q = abs(qty)
        cost = charges(fpx, q, side)
        gross = qty * (px - trades_open_px[0])  # mid to mid, before any charge
        cash += qty * (fpx - entry_px) - cost
        total_cost += cost
        net = qty * (fpx - entry_px) - cost - entry_cost - fin_cost
        trades.append(dict(
            entry_date=idx[entry_i], exit_date=idx[i], side="Long" if qty > 0 else "Short", qty=q,
            entry_price=entry_px, exit_price=fpx, bars=i - entry_i, exit_reason=reason, gross_pnl=gross,
            charges=gross - net, net_pnl=net, r_multiple=net / risk_amt if risk_amt > 0 else np.nan))
        qty = 0.0

    trades_open_px = [0.0]
    for i in range(1, n):
        # 1. overnight financing on the position carried into this bar
        if qty != 0:
            rate = c["financing_long_pct"] if qty > 0 else c["financing_short_pct"]
            fee = abs(qty) * C[i - 1] * rate / 100 * gap_days[i] / 365
            cash -= fee
            fin_cost += fee
            total_cost += fee

        # 2. orders decided on yesterday's close are filled at today's open
        if want_exit and qty != 0:
            close(i, O[i], "Signal")
        if want_dir != 0 and qty == 0 and cash > 0 and ATR[i - 1] > 0:
            stop_dist = p["sl_atr_mult"] * ATR[i - 1]
            fpx = fill(O[i], want_dir)
            q = cash * p["risk_pct"] / 100 / stop_dist if p["sizing_mode"] == "risk" else cash * p["alloc_pct"] / 100 / fpx
            q = min(q, cash * p["max_leverage"] / fpx)
            if p["scale_by_score"]:
                q *= min(1.0, abs(want_score) / FULL_SIZE_SCORE)
            if q > 0:
                qty, entry_px, entry_i, fin_cost = want_dir * q, fpx, i, 0.0
                trades_open_px[0] = O[i]
                entry_cost = charges(fpx, q, want_dir)
                cash -= entry_cost
                total_cost += entry_cost
                risk_amt = q * stop_dist
                sl = fpx - want_dir * stop_dist
                tp = fpx + want_dir * stop_dist * p["tp_rr"]
        want_exit, want_dir = False, 0

        # 3. stop and target inside the bar. If both are touched the stop is assumed first. Gaps fill at the open.
        if qty > 0:
            if use_sl and L[i] <= sl:
                close(i, min(sl, O[i]), "Stop loss"); cooldown = p["cooldown_bars"]
            elif use_tp and H[i] >= tp:
                close(i, max(tp, O[i]), "Take profit"); cooldown = p["cooldown_bars"]
        elif qty < 0:
            if use_sl and H[i] >= sl:
                close(i, max(sl, O[i]), "Stop loss"); cooldown = p["cooldown_bars"]
            elif use_tp and L[i] <= tp:
                close(i, min(tp, O[i]), "Take profit"); cooldown = p["cooldown_bars"]

        # 4. mark to market
        equity[i] = cash + qty * (C[i] - entry_px)
        notional[i] = abs(qty) * C[i]

        # 5. decide on the close, act on the next open
        s = S[i]
        if np.isnan(s):
            continue
        go_long, go_short = s >= entry_th, allow_short and s <= -entry_th
        if qty == 0:
            if cooldown > 0:
                cooldown -= 1
            elif go_long or go_short:
                want_dir, want_score = (1 if go_long else -1), s
        elif qty > 0 and s <= exit_th:
            want_exit = True
            if go_short:
                want_dir, want_score = -1, s
        elif qty < 0 and s >= -exit_th:
            want_exit = True
            if go_long:
                want_dir, want_score = 1, s

    if qty != 0:
        close(n - 1, C[-1], "End of test")
        equity[-1] = cash
    eq = pd.DataFrame({"equity": equity, "notional": notional}, index=idx)
    return eq, pd.DataFrame(trades), total_cost


def metrics(equity, rf=0.04):
    """Return and risk statistics for a daily equity curve."""
    eq = equity.dropna()
    ret = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0]
    cagr = total ** (1 / years) - 1 if total > 0 else -1.0
    dd = eq / eq.cummax() - 1
    sd, downside = ret.std(), ret[ret < 0].std()
    excess = ret.mean() - rf / 252
    return {
        "Total return %": (total - 1) * 100,
        "CAGR %": cagr * 100,
        "Max drawdown %": dd.min() * 100,
        "Volatility %": sd * np.sqrt(252) * 100,
        "Sharpe": np.sqrt(252) * excess / sd if sd > 0 else 0.0,
        "Sortino": np.sqrt(252) * excess / downside if downside and downside > 0 else 0.0,
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else 0.0,
    }


def trade_stats(trades):
    if trades.empty:
        return {"Trades": 0, "Winning": 0, "Losing": 0, "Breakeven": 0, "Win rate %": 0.0, "Profit factor": 0.0,
                "Avg win": 0.0, "Avg loss": 0.0, "Avg R": 0.0, "Avg bars held": 0.0}
    r, pnl = trades["r_multiple"].fillna(0), trades["net_pnl"]
    flat = r.abs() < 0.1  # within a tenth of the risk taken counts as breakeven
    win, loss = (pnl > 0) & ~flat, (pnl < 0) & ~flat
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "Trades": len(trades), "Winning": int(win.sum()), "Losing": int(loss.sum()), "Breakeven": int(flat.sum()),
        "Win rate %": win.sum() / len(trades) * 100,
        "Profit factor": gains / losses if losses > 0 else float("inf"),
        "Avg win": pnl[win].mean() if win.any() else 0.0,
        "Avg loss": pnl[loss].mean() if loss.any() else 0.0,
        "Avg R": r.mean(), "Avg bars held": trades["bars"].mean(),
    }


def run_backtest(gold, raw, params=None, costs=None, capital=100_000, start=None, end=None, rf=0.04, feat=None):
    """Full run. Returns features, equity curves (net, gross, buy and hold), trade log and metric tables."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    c = {**ZERO_COSTS, **(costs or {})}
    if feat is None:
        feat = build_features(gold, raw, p)
    # Indicators are built on full history, then the test window is cut, so no warm-up is lost.
    f = feat.loc[start:end]
    f = f.loc[f["score"].first_valid_index():]
    if len(f) < 30:
        raise ValueError("Not enough data in the selected date range.")

    net, trades, total_cost = _simulate(f, p, c, capital)
    gross, _, _ = _simulate(f, p, ZERO_COSTS, capital)
    curves = pd.DataFrame({
        "Strategy (net of charges)": net["equity"],
        "Strategy (gross)": gross["equity"],
        "Buy and hold gold": capital * f["Close"] / f["Close"].iloc[0],
    })
    table = pd.DataFrame({name: metrics(curves[name], rf) for name in curves})
    lev = net["notional"] / net["equity"].clip(lower=1e-9)
    extra = {
        **trade_stats(trades),
        "Total charges": total_cost,
        "Charges % of capital": total_cost / capital * 100,
        "CAGR lost to charges %": table.loc["CAGR %", "Strategy (gross)"] - table.loc["CAGR %", "Strategy (net of charges)"],
        "Time in market %": (net["notional"] > 0).mean() * 100,
        "Avg leverage when in": lev[lev > 0].mean() if (lev > 0).any() else 0.0,
    }
    return dict(feat=f, curves=curves, leverage=lev, trades=trades, metrics=table, extra=extra, params=p, costs=c)


def summary_row(res):
    """One flat row per run, for sweeps."""
    m, x = res["metrics"], res["extra"]
    net, bh = m["Strategy (net of charges)"], m["Buy and hold gold"]
    return {
        "CAGR %": net["CAGR %"], "Max DD %": net["Max drawdown %"], "Sharpe": net["Sharpe"], "Calmar": net["Calmar"],
        "Total return %": net["Total return %"], "Gross CAGR %": m.loc["CAGR %", "Strategy (gross)"],
        "CAGR vs B&H %": net["CAGR %"] - bh["CAGR %"], "Trades": x["Trades"], "Win rate %": x["Win rate %"],
        "Profit factor": x["Profit factor"], "Time in market %": x["Time in market %"], "Total charges": x["Total charges"],
    }


def run_sweep(gold, raw, base_params, grid, costs=None, capital=100_000, start=None, end=None, rf=0.04, progress=None):
    """Run every combination in `grid` ({param: [values]}) on top of base_params. Returns one row per run."""
    names = list(grid)
    combos = list(itertools.product(*(grid[k] for k in names)))
    return _run_many(gold, raw, base_params, [dict(zip(names, v)) for v in combos], costs, capital, start, end, rf, progress)


def run_random_weights(gold, raw, base_params, weight_keys, n, seed=0, **kw):
    """Try n random 0-10 score sets for the chosen factors."""
    rng = np.random.default_rng(seed)
    sets = [dict(zip(weight_keys, map(int, rng.integers(0, 11, len(weight_keys))))) for _ in range(n)]
    sets = [s for s in sets if sum(s.values()) > 0]
    return _run_many(gold, raw, base_params, sets, kw.get("costs"), kw.get("capital", 100_000),
                     kw.get("start"), kw.get("end"), kw.get("rf", 0.04), kw.get("progress"))


def run_random_params(gold, raw, base_params, ranges, n, seed=0, **kw):
    """Try n random combinations of any number of parameters.
    `ranges` is {param: (lo, hi, step, kind)}; each value is drawn from lo, lo+step, ... up to hi."""
    rng = np.random.default_rng(seed)
    pools = {}
    for k, (lo, hi, step, kind) in ranges.items():
        vals = np.arange(lo, hi + step / 2, step)
        pools[k] = [int(round(v)) if kind == "int" else float(round(v, 6)) for v in vals]
    sets = [{k: pool[rng.integers(len(pool))] for k, pool in pools.items()} for _ in range(n)]
    w_keys = [k for k in pools if k.startswith("w_")]
    if len(w_keys) == len(pools):  # an all-zero score set has no signal
        sets = [s for s in sets if sum(s.values()) > 0]
    return _run_many(gold, raw, base_params, sets, kw.get("costs"), kw.get("capital", 100_000),
                     kw.get("start"), kw.get("end"), kw.get("rf", 0.04), kw.get("progress"))


def _run_many(gold, raw, base_params, overrides, costs, capital, start, end, rf, progress):
    rows = []
    for j, ov in enumerate(overrides):
        try:
            res = run_backtest(gold, raw, {**base_params, **ov}, costs, capital, start, end, rf)
            rows.append({**ov, **summary_row(res)})
        except Exception:
            pass
        if progress:
            progress((j + 1) / len(overrides))
    return pd.DataFrame(rows)
