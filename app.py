"""Gold (XAUUSD) macro strategy backtester. Run with:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gold_bot import binance, data, engine, features, live, mt5, mtf, store, ticks
from gold_bot import regime_hmm as hmm
from gold_bot.config import (CATEGORY_ORDER, COST_FIELDS, COST_PRESETS, DATA_INPUTS, DEFAULT_COST_PRESET,
                             DEFAULT_PARAMS, FACTORS, GROUP_ORDER, PARAMS, TIMEFRAMES)

st.set_page_config(page_title="Gold macro backtester", layout="wide")

BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984"
NET, GROSS, BH = "Strategy (net of charges)", "Strategy (gross)", "Buy and hold gold"
CURVE_STYLE = {NET: dict(color=BLUE, width=2), GROSS: dict(color=ORANGE, width=2), BH: dict(color=GREY, width=2, dash="dot")}
CHOICE_TEXT = {1: "+1  rising is bullish for gold", -1: "-1  rising is bearish for gold"}
MAX_RUNS = 3000
REGIME_COLOURS = {features.UP: AQUA, features.MIXED: ORANGE, features.DOWN: RED, features.UNKNOWN: GREY}
STATE_COLOURS = [AQUA, BLUE, ORANGE, RED]
# HMM only has no stop loss, so measures in R (multiples of the stop distance) do not apply to it.
R_ROWS = ("Avg loss (R)", "Stop overruns (< -1.1R)", "R lost beyond the stop", "Worst trade (R)", "Sum of R")
ORIG, RULES, HMM_ONLY = "Original strategy", "Strategy + rule filter (Phase 1)", "HMM only (Phase 2a)"
HMM_GATE, COMBINED = "Strategy + HMM gate", "Strategy + rules + HMM (Phase 2b)"


def chart(fig, height=340, **layout):
    layout.setdefault("hovermode", "x unified")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), **layout)
    # The same chart can appear in two tabs, so every chart gets its own key.
    _chart_n[0] += 1
    st.plotly_chart(fig, width="stretch", key=f"chart_{_chart_n[0]}")


_chart_n = [0]


@st.cache_data(show_spinner="Downloading market and macro data...")
def load(refresh_id):
    return data.load_all(refresh=refresh_id > 0)


@st.cache_data(show_spinner="Loading Binance 15m gold bars (the first download takes a few minutes)...", ttl=900)
def load_bars(symbol, refresh_id):
    return mt5.klines(symbol, update=False) if mt5.is_mt5(symbol) else binance.klines(symbol)


def mt5_markets():
    """MT5 symbols that can be used as a data source: the one in use now, plus any history already downloaded."""
    names = ([mt5.STATE["symbol"]] if mt5.STATE["connected"] and mt5.STATE["symbol"] else []) + mt5.cached_symbols()
    return [mt5.PREFIX + n for n in dict.fromkeys(names)]


def market_label(k):
    if mt5.is_mt5(k):
        return f"{mt5.name_of(k)} from MetaTrader 5 (broker XAUUSD, tick volume)"
    return binance.MARKETS[k]["label"] if k in binance.MARKETS else k


# ---------------------------------------------------------
# Session state: every parameter lives under p_<name>, every charge under c_<name>
# ---------------------------------------------------------
ss = st.session_state
ss.setdefault("refresh_id", 0)


def _cast(name, v):
    """A saved value in the type the widget expects (JSON turns 1.0 into 1 and so on)."""
    d = DEFAULT_PARAMS[name]
    return bool(v) if isinstance(d, bool) else (v if isinstance(d, str) or PARAMS[name]["kind"] == "choice" else type(d)(v))


if "settings_loaded" not in ss:
    # first run of this browser session: start from the settings saved with the Save button, if there are any
    saved = store.load_settings() or {}
    ss["saved_info"] = saved.get("saved")
    for k, v in (saved.get("params") or {}).items():
        if k in DEFAULT_PARAMS:
            try:
                ss[f"p_{k}"] = _cast(k, v)
            except (TypeError, ValueError):
                pass
    if saved.get("cost_preset") in COST_PRESETS:
        ss["cost_preset"] = saved["cost_preset"]
    for k, v in (saved.get("costs") or {}).items():
        if k in COST_FIELDS:
            ss[f"c_{k}"] = float(v)
    ss["capital"] = int(saved.get("capital", 100_000))
    ss["rf_pct"] = float(saved.get("rf_pct", 4.0))
    ss["settings_loaded"] = True
for k, v in DEFAULT_PARAMS.items():
    ss.setdefault(f"p_{k}", v)
ss.setdefault("cost_preset", DEFAULT_COST_PRESET)
for k, v in COST_PRESETS[ss["cost_preset"]].items():
    ss.setdefault(f"c_{k}", float(v))
ss.setdefault("capital", 100_000)
ss.setdefault("rf_pct", 4.0)


def current_settings():
    return dict(params={k: ss[f"p_{k}"] for k in PARAMS}, cost_preset=ss["cost_preset"],
                costs={k: ss[f"c_{k}"] for k in COST_FIELDS}, capital=ss["capital"], rf_pct=ss["rf_pct"])


def save_settings():
    store.save_settings(current_settings())
    ss["saved_info"] = "just now"
    ss["saved_snapshot"] = repr(current_settings())


ss.setdefault("saved_snapshot", repr(current_settings()) if ss.get("saved_info") else None)


def reset_params():
    for k, v in DEFAULT_PARAMS.items():
        ss[f"p_{k}"] = v


def apply_preset():
    for k, v in COST_PRESETS[ss["cost_preset"]].items():
        ss[f"c_{k}"] = float(v)


def load_into_sidebar(values):
    for k, v in values.items():
        ss[f"p_{k}"] = type(DEFAULT_PARAMS[k])(v)


def regime_chart(f, title="Market type on each bar (Phase 1 rules)"):
    """Gold price coloured by the rule-based market type, with high volatility days marked."""
    fig = go.Figure(go.Scatter(x=f.index, y=f["Close"], name="Gold close", line=dict(color=GREY, width=1), showlegend=False))
    for name, colour in REGIME_COLOURS.items():
        d = f[f["regime"] == name]
        if len(d):
            fig.add_scatter(x=d.index, y=d["Close"], mode="markers", name=name, marker=dict(size=3, color=colour))
    hv = f[f["rg_hivol"]]
    if len(hv):
        fig.add_scatter(x=hv.index, y=hv["Close"], mode="markers", name="High volatility",
                        marker=dict(size=7, color="rgba(0,0,0,0)", line=dict(width=1, color=RED), symbol="circle-open"))
    chart(fig, 380, yaxis_title="USD per oz", yaxis_type="log", title=title, hovermode="closest")


def group_table(trades, key, label):
    g = engine.by_group(trades, key)
    if g.empty:
        st.info("No trades.")
    else:
        st.dataframe(g.rename_axis(label).round(2), width="stretch")


def compare_table(cols):
    """cols = {column name: comparison_row dict}; shows the metrics as rows, one column per approach."""
    t = pd.DataFrame(cols)
    st.dataframe(t.apply(pd.to_numeric, errors="coerce").round(2), width="stretch", height=560)
    return t


def param_widget(name):
    s, key = PARAMS[name], f"p_{name}"
    if s["kind"] == "bool":
        st.checkbox(s["label"], key=key, help=s["help"] or None)
    elif s["kind"] == "choice":
        st.selectbox(s["label"], s["choices"], key=key, help=s["help"] or None,
                     format_func=lambda v: CHOICE_TEXT.get(v, str(v)) if name.startswith("dir_") else str(v))
    elif name.startswith("w_"):
        st.slider(s["label"], s["lo"], s["hi"], step=s["step"], key=key, help=s["help"] or None)
    else:
        cast = int if s["kind"] == "int" else float
        st.number_input(s["label"], cast(s["lo"]), cast(s["hi"]), step=cast(s["step"]), key=key, help=s["help"] or None)


gold, raw, status = load(ss["refresh_id"])

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Backtest setup")
    first, last = gold.index[0].date(), gold.index[-1].date()
    start, end = st.slider("Test period", first, last, (first, last), format="YYYY-MM-DD")
    capital = st.number_input("Initial capital (USD)", 1_000, 100_000_000, step=10_000, key="capital")
    rf = st.number_input("Risk-free rate for Sharpe (% a year)", 0.0, 15.0, step=0.25, key="rf_pct") / 100

    with st.expander("Charges and brokerage", expanded=False):
        st.selectbox("Preset", list(COST_PRESETS), key="cost_preset", on_change=apply_preset)
        for k, (label, lo, hi, step) in COST_FIELDS.items():
            st.number_input(label, float(lo), float(hi), step=float(step), key=f"c_{k}", format="%.4f")
        st.caption("Presets are typical retail levels. Replace them with the numbers on your broker's contract note.")

    st.header("Strategy parameters")
    st.button("Save settings", type="primary", on_click=save_settings, key="save_settings_btn",
              help="Saves every setting in this sidebar (parameters, ticks, charges, capital). They are loaded again whenever the app "
                   "starts. Without this they reset each time the server restarts.")
    if ss.get("saved_snapshot") is None:
        st.caption("Nothing saved yet: settings reset when the app restarts.")
    elif ss["saved_snapshot"] != repr(current_settings()):
        st.caption(f"Unsaved changes (last saved: {ss.get('saved_info')}). Press Save settings to keep them.")
    else:
        st.caption(f"Saved ({ss.get('saved_info')}) in {store.store.name}.")
    if store.error:
        st.warning(store.error)
    for group in GROUP_ORDER:
        with st.expander(group, expanded=group in GROUP_ORDER[:2]):
            for name, spec in PARAMS.items():
                if spec["group"] == group:
                    param_widget(name)
    st.button("Reset parameters to defaults", on_click=reset_params)
    if st.button("Refresh data from sources"):
        ss["refresh_id"] += 1
        st.rerun()

params = {k: ss[f"p_{k}"] for k in PARAMS}
costs = {k: ss[f"c_{k}"] for k in COST_FIELDS}
run_kw = dict(costs=costs, capital=capital, start=str(start), end=str(end), rf=rf, hmm=ss.get("hmm"))

st.title("Gold (XAUUSD) macro strategy backtester")
tab_data, tab_bt, tab_regime, tab_sweep, tab_mtf, tab_mt5, tab_live = st.tabs(
    ["Data and parameters", "Backtest", "Regime lab (rules vs HMM)", "Parameter sweep", "Multi-timeframe", "MT5 (XAUUSD)", "Live (paper)"])

# ---------------------------------------------------------
# Tab 1: what goes in, how often it changes, what can be tweaked
# ---------------------------------------------------------
with tab_data:
    failed = [DATA_INPUTS[k]["label"] for k, v in status.items() if v.startswith("FAILED")]
    if failed:
        st.warning("Could not load: " + ", ".join(failed) + ". Factors that depend on them are left out of the score.")

    st.subheader("Inputs by how often the data changes")
    st.caption("The backtest runs on daily bars, so tick-by-tick inputs are sampled once a day at the close. "
               "Lagged inputs are only used after their publication delay.")
    for cat in CATEGORY_ORDER:
        rows = []
        for key, d in DATA_INPUTS.items():
            if d["category"] != cat:
                continue
            s = raw.get(key)
            has = s is not None and len(s) > 0
            latest = "" if not has or isinstance(s, pd.DataFrame) else f"{s.iloc[-1]:,.2f}"
            rows.append({
                "Input": d["label"], "Source": d["source"], "Changes": d["native"],
                "Delay used": "CPI lag parameter" if d["lag_days"] is None else f"{d['lag_days']} day",
                "History from": s.index[0].date() if has else None, "Latest": s.index[-1].date() if has else None,
                "Latest value": latest, "Real or proxy": "Proxy" if d["proxy"] else "Real", "Status": status.get(key, "not loaded"),
                "Note": d["note"],
            })
        st.markdown(f"**{cat}**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Strategy factors")
    st.caption("You cannot change the data itself (the repo rate, CPI, the dollar). You can change how much each "
               "factor counts (score 0-10), which way it points, and the lookback it is measured over.")
    feat_now = features.build_features(gold, raw, params)
    total_w = sum(features.weight(params, k) for k in FACTORS) or 1
    st.dataframe(pd.DataFrame([{
        "Factor": f["label"], "Included": "Yes" if params[f"inc_{k}"] else "No", "Data type": f["category"],
        "Score (0-10)": params[f"w_{k}"],
        "Share of final score %": round(features.weight(params, k) / total_w * 100, 1),
        "Direction": "Positive" if params[f"dir_{k}"] > 0 else "Inverse", "Lookback (days)": params[f"win_{k}"],
        "Usable from": d.date() if (d := feat_now[f"sig_{k}"].first_valid_index()) is not None else None,
        "Logic": f["logic"],
    } for k, f in FACTORS.items()]), hide_index=True, width="stretch")

    st.subheader("Pearson correlation with gold")
    st.caption("r = cov(gold, factor) / (std(gold) x std(factor)). Close to +1: moves with gold. Close to -1: moves against gold. "
               "Close to 0: no linear link. It is measured over the Test period in the sidebar, on the same lagged inputs the backtest uses.")
    ca, cb = st.columns(2)
    basis = ca.radio("Measured on", ["change", "level"], horizontal=True,
                     format_func=lambda v: "Changes (recommended)" if v == "change" else "Price levels",
                     help="Levels of two trending series often look correlated by accident. Changes are the honest test.")
    horizon = cb.selectbox("Change horizon (days)", [1, 5, 21], disabled=basis == "level")
    corr = features.correlation_table(gold, raw, params, start, end, basis, horizon)
    if corr.empty:
        st.info("No factor data loaded to correlate.")
    else:
        corr["Direction"] = np.where(corr["r"] >= 0, 1, -1)
        corr_view = pd.DataFrame({
            "Factor": [FACTORS[k]["label"] for k in corr["key"]],
            "Pearson r": corr["r"].round(3),
            "Strength": [features.strength(r) for r in corr["r"]],
            "Observed relation": ["Same way as gold" if r > 0 else "Opposite to gold" for r in corr["r"]],
            "Suggested direction": corr["Direction"],
            "Current direction": [params[f"dir_{k}"] for k in corr["key"]],
            "Agrees?": ["Yes" if params[f"dir_{k}"] == d else "No" for k, d in zip(corr["key"], corr["Direction"])],
            "Days used": corr["n"],
        })
        st.dataframe(corr_view, hide_index=True, width="stretch")
        order = corr.sort_values("r")
        fig = go.Figure(go.Bar(x=order["r"], y=[FACTORS[k]["label"] for k in order["key"]], orientation="h",
                               marker_color=[AQUA if r > 0 else RED for r in order["r"]],
                               text=order["r"].round(2), hovertemplate="%{y}: r = %{x:.3f}<extra></extra>"))
        chart(fig, 300, xaxis_title="Pearson r with gold", xaxis_range=[-1, 1], showlegend=False, hovermode="closest")
        st.button("Set factor directions from correlation sign",
                  on_click=load_into_sidebar, args=({f"dir_{k}": int(d) for k, d in zip(corr["key"], corr["Direction"])},),
                  help="Copies the sign of r into the Factor direction settings (+1 if r > 0, -1 if r < 0).")

    st.subheader("Everything you can tweak")
    st.dataframe(pd.DataFrame([{
        "Group": s["group"], "Parameter": s["label"], "Current": str(params[k]), "Default": str(s["default"]),
        "Range / options": f"{s['lo']} to {s['hi']}" if s["choices"] is None and s["kind"] != "bool" else
                           ("on / off" if s["kind"] == "bool" else ", ".join(map(str, s["choices"]))),
        "Can be swept": "Yes" if s["sweep"] else "No", "Key": k,
    } for k, s in PARAMS.items()]), hide_index=True, width="stretch", height=420)

    st.subheader("Look at an input")
    pick = st.selectbox("Input", [k for k in DATA_INPUTS if k in raw], format_func=lambda k: DATA_INPUTS[k]["label"])
    s = raw[pick]
    if pick == "gold":
        s, ytitle = s["Close"], "USD per oz"
    elif isinstance(s, pd.DataFrame):
        s, ytitle = features._flow(s, pick, int(params["flow_window"])), f"Net money flow over {params['flow_window']} days (-1 to +1)"
    else:
        ytitle = ("Index level" if pick in ("dxy", "cpi") else "INR per USD" if pick == "usdinr"
                  else "Shares traded" if pick == "gld_volume" else "Percent")
    chart(go.Figure(go.Scatter(x=s.index, y=s.values, line=dict(color=BLUE, width=2), name=DATA_INPUTS[pick]["label"])),
          yaxis_title=ytitle, showlegend=False)

# ---------------------------------------------------------
# Tab 2: single backtest
# ---------------------------------------------------------
with tab_bt:
    if params["use_hmm_gate"] and ss.get("hmm") is None:
        st.warning("'Use HMM gate' is on but no HMM is trained yet, so the gate does nothing. Train it in the Regime lab tab.")
    if st.button("Run backtest", type="primary"):
        try:
            ss["result"] = engine.run_backtest(gold, raw, params, **run_kw)
            ss["result_inputs"] = (params, costs, capital, start, end, rf, id(ss.get("hmm")))
        except ValueError as e:
            st.error(str(e))
    res = ss.get("result")
    if res is None:
        st.info("Set the scores and parameters in the sidebar, then press Run backtest.")
    else:
        if ss["result_inputs"] != (params, costs, capital, start, end, rf, id(ss.get("hmm"))):
            st.warning("Settings have changed since this run. Press Run backtest to update.")
        m, x, f, curves, trades = res["metrics"], res["extra"], res["feat"], res["curves"], res["trades"]
        st.caption(f"Test period {f.index[0].date()} to {f.index[-1].date()}, {len(f):,} daily bars. "
                   "Signals are taken on the close and filled at the next open.")

        c = st.columns(6)
        c[0].metric("CAGR, net of charges", f"{m.loc['CAGR %', NET]:.2f}%", f"{m.loc['CAGR %', NET] - m.loc['CAGR %', BH]:+.2f}% vs buy and hold")
        c[1].metric("CAGR, gross", f"{m.loc['CAGR %', GROSS]:.2f}%")
        c[2].metric("CAGR, buy and hold", f"{m.loc['CAGR %', BH]:.2f}%")
        c[3].metric("Max drawdown, net", f"{m.loc['Max drawdown %', NET]:.1f}%", f"buy and hold {m.loc['Max drawdown %', BH]:.1f}%", delta_color="off")
        c[4].metric("Sharpe, net", f"{m.loc['Sharpe', NET]:.2f}", f"buy and hold {m.loc['Sharpe', BH]:.2f}", delta_color="off")
        c[5].metric("Total charges paid", f"{x['Total charges']:,.0f}", f"-{x['CAGR lost to charges %']:.2f}% CAGR", delta_color="off")

        log_y = st.toggle("Log scale", value=True)
        fig = go.Figure([go.Scatter(x=curves.index, y=curves[n], name=n, line=CURVE_STYLE[n]) for n in (NET, GROSS, BH)])
        chart(fig, 420, yaxis_title="Account value (USD)", yaxis_type="log" if log_y else "linear", title="Account value")

        dd = (curves / curves.cummax() - 1) * 100
        fig = go.Figure([go.Scatter(x=dd.index, y=dd[n], name=n, line={**CURVE_STYLE[n], "width": 1.5}) for n in (NET, BH)])
        chart(fig, 260, yaxis_title="Drawdown %", title="Drawdown from peak")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Return and risk**")
            st.dataframe(m.round(2), width="stretch")
        with right:
            st.markdown("**Trades**")
            order = ["Trades", "Winning", "Losing", "Breakeven", "Win rate %", "Profit factor", "Avg win", "Avg loss", "Avg R",
                     "Avg loss (R)", "Stop overruns (< -1.1R)", "Longest losing streak",
                     "Avg bars held", "Time in market %", "Avg leverage when in", "Total charges", "Charges % of capital"]
            st.dataframe(pd.DataFrame({"Value": [f"{x[k]:,.2f}" if isinstance(x[k], float) else f"{x[k]:,}" for k in order]}, index=order),
                         width="stretch", height=280)

        fig = go.Figure(go.Scatter(x=f.index, y=f["Close"], name="Gold close", line=dict(color=GREY, width=1.5)))
        if not trades.empty:
            fig.add_scatter(x=trades["entry_date"], y=trades["entry_price"], mode="markers", name="Entry",
                            marker=dict(symbol="triangle-up", size=9, color=AQUA, line=dict(width=1, color="white")))
            fig.add_scatter(x=trades["exit_date"], y=trades["exit_price"], mode="markers", name="Exit",
                            marker=dict(symbol="triangle-down", size=9, color=RED, line=dict(width=1, color="white")))
        chart(fig, 380, yaxis_title="USD per oz", yaxis_type="log" if log_y else "linear", title="Gold price with entries and exits", hovermode="closest")

        fig = go.Figure(go.Scatter(x=f.index, y=f["score"], name="Final score", line=dict(color=BLUE, width=1.5)))
        fig.add_hline(y=res["params"]["entry_threshold"], line=dict(color=AQUA, dash="dash"), annotation_text="entry")
        fig.add_hline(y=res["params"]["exit_threshold"], line=dict(color=RED, dash="dash"), annotation_text="exit", annotation_position="bottom right")
        chart(fig, 260, yaxis_title="Final score (-1 to +1)", yaxis_range=[-1, 1], showlegend=False, title="Final weighted score")

        st.markdown("**Scorecard on the last bar** (the sheet 2 view)")
        p_used, last_row = res["params"], f.iloc[-1]
        tw = sum(features.weight(p_used, k) for k in FACTORS if pd.notna(last_row[f"sig_{k}"])) or 1
        card = pd.DataFrame([{
            "Factor": FACTORS[k]["label"], "Raw value": last_row[f"raw_{k}"], "Normalised reading": last_row[f"z_{k}"],
            "Signal (-1 to +1)": last_row[f"sig_{k}"], "Score (0-10)": features.weight(p_used, k),
            "Contribution to final": features.weight(p_used, k) * last_row[f"sig_{k}"] / tw,
        } for k in FACTORS])
        st.dataframe(card.round(3), hide_index=True, width="stretch")
        decision = "Long" if last_row["score"] >= p_used["entry_threshold"] else ("Short" if p_used["allow_short"] and last_row["score"] <= -p_used["entry_threshold"] else "No new trade")
        vol_note = ""
        if p_used["use_rvol"] or p_used["use_vdu"]:
            ok = last_row["vol_ok_long"] if decision == "Long" else last_row["vol_ok_short"] if decision == "Short" else True
            vol_note = f" Volume z-score {last_row['rv_z']:+.2f}" + ("" if ok else "; the volume filter blocks this entry") + "."
        st.caption(f"Final score on {f.index[-1].date()}: {last_row['score']:+.3f}. Decision: {decision}.{vol_note}")

        yearly = curves.resample("YE").last()
        yearly = (yearly / yearly.shift(1).fillna(curves.iloc[0]) - 1) * 100
        yearly.index = yearly.index.year
        fig = go.Figure([go.Bar(x=yearly.index, y=yearly[n], name=n, marker_color=col) for n, col in ((NET, BLUE), (BH, GREY))])
        chart(fig, 300, yaxis_title="Return %", barmode="group", title="Return by calendar year")

        st.markdown("**Trades by market type** (on the signal bar)" + ("  |  regime filter ON" if res["params"]["use_regime"] else "  |  regime filter OFF"))
        st.caption("Shows where the strategy makes and loses money. With the filter on, the size multiplier column in the trade log "
                   "shows how much each trade was cut.")
        group_table(trades, "regime", "Market type")
        if "hmm_state" in trades and (trades["hmm_state"] != "").any():
            st.markdown("**Trades by HMM state**")
            group_table(trades, "hmm_state", "HMM state")
        regime_chart(f)

        with st.expander(f"Trade log ({len(trades)} trades)"):
            st.dataframe(trades.round(2), hide_index=True, width="stretch")
            st.download_button("Download trade log (CSV)", trades.to_csv(index=False), "trades.csv", "text/csv")

    # ---------- Volume filter: does it help? ----------
    st.divider()
    st.subheader("Volume filter: does it help?")
    st.caption("The same sidebar settings run four ways: no volume filter, RVOL spike only, dry-up only, and both. "
               "The filter only removes entries, so look for fewer losing trades and a higher profit factor without losing much CAGR. "
               "Logic: VOLUME_SIGNALS_LOGIC.md in the project folder.")
    VOL_VARIANTS = {"No volume filter": dict(use_rvol=False, use_vdu=False), "RVOL spike only": dict(use_rvol=True, use_vdu=False),
                    "Dry-up (VDU) only": dict(use_rvol=False, use_vdu=True), "RVOL + VDU": dict(use_rvol=True, use_vdu=True)}
    VOL_COLOURS = dict(zip(VOL_VARIANTS, (GREY, BLUE, ORANGE, AQUA)))
    if st.button("Run volume filter comparison"):
        try:
            ss["vol_cmp"] = {n: engine.run_backtest(gold, raw, {**params, **v}, **run_kw) for n, v in VOL_VARIANTS.items()}
            ss["vol_cmp_inputs"] = (params, costs, capital, start, end, rf)
        except ValueError as e:
            st.error(str(e))
    vc = ss.get("vol_cmp")
    if vc is not None:
        if ss["vol_cmp_inputs"] != (params, costs, capital, start, end, rf):
            st.warning("Settings have changed since this run. Press the button again to update.")
        compare_table({n: engine.comparison_row(r) for n, r in vc.items()})
        fig = go.Figure([go.Scatter(x=r["curves"].index, y=r["curves"][NET], name=n, line=dict(color=VOL_COLOURS[n], width=1.5))
                         for n, r in vc.items()])
        chart(fig, 380, yaxis_type="log", yaxis_title="Account value (USD)", title="Account value with and without the volume filter")
        vf = vc["No volume filter"]["feat"]
        fig = go.Figure(go.Scatter(x=vf.index, y=vf["rv_z"], name="Volume z-score", line=dict(color=GREY, width=1)))
        sp = vf[vf["rv_spike"]]
        for side, name, colour in ((1, "Spike on an up day", AQUA), (-1, "Spike on a down day", RED)):
            d = sp[sp["rv_side"] == side]
            fig.add_scatter(x=d.index, y=d["rv_z"], mode="markers", name=name, marker=dict(size=6, color=colour))
        fig.add_hline(y=params["rv_z"], line=dict(color=RED, dash="dash"), annotation_text="spike threshold")
        chart(fig, 260, yaxis_title="z-score", title=f"Volume z-score ({'GLD' if params['vol_source'] == 'gld' else 'GC=F'} volume)")
        st.caption(f"{int(vf['rv_spike'].sum())} RVOL spike days and {int(vf['vdu'].sum())} dry-up days in the test period.")

# ---------------------------------------------------------
# Tab 3: Regime lab. Phase 1 (rules) and Phase 2a (HMM alone), side by side
# ---------------------------------------------------------
with tab_regime:
    st.caption("Phase 1 adds a rule-based market-type filter on top of the strategy. Phase 2a trains a Hidden Markov Model (HMM) "
               "on gold's own price behaviour and tests it alone. Both use only data that was available at the time. "
               "Plan: REGIME_PLAN.md in the project folder.")

    # ---------- Phase 1 ----------
    st.subheader("Phase 1: rule-based regime filter")
    st.caption("Uses every sidebar setting, including the 'Regime filter' group. The original strategy is the same run with the filter off.")
    if st.button("Run rule-based comparison", type="primary"):
        try:
            ss["lab_orig"] = engine.run_backtest(gold, raw, {**params, "use_regime": False, "use_hmm_gate": False}, **run_kw)
            ss["lab_rules"] = engine.run_backtest(gold, raw, {**params, "use_regime": True, "use_hmm_gate": False}, **run_kw)
            ss["lab_inputs"] = (params, costs, capital, start, end, rf)
        except ValueError as e:
            st.error(str(e))
    lo, lr = ss.get("lab_orig"), ss.get("lab_rules")
    if lo is None:
        st.info("Press 'Run rule-based comparison'. Tune the filter in the sidebar under 'Regime filter (Phase 1, rule based)'.")
    else:
        if ss["lab_inputs"] != (params, costs, capital, start, end, rf):
            st.warning("Settings have changed since this run. Press the button again to update.")
        c = st.columns(5)
        a, b = engine.comparison_row(lo), engine.comparison_row(lr)
        c[0].metric("CAGR, with filter", f"{b['CAGR %']:.2f}%", f"{b['CAGR %'] - a['CAGR %']:+.2f}% vs original")
        c[1].metric("Max drawdown, with filter", f"{b['Max drawdown %']:.1f}%", f"{b['Max drawdown %'] - a['Max drawdown %']:+.1f}% vs original")
        c[2].metric("Losing trades", f"{b['Losing trades']}", f"{b['Losing trades'] - a['Losing trades']:+d}", delta_color="inverse")
        c[3].metric("Longest losing streak", f"{b['Longest losing streak']}",
                    f"{b['Longest losing streak'] - a['Longest losing streak']:+d}", delta_color="inverse")
        c[4].metric("Profit factor", f"{b['Profit factor']:.2f}", f"{b['Profit factor'] - a['Profit factor']:+.2f}")
        compare_table({ORIG: a, RULES: b})
        curves = pd.DataFrame({ORIG: lo["curves"][NET], RULES: lr["curves"][NET], BH: lo["curves"][BH]})
        fig = go.Figure([go.Scatter(x=curves.index, y=curves[n], name=n, line=sty) for n, sty in
                         ((ORIG, dict(color=GREY, width=1.5)), (RULES, dict(color=BLUE, width=2)), (BH, dict(color=GREY, width=1, dash="dot")))])
        chart(fig, 380, yaxis_type="log", yaxis_title="Account value (USD)", title="Account value: original vs with rule filter")
        l, r = st.columns(2)
        with l:
            st.markdown("**Original strategy: trades by market type**")
            group_table(lo["trades"], "regime", "Market type")
        with r:
            st.markdown("**With rule filter: trades by market type**")
            group_table(lr["trades"], "regime", "Market type")
        regime_chart(lo["feat"])
        f = lo["feat"]
        fig = go.Figure(go.Scatter(x=f.index, y=f["rg_vol_pct"] * 100, line=dict(color=BLUE, width=1), name="Volatility percentile"))
        fig.add_hline(y=params["rg_hivol_pct"] * 100, line=dict(color=RED, dash="dash"), annotation_text="high volatility above")
        chart(fig, 220, yaxis_title="Percentile", yaxis_range=[0, 100], showlegend=False, title="Today's volatility vs its own history")

    st.divider()

    # ---------- Phase 2a ----------
    st.subheader("Phase 2a: HMM alone")
    st.caption("The HMM groups days into hidden states from price behaviour only (no macro). It is refitted every N days on past "
               "data, and each day's state comes from the forward filter (data up to that day only). States are sorted by "
               "volatility, so S0 is always the calmest.")
    a1, a2, a3 = st.columns(3)
    h_keys = a1.multiselect("Model inputs", list(hmm.HMM_FEATURES), default=hmm.DEFAULT_FEATURES, format_func=hmm.HMM_FEATURES.get)
    h_states = a2.selectbox("Number of states", [2, 3, 4], index=1)
    h_rule = a3.selectbox("HMM-only trading rule", list(hmm.SIGNAL_RULES), format_func=hmm.SIGNAL_RULES.get)
    b1, b2, b3, b4 = st.columns(4)
    h_train = b1.number_input("First training period (years)", 2, 15, 5)
    h_refit = b2.number_input("Refit every (trading days)", 21, 756, 252, step=21)
    h_window = b3.selectbox("Training window", ["expanding", "rolling"],
                            format_func=lambda v: "Expanding (all history so far)" if v == "expanding" else "Rolling (last N years only)")
    h_init = b4.number_input("Random restarts per fit", 1, 10, 2, help="More restarts = less chance of a poor fit, slower training.")
    h_short = st.checkbox("HMM only: go short instead of flat when the rule is off", value=False)
    h_settings = dict(keys=tuple(h_keys), n_states=int(h_states), train_years=int(h_train), refit_days=int(h_refit),
                      window=h_window, n_init=int(h_init))
    if st.button("Train HMM", type="primary", disabled=not h_keys):
        bar = st.progress(0.0, text="Training the HMM walk-forward...")
        try:
            ss["hmm"] = hmm.walk_forward(gold, list(h_keys), int(h_states), int(h_train), int(h_refit), h_window, n_init=int(h_init),
                                         progress=lambda v: bar.progress(v, text=f"Training the HMM walk-forward... {v:.0%}"))
            ss["hmm_settings"] = h_settings
        except ValueError as e:
            st.error(str(e))
        bar.empty()

    H = ss.get("hmm")
    if H is None:
        st.info("Pick the inputs and press 'Train HMM' (about 30 seconds with the defaults).")
    else:
        if ss["hmm_settings"] != h_settings:
            st.warning("HMM settings have changed since training. Press 'Train HMM' to retrain. "
                       "(The trading rule and the short option apply without retraining.)")
        feat_all = features.build_features(gold, raw, params)
        h_start = max(pd.Timestamp(start), H["state"].index[0])
        h_end = pd.Timestamp(end)
        if h_end <= h_start:
            st.error("The test period ends before the HMM has finished its first training period.")
        else:
            period = (str(h_start.date()), str(h_end.date()))
            st.caption(f"HMM states are out of sample from {H['state'].index[0].date()} (after the first "
                       f"{ss['hmm_settings']['train_years']} years). Everything below is measured from {period[0]} to {period[1]}, "
                       f"{len(H['fits'])} refits.")
            try:
                h_res = engine.run_switch(feat_all, hmm.signal(H, h_rule), costs, capital, *period, rf, h_short)
            except ValueError as e:
                h_res = None
                st.error(str(e))

            st.markdown("**What each state looked like (out of sample, the day after the state was read)**")
            st.dataframe(hmm.state_summary(H, gold, h_start, h_end).round(2), hide_index=True, width="stretch")

            P, stt = H["probs"].loc[h_start:h_end], H["state"].loc[h_start:h_end]
            px = gold["Close"].reindex(stt.index)
            fig = go.Figure(go.Scatter(x=px.index, y=px.values, line=dict(color=GREY, width=1), showlegend=False, name="Gold close"))
            for k, nm in enumerate(H["names"]):
                d = px[stt == k]
                fig.add_scatter(x=d.index, y=d.values, mode="markers", name=nm, marker=dict(size=3, color=STATE_COLOURS[k % 4]))
            chart(fig, 380, yaxis_type="log", yaxis_title="USD per oz", title="Gold price coloured by the most likely HMM state", hovermode="closest")
            fig = go.Figure([go.Scatter(x=P.index, y=P[nm], name=nm, stackgroup="p", line=dict(width=0.5, color=STATE_COLOURS[k % 4]))
                             for k, nm in enumerate(H["names"])])
            chart(fig, 240, yaxis_title="Probability", yaxis_range=[0, 1], title="State probabilities (forward filter)")

            if h_res is not None:
                st.markdown(f"**HMM-only backtest:** {hmm.SIGNAL_RULES[h_rule].lower()}; fully invested (1x) while on, "
                            "no stop or target, same charges as the strategy.")
                hc = h_res["curves"]
                fig = go.Figure([go.Scatter(x=hc.index, y=hc[NET], name=HMM_ONLY, line=dict(color=BLUE, width=2)),
                                 go.Scatter(x=hc.index, y=hc[BH], name=BH, line=dict(color=GREY, width=1.5, dash="dot"))])
                chart(fig, 320, yaxis_type="log", yaxis_title="Account value (USD)", title="HMM only vs buy and hold")

            st.markdown("**Does the HMM separate the strategy's winners from its losers?** "
                        "The original strategy's trades, grouped by the HMM state on the signal bar.")
            try:
                o_res = engine.run_backtest(gold, raw, {**params, "use_regime": False, "use_hmm_gate": False}, costs, capital, *period, rf)
                ot = o_res["trades"].copy()
                if not ot.empty:
                    ot["hmm_state"] = hmm.label_at_entry(ot, stt.map(dict(enumerate(H["names"]))), o_res["feat"].index)
                l, r = st.columns(2)
                with l:
                    st.caption("By HMM state")
                    group_table(ot, "hmm_state", "HMM state")
                with r:
                    st.caption("By rule-based market type (same trades)")
                    group_table(ot, "regime", "Market type")
            except ValueError as e:
                o_res = None
                st.error(str(e))

            with st.expander(f"Refit log ({len(H['fits'])} fits): what the model learned each time"):
                st.dataframe(H["fits"], hide_index=True, width="stretch")

            # ---------- Side by side ----------
            st.divider()
            st.subheader("Side by side: original vs rule filter vs HMM only")
            st.caption(f"All on the same period, {period[0]} to {period[1]}, with the same charges.")
            if o_res is not None and h_res is not None:
                try:
                    r_res = engine.run_backtest(gold, raw, {**params, "use_regime": True, "use_hmm_gate": False}, costs, capital, *period, rf)
                    bh = o_res["metrics"][BH]
                    bh_row = {"CAGR %": bh["CAGR %"], "Max drawdown %": bh["Max drawdown %"], "Sharpe": bh["Sharpe"],
                              "Calmar": bh["Calmar"], "Volatility %": bh["Volatility %"], "Time in market %": 100.0}
                    compare_table({ORIG: engine.comparison_row(o_res), RULES: engine.comparison_row(r_res),
                                   HMM_ONLY: {k: (np.nan if k in R_ROWS else v) for k, v in engine.comparison_row(h_res).items()},
                                   BH: bh_row})
                    cv = pd.DataFrame({ORIG: o_res["curves"][NET], RULES: r_res["curves"][NET], HMM_ONLY: h_res["curves"][NET]})
                    fig = go.Figure([go.Scatter(x=cv.index, y=cv[n], name=n, line=sty) for n, sty in
                                     ((ORIG, dict(color=GREY, width=1.5)), (RULES, dict(color=BLUE, width=2)), (HMM_ONLY, dict(color=ORANGE, width=2)))])
                    fig.add_scatter(x=o_res["curves"].index, y=o_res["curves"][BH], name=BH, line=dict(color=GREY, width=1, dash="dot"))
                    chart(fig, 400, yaxis_type="log", yaxis_title="Account value (USD)", title="Account value, same period")
                    dd = (cv / cv.cummax() - 1) * 100
                    fig = go.Figure([go.Scatter(x=dd.index, y=dd[n], name=n, line=dict(color=col, width=1.5))
                                     for n, col in ((ORIG, GREY), (RULES, BLUE), (HMM_ONLY, ORANGE))])
                    chart(fig, 260, yaxis_title="Drawdown %", title="Drawdown from peak")
                    st.caption("HMM only is a different kind of strategy (a fully invested on/off switch), so compare drawdown, Sharpe "
                               "and Calmar more than trade counts. For Phase 2b the deciding table is the one above it: if the HMM "
                               "state separates winning and losing trades better than the rule-based market type, it is worth combining.")
                except ValueError as e:
                    st.error(str(e))

            # ---------- Phase 2b ----------
            st.divider()
            st.subheader("Phase 2b: strategy + rules + HMM combined")
            st.caption("The same strategy run four ways. The HMM gate multiplies the entry size by the multiplier of the HMM state "
                       "(sidebar group 'HMM gate (Phase 2b)'); the rule filter uses the 'Regime filter' group. The deciding numbers "
                       "are drawdown, Calmar, profit factor and how they hold up in both halves of the period.")
            variants = {ORIG: dict(use_regime=False, use_hmm_gate=False), RULES: dict(use_regime=True, use_hmm_gate=False),
                        HMM_GATE: dict(use_regime=False, use_hmm_gate=True), COMBINED: dict(use_regime=True, use_hmm_gate=True)}
            v_colours = {ORIG: GREY, RULES: BLUE, HMM_GATE: ORANGE, COMBINED: AQUA}
            try:
                v_res = {n: engine.run_backtest(gold, raw, {**params, **v}, costs, capital, *period, rf, hmm=H) for n, v in variants.items()}
                compare_table({n: engine.comparison_row(r) for n, r in v_res.items()})
                fig = go.Figure([go.Scatter(x=r["curves"].index, y=r["curves"][NET], name=n, line=dict(color=v_colours[n], width=2 if n == COMBINED else 1.5))
                                 for n, r in v_res.items()])
                chart(fig, 400, yaxis_type="log", yaxis_title="Account value (USD)", title="Account value: four ways")
                dd = pd.DataFrame({n: r["curves"][NET] for n, r in v_res.items()})
                dd = (dd / dd.cummax() - 1) * 100
                fig = go.Figure([go.Scatter(x=dd.index, y=dd[n], name=n, line=dict(color=v_colours[n], width=1.5)) for n in dd])
                chart(fig, 260, yaxis_title="Drawdown %", title="Drawdown from peak")

                st.markdown("**Does it hold up? Same four runs on each half of the period**")
                st.caption("A filter that only helps in one half is probably fitted to that half. Look for the same ranking in both.")
                mid = H["state"].loc[h_start:h_end].index[len(H["state"].loc[h_start:h_end]) // 2]
                halves = {f"1st half ({period[0]} to {mid.date()})": (period[0], str(mid.date())),
                          f"2nd half ({mid.date()} to {period[1]})": (str(mid.date()), period[1])}
                keep = ["CAGR %", "Max drawdown %", "Sharpe", "Calmar", "Profit factor", "Trades", "Longest losing streak"]
                cols = st.columns(2)
                for col, (label, (a_, b_)) in zip(cols, halves.items()):
                    with col:
                        st.caption(label)
                        half = {n: engine.comparison_row(engine.run_backtest(gold, raw, {**params, **v}, costs, capital, a_, b_, rf, hmm=H))
                                for n, v in variants.items()}
                        st.dataframe(pd.DataFrame(half).loc[keep].round(2), width="stretch")

                st.markdown("**Combined run: trades by HMM state and by market type**")
                st.caption("size_mult in the trade log is the final multiplier (rules x HMM). Sizing does not remove trades, "
                           "so the HMM gate alone leaves the trade count unchanged; it shrinks the money at risk.")
                ct = v_res[COMBINED]["trades"]
                l, r = st.columns(2)
                with l:
                    group_table(ct, "hmm_state", "HMM state")
                with r:
                    group_table(ct, "regime", "Market type")
                with st.expander(f"Combined run trade log ({len(ct)} trades)"):
                    st.dataframe(ct.round(2), hide_index=True, width="stretch")
                    st.download_button("Download combined trade log (CSV)", ct.to_csv(index=False), "trades_combined.csv", "text/csv")
                st.button("Use rules + HMM in the sidebar", on_click=load_into_sidebar, args=({"use_regime": True, "use_hmm_gate": True},),
                          help="Turns both gates on, so the Backtest and Parameter sweep tabs use the combined setup.")
            except ValueError as e:
                st.error(str(e))


# ---------------------------------------------------------
# Tab 4: vary parameters and scores
# ---------------------------------------------------------
with tab_sweep:
    st.caption("Every run uses the sidebar settings as the base and changes only what you pick here. "
               "The best row of a sweep is fitted to the past; check it on a period it has not seen (Test period slider) before trusting it.")
    mode = st.radio("What to vary", ["Chosen parameters (grid)", "Many parameters (random search)", "Factor scores (random search)"], horizontal=True)
    sweepable = [k for k, s in PARAMS.items() if s["sweep"]]
    label_of = lambda k: f"{PARAMS[k]['group']}: {PARAMS[k]['label']}"
    progress = st.empty()

    if mode.startswith("Chosen"):
        chosen = st.multiselect("Parameters to vary (up to 3)", sweepable, default=["atr_period"], format_func=label_of, max_selections=3)
        grid = {}
        for k in chosen:
            s = PARAMS[k]
            cast = int if s["kind"] == "int" else float
            a, b, c3 = st.columns(3)
            lo = a.number_input(f"From: {s['label']}", cast(s["lo"]), cast(s["hi"]), cast(s["lo"]), step=cast(s["step"]), key=f"sw_lo_{k}")
            hi = b.number_input("To", cast(s["lo"]), cast(s["hi"]), cast(s["hi"]), step=cast(s["step"]), key=f"sw_hi_{k}")
            step = c3.number_input("Step", cast(s["step"]), cast(s["hi"]), cast(s["step"]), step=cast(s["step"]), key=f"sw_st_{k}")
            grid[k] = [cast(round(v, 6)) for v in np.arange(lo, hi + step / 2, step)]
        n_runs = int(np.prod([len(v) for v in grid.values()])) if grid else 0
        st.write(f"{n_runs:,} runs")
        if n_runs > MAX_RUNS:
            st.error(f"More than {MAX_RUNS:,} runs. Use a bigger step or fewer parameters.")
        elif st.button("Run sweep", type="primary", disabled=n_runs == 0):
            bar = progress.progress(0.0)
            ss["sweep"] = engine.run_sweep(gold, raw, params, grid, progress=bar.progress, **run_kw)
            ss["sweep_keys"] = chosen
            progress.empty()
    elif mode.startswith("Many"):
        chosen = st.multiselect("Parameters to vary (any number)", sweepable, default=["atr_period", "sl_atr_mult", "entry_threshold", "tp_rr"], format_func=label_of)
        ranges = {}
        for k in chosen:
            s = PARAMS[k]
            cast = int if s["kind"] == "int" else float
            a, b, c3 = st.columns(3)
            lo = a.number_input(f"From: {s['label']}", cast(s["lo"]), cast(s["hi"]), cast(s["lo"]), step=cast(s["step"]), key=f"rs_lo_{k}")
            hi = b.number_input("To", cast(s["lo"]), cast(s["hi"]), cast(s["hi"]), step=cast(s["step"]), key=f"rs_hi_{k}")
            step = c3.number_input("Step", cast(s["step"]), cast(s["hi"]), cast(s["step"]), step=cast(s["step"]), key=f"rs_st_{k}")
            ranges[k] = (lo, hi, step, s["kind"])
        a, b = st.columns(2)
        n_rand = a.number_input("Number of random runs", 10, MAX_RUNS, 500, step=50, key="rs_n")
        seed = b.number_input("Random seed", 0, 9999, 0, key="rs_seed")
        st.caption("Each run draws one value for every chosen parameter from its range, so the cost is the number of runs however many "
                   "parameters you pick. Use the sensitivity charts below the results to find the few parameters that matter, "
                   "then run a grid on those.")
        if st.button("Run random search", type="primary", disabled=not ranges, key="rs_go"):
            bar = progress.progress(0.0)
            ss["sweep"] = engine.run_random_params(gold, raw, params, ranges, int(n_rand), int(seed), progress=bar.progress, **run_kw)
            ss["sweep_keys"] = chosen
            progress.empty()
    else:
        w_keys = st.multiselect("Factor scores to vary", [f"w_{k}" for k in FACTORS], default=[f"w_{k}" for k in FACTORS],
                                format_func=lambda k: PARAMS[k]["label"])
        a, b = st.columns(2)
        n_rand = a.number_input("Number of random score sets", 10, MAX_RUNS, 200, step=50)
        seed = b.number_input("Random seed", 0, 9999, 0)
        st.caption("All 8 scores from 0 to 10 is 11^8 = 214 million combinations, so this samples them instead of running every one.")
        if st.button("Run random search", type="primary", disabled=not w_keys):
            bar = progress.progress(0.0)
            ss["sweep"] = engine.run_random_weights(gold, raw, params, w_keys, int(n_rand), int(seed), progress=bar.progress, **run_kw)
            ss["sweep_keys"] = w_keys
            progress.empty()

    sw = ss.get("sweep")
    if sw is not None and not sw.empty:
        keys = ss["sweep_keys"]
        metric_cols = [c for c in sw.columns if c not in keys]
        rank_by = st.selectbox("Rank by", ["CAGR %", "Sharpe", "Calmar", "Max DD %", "Profit factor", "CAGR vs B&H %"])
        ranked = sw.sort_values(rank_by, ascending=False).reset_index(drop=True)
        best = ranked.iloc[0]
        st.markdown(f"**Best by {rank_by}:** " + ", ".join(f"{PARAMS[k]['label']} = {best[k]:g}" for k in keys)
                    + f"  |  CAGR {best['CAGR %']:.2f}%, max drawdown {best['Max DD %']:.1f}%, Sharpe {best['Sharpe']:.2f}")
        st.button("Load the best row into the sidebar", on_click=load_into_sidebar, args=({k: best[k] for k in keys},))

        if len(keys) == 1:
            line = sw.sort_values(keys[0])
            for col, colour in (("CAGR %", BLUE), ("Max DD %", RED)):
                fig = go.Figure(go.Scatter(x=line[keys[0]], y=line[col], mode="lines+markers", line=dict(color=colour, width=2), marker=dict(size=8)))
                chart(fig, 260, xaxis_title=PARAMS[keys[0]]["label"], yaxis_title=col, showlegend=False, title=f"{col} (net of charges)")
        elif len(keys) == 2:
            pivot = sw.pivot(index=keys[1], columns=keys[0], values=rank_by)
            fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Blues", colorbar=dict(title=rank_by),
                                       hovertemplate=f"{PARAMS[keys[0]]['label']}: %{{x}}<br>{PARAMS[keys[1]]['label']}: %{{y}}<br>{rank_by}: %{{z:.2f}}<extra></extra>"))
            chart(fig, 420, xaxis_title=PARAMS[keys[0]]["label"], yaxis_title=PARAMS[keys[1]]["label"], hovermode="closest", title=f"{rank_by} (net of charges)")
        else:
            text = [", ".join(f"{k[2:] if k.startswith('w_') else k}={r[k]:g}" for k in keys) for _, r in sw.iterrows()]
            fig = go.Figure(go.Scatter(x=sw["Max DD %"], y=sw["CAGR %"], mode="markers", text=text, marker=dict(size=8, color=BLUE, opacity=0.6),
                                       hovertemplate="%{text}<br>CAGR %{y:.2f}%<br>Max DD %{x:.1f}%<extra></extra>"))
            chart(fig, 420, xaxis_title="Max drawdown % (closer to 0 is better)", yaxis_title="CAGR % (net of charges)", hovermode="closest",
                  showlegend=False, title="Each dot is one run")

        if len(keys) > 3:
            st.subheader("Sensitivity")
            st.caption(f"Each dot is one run; the line is the average {rank_by} in 10 bins of that parameter. "
                       "A flat line means the parameter barely matters. Fix those and grid the ones with a clear slope or peak.")
            cols = st.columns(2)
            for i, k in enumerate(keys):
                d = sw[[k, rank_by]].dropna()
                fig = go.Figure(go.Scatter(x=d[k], y=d[rank_by], mode="markers", marker=dict(size=5, color=BLUE, opacity=0.35), name="runs"))
                if d[k].nunique() > 1:
                    binned = d.groupby(pd.cut(d[k], 10, duplicates="drop"), observed=True).agg(x=(k, "mean"), y=(rank_by, "mean"))
                    fig.add_trace(go.Scatter(x=binned["x"], y=binned["y"], mode="lines+markers", line=dict(color=RED, width=2), name="bin average"))
                with cols[i % 2]:
                    chart(fig, 260, xaxis_title=PARAMS[k]["label"], yaxis_title=rank_by, showlegend=False, hovermode="closest")

        st.dataframe(ranked.round(2), width="stretch", height=420,
                     column_config={k: st.column_config.NumberColumn(PARAMS[k]["label"]) for k in keys})
        st.download_button("Download sweep results (CSV)", ranked.to_csv(index=False), "sweep_results.csv", "text/csv")
    elif sw is not None:
        st.warning("No run produced a result. Check the test period.")


# ---------------------------------------------------------
# Tab 5: multi-timeframe backtest and walk-forward training on Binance bars
# ---------------------------------------------------------
TF_COLOURS = {"15m": ORANGE, "1h": AQUA, "4h": BLUE, "1d": RED}
GOLD_DAILY = "gold_daily"  # long-history source: the downloaded daily gold futures data (back to 2000), Daily layer only
SOURCE_LABEL = {**{k: v["label"] for k, v in binance.MARKETS.items()},
                GOLD_DAILY: "Gold futures daily, GC=F (back to 2000, Daily timeframe only)"}
WF, SB = "Walk-forward (tuned, out of sample)", "Sidebar settings (no tuning)"


def tf_summary(p):
    tfs = mtf.ticked(p)
    if not tfs:
        return "no timeframe ticked"
    short = lambda t: TIMEFRAMES[t]["label"].split(" (")[0]
    slower = ", ".join(short(t) for t in tfs[1:]) or "none"
    how = "top-down (slower timeframes must agree)" if p["mtf_mode"] == "top_down" else "weighted blend"
    return f"trades on **{short(tfs[0])}** bars | direction from: {slower} | combining: {how}"


def save_strategy_box(p, key):
    a, b = st.columns([3, 1])
    name = a.text_input("Save these settings as a strategy (name)", key=f"save_name_{key}", placeholder="e.g. 4h + daily, 1h reversion")
    b.markdown("&nbsp;")
    if b.button("Save strategy", key=f"save_btn_{key}", disabled=not name):
        st.success(f"Saved as '{live.save_strategy(name, p)}'. Pick it in the Live tab.")


def trades_by(trades, key, label):
    if trades.empty:
        st.info("No trades.")
        return
    t = trades.assign(_k=key(trades), _win=trades["net_pnl"] > 0)
    g = t.groupby("_k").agg(Trades=("net_pnl", "size"), **{"Win rate %": ("_win", "mean"), "Net P&L": ("net_pnl", "sum"),
                                                            "Avg R": ("r_multiple", "mean")})
    g["Win rate %"] *= 100
    st.dataframe(g.rename_axis(label).round(2), width="stretch")


with tab_mtf:
    st.caption("The bot on several timeframes at once, on Binance gold bars (24/7). Slower timeframes set the direction, the fastest "
               "ticked one times the entry. Tick timeframes in the sidebar under 'Include in backtest'; settings under 'Timeframes'. "
               "Every other sidebar setting (stops, sizing, volume / regime filters, charges) applies here too. Plan: MULTI_TIMEFRAME_PLAN.md.")
    a, b = st.columns([2, 3])
    mtf_symbol = a.selectbox("Data", list(SOURCE_LABEL) + mt5_markets(), format_func=lambda k: SOURCE_LABEL.get(k) or market_label(k),
                             key="mtf_symbol",
                             help="PAXGUSDT: intraday from Aug 2020 (the longest Binance has). XAUUSDT perpetual: only from Dec 2025, "
                                  "about 11 months, so results on it say little. Gold futures daily: 26 years, but only the Daily "
                                  "timeframe exists. Use it to check that settings found on a short intraday window also hold up over decades.")
    daily_only = mtf_symbol == GOLD_DAILY
    b.markdown("&nbsp;")
    b.markdown("trades on **daily** bars, the same strategy as the Backtest tab (timeframe ticks do not apply)" if daily_only else tf_summary(params))
    if daily_only:
        st.info("Long-history mode: the downloaded daily gold futures data, 26 years. Pick a charges preset such as 'XAUUSD CFD ...' "
                "in the sidebar, since Binance fees do not apply to this data.")
    elif not ss["cost_preset"].startswith("Binance") or mt5.is_mt5(mtf_symbol):
        if not mt5.is_mt5(mtf_symbol):
            st.info("Charges come from the sidebar. For Binance pick a 'Binance ...' preset under 'Charges and brokerage'. "
                    "Intraday trades chase small moves, so charges decide most results here.")
        else:
            st.info("MT5 data: pick an 'XAUUSD CFD ...' charges preset in the sidebar (spread, commission and swap of your broker).")
    bars = None
    if not daily_only:
        try:
            bars = load_bars(mtf_symbol, ss["refresh_id"])
        except Exception as e:
            st.error(f"Could not load Binance data: {e}")
    if not daily_only and bars is not None and not mtf.ticked(params):
        st.error("No timeframe is ticked. Tick at least one under 'Include in backtest'.")
    if daily_only or (bars is not None and mtf.ticked(params)):
        idx_ = gold.index if daily_only else bars.index
        bfirst, blast = idx_[0].date(), idx_[-1].date()
        mstart, mend = st.slider("Test period", bfirst, blast, (bfirst, blast), format="YYYY-MM-DD", key=f"mtf_period_{mtf_symbol}")
        hmm_obj = ss.get("hmm") if params["use_hmm_gate"] else None
        if params["use_hmm_gate"] and hmm_obj is None:
            st.warning("'HMM gate' is ticked but no HMM is trained yet, so it does nothing. Train it in the Regime lab tab.")
        mtf_inputs = (params, costs, capital, mstart, mend, rf, mtf_symbol)
        c1, c2 = st.columns(2)
        if c1.button("Run multi-timeframe backtest", type="primary"):
            with st.spinner("Building signals on every timeframe and running the backtest..."):
                try:
                    ss["mtf_res"] = mtf.run(bars, gold, raw, params, costs, capital, str(mstart), str(mend), rf, hmm_obj)
                    ss["mtf_inputs"] = mtf_inputs
                except ValueError as e:
                    st.error(str(e))
        if c2.button("Does each timeframe add value?", help="Runs: all ticked, each one removed, and each one alone.", disabled=daily_only):
            tfs = mtf.ticked(params)
            runs = {"All ticked": params}
            for tf in tfs if len(tfs) > 1 else []:
                runs[f"Without {tf}"] = {**params, f"inc_tf_{tf}": False}
            for tf in tfs if len(tfs) > 1 else []:
                runs[f"{tf} only"] = {**params, **{f"inc_tf_{k}": k == tf for k in TIMEFRAMES}}
            bar = st.progress(0.0, text="Running...")
            out = {}
            for j, (n, pv) in enumerate(runs.items()):
                try:
                    out[n] = engine.comparison_row(mtf.run(bars, gold, raw, pv, costs, capital, str(mstart), str(mend), rf, hmm_obj,
                                                           with_gross=False))
                except ValueError:
                    pass
                bar.progress((j + 1) / len(runs), text=f"Running... {n}")
            bar.empty()
            ss["mtf_loo"], ss["mtf_loo_inputs"] = out, mtf_inputs

        if ss.get("mtf_loo"):
            st.markdown("**Does each timeframe add value?** (net of charges)")
            if ss["mtf_loo_inputs"] != mtf_inputs:
                st.warning("Settings have changed since this run.")
            keep = ["CAGR %", "Max drawdown %", "Sharpe", "Calmar", "Profit factor", "Trades", "Win rate %", "Longest losing streak",
                    "Time in market %", "Total charges"]
            st.dataframe(pd.DataFrame(ss["mtf_loo"]).loc[keep].apply(pd.to_numeric, errors="coerce").round(2), width="stretch")
            st.caption("If removing a timeframe improves the result, it is not earning its place with these settings.")

        r = ss.get("mtf_res")
        if r is not None:
            if ss["mtf_inputs"] != mtf_inputs:
                st.warning("Settings have changed since this run. Press 'Run multi-timeframe backtest' to update.")
            m, x, f, curves, trades = r["metrics"], r["extra"], r["feat"], r["curves"], r["trades"]
            st.caption(f"{f.index[0]:%Y-%m-%d} to {f.index[-1]:%Y-%m-%d %H:%M} UTC, {len(f):,} bars of {r['base_tf']}. "
                       "Decided on each bar's close, filled at the next bar's open.")
            c = st.columns(6)
            c[0].metric("CAGR, net", f"{m.loc['CAGR %', NET]:.2f}%", f"{m.loc['CAGR %', NET] - m.loc['CAGR %', BH]:+.2f}% vs buy and hold")
            c[1].metric("CAGR, gross", f"{m.loc['CAGR %', GROSS]:.2f}%")
            c[2].metric("Max drawdown, net", f"{m.loc['Max drawdown %', NET]:.1f}%", f"buy and hold {m.loc['Max drawdown %', BH]:.1f}%",
                        delta_color="off")
            c[3].metric("Sharpe, net", f"{m.loc['Sharpe', NET]:.2f}")
            c[4].metric("Trades", f"{x['Trades']:,}", f"win rate {x['Win rate %']:.0f}%", delta_color="off")
            c[5].metric("Total charges", f"{x['Total charges']:,.0f}", f"-{x['CAGR lost to charges %']:.2f}% CAGR", delta_color="off")
            fig = go.Figure([go.Scatter(x=curves.index, y=curves[n], name=n, line=CURVE_STYLE[n]) for n in (NET, GROSS, BH)])
            chart(fig, 400, yaxis_title="Account value", yaxis_type="log", title="Account value")
            left, right = st.columns([3, 2])
            with left:
                st.dataframe(m.round(2), width="stretch")
            with right:
                order = ["Trades", "Win rate %", "Profit factor", "Avg win", "Avg loss", "Avg R", "Avg bars held",
                         "Longest losing streak", "Time in market %", "Total charges"]
                st.dataframe(pd.DataFrame({"Value": [f"{x[k]:,.2f}" if isinstance(x[k], float) else f"{x[k]:,}" for k in order]}, index=order),
                             width="stretch")

            if mt5.STATE["connected"] and not trades.empty:
                tgt = mt5.name_of(mtf_symbol) if mt5.is_mt5(mtf_symbol) else (mt5.STATE["symbol"] or "")
                if tgt and st.button(f"Draw these {len(trades)} trades on the MT5 {tgt} chart",
                                     help="Writes the entries, exits and lines to MT5; the GoldBotSignals indicator on the chart draws them. "
                                          "Set up the indicator in the MT5 tab first."):
                    try:
                        path_, n_ = mt5.write_chart_file(tgt, mt5.chart_rows(tgt, trades), title="backtest")
                        st.success(f"{n_} rows sent to MT5 ({path_.name}). Open the {tgt} chart; scroll back to the trade dates.")
                    except Exception as e:
                        st.error(f"Could not write the chart file: {e}")

            days = st.select_slider("Show the last", [3, 7, 14, 30, 90], value=14, format_func=lambda d: f"{d} days", key="mtf_days")
            w = f[f.index >= f.index[-1] - pd.Timedelta(days=days)]
            fig = go.Figure(go.Candlestick(x=w.index, open=w["Open"], high=w["High"], low=w["Low"], close=w["Close"], name="Price",
                                           showlegend=False))
            if not trades.empty:
                tw = trades[trades["entry_date"] >= w.index[0]]
                fig.add_scatter(x=tw["entry_date"], y=tw["entry_price"], mode="markers", name="Entry",
                                marker=dict(symbol="triangle-up", size=10, color=AQUA))
                tx = trades[trades["exit_date"] >= w.index[0]]
                fig.add_scatter(x=tx["exit_date"], y=tx["exit_price"], mode="markers", name="Exit",
                                marker=dict(symbol="triangle-down", size=10, color=RED))
            chart(fig, 380, yaxis_title="USD", title=f"{r['base_tf']} bars with entries and exits", xaxis_rangeslider_visible=False,
                  hovermode="closest")
            sig_cols = [c_ for c_ in w.columns if c_.startswith("sig_") and c_[4:] in TIMEFRAMES]
            fig = go.Figure([go.Scatter(x=w.index, y=w[c_], name=TIMEFRAMES[c_[4:]]["label"].split(" (")[0],
                                        line=dict(color=TF_COLOURS[c_[4:]], width=1.2)) for c_ in sig_cols])
            fig.add_scatter(x=w.index, y=w["score"], name="Combined score", line=dict(color="black", width=2))
            fig.add_hline(y=r["params"]["entry_threshold"], line=dict(color=AQUA, dash="dash"))
            fig.add_hline(y=-r["params"]["entry_threshold"], line=dict(color=RED, dash="dash"))
            chart(fig, 300, yaxis_title="Signal (-1 to +1)", yaxis_range=[-1.05, 1.05], title="Signal on each timeframe and the combined score")

            l, rr = st.columns(2)
            with l:
                if not daily_only:
                    st.markdown("**Trades by hour of entry (UTC)**")
                    trades_by(trades, lambda t: t["entry_date"].dt.hour, "Hour (UTC)")
            with rr:
                st.markdown("**Trades by year**")
                trades_by(trades, lambda t: t["entry_date"].dt.year, "Year")
            with st.expander(f"Trade log ({len(trades)} trades)"):
                st.dataframe(trades.round(4), hide_index=True, width="stretch")
                st.download_button("Download trade log (CSV)", trades.to_csv(index=False), "trades_mtf.csv", "text/csv")

        # ---------- training ----------
        st.divider()
        st.subheader("Train the bot (walk-forward)")
        st.caption("Tunes the chosen parameters on a past window, tests the winner on the next window it has never seen, then rolls "
                   "forward. Only the unseen test windows count in the result. The winner on the latest window (ending today) is the "
                   "set to use live. Uses the market and every sidebar setting above as the base.")
        sweepable = [k for k, s_ in PARAMS.items() if s_["sweep"]]
        label_of = lambda k: f"{PARAMS[k]['group']}: {PARAMS[k]['label']}"
        chosen = st.multiselect("Parameters to tune", sweepable, default=["entry_threshold", "sl_atr_mult", "tp_rr", "mtf_win"],
                                format_func=label_of, key="wf_keys")
        ranges = {}
        for k in chosen:
            s_ = PARAMS[k]
            cast = int if s_["kind"] == "int" else float
            a, b, c3 = st.columns(3)
            lo = a.number_input(f"From: {s_['label']}", cast(s_["lo"]), cast(s_["hi"]), cast(s_["lo"]), step=cast(s_["step"]), key=f"wf_lo_{k}")
            hi = b.number_input("To", cast(s_["lo"]), cast(s_["hi"]), cast(s_["hi"]), step=cast(s_["step"]), key=f"wf_hi_{k}")
            step = c3.number_input("Step", cast(s_["step"]), cast(s_["hi"]), cast(s_["step"]), step=cast(s_["step"]), key=f"wf_st_{k}")
            ranges[k] = (lo, hi, step, s_["kind"])
        a, b, c3, d, e, g = st.columns(6)
        n_sets = a.number_input("Parameter sets to try", 5, 500, 20, step=5, key="wf_n")
        train_days = b.number_input("Training window (days)", 60, 9000, 1095 if daily_only else 365, step=30, key=f"wf_train_{mtf_symbol}")
        test_days = c3.number_input("Test window (days)", 14, 1500, 365 if daily_only else 90, step=14, key=f"wf_test_{mtf_symbol}")
        score_by = d.selectbox("Pick the winner by", list(mtf.SCORERS), key="wf_score")
        min_tr = e.number_input("Min trades in training", 0, 500, 10, key="wf_min")
        seed = g.number_input("Random seed", 0, 9999, 0, key="wf_seed")
        base_now = "1d" if daily_only else mtf.ticked(params)[0]
        per_set = {"15m": 12, "1h": 3, "4h": 1, "1d": 0.5}[base_now]
        st.caption(f"Rough time: about {max(1, round(per_set * (n_sets + 1) / 60))} minute(s) on {base_now} bars "
                   "(each parameter set rebuilds the signals on the full history).")
        if st.button("Train (walk-forward)", type="primary", disabled=not ranges, key="wf_go"):
            bar = st.progress(0.0, text="Training...")
            try:
                ss["mtf_wf"] = mtf.walk_forward(bars, gold, raw, params, ranges, int(n_sets), int(train_days), int(test_days), score_by,
                                                int(min_tr), costs, capital, rf, hmm_obj, int(seed),
                                                progress=lambda v: bar.progress(min(v, 1.0), text=f"Training... {v:.0%}"))
                ss["mtf_wf_inputs"] = (params, costs, mtf_symbol)
            except ValueError as e_:
                st.error(str(e_))
            bar.empty()
        wf = ss.get("mtf_wf")
        if wf is not None:
            if ss["mtf_wf_inputs"] != (params, costs, mtf_symbol):
                st.warning("Sidebar settings have changed since this training run.")
            wm = wf["metrics"]
            c = st.columns(4)
            c[0].metric("CAGR, tuned (unseen data)", f"{wm.loc['CAGR %', WF]:.2f}%", f"{wm.loc['CAGR %', WF] - wm.loc['CAGR %', SB]:+.2f}% vs untuned")
            c[1].metric("Max drawdown, tuned", f"{wm.loc['Max drawdown %', WF]:.1f}%", f"untuned {wm.loc['Max drawdown %', SB]:.1f}%",
                        delta_color="off")
            c[2].metric("Sharpe, tuned", f"{wm.loc['Sharpe', WF]:.2f}", f"buy and hold {wm.loc['Sharpe', BH]:.2f}", delta_color="off")
            c[3].metric("Trades (unseen data)", f"{wf['trade_stats']['Trades']:,}", f"win rate {wf['trade_stats']['Win rate %']:.0f}%",
                        delta_color="off")
            cv = wf["curves"]
            fig = go.Figure([go.Scatter(x=cv.index, y=cv[n], name=n, line=sty) for n, sty in
                             ((WF, dict(color=BLUE, width=2)), (SB, dict(color=GREY, width=1.5)), (BH, dict(color=GREY, width=1, dash="dot")))])
            chart(fig, 380, yaxis_type="log", yaxis_title="Account value", title="Out-of-sample account value (test windows joined)")
            st.dataframe(wm.round(2), width="stretch")
            st.markdown("**Each fold: what won in training and how it did on the next, unseen window**")
            st.dataframe(wf["folds"].round(3), hide_index=True, width="stretch")
            a_, b_ = wf["latest_window"]
            st.markdown(f"**Settings to use live:** winner on the latest training window ({a_:%Y-%m-%d} to {b_:%Y-%m-%d}), "
                        f"{score_by} {wf['latest_score']:.2f}")
            if wf["latest_params"]:
                st.dataframe(pd.DataFrame([wf["latest_params"]]), hide_index=True)
                st.button("Load these settings into the sidebar", on_click=load_into_sidebar, args=(wf["latest_params"],), key="wf_load")
            else:
                st.info("The sidebar settings themselves won on the latest window (or no set had enough trades).")
            st.caption("Load them into the sidebar, then save them as a strategy below to run them in the Live tab. "
                       "If the tuned result on unseen data is not better than untuned, the tuning is fitting noise.")

        st.divider()
        save_strategy_box(params, "mtf")

# ---------------------------------------------------------
# Tab 6: MetaTrader 5 (XAUUSD): connect, history, and drawing the bot's trades on the MT5 chart
# ---------------------------------------------------------
with tab_mt5:
    st.caption("Use your broker's XAUUSD from the MetaTrader 5 terminal installed on this PC. Read only: the app reads prices and history and "
               "draws on the chart, and never places an order. The password is kept in memory only, not saved anywhere.")
    if not mt5.available():
        st.error("The MetaTrader5 Python package is not installed. Run:  pip install MetaTrader5")
    else:
        sc = mt5.saved_connection()
        with st.expander("Connection", expanded=not mt5.STATE["connected"]):
            c1, c2, c3 = st.columns(3)
            m_login_txt = c1.text_input("Login (account number)", str(sc.get("login", "") or ""), key="mt5_login",
                                        help="Leave empty (with the password empty) to use the account the MT5 terminal is already logged in to.")
            m_login = int(m_login_txt) if m_login_txt.strip().isdigit() else 0
            if m_login_txt.strip() and not m_login_txt.strip().isdigit():
                c1.error("The login is digits only.")
            m_pass = c2.text_input("Password", type="password", key="mt5_pass")
            m_server = c3.text_input("Server", sc.get("server", ""), key="mt5_server", help="Exactly as in MT5, for example  ICMarkets-Demo.")
            m_path = st.text_input("Terminal path (optional)", sc.get("path", ""), key="mt5_path",
                                   placeholder=r"C:\Program Files\MetaTrader 5\terminal64.exe",
                                   help="Only needed if you have several MT5 terminals installed.")
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("Connect", type="primary", key="mt5_connect"):
                with st.spinner("Connecting to MT5 (this starts the terminal if it is closed)..."):
                    ok = mt5.connect(m_login, m_pass, m_server, m_path)
                if ok:
                    st.rerun()
            if b2.button("Disconnect", key="mt5_disconnect", disabled=not mt5.STATE["connected"]):
                mt5.disconnect()
                st.rerun()
        if mt5.STATE["error"] and not mt5.STATE["connected"]:
            st.error(mt5.STATE["error"])
        inf = mt5.info() if mt5.STATE["connected"] else None
        if inf is None:
            st.info("Not connected. Enter your broker login, password and server (a demo account is fine), then press Connect.")
        else:
            a_, t_ = inf["account"], inf["terminal"]
            kind = "DEMO account" if inf["demo"] else "REAL account"
            (st.success if inf["demo"] else st.warning)(f"Connected to {a_['server']} as {a_['name']} ({a_['login']}), {kind}. "
                                                         f"Nothing is traded by this app.")
            c = st.columns(5)
            c[0].metric("Balance", f"{a_['balance']:,.2f} {a_['currency']}")
            c[1].metric("Equity", f"{a_['equity']:,.2f}")
            c[2].metric("Leverage", f"1:{a_['leverage']}")
            c[3].metric("Terminal build", f"{t_['build']}")
            c[4].metric("Terminal link", "online" if t_["connected"] else "OFFLINE")

            syms = mt5.find_symbols()
            if not syms:
                st.error("No gold symbol found. In MT5 open Market Watch, right click, Symbols, search 'XAU' and show it.")
            else:
                pick = st.selectbox("Gold symbol at your broker", syms, key="mt5_symbol",
                                    help="Names differ by broker (XAUUSD, GOLD, XAUUSDm ...). Pick the plain spot gold one.")
                try:
                    si = mt5.select_symbol(pick)
                    q = mt5.live_price(mt5.PREFIX + pick)
                    spread_usd = q["ask"] - q["bid"]
                    c = st.columns(5)
                    c[0].metric("Bid", f"{q['bid']:,.{si['digits']}f}")
                    c[1].metric("Ask", f"{q['ask']:,.{si['digits']}f}", f"spread {spread_usd:.2f} USD", delta_color="off")
                    c[2].metric("Contract size", f"{si['trade_contract_size']:g} oz / lot", f"min lot {si['volume_min']:g}", delta_color="off")
                    c[3].metric("Swap long / short", f"{si['swap_long']:g} / {si['swap_short']:g}", "per lot per night, in broker units", delta_color="off")
                    c[4].metric("Server clock", f"UTC{mt5.server_offset_hours(pick):+g}",
                                "bars are converted to UTC", delta_color="off")
                    st.caption("Put your broker's spread, swap and commission into the sidebar charges (preset 'XAUUSD CFD ...'). "
                               "A closed market (weekend) shows the last quote.")
                except Exception as e:
                    st.error(f"Could not read {pick}: {e}")
                    pick = None

                if pick:
                    st.subheader("History for backtests and training")
                    cached = mt5.STORE / f"{pick}_15m.csv"
                    h1, h2 = st.columns([1, 2])
                    yrs = h1.number_input("Years to download", 1, 20, 10, key="mt5_years",
                                          help="Brokers keep a limited history. You get what the server has, often 5 to 10 years of 15m bars.")
                    if h2.button(f"Download / update {pick} 15m history", key="mt5_dl"):
                        with st.spinner("Downloading from MT5..."):
                            try:
                                mt5.klines(mt5.PREFIX + pick, update=True, years=int(yrs))
                                load_bars.clear()
                                st.success("Done.")
                            except Exception as e:
                                st.error(f"Download failed: {e}")
                    if cached.exists():
                        hb = mt5.klines(mt5.PREFIX + pick, update=False)
                        st.caption(f"Stored: {len(hb):,} bars from {hb.index[0]:%Y-%m-%d} to {hb.index[-1]:%Y-%m-%d %H:%M} UTC "
                                   f"({(hb.index[-1] - hb.index[0]).days / 365.25:.1f} years). Choose it in the Multi-timeframe tab under 'Data', "
                                   f"and as the 'Live market' in the Live tab. Volume is MT5 tick volume (number of quotes), not traded lots.")
                    else:
                        st.caption("No history stored yet.")

                    st.subheader("Signals on the MT5 chart")
                    st.markdown(
                        "1. Press **Install indicator** (copies and compiles `GoldBotSignals` into your terminal).  \n"
                        f"2. In MT5 open a **{pick}** chart (any timeframe), then Navigator, Indicators, **GoldBotSignals**, drag it onto the chart.  \n"
                        "3. In the **Live** tab keep 'Draw signals on the MT5 chart' on, or press the draw button under a Multi-timeframe backtest. "
                        "The chart shows buy / sell arrows, an x where each trade exits, a line from entry to exit (green win, red loss, "
                        "yellow open) and the stop and target of an open trade.")
                    i1, i2 = st.columns([1, 2])
                    if i1.button("Install indicator", key="mt5_install"):
                        with st.spinner("Copying and compiling..."):
                            ok_, msg_ = mt5.install_indicator()
                        (st.success if ok_ else st.warning)(msg_)
                    if i2.button("Draw a test marker on the chart", key="mt5_test",
                                 help="Puts a buy arrow at the current price and a sell arrow 1 hour ago, to check the indicator is working."):
                        now_ = pd.Timestamp.now(tz="UTC").tz_localize(None)
                        rows_ = mt5.chart_rows(pick, pd.DataFrame(), signals=[(now_ - pd.Timedelta(hours=1), q["ask"], "SELL", "test sell"),
                                                                              (now_, q["bid"], "BUY", "test buy")])
                        path_, n_ = mt5.write_chart_file(pick, rows_, title="test")
                        st.success(f"Wrote {n_} rows to {path_}. The chart should update within 2 seconds.")

# ---------------------------------------------------------
# Tab 7: live paper trading (Binance or MT5 feed)
# ---------------------------------------------------------
with tab_live:
    st.caption("Paper trading on the live Binance gold feed. Nothing is sent to the exchange. The session is the same backtest "
               "engine run from the moment you press Start to the last closed bar, so live and backtest can be compared directly. "
               "It keeps going when the browser is closed, because the result is recalculated from the start time. For a log "
               "without the browser, run  python live_runner.py  in a terminal.")
    sess = live.get_session()
    if sess is None:
        CURRENT = "Current sidebar settings"
        a, b = st.columns(2)
        pick = a.selectbox("Strategy", [CURRENT] + live.list_strategies(), key="live_pick",
                           help="Save strategies in the Multi-timeframe tab (or below).")
        sym = b.selectbox("Live market", mt5_markets() + list(binance.MARKETS), format_func=market_label, key="live_symbol",
                          help="XAUUSD from MetaTrader 5 needs the terminal running on this PC (connect in the MT5 tab). "
                               "Binance works anywhere except US servers.")
        if mt5.is_mt5(sym) and not mt5.STATE["connected"]:
            st.warning("MT5 is not connected. Connect in the MT5 tab first, or the session will have no price.")
        p_sel = params if pick == CURRENT else live.load_strategy(pick)
        c1, c2 = st.columns(2)
        lcap = c1.number_input("Paper capital (USD)" if mt5.is_mt5(sym) else "Paper capital (USDT)", 100, 10_000_000, 10_000, step=1_000, key="live_cap")
        presets = list(COST_PRESETS)
        default = ("XAUUSD CFD, spread only" if mt5.is_mt5(sym) else
                   "Binance PAXGUSDT spot (0.1% fee)" if sym == "PAXGUSDT" else "Binance XAUUSDT perpetual (0.05% taker)")
        lpreset = c2.selectbox("Charges", presets, index=presets.index(default), key=f"live_preset_{sym}")
        factors_on = ", ".join(FACTORS[k]["label"] for k in FACTORS if p_sel[f"inc_{k}"] and p_sel[f"w_{k}"] > 0) or "none"
        filters_on = ", ".join(n for n, on in (("RVOL spike", p_sel["use_rvol"]), ("volume dry-up", p_sel["use_vdu"]),
                                               ("regime rules", p_sel["use_regime"]), ("HMM gate", p_sel["use_hmm_gate"])) if on) or "none"
        st.markdown(f"**{pick}:** {tf_summary(p_sel)}  \n"
                    f"Factors in the daily score: {factors_on}  \n"
                    f"Filters: {filters_on} | entry at score {p_sel['entry_threshold']:+.2f}, stop {p_sel['sl_atr_mult']} x ATR, "
                    f"target {p_sel['tp_rr']} x stop, shorts {'on' if p_sel['allow_short'] else 'off'}")
        if sym == "PAXGUSDT" and p_sel["allow_short"]:
            st.warning("PAXGUSDT is spot and cannot be shorted on Binance. Short paper trades are shown, but you could not place them.")
        if not mtf.ticked(p_sel):
            st.error("This strategy has no timeframe ticked.")
        elif st.button("Start live paper session", type="primary"):
            live.start_session(pick, p_sel, sym, COST_PRESETS[lpreset], float(lcap))
            st.rerun()
        st.divider()
        save_strategy_box(params, "live")
    else:
        sp = {**DEFAULT_PARAMS, **sess["params"]}
        st.markdown(f"**Running:** strategy '{sess['strategy']}' on **{sess['symbol']}** since {sess['started']} UTC, "
                    f"paper capital {sess['capital']:,.0f}  \n{tf_summary(sp)}")
        a, b, c3, d = st.columns(4)
        auto = a.toggle("Auto refresh", value=True, key="live_auto")
        every = b.selectbox("Strategy panel refresh", [2, 5, 15, 30, 60, 120], index=1, format_func=lambda v: f"every {v} seconds",
                            key="live_every",
                            help="Signals are only rebuilt when a new 15m bar closes; between bars this just re-checks the live price against the position.")
        rec = c3.toggle("Record ticks to disk", value=False, key="live_rec",
                        help="Saves trades and 1-second quote snapshots to data/ticks/<symbol>_<date>.csv, for later study.")
        if d.button("Stop session"):
            live.stop_session()
            st.rerun()
        to_mt5 = False
        if mt5.is_mt5(sess["symbol"]):
            to_mt5 = st.toggle("Draw signals on the MT5 chart", value=True, key="live_to_mt5", disabled=not mt5.STATE["connected"],
                               help="Every refresh writes the session's entries, exits, stop and target for the GoldBotSignals "
                                    "indicator on the chart. Needs MT5 connected and the indicator on the chart (MT5 tab).")
        stream = ticks.get_stream(sess["symbol"])
        stream.record = bool(rec)

        @st.fragment(run_every="1s" if auto else None)
        def tick_panel():
            """Tick by tick: best bid / ask, trades, and the bar in progress, refreshed every second."""
            stt = stream.status()
            if not stt["connected"]:
                st.warning(f"Tick feed not connected ({stt['error'] or 'connecting...'}). The REST price is used instead. "
                           f"Reconnect attempts: {stt['reconnects']}. " + ("Check MT5 is connected and the market is open."
                                                                          if mt5.is_mt5(sess["symbol"]) else "Binance blocks US-hosted servers (HTTP 451)."))
                return
            fb = stream.forming()
            c = st.columns(6)
            c[0].metric("Best bid", f"{stream.bid:,.2f}" if stream.bid == stream.bid else "-")
            c[1].metric("Best ask", f"{stream.ask:,.2f}" if stream.ask == stream.ask else "-",
                        f"spread {stream.ask - stream.bid:.2f}" if stream.ask == stream.ask else None, delta_color="off")
            c[2].metric("Last trade", f"{stream.last:,.2f}" if stream.last == stream.last else "waiting",
                        f"{stream.last_qty:g} oz, {stream.last_side}" if stream.last == stream.last else None, delta_color="off")
            c[3].metric("Ticks per second", f"{stt['rate']:.1f}", f"{stt['quotes']:,} quotes, {stt['trades']:,} trades", delta_color="off")
            c[4].metric("Last tick", f"{stt['age']:.1f}s ago", f"{stt['reconnects']} reconnects", delta_color="off")
            c[5].metric("15m bar so far", f"{fb['close']:,.2f}" if fb else "-",
                        f"O {fb['open']:,.2f}  H {fb['high']:,.2f}  L {fb['low']:,.2f}" if fb else None, delta_color="off")
            left, right = st.columns([3, 2])
            mids = stream.recent_mid(300)
            with left:
                if len(mids) > 1:
                    fig = go.Figure(go.Scatter(x=mids["time"], y=mids["mid"], mode="lines", line=dict(color=BLUE, width=1.2, shape="hv"), name="Mid"))
                    chart(fig, 240, title="Last 5 minutes, every quote change", showlegend=False, yaxis_title="USD per oz")
                else:
                    st.caption("Waiting for quotes...")
            with right:
                st.caption("Trade tape (aggressor = the side that hit the book)")
                tp = stream.tape(8)
                if len(tp):
                    st.dataframe(tp.round(4), hide_index=True, width="stretch")
                else:
                    st.caption("No trades yet (spot gold trades only every few seconds).")

        tick_panel()

        @st.fragment(run_every=f"{every}s" if auto else None)
        def live_panel():
            try:
                snap = live.snapshot(sess, gold, raw, ss.get("hmm") if sp["use_hmm_gate"] else None)
            except Exception as e:
                st.error(f"Live update failed: {type(e).__name__}: {e}")
                return
            live.log_events(snap)
            if to_mt5 and mt5.STATE["connected"]:
                try:
                    live.write_mt5_chart(snap)
                except Exception as e:
                    st.warning(f"Could not update the MT5 chart: {e}")
            pr, pos, f, closed = snap["price"], snap["position"], snap["feat"], snap["closed"]
            realised = closed["net_pnl"].sum() if not closed.empty else 0.0
            eq_live = sess["capital"] + realised + (pos["unrealised_pnl"] if pos is not None else 0.0)
            c = st.columns(5)
            c[0].metric(f"{sess['symbol']} live", f"{pr['price']:,.2f}", f"bid {pr['bid']:,.2f} / ask {pr['ask']:,.2f} ({pr.get('source')})", delta_color="off")
            c[1].metric("Last closed bar", f"{snap['last_bar']:%H:%M} UTC", f"{snap['base_tf']} bars, next close {snap['next_close']:%H:%M}",
                        delta_color="off")
            c[2].metric("Position", "Flat" if pos is None else f"{pos['side']} {pos['qty']:.4f}",
                        None if pos is None else f"entry {pos['entry_price']:,.2f}", delta_color="off")
            c[3].metric("Unrealised P&L", "-" if pos is None else f"{pos['unrealised_pnl']:+,.2f}")
            c[4].metric("Paper equity", f"{eq_live:,.2f}", f"{(eq_live / sess['capital'] - 1) * 100:+.2f}% since start")
            st.markdown(f"**Signal on the last closed bar:** {live.decision_text(f, sp)}")
            if pr.get("source") != "websocket":
                st.warning(f"Price source: {pr.get('source')}. " + (f"Live price unavailable ({pr['error']}). " if "error" in pr else "")
                           + "Stops and fills use only closed bars until the tick feed connects.")

            row = f.iloc[-1]
            sig_rows = [{"Layer": TIMEFRAMES[t]["label"], "Signal (-1 to +1)": row.get(f"sig_{t}")} for t in mtf.ticked(sp)]
            sig_rows.append({"Layer": "Combined score", "Signal (-1 to +1)": row["score"]})
            l, r_ = st.columns([2, 3])
            with l:
                st.dataframe(pd.DataFrame(sig_rows).round(3), hide_index=True, width="stretch")
                st.caption(f"Volume z-score {row['rv_z']:+.2f}" + (f" | market type: {row['regime_label']}" if sp["use_regime"] else "")
                           + (" | volume filter blocks longs" if not row["vol_ok_long"] else ""))
            with r_:
                fl = snap["feat_live"]
                w = fl[fl.index >= fl.index[-1] - pd.Timedelta(days=3)]
                fig = go.Figure(go.Candlestick(x=w.index, open=w["Open"], high=w["High"], low=w["Low"], close=w["Close"],
                                               showlegend=False, name="Price"))
                fig.add_hline(y=pr["price"], line=dict(color=BLUE, dash="dot"), annotation_text="live")
                if snap["started"] >= w.index[0]:
                    fig.add_vline(x=snap["started"], line=dict(color=GREY, dash="dash"))
                res = snap["res"]
                if res is not None and not res["trades"].empty:
                    t = res["trades"]
                    fig.add_scatter(x=t["entry_date"], y=t["entry_price"], mode="markers", name="Entry",
                                    marker=dict(symbol="triangle-up", size=11, color=AQUA))
                    tc = t[t["exit_reason"] != "End of test"]
                    fig.add_scatter(x=tc["exit_date"], y=tc["exit_price"], mode="markers", name="Exit",
                                    marker=dict(symbol="triangle-down", size=11, color=RED))
                if pos is not None:
                    p_ = res["params"]
                    atr_prev = f["ATR"].loc[:pos["entry_date"]]
                    stop_dist = p_["sl_atr_mult"] * (atr_prev.iloc[-2] if len(atr_prev) > 1 else atr_prev.iloc[-1])
                    sgn = 1 if pos["side"] == "Long" else -1
                    if p_["use_sl"]:
                        fig.add_hline(y=pos["entry_price"] - sgn * stop_dist, line=dict(color=RED, width=1), annotation_text="stop")
                    if p_["use_tp"]:
                        fig.add_hline(y=pos["entry_price"] + sgn * stop_dist * p_["tp_rr"], line=dict(color=AQUA, width=1),
                                      annotation_text="target")
                chart(fig, 340, title=f"Last 3 days, {snap['base_tf']} bars; the last bar is still forming (dashed line = session start)",
                      xaxis_rangeslider_visible=False, hovermode="closest")
            if snap["equity"] is not None and len(snap["equity"]) > 1:
                fig = go.Figure(go.Scatter(x=snap["equity"].index, y=snap["equity"].values, line=dict(color=BLUE, width=2), name="Paper equity"))
                chart(fig, 220, yaxis_title="Equity", title="Paper equity since start (closed bars)", showlegend=False)
            st.markdown(f"**Closed paper trades ({len(closed)})**")
            if closed.empty:
                st.caption("None yet.")
            else:
                st.dataframe(closed.round(4), hide_index=True, width="stretch")
            st.caption(f"Updated {pd.Timestamp.now(tz='UTC'):%H:%M:%S} UTC. Entries and exits are also logged to data/live/events.csv.")

        live_panel()
