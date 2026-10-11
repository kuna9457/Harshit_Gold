"""Gold (XAUUSD) macro strategy backtester. Run with:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gold_bot import data, engine, features
from gold_bot.config import (CATEGORY_ORDER, COST_FIELDS, COST_PRESETS, DATA_INPUTS, DEFAULT_COST_PRESET,
                             DEFAULT_PARAMS, FACTORS, GROUP_ORDER, PARAMS)

st.set_page_config(page_title="Gold macro backtester", layout="wide")

BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984"
NET, GROSS, BH = "Strategy (net of charges)", "Strategy (gross)", "Buy and hold gold"
CURVE_STYLE = {NET: dict(color=BLUE, width=2), GROSS: dict(color=ORANGE, width=2), BH: dict(color=GREY, width=2, dash="dot")}
CHOICE_TEXT = {1: "+1  rising is bullish for gold", -1: "-1  rising is bearish for gold"}
MAX_RUNS = 3000


def chart(fig, height=340, **layout):
    layout.setdefault("hovermode", "x unified")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), **layout)
    st.plotly_chart(fig, width="stretch")


@st.cache_data(show_spinner="Downloading market and macro data...")
def load(refresh_id):
    return data.load_all(refresh=refresh_id > 0)


# ---------------------------------------------------------
# Session state: every parameter lives under p_<name>, every charge under c_<name>
# ---------------------------------------------------------
ss = st.session_state
ss.setdefault("refresh_id", 0)
for k, v in DEFAULT_PARAMS.items():
    ss.setdefault(f"p_{k}", v)
ss.setdefault("cost_preset", DEFAULT_COST_PRESET)
for k, v in COST_PRESETS[DEFAULT_COST_PRESET].items():
    ss.setdefault(f"c_{k}", float(v))


def reset_params():
    for k, v in DEFAULT_PARAMS.items():
        ss[f"p_{k}"] = v


def apply_preset():
    for k, v in COST_PRESETS[ss["cost_preset"]].items():
        ss[f"c_{k}"] = float(v)


def load_into_sidebar(values):
    for k, v in values.items():
        ss[f"p_{k}"] = type(DEFAULT_PARAMS[k])(v)


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
    capital = st.number_input("Initial capital (USD)", 1_000, 100_000_000, 100_000, step=10_000)
    rf = st.number_input("Risk-free rate for Sharpe (% a year)", 0.0, 15.0, 4.0, step=0.25) / 100

    with st.expander("Charges and brokerage", expanded=False):
        st.selectbox("Preset", list(COST_PRESETS), key="cost_preset", on_change=apply_preset)
        for k, (label, lo, hi, step) in COST_FIELDS.items():
            st.number_input(label, float(lo), float(hi), step=float(step), key=f"c_{k}", format="%.4f")
        st.caption("Presets are typical retail levels. Replace them with the numbers on your broker's contract note.")

    st.header("Strategy parameters")
    for group in GROUP_ORDER:
        with st.expander(group, expanded=group == GROUP_ORDER[0]):
            for name, spec in PARAMS.items():
                if spec["group"] == group:
                    param_widget(name)
    st.button("Reset parameters to defaults", on_click=reset_params)
    if st.button("Refresh data from sources"):
        ss["refresh_id"] += 1
        st.rerun()

params = {k: ss[f"p_{k}"] for k in PARAMS}
costs = {k: ss[f"c_{k}"] for k in COST_FIELDS}
run_kw = dict(costs=costs, capital=capital, start=str(start), end=str(end), rf=rf)

st.title("Gold (XAUUSD) macro strategy backtester")
tab_data, tab_bt, tab_sweep = st.tabs(["Data and parameters", "Backtest", "Parameter sweep"])

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
    total_w = sum(params[f"w_{k}"] for k in FACTORS) or 1
    st.dataframe(pd.DataFrame([{
        "Factor": f["label"], "Data type": f["category"], "Score (0-10)": params[f"w_{k}"],
        "Share of final score %": round(params[f"w_{k}"] / total_w * 100, 1),
        "Direction": "Positive" if params[f"dir_{k}"] > 0 else "Inverse", "Lookback (days)": params[f"win_{k}"],
        "Usable from": d.date() if (d := feat_now[f"sig_{k}"].first_valid_index()) is not None else None,
        "Logic": f["logic"],
    } for k, f in FACTORS.items()]), hide_index=True, width="stretch")

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
        ytitle = "Index level" if pick in ("dxy", "cpi") else ("INR per USD" if pick == "usdinr" else "Percent")
    chart(go.Figure(go.Scatter(x=s.index, y=s.values, line=dict(color=BLUE, width=2), name=DATA_INPUTS[pick]["label"])),
          yaxis_title=ytitle, showlegend=False)

# ---------------------------------------------------------
# Tab 2: single backtest
# ---------------------------------------------------------
with tab_bt:
    if st.button("Run backtest", type="primary"):
        try:
            ss["result"] = engine.run_backtest(gold, raw, params, **run_kw)
            ss["result_inputs"] = (params, costs, capital, start, end, rf)
        except ValueError as e:
            st.error(str(e))
    res = ss.get("result")
    if res is None:
        st.info("Set the scores and parameters in the sidebar, then press Run backtest.")
    else:
        if ss["result_inputs"] != (params, costs, capital, start, end, rf):
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
        tw = sum(p_used[f"w_{k}"] for k in FACTORS if pd.notna(last_row[f"sig_{k}"])) or 1
        card = pd.DataFrame([{
            "Factor": FACTORS[k]["label"], "Raw value": last_row[f"raw_{k}"], "Normalised reading": last_row[f"z_{k}"],
            "Signal (-1 to +1)": last_row[f"sig_{k}"], "Score (0-10)": p_used[f"w_{k}"],
            "Contribution to final": p_used[f"w_{k}"] * last_row[f"sig_{k}"] / tw,
        } for k in FACTORS])
        st.dataframe(card.round(3), hide_index=True, width="stretch")
        decision = "Long" if last_row["score"] >= p_used["entry_threshold"] else ("Short" if p_used["allow_short"] and last_row["score"] <= -p_used["entry_threshold"] else "No new trade")
        st.caption(f"Final score on {f.index[-1].date()}: {last_row['score']:+.3f}. Decision: {decision}.")

        yearly = curves.resample("YE").last()
        yearly = (yearly / yearly.shift(1).fillna(curves.iloc[0]) - 1) * 100
        yearly.index = yearly.index.year
        fig = go.Figure([go.Bar(x=yearly.index, y=yearly[n], name=n, marker_color=col) for n, col in ((NET, BLUE), (BH, GREY))])
        chart(fig, 300, yaxis_title="Return %", barmode="group", title="Return by calendar year")

        with st.expander(f"Trade log ({len(trades)} trades)"):
            st.dataframe(trades.round(2), hide_index=True, width="stretch")
            st.download_button("Download trade log (CSV)", trades.to_csv(index=False), "trades.csv", "text/csv")

# ---------------------------------------------------------
# Tab 3: vary parameters and scores
# ---------------------------------------------------------
with tab_sweep:
    st.caption("Every run uses the sidebar settings as the base and changes only what you pick here. "
               "The best row of a sweep is fitted to the past; check it on a period it has not seen (Test period slider) before trusting it.")
    mode = st.radio("What to vary", ["Chosen parameters (grid)", "Factor scores (random search)"], horizontal=True)
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

        st.dataframe(ranked.round(2), width="stretch", height=420,
                     column_config={k: st.column_config.NumberColumn(PARAMS[k]["label"]) for k in keys})
        st.download_button("Download sweep results (CSV)", ranked.to_csv(index=False), "sweep_results.csv", "text/csv")
    elif sw is not None:
        st.warning("No run produced a result. Check the test period.")
