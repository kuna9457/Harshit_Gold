"""Phase 2a: Hidden Markov Model (HMM) market states for gold, trained walk-forward.

How it avoids looking into the future:
  * The model is refitted every `refit_days` bars using only bars before that point.
  * Inputs are standardised with the mean / std of the training bars only.
  * The state on each test bar comes from the forward filter, P(state today | data up to today),
    never from the smoothed probabilities, which use later bars.
  * States are sorted by volatility after every fit, so state 0 is always the calmest one
    (an HMM gives its states arbitrary numbers, which would otherwise swap between fits).
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")  # hmmlearn's k-means start otherwise probes the CPU count noisily on Windows

HMM_FEATURES = {
    "ret_1d": "Daily return",
    "ret_5d": "5-day return",
    "ret_20d": "20-day return (trend)",
    "vol_20d": "20-day volatility (log)",
    "range": "Daily high-low range vs its 20-day average (log)",
}
DEFAULT_FEATURES = ["ret_1d", "ret_20d", "vol_20d"]
STATE_NAMES = {2: ["Calm", "Volatile"], 3: ["Calm", "Normal", "Volatile"], 4: ["Calm", "Normal", "Active", "Volatile"]}
SIGNAL_RULES = {
    "expected_return": "Long when the model's expected next-day return is above 0",
    "avoid_volatile": "Long unless the most likely state is the most volatile one",
    "calm_only": "Long only when the most likely state is the calmest one",
}


def state_names(k):
    return [f"S{i} {n}" for i, n in enumerate(STATE_NAMES.get(k, [str(i) for i in range(k)]))]


def hmm_features(gold, keys):
    """Model inputs on each bar, from data up to that bar's close."""
    c = gold["Close"]
    r = np.log(c).diff()
    rng = np.log(gold["High"] / gold["Low"]).replace(0, np.nan)
    every = {
        "ret_1d": r,
        "ret_5d": np.log(c).diff(5),
        "ret_20d": np.log(c).diff(20),
        "vol_20d": np.log(r.rolling(20).std()),
        "range": np.log(rng / rng.rolling(20).mean()),
    }
    return pd.DataFrame({k: every[k] for k in keys}).replace([np.inf, -np.inf], np.nan).dropna()


def _fit(Z, n_states, n_init, seed):
    from hmmlearn.hmm import GaussianHMM

    best, best_ll = None, -np.inf
    for j in range(n_init):
        m = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, tol=1e-4,
                        random_state=seed + j, min_covar=1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Z)
            ll = m.score(Z)
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise ValueError("The HMM did not converge on this training window.")
    return best, best_ll


def _log_emission(m, Z):
    var = np.diagonal(m.covars_, axis1=1, axis2=2)  # (states, inputs)
    d2 = ((Z[:, None, :] - m.means_[None]) ** 2 / var[None]).sum(2)
    return -0.5 * (np.log(2 * np.pi * var).sum(1)[None, :] + d2)


def _filter(m, Z):
    """Forward pass only: probability of each state on each bar given data up to that bar."""
    log_a, log_b = np.log(m.transmat_ + 1e-300), _log_emission(m, Z)
    out = np.empty_like(log_b)
    a = np.log(m.startprob_ + 1e-300) + log_b[0]
    out[0] = a = a - logsumexp(a)
    for t in range(1, len(Z)):
        a = logsumexp(a[:, None] + log_a, axis=0) + log_b[t]
        out[t] = a = a - logsumexp(a)
    return np.exp(out)


def walk_forward(gold, keys=None, n_states=3, train_years=5, refit_days=252, window="expanding",
                 warmup=252, n_init=2, seed=0, progress=None):
    """Train and apply the HMM block by block. Returns filtered state probabilities, the most likely
    state, the model's expected next-day return, and one row per refit."""
    keys = list(keys or DEFAULT_FEATURES)
    F = hmm_features(gold, keys)
    r1 = np.log(gold["Close"]).diff().reindex(F.index).to_numpy()
    X, idx, n = F.to_numpy(float), F.index, len(F)
    first = int(round(train_years * 252))
    if n < first + 20:
        raise ValueError(f"Need more than {train_years} years of data to train the HMM.")

    probs = np.full((n, n_states), np.nan)
    exp_ret = np.full(n, np.nan)
    fits = []
    starts = list(range(first, n, int(refit_days)))
    for b, s0 in enumerate(starts):
        s1 = min(s0 + int(refit_days), n)
        lo = 0 if window == "expanding" else max(0, s0 - first)
        mu, sd = X[lo:s0].mean(0), X[lo:s0].std(0) + 1e-12
        Z = (X - mu) / sd
        m, ll = _fit(Z[lo:s0], n_states, n_init, seed)

        # What each state looked like in the training data (smoothed weights are fine here: it is all past data).
        g = m.predict_proba(Z[lo:s0])
        w = g / g.sum(0).clip(min=1e-12)
        rt = np.nan_to_num(r1[lo:s0])
        s_mu = (w * rt[:, None]).sum(0)
        s_vol = np.sqrt((w * (rt[:, None] - s_mu) ** 2).sum(0)) * np.sqrt(252)
        order = np.argsort(s_vol)
        A = m.transmat_[np.ix_(order, order)]

        w0 = max(lo, s0 - int(warmup))  # warm the filter up on the last part of the training window
        pf = _filter(m, Z[w0:s1])[s0 - w0:][:, order]
        probs[s0:s1] = pf
        exp_ret[s0:s1] = (pf @ A) @ s_mu[order]
        fits.append(dict(refit_on=idx[s0].date(), applied_until=idx[s1 - 1].date(), trained_from=idx[lo].date(),
                         training_bars=s0 - lo, log_likelihood=round(ll, 1),
                         **{f"vol % {nm}": round(v * 100, 1) for nm, v in zip(state_names(n_states), s_vol[order])},
                         **{f"ann. return % {nm}": round(v * 252 * 100, 1) for nm, v in zip(state_names(n_states), s_mu[order])},
                         **{f"stay prob {nm}": round(A[k, k], 3) for k, nm in enumerate(state_names(n_states))}))
        if progress:
            progress((b + 1) / len(starts))

    names = state_names(n_states)
    P = pd.DataFrame(probs, index=idx, columns=names)
    valid = P.notna().all(1)
    state = pd.Series(np.where(valid, np.nan_to_num(probs).argmax(1), -1), index=idx).where(valid)
    return dict(probs=P[valid], state=state[valid].astype(int), exp_ret=pd.Series(exp_ret, index=idx)[valid],
                fits=pd.DataFrame(fits), names=names, n_states=n_states,
                settings=dict(keys=keys, n_states=n_states, train_years=train_years, refit_days=refit_days,
                              window=window, n_init=n_init, seed=seed))


def signal(hmm, rule="expected_return"):
    """+1 = be long, -1 = be flat (or short). Decided on the close, filled next open by the backtest."""
    st, k = hmm["state"], hmm["n_states"]
    if rule == "avoid_volatile":
        on = st != k - 1
    elif rule == "calm_only":
        on = st == 0
    else:
        on = hmm["exp_ret"] > 0
    return pd.Series(np.where(on, 1.0, -1.0), index=st.index)


def state_summary(hmm, gold, start=None, end=None):
    """How gold actually behaved on the day AFTER each state was read, over the test period (out of sample)."""
    st = hmm["state"].loc[start:end]
    nxt = np.log(gold["Close"]).diff().shift(-1).reindex(st.index)
    runs = (st != st.shift()).cumsum()
    rows = []
    for k, nm in enumerate(hmm["names"]):
        mask = st == k
        r = nxt[mask].dropna()
        lens = runs[mask].value_counts()
        rows.append({"State": nm, "Share of days %": mask.mean() * 100,
                     "Next-day return, annualised %": r.mean() * 252 * 100 if len(r) else np.nan,
                     "Next-day volatility, annualised %": r.std() * np.sqrt(252) * 100 if len(r) > 1 else np.nan,
                     "Up days %": (r > 0).mean() * 100 if len(r) else np.nan,
                     "Average stay (days)": lens.mean() if len(lens) else np.nan, "Days": int(mask.sum())})
    return pd.DataFrame(rows)


def label_at_entry(trades, series, index):
    """The value `series` had on each trade's signal bar (the bar before the entry fill)."""
    if trades.empty:
        return pd.Series(dtype=object)
    s = series.reindex(index).shift(1)
    return s.reindex(pd.to_datetime(trades["entry_date"])).to_numpy()
