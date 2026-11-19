"""Run the live paper session in a terminal (no browser needed):

    python live_runner.py                  # uses the session started in the app's Live tab
    python live_runner.py --every 60       # check every 60 seconds (default 5)
    python live_runner.py --record         # also save ticks to data/ticks/

Prints the price, the signal and every new entry / exit, and appends them to data/live/events.csv.
Paper trading only: nothing is sent to Binance.
"""
import argparse
import time
import warnings

import pandas as pd

from gold_bot import data, live, ticks

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=5, help="seconds between checks")
    ap.add_argument("--record", action="store_true", help="save ticks to data/ticks/")
    args = ap.parse_args()
    session = live.get_session()
    if session is None:
        raise SystemExit("No live session. Start one in the app's Live tab first.")
    print(f"Strategy '{session['strategy']}' on {session['symbol']} since {session['started']} UTC. Ctrl+C to stop.")
    gold, raw, _ = data.load_all()
    ticks.get_stream(session["symbol"]).record = args.record
    loaded = pd.Timestamp.now()
    while True:
        try:
            if pd.Timestamp.now() - loaded > pd.Timedelta(hours=6):  # pick up new daily / macro data
                gold, raw, _ = data.load_all()
                loaded = pd.Timestamp.now()
            snap = live.snapshot(session, gold, raw)
            p = {**live.DEFAULT_PARAMS, **session["params"]}
            pos = snap["position"]
            eq = snap["equity"].iloc[-1] if snap["equity"] is not None else session["capital"]
            line = (f"{pd.Timestamp.now():%H:%M:%S}  {session['symbol']} {snap['price']['price']:,.2f}  "
                    f"last bar {snap['last_bar']:%m-%d %H:%M} ({snap['base_tf']})  {live.decision_text(snap['feat'], p)}  "
                    f"position: {'flat' if pos is None else pos['side'] + ' ' + format(pos['qty'], '.4f') + ' @ ' + format(pos['entry_price'], ',.2f')}  "
                    f"equity {eq:,.0f}  [{snap['price'].get('source')}, {snap['stream'].rate():.1f} ticks/s]")
            print(line)
            for e in live.log_events(snap):
                print(f"   >>> {e['time']}  {e['event']}  price {e['price']:,.2f}" + ("" if e["pnl"] is None else f"  P&L {e['pnl']:,.2f}"))
        except Exception as e:
            print(f"{pd.Timestamp.now():%H:%M:%S}  error: {type(e).__name__}: {e}")
        time.sleep(args.every)


if __name__ == "__main__":
    main()
