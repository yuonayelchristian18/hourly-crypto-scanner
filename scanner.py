"""scanner.py - scan the coin list on closed candles, print a table, log/alert signals.

  python scanner.py                      # course default rules
  python scanner.py --mode more          # more trades (relaxes HTF/phase/room)
  python scanner.py --min-score 70       # only alert at 70+ (needs final score 11+/15)

Score = final entry score / 15 * 100.  Course minimum (9/15) = 60.
Optional: set env DISCORD_WEBHOOK_URL to get alerts in Discord.
"""
import argparse
import os

import pandas as pd
import requests

from data import PRODUCTS, fetch_candles
from signals import analyze, make_params

ALERTS = "alerts.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--mode", default="original", choices=["original", "more", "max"])
    ap.add_argument("--min-score", type=int, default=60)
    ap.add_argument("--products", nargs="*", default=None)
    args = ap.parse_args()

    p = make_params(args.mode)
    prods = args.products or PRODUCTS
    seen = set()
    if os.path.exists(ALERTS):
        old = pd.read_csv(ALERTS)
        seen = set(zip(old["product"], old["time"].astype(str)))

    scanned, alerts = 0, []
    print(f"{'pair':<14}{'close':>14}  {'trend':<6} signal")
    for pr in prods:
        df = fetch_candles(pr, args.interval, args.days)
        if df is None or len(df) < 150:
            print(f"{pr:<14}{'-':>14}  skipped (not listed / too little data)")
            continue
        scanned += 1
        res = analyze(df, p)
        s = res["signal"]
        txt = "-"
        if s is not None:
            txt = (f"{s.direction.upper()} {s.pattern} [{s.entry_type}] score {s.scanner_score} "
                   f"(setup {s.setup_score}/13, final {s.final_score}/15, grade {s.grade}) "
                   f"entry~{s.entry_ref:.8g} stop {s.stop:.8g} T2R {s.target_2r:.8g}")
            if s.scanner_score >= args.min_score and (pr, str(s.time)) not in seen:
                alerts.append((pr, s))
        print(f"{pr:<14}{res['close']:>14.8g}  {res['trend']:<6} {txt}")

    print(f"\nscanned {scanned}/{len(prods)} pairs | new alerts: {len(alerts)}")
    if alerts:
        rows = [dict(product=pr, time=str(s.time), direction=s.direction, pattern=s.pattern,
                     entry_type=s.entry_type, score=s.scanner_score, setup=s.setup_score,
                     final=s.final_score, entry=s.entry_ref, stop=s.stop, target_2r=s.target_2r)
                for pr, s in alerts]
        pd.DataFrame(rows).to_csv(ALERTS, mode="a", header=not os.path.exists(ALERTS), index=False)
        hook = os.getenv("DISCORD_WEBHOOK_URL")
        if hook:
            msg = "\n".join(f"**{pr}** {s.direction.upper()} {s.pattern} score {s.scanner_score} "
                            f"entry~{s.entry_ref:.8g} stop {s.stop:.8g} target {s.target_2r:.8g}"
                            for pr, s in alerts)
            try:
                requests.post(hook, json={"content": msg[:1900]}, timeout=15)
            except Exception as e:
                print("discord post failed:", e)


if __name__ == "__main__":
    main()
