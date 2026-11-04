name: Walkforward

on:
  workflow_dispatch:

jobs:
  backtest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - run: pip install requests pandas numpy

      - run: python walkforward.py --interval 15 --days 90 --pairs BTC-USD,ETH-USD,SOL-USD,XRP-USD,ADA-USD,AVAX-USD,LINK-USD,DOGE-USD,LTC-USD

      - name: Save trades CSV
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: walkforward-trades
          path: walkforward_trades.csv
