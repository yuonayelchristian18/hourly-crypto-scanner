
import requests
import time

COINS = ["bitcoin", "ethereum", "solana", "ripple", "cardano"]

DAYS = 90
SHORT_WINDOW = 6
LONG_WINDOW = 24

FEE = 0.001
STARTING_CASH = 1000


def get_prices(coin):
    url = (
        f"https://api.coingecko.com/api/v3/coins/"
        f"{coin}/market_chart"
    )

    params = {
        "vs_currency": "usd",
        "days": DAYS
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    prices = data.get("prices", [])

    return prices


def backtest(coin):
    prices = get_prices(coin)

    if len(prices) < LONG_WINDOW + 2:
        print(f"{coin}: Not enough price data.")
        return

    # Check the time gaps between observations.
    gaps = [
        (prices[i][0] - prices[i - 1][0]) / 60000
        for i in range(1, len(prices))
    ]

    average_gap = sum(gaps) / len(gaps)

    print(f"\n{coin.upper()}")
    print(f"Price observations: {len(prices)}")
    print(f"Average time between observations: {average_gap:.1f} minutes")

    if average_gap > 90:
        print("Warning: Data is not hourly enough for an hourly backtest.")

    values = [item[1] for item in prices]

    cash = STARTING_CASH
    quantity = 0
    trades = 0
    wins = 0
    entry_price = 0

    for i in range(LONG_WINDOW, len(values) - 1):
        short_average = sum(
            values[i - SHORT_WINDOW:i]
        ) / SHORT_WINDOW

        long_average = sum(
            values[i - LONG_WINDOW:i]
        ) / LONG_WINDOW

        previous_short = sum(
            values[i - SHORT_WINDOW - 1:i - 1]
        ) / SHORT_WINDOW

        previous_long = sum(
            values[i - LONG_WINDOW - 1:i - 1]
        ) / LONG_WINDOW

        buy_signal = (
            previous_short <= previous_long
            and short_average > long_average
        )

        sell_signal = (
            previous_short >= previous_long
            and short_average < long_average
        )

        # Execute at the next available observation.
        next_price = values[i + 1]

        if buy_signal and quantity == 0:
            entry_price = next_price
            quantity = (cash * (1 - FEE)) / next_price
            cash = 0
            trades += 1

        elif sell_signal and quantity > 0:
            cash = quantity * next_price * (1 - FEE)

            if next_price > entry_price:
                wins += 1

            quantity = 0

    final_value = cash + quantity * values[-1]
    profit = final_value - STARTING_CASH
    return_percent = (profit / STARTING_CASH) * 100

    win_rate = (wins / trades * 100) if trades else 0

    print(f"Starting balance: ${STARTING_CASH:.2f}")
    print(f"Final balance: ${final_value:.2f}")
    print(f"Profit/loss: ${profit:.2f}")
    print(f"Return: {return_percent:.2f}%")
    print(f"Completed trades: {wins} wins out of {trades} entries")
    print(f"Win rate: {win_rate:.2f}%")


for coin in COINS:
    try:
        backtest(coin)
        time.sleep(3)
    except Exception as error:
        print(f"{coin}: Error - {error}")
