"""
Quick connectivity and data pull test.
"""

import os
import datetime as dt

import pandas as pd
import MetaTrader5 as mt5

from execution import mt5_client


def main():
    login = int(os.environ["MT5_LOGIN"])
    password = os.environ["MT5_PASSWORD"]
    server = os.environ["MT5_SERVER"]
    symbol = os.environ.get("MT5_SYMBOL", "EURUSD")

    mt5_client.init(login=login, password=password, server=server)

    info = mt5_client.account_info()
    print("Account:", info)

    end = dt.datetime.now()
    start = end - dt.timedelta(days=5)
    rates = mt5_client.get_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
    df = pd.DataFrame(rates)
    print(df.head())
    print(df.tail())

    mt5_client.shutdown()


if __name__ == "__main__":
    main()
