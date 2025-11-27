"""
Simple paper/live trading loop skeleton.
"""

from __future__ import annotations

import os
import time
import datetime as dt

import MetaTrader5 as mt5
import pandas as pd

from execution import mt5_client
from execution import risk
from ml import features
from ml import model as model_utils
from ml import labels


def load_credentials():
    return {
        "login": int(os.environ["MT5_LOGIN"]),
        "password": os.environ["MT5_PASSWORD"],
        "server": os.environ["MT5_SERVER"],
    }


def fetch_latest_bars(symbol: str, timeframe: int, count: int = 200) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
    return pd.DataFrame(rates)


def run_loop(
    symbol: str,
    timeframe: int,
    model_path: str,
    risk_fraction: float = 0.005,
    sl_pips: float = 20,
    tp_pips: float = 30,
    sleep_seconds: int = 60,
    paper: bool = True,
):
    creds = load_credentials()
    mt5_client.init(**creds)
    model = model_utils.load_model(model_path)
    equity_start = mt5_client.account_info().equity

    try:
        while True:
            bars = fetch_latest_bars(symbol, timeframe)
            feats = features.make_features(bars)
            aligned = feats.iloc[-1:]
            prob_long = model.predict_proba(aligned)[0][1]

            spread = bars["spread"].iloc[-1] if "spread" in bars else 0
            if spread > 30:  # basic spread filter in points
                time.sleep(sleep_seconds)
                continue

            direction = 1 if prob_long > 0.55 else -1 if prob_long < 0.45 else 0
            if direction == 0:
                time.sleep(sleep_seconds)
                continue

            info = mt5_client.account_info()
            if risk.enforce_daily_loss_cap(equity_start, info.equity, loss_cap_fraction=0.02):
                break

            balance = info.equity
            pip_value = 0.0001  # simplify for majors; adjust per symbol
            size = risk.position_size(balance, risk_fraction, sl_pips, pip_value)

            if paper:
                print(f"[PAPER] {dt.datetime.utcnow()} {symbol} dir={direction} size={size:.2f} prob={prob_long:.3f}")
            else:
                order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
                price = bars["close"].iloc[-1]
                result = mt5_client.place_order(
                    symbol=symbol,
                    volume=size,
                    order_type=order_type,
                    price=price,
                    sl=price - sl_pips * pip_value * direction,
                    tp=price + tp_pips * pip_value * direction,
                    comment="ml-signal",
                )
                print(f"Order result: {result}")

            time.sleep(sleep_seconds)
    finally:
        mt5_client.shutdown()
