"""
Feature engineering for FX bar data.
"""

from __future__ import annotations

import pandas as pd
import ta


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame with columns: ['time','open','high','low','close','tick_volume','spread'].
    Returns DataFrame with engineered features aligned to df index.
    """
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = df["close"].pct_change()
    out["ret_5"] = df["close"].pct_change(5)
    out["ret_20"] = df["close"].pct_change(20)

    out["vol_20"] = df["ret_1"].rolling(20).std()
    out["vol_50"] = df["ret_1"].rolling(50).std()

    out["ema_20"] = df["close"].ewm(span=20).mean()
    out["ema_50"] = df["close"].ewm(span=50).mean()
    out["ema_gap"] = out["ema_20"] - out["ema_50"]

    out["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    macd = ta.trend.MACD(df["close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    out["spread"] = df.get("spread", pd.Series(index=df.index))
    out["tick_volume"] = df.get("tick_volume", pd.Series(index=df.index))

    # Time features
    dt_index = pd.to_datetime(df["time"], unit="s") if "time" in df.columns else df.index
    out["hour"] = dt_index.hour
    out["day_of_week"] = dt_index.dayofweek

    return out.dropna()
