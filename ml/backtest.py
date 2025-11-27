"""
Simple threshold-based backtest for directional classifier outputs.
"""

from __future__ import annotations

import pandas as pd


def backtest_threshold(
    df: pd.DataFrame,
    probs: pd.Series,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    sl_pips: float = 20,
    tp_pips: float = 30,
    pip_value: float = 0.0001,
):
    """
    df should have 'close' and 'time'. probs is P(long). Returns equity curve DataFrame.
    """
    equity = [0.0]
    returns = []
    for i in range(len(df) - 1):
        p = probs.iloc[i]
        direction = 0
        if p > long_threshold:
            direction = 1
        elif p < short_threshold:
            direction = -1
        if direction == 0:
            returns.append(0.0)
            equity.append(equity[-1])
            continue
        entry = df["close"].iloc[i]
        future = df["close"].iloc[i + 1]
        move_pips = (future - entry) / pip_value * direction
        pnl = 0.0
        if move_pips <= -sl_pips:
            pnl = -sl_pips * pip_value
        elif move_pips >= tp_pips:
            pnl = tp_pips * pip_value
        else:
            pnl = move_pips * pip_value
        returns.append(pnl)
        equity.append(equity[-1] + pnl)
    equity_df = pd.DataFrame({"equity": equity[1:], "pnl": returns}, index=df.index[: len(returns)])
    return equity_df
