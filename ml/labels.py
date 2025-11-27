"""
Label generation for supervised learning on FX bars.
"""

from __future__ import annotations

import pandas as pd


def forward_return_labels(df: pd.DataFrame, horizon: int = 12) -> pd.Series:
    """
    Computes forward return over 'horizon' bars and returns sign label {-1,0,1}.
    Expects 'close' column.
    """
    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    labels = fwd_ret.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return labels
