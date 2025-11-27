"""
Risk and sizing utilities.
"""

from __future__ import annotations


def position_size(balance: float, risk_fraction: float, stop_pips: float, pip_value: float) -> float:
    """
    Compute position size (lots) given account balance, risk %, stop distance in pips, and pip value.
    """
    if stop_pips <= 0:
        raise ValueError("stop_pips must be positive")
    risk_amount = balance * risk_fraction
    return risk_amount / (stop_pips * pip_value)


def enforce_daily_loss_cap(equity_start: float, equity_now: float, loss_cap_fraction: float) -> bool:
    """
    Returns True if trading should pause because daily loss cap is hit.
    """
    drawdown = (equity_start - equity_now) / equity_start
    return drawdown >= loss_cap_fraction
