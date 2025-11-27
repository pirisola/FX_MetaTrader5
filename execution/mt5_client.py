"""
MT5 client helpers for initialization, data retrieval, and order placement.
This keeps MetaTrader5 interactions in one place so other modules stay clean.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import MetaTrader5 as mt5


def init(login: int, password: str, server: str, path: Optional[str] = None) -> None:
    """Initialize MT5 and log in; raises RuntimeError on failure."""
    if path:
        initialized = mt5.initialize(path=path, login=login, password=password, server=server)
    else:
        initialized = mt5.initialize(login=login, password=password, server=server)
    if not initialized:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    authorized = mt5.login(login=login, password=password, server=server)
    if not authorized:
        raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")


def shutdown() -> None:
    """Shutdown MT5 connection."""
    mt5.shutdown()


def account_info():
    """Return account info or raise."""
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"account_info failed: {mt5.last_error()}")
    return info


def symbol_info(symbol: str):
    """Return symbol info or raise."""
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info failed for {symbol}: {mt5.last_error()}")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol {symbol}: {mt5.last_error()}")
    return info


def get_rates_range(symbol: str, timeframe: int, start: dt.datetime, end: dt.datetime):
    """Fetch rates between start and end."""
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None:
        raise RuntimeError(f"copy_rates_range failed for {symbol}: {mt5.last_error()}")
    return rates


def place_order(
    symbol: str,
    volume: float,
    order_type: int,
    price: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    deviation: int = 10,
    comment: str = "",
) -> dict:
    """Submit an order_send request; returns result._asdict()."""
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "deviation": deviation,
        "comment": comment,
    }
    if price is not None:
        request["price"] = price
    if sl is not None:
        request["sl"] = sl
    if tp is not None:
        request["tp"] = tp
    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send failed: {mt5.last_error()}")
    return result._asdict()
