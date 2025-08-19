import time
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache

import MetaTrader5 as mt5
import pandas as pd


TIMEFRAME_NAME_TO_CONST = {
	"M1": mt5.TIMEFRAME_M1,
	"M5": mt5.TIMEFRAME_M5,
	"M15": mt5.TIMEFRAME_M15,
	"M30": mt5.TIMEFRAME_M30,
	"H1": mt5.TIMEFRAME_H1,
	"H4": mt5.TIMEFRAME_H4,
	"D1": mt5.TIMEFRAME_D1,
}

CONST_TO_TIMEFRAME_NAME = {v: k for k, v in TIMEFRAME_NAME_TO_CONST.items()}


@dataclass
class OrderResult:
	retcode: int
	order: Optional[int]
	deal: Optional[int]
	price: Optional[float]
	comment: str
	request: Dict[str, Any]
	response_raw: Any


def timeframe_to_str(timeframe_const: int) -> str:
	return CONST_TO_TIMEFRAME_NAME.get(timeframe_const, str(timeframe_const))


def initialize_mt5(login: Optional[int] = None,
					 password: Optional[str] = None,
					 server: Optional[str] = None,
					 path: Optional[str] = None) -> bool:
	if not mt5.initialize():
		raise RuntimeError("Could not initialize MT5")
	if login and password and server:
		authorized = mt5.login(login=login, password=password, server=server)
		if not authorized:
			logging.error(f"MT5 login failed: {mt5.last_error()}")
			return False
	account_info = mt5.account_info()
	if account_info:
		logging.info(f"Connected to MT5 account: {account_info.login}")
	return True


def shutdown_mt5() -> None:
	try:
		mt5.shutdown()
	except Exception:
		pass


@lru_cache(maxsize=256)
def resolve_symbol_name(candidate: str) -> Optional[str]:
	"""Resolve a broker-specific symbol name (e.g., EURUSD.a) from a canonical symbol (e.g., EURUSD).

	Returns the exact tradable symbol name if found; otherwise None.
	"""
	info = mt5.symbol_info(candidate)
	if info is not None:
		return candidate
	try:
		pref_matches = mt5.symbols_get(f"{candidate}*") or []
	except Exception:
		pref_matches = []
	try:
		any_matches = mt5.symbols_get(f"*{candidate}*") or []
	except Exception:
		any_matches = []
	seen = set()
	merged = []
	for s in list(pref_matches) + list(any_matches):
		if s.name not in seen:
			merged.append(s)
			seen.add(s.name)
	if not merged:
		return None
	def sort_key(s):
		trade_mode = getattr(s, 'trade_mode', None)
		is_tradable = 1 if (trade_mode is not None and trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED) else 0
		is_visible = 1 if getattr(s, 'visible', False) else 0
		starts_with = 1 if s.name.upper().startswith(candidate.upper()) else 0
		return (-is_tradable, -is_visible, -starts_with, len(s.name))
	best = sorted(merged, key=sort_key)[0]
	if best and best.name.upper() != candidate.upper():
		logging.info(f"Resolved symbol alias: {candidate} -> {best.name}")
	return best.name


_ENSURED_SYMBOLS = set()


def ensure_symbol(symbol: str) -> bool:
	resolved = resolve_symbol_name(symbol)
	if resolved is None:
		logging.error(f"Symbol not found: {symbol}")
		return False
	info = mt5.symbol_info(resolved)
	if info is None:
		logging.error(f"Symbol not found: {symbol}")
		return False
	if info.visible:
		_ENSURED_SYMBOLS.add(resolved)
		return True
	if resolved in _ENSURED_SYMBOLS:
		return True
	if not mt5.symbol_select(resolved, True):
		logging.error(f"Failed to select symbol: {resolved}")
		return False
	_ENSURED_SYMBOLS.add(resolved)
	return True


def fetch_rates(symbol: str, timeframe: int, num_bars: int = 1000) -> Optional[pd.DataFrame]:
	resolved = resolve_symbol_name(symbol)
	if resolved is None:
		logging.error(f"Symbol not found: {symbol}")
		return None
	if not ensure_symbol(resolved):
		return None
	rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, num_bars)
	if rates is None or len(rates) == 0:
		logging.warning(f"No rates for {symbol} {timeframe_to_str(timeframe)}")
		return None
	df = pd.DataFrame(rates)
	df['time'] = pd.to_datetime(df['time'], unit='s')
	return df


def _round_volume(symbol: str, volume: float) -> float:
	resolved = resolve_symbol_name(symbol) or symbol
	info = mt5.symbol_info(resolved)
	step = info.volume_step if info else 0.01
	min_lot = info.volume_min if info else 0.01
	max_lot = info.volume_max if info else 100.0
	# Round down to nearest step, then clamp
	n_steps = math.floor(volume / step)
	vol = n_steps * step
	vol = max(min_lot, min(max_lot, vol))
	return round(vol, 3)


def compute_volume_for_risk(symbol: str, risk_amount: float, entry_price: float, stop_loss_price: float) -> float:
	resolved = resolve_symbol_name(symbol)
	info = mt5.symbol_info(resolved) if resolved else None
	if info is None:
		return 0.0
	point = info.point
	tick_value = info.trade_tick_value
	if tick_value is None or tick_value == 0:
		# Fallback approximation if MT5 doesn't provide tick value
		tick_value = 1.0
	ticks = max(1.0, abs(entry_price - stop_loss_price) / max(point, 1e-10))
	if ticks <= 0:
		return 0.0
	volume = risk_amount / (ticks * tick_value)
	return _round_volume(symbol, volume)


def current_price(symbol: str) -> Optional[Tuple[float, float]]:
	resolved = resolve_symbol_name(symbol)
	tick = mt5.symbol_info_tick(resolved) if resolved else None
	if tick is None:
		return None
	return tick.bid, tick.ask


def place_market_order(symbol: str,
						 is_buy: bool,
						 volume: float,
						 sl: Optional[float],
						 tp: Optional[float],
						 deviation: int = 30,
						 magic: int = 123456,
						 comment: str = "HS Bot") -> OrderResult:
	if volume <= 0:
		return OrderResult(retcode=-1, order=None, deal=None, price=None, comment="Invalid volume", request={}, response_raw=None)
	resolved = resolve_symbol_name(symbol)
	if resolved is None:
		return OrderResult(retcode=-2, order=None, deal=None, price=None, comment="Symbol not available", request={}, response_raw=None)
	if not ensure_symbol(resolved):
		return OrderResult(retcode=-2, order=None, deal=None, price=None, comment="Symbol not available", request={}, response_raw=None)
	prices = current_price(resolved)
	if prices is None:
		return OrderResult(retcode=-3, order=None, deal=None, price=None, comment="No tick price", request={}, response_raw=None)
	bid, ask = prices
	order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
	price = ask if is_buy else bid
	# Sanitize SL/TP: MT5 expects 0 if not set and correct side
	sl_sane = float(sl) if sl and sl > 0 else 0.0
	tp_sane = float(tp) if tp and tp > 0 else 0.0
	if is_buy and sl_sane and sl_sane >= price:
		sl_sane = 0.0
	if (not is_buy) and sl_sane and sl_sane <= price:
		sl_sane = 0.0
	# Choose filling type based on symbol settings when available
	fill_type = mt5.ORDER_FILLING_IOC
	info = mt5.symbol_info(resolved)
	if info and getattr(info, 'filling_mode', None) in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
		fill_type = info.filling_mode
	request = {
		"action": mt5.TRADE_ACTION_DEAL,
		"symbol": resolved,
		"volume": volume,
		"type": order_type,
		"price": price,
		"sl": sl_sane,
		"tp": tp_sane,
		"deviation": deviation,
		"magic": magic,
		"comment": comment,
		"type_filling": fill_type,
		"type_time": mt5.ORDER_TIME_GTC,
	}
	result = mt5.order_send(request)
	if result is None:
		return OrderResult(retcode=-4, order=None, deal=None, price=None, comment=str(mt5.last_error()), request=request, response_raw=None)
	logging.info(f"Order send result: retcode={result.retcode} comment={result.comment}")
	return OrderResult(
		retcode=result.retcode,
		order=getattr(result, 'order', None),
		deal=getattr(result, 'deal', None),
		price=getattr(result, 'price', price),
		comment=getattr(result, 'comment', ''),
		request=request,
		response_raw=result,
	)


def format_price(symbol: str, price: float) -> float:
	resolved = resolve_symbol_name(symbol) or symbol
	info = mt5.symbol_info(resolved)
	if info is None:
		return round(price, 5)
	digits = info.digits
	return round(price, digits)


def get_last_bar_time(symbol: str, timeframe: int) -> Optional[pd.Timestamp]:
	"""Returns the last bar's timestamp for a symbol/timeframe or None if unavailable."""
	resolved = resolve_symbol_name(symbol)
	if resolved is None:
		return None
	if not ensure_symbol(resolved):
		return None
	rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 1)
	if rates is None or len(rates) == 0:
		return None
	try:
		last_ts = int(rates[0]['time'])
	except Exception:
		try:
			last_ts = int(rates[0]["time"])  # dict-like
		except Exception:
			return None
	return pd.to_datetime(last_ts, unit='s')