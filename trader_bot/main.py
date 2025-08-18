import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd
import tkinter as tk

from mt5_client import (
	initialize_mt5, shutdown_mt5, fetch_rates, timeframe_to_str,
	compute_volume_for_risk, place_market_order, format_price
)
from hs_detector import DetectorConfig, detect_head_shoulders, HeadShouldersPattern
from ui_panel import PatternTableUI


# Configuration
SYMBOLS = ["EURUSD", "XAUUSD", "EURJPY"]
TIMEFRAMES = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1]
SCAN_INTERVAL_SECONDS = 15
RISK_PER_TRADE_PCT = 1.0  # percent of account balance
MAGIC_NUMBER = 902145


@dataclass
class StoredPattern:
	pattern: HeadShouldersPattern
	traded: bool = False
	order_ticket: Optional[int] = None


class PatternStore:
	def __init__(self):
		self._lock = threading.Lock()
		self._data: Dict[str, StoredPattern] = {}

	def upsert(self, p: HeadShouldersPattern):
		with self._lock:
			key = p.id
			if key in self._data:
				# Update mutable fields
				sp = self._data[key]
				sp.pattern.is_breakout = p.is_breakout
				sp.pattern.breakout_ts = p.breakout_ts
				sp.pattern.entry_price = p.entry_price
				sp.pattern.stop_loss = p.stop_loss
				sp.pattern.take_profit = p.take_profit
			else:
				self._data[key] = StoredPattern(pattern=p)

	def mark_traded(self, pattern_id: str, order_ticket: Optional[int]):
		with self._lock:
			if pattern_id in self._data:
				self._data[pattern_id].traded = True
				self._data[pattern_id].order_ticket = order_ticket

	def rows(self) -> List[Dict[str, str]]:
		with self._lock:
			out = []
			for sp in self._data.values():
				p = sp.pattern
				status = "BREAKOUT" if p.is_breakout else "DETECTED"
				if sp.traded:
					status += ", TRADED"
				out.append({
					"id": p.id,
					"symbol": p.symbol,
					"timeframe": p.timeframe_name,
					"type": p.pattern_type,
					"status": status,
					"detected_at": str(p.detected_at_ts),
					"breakout_at": str(p.breakout_ts) if p.breakout_ts else "",
					"entry": f"{p.entry_price:.5f}" if p.entry_price else "",
					"neckline_a": f"{p.neckline_a_price:.5f}",
					"neckline_b": f"{p.neckline_b_price:.5f}",
					"sl": f"{p.stop_loss:.5f}" if p.stop_loss else "",
					"tp": f"{p.take_profit:.5f}" if p.take_profit else "",
				})
			return sorted(out, key=lambda r: r['detected_at'], reverse=True)

	def items(self):
		with self._lock:
			return list(self._data.items())


class ScannerThread(threading.Thread):
	def __init__(self, store: PatternStore, detector_cfg: DetectorConfig, symbols: List[str], timeframes: List[int], scan_interval_sec: int, enable_trading_getter):
		super().__init__(daemon=True)
		self._store = store
		self._cfg = detector_cfg
		self._symbols = symbols
		self._timeframes = timeframes
		self._interval = scan_interval_sec
		self._stop_event = threading.Event()
		self._enable_trading_getter = enable_trading_getter

	def stop(self):
		self._stop_event.set()

	def run(self):
		while not self._stop_event.is_set():
			try:
				self._scan_once()
			except Exception as exc:
				logging.exception(f"Scanner error: {exc}")
			time.sleep(self._interval)

	def _scan_once(self):
		for symbol in self._symbols:
			for tf in self._timeframes:
				df = fetch_rates(symbol, tf, num_bars=max(800, self._cfg.lookback_bars + 50))
				if df is None or len(df) == 0:
					continue
				patterns = detect_head_shoulders(df, symbol, timeframe_to_str(tf), self._cfg)
				for p in patterns:
					self._store.upsert(p)
					if p.is_breakout and self._enable_trading_getter():
						self._maybe_trade(p)

	def _maybe_trade(self, p: HeadShouldersPattern):
		# Avoid duplicate trades
		for pattern_id, sp in self._store.items():
			if pattern_id == p.id and sp.traded:
				return
		# Compute risk amount based on account balance
		acct = mt5.account_info()
		if acct is None:
			logging.warning("No MT5 account info; cannot size position")
			return
		risk_amount = acct.balance * (RISK_PER_TRADE_PCT / 100.0)
		is_buy = p.pattern_type == 'INV_HS'
		entry = p.entry_price
		sl = p.stop_loss
		tp = p.take_profit
		if entry is None or sl is None or entry == sl:
			return
		volume = compute_volume_for_risk(p.symbol, risk_amount, entry, sl)
		if volume <= 0:
			logging.info(f"Zero volume computed for {p.symbol} {p.id}")
			return
		res = place_market_order(
			symbol=p.symbol,
			is_buy=is_buy,
			volume=volume,
			sl=sl,
			tp=tp,
			magic=MAGIC_NUMBER,
			comment=f"HS_BOT {p.pattern_type} {p.timeframe_name}"
		)
		if res and res.retcode == mt5.TRADE_RETCODE_DONE:
			self._store.mark_traded(p.id, res.order or res.deal)
			logging.info(f"Trade placed: {p.symbol} {p.pattern_type} vol={volume} entry={entry} sl={sl} tp={tp}")
		else:
			logging.warning(f"Order failed for {p.symbol}: {res.retcode if res else 'NO_RES'} {res.comment if res else ''}")


def setup_logging():
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s | %(levelname)-7s | %(message)s',
	)


def main():
	setup_logging()
	if not initialize_mt5():
		print("Failed to initialize MT5. Ensure your terminal is running and logged in.")
		return

	detector_cfg = DetectorConfig()
	store = PatternStore()

	root = tk.Tk()
	trading_enabled_flag = {"value": True}

	def on_toggle_trading(enabled: bool):
		trading_enabled_flag["value"] = enabled
		logging.info(f"Trading enabled set to {enabled}")

	ui = PatternTableUI(root, on_toggle_trading)

	def refresh_table():
		rows = store.rows()
		ui.update_rows(rows)
		root.after(2000, refresh_table)

	refresh_table()

	scanner = ScannerThread(
		store=store,
		detector_cfg=detector_cfg,
		symbols=SYMBOLS,
		timeframes=TIMEFRAMES,
		scan_interval_sec=SCAN_INTERVAL_SECONDS,
		enable_trading_getter=lambda: trading_enabled_flag["value"],
	)
	scanner.start()

	def handle_exit(*_):
		logging.info("Shutting down...")
		scanner.stop()
		shutdown_mt5()
		root.quit()

	signal.signal(signal.SIGINT, handle_exit)
	signal.signal(signal.SIGTERM, handle_exit)

	try:
		root.mainloop()
	finally:
		handle_exit()


if __name__ == "__main__":
	main()