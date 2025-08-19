import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class HeadShouldersPattern:
	symbol: str
	timeframe_name: str
	pattern_type: str  # 'HS' or 'INV_HS'
	left_idx: int
	head_idx: int
	right_idx: int
	left_ts: pd.Timestamp
	head_ts: pd.Timestamp
	right_ts: pd.Timestamp
	low1_idx: int
	low2_idx: int
	left_price: float
	head_price: float
	right_price: float
	low1_price: float
	low2_price: float
	neckline_a_idx: int
	neckline_a_price: float
	neckline_b_idx: int
	neckline_b_price: float
	detected_at_ts: pd.Timestamp
	is_breakout: bool
	breakout_ts: Optional[pd.Timestamp]
	entry_price: Optional[float]
	stop_loss: Optional[float]
	take_profit: Optional[float]

	@property
	def id(self) -> str:
		return f"{self.symbol}-{self.timeframe_name}-{int(self.detected_at_ts.value//1e9)}-{self.pattern_type}-{self.head_idx}"

	def neckline_value_at_index(self, idx: int) -> float:
		# Linear interpolation between neckline points
		x1, y1 = self.neckline_a_idx, self.neckline_a_price
		x2, y2 = self.neckline_b_idx, self.neckline_b_price
		if x2 == x1:
			return (y1 + y2) / 2.0
		slope = (y2 - y1) / (x2 - x1)
		return y1 + slope * (idx - x1)


@dataclass
class DetectorConfig:
	pivot_lookback: int = 5
	shoulder_tolerance_ratio: float = 0.12  # 12% tolerance between shoulders
	min_bars_between_pivots: int = 3
	lookback_bars: int = 100
	min_head_to_shoulder_ratio: float = 1.03  # head must be at least 3% higher/lower than shoulders
	atr_period: int = 14
	use_measured_move_target: bool = True
	risk_reward_ratio: float = 2.0
	# Only consider breakouts that occurred within the most recent N bars of the data window
	# This ensures we do not trade on stale historical signals
	trade_recent_n_bars: int = 3
	# Only place trades when the expected reward/risk is at least this threshold
	min_trade_rr: float = 1.2


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
	high = df['high'].values
	low = df['low'].values
	close = df['close'].values
	tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
	atr = np.empty_like(close)
	atr[:] = np.nan
	atr[period] = np.nanmean(tr[:period]) if len(tr) > period else np.nan
	for i in range(period + 1, len(close)):
		atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
	return pd.Series(atr, index=df.index)


def _is_pivot_high(high: np.ndarray, idx: int, lb: int) -> bool:
	left = high[idx - lb: idx]
	right = high[idx + 1: idx + lb + 1]
	if len(left) < lb or len(right) < lb:
		return False
	return high[idx] > np.max(left) and high[idx] > np.max(right)


def _is_pivot_low(low: np.ndarray, idx: int, lb: int) -> bool:
	left = low[idx - lb: idx]
	right = low[idx + 1: idx + lb + 1]
	if len(left) < lb or len(right) < lb:
		return False
	return low[idx] < np.min(left) and low[idx] < np.min(right)


def _find_pivots(df: pd.DataFrame, lb: int) -> Tuple[List[int], List[int]]:
	high = df['high'].values
	low = df['low'].values
	pivot_highs = []
	pivot_lows = []
	for i in range(lb, len(df) - lb):
		if _is_pivot_high(high, i, lb):
			pivot_highs.append(i)
		if _is_pivot_low(low, i, lb):
			pivot_lows.append(i)
	return pivot_highs, pivot_lows


def _similar(value_a: float, value_b: float, tolerance_ratio: float) -> bool:
	if value_b == 0:
		return False
	return abs(value_a - value_b) / abs(value_b) <= tolerance_ratio


def _select_trough_between(lows: List[int], start_idx: int, end_idx: int, low_values: np.ndarray) -> Optional[int]:
	candidates = [i for i in lows if start_idx < i < end_idx]
	if not candidates:
		return None
	return int(min(candidates, key=lambda i: low_values[i]))


def detect_head_shoulders(df: pd.DataFrame, symbol: str, timeframe_name: str, cfg: DetectorConfig) -> List[HeadShouldersPattern]:
	if df is None or len(df) < 100:
		return []
	work_df = df.tail(cfg.lookback_bars).copy()
	atr_series = compute_atr(work_df, cfg.atr_period)
	work_df['atr'] = atr_series
	pivot_highs, pivot_lows = _find_pivots(work_df, cfg.pivot_lookback)

	patterns: List[HeadShouldersPattern] = []
	high = work_df['high'].values
	low = work_df['low'].values
	close = work_df['close'].values
	times = work_df['time']

	# Regular Head & Shoulders (peaks)
	for i in range(0, len(pivot_highs) - 2):
		lh, hd, rh = pivot_highs[i], pivot_highs[i + 1], pivot_highs[i + 2]
		if not (lh + cfg.min_bars_between_pivots <= hd and hd + cfg.min_bars_between_pivots <= rh):
			continue
		if not (high[hd] > high[lh] and high[hd] > high[rh]):
			continue
		if (high[hd] / max(high[lh], high[rh])) < cfg.min_head_to_shoulder_ratio:
			continue
		if not _similar(high[lh], high[rh], cfg.shoulder_tolerance_ratio):
			continue
		low1 = _select_trough_between(pivot_lows, lh, hd, low)
		low2 = _select_trough_between(pivot_lows, hd, rh, low)
		if low1 is None or low2 is None:
			continue
		neck_a, neck_b = low1, low2
		# Breakout check: close below neckline after right shoulder
		last_idx = len(work_df) - 1
		neck_prev = None
		neck_cur = None
		is_break = False
		break_ts = None
		entry = None
		sl = None
		tp = None
		# Accept breakouts only if they occur within the most recent window
		min_break_k = max(rh + 1, last_idx - max(0, int(cfg.trade_recent_n_bars)) + 1)
		for k in range(min_break_k, last_idx + 1):
			# Linearly interpolate neckline level at k
			y1 = low[neck_a]
			y2 = low[neck_b]
			x1 = neck_a
			x2 = neck_b
			lvl = y1 if x2 == x1 else y1 + (y2 - y1) * (k - x1) / (x2 - x1)
			prev_close = close[k - 1] if k - 1 >= 0 else close[k]
			if prev_close > lvl and close[k] < lvl:
				is_break = True
				break_ts = times.iloc[k]
				entry = close[k]
				atr_val = work_df['atr'].iloc[k]
				sl = max(high[rh], high[lh]) + (atr_val if not np.isnan(atr_val) else 0.0) * 0.2
				if cfg.use_measured_move_target:
					# Measured target: vertical distance head to neckline projected downward
					neck_at_hd = y1 if x2 == x1 else y1 + (y2 - y1) * (hd - x1) / (x2 - x1)
					height = max(0.0, high[hd] - neck_at_hd)
					tp = entry - height
				else:
					risk = abs(entry - sl)
					tp = entry - risk * cfg.risk_reward_ratio
				# Enforce minimum RR if SL/TP available
				risk = abs(entry - sl) if sl is not None else float('nan')
				reward = abs(entry - tp) if tp is not None else float('nan')
				if not (risk > 0 and reward > 0 and (reward / risk) >= cfg.min_trade_rr):
					# Ignore this breakout; keep scanning
					is_break = False
					break_ts = None
					entry = None
					sl = None
					tp = None
					continue
				break
		if True:
			pattern = HeadShouldersPattern(
				symbol=symbol,
				timeframe_name=timeframe_name,
				pattern_type='HS',
				left_idx=lh,
				head_idx=hd,
				right_idx=rh,
				left_ts=times.iloc[lh],
				head_ts=times.iloc[hd],
				right_ts=times.iloc[rh],
				low1_idx=low1,
				low2_idx=low2,
				left_price=high[lh],
				head_price=high[hd],
				right_price=high[rh],
				low1_price=low[low1],
				low2_price=low[low2],
				neckline_a_idx=neck_a,
				neckline_a_price=low[neck_a],
				neckline_b_idx=neck_b,
				neckline_b_price=low[neck_b],
				detected_at_ts=times.iloc[rh],
				is_breakout=is_break,
				breakout_ts=break_ts,
				entry_price=entry,
				stop_loss=sl,
				take_profit=tp,
			)
			patterns.append(pattern)

	# Inverse Head & Shoulders (troughs)
	for i in range(0, len(pivot_lows) - 2):
		ls, hd, rs = pivot_lows[i], pivot_lows[i + 1], pivot_lows[i + 2]
		if not (ls + cfg.min_bars_between_pivots <= hd and hd + cfg.min_bars_between_pivots <= rs):
			continue
		if not (low[hd] < low[ls] and low[hd] < low[rs]):
			continue
		# For inverse, enforce head sufficiently lower than shoulders (by ratio)
		if (min(low[ls], low[rs]) - low[hd]) / max(abs(min(low[ls], low[rs])), 1e-9) < (cfg.min_head_to_shoulder_ratio - 1.0):
			continue
		# Shoulders similarity
		if not _similar(low[ls], low[rs], cfg.shoulder_tolerance_ratio):
			continue
		# Select peaks between troughs for neckline
		# Convert by reusing select_trough but on highs by negating values; instead, pick highest highs between
		def _select_peak_between(start_i: int, end_i: int) -> Optional[int]:
			candidates = [i for i in pivot_highs if start_i < i < end_i]
			if not candidates:
				return None
			return int(max(candidates, key=lambda i: high[i]))

		peak1 = _select_peak_between(ls, hd)
		peak2 = _select_peak_between(hd, rs)
		if peak1 is None or peak2 is None:
			continue
		neck_a, neck_b = peak1, peak2

		# Breakout check: close above neckline after right shoulder
		last_idx = len(work_df) - 1
		is_break = False
		break_ts = None
		entry = None
		sl = None
		tp = None
		min_break_k = max(rs + 1, last_idx - max(0, int(cfg.trade_recent_n_bars)) + 1)
		for k in range(min_break_k, last_idx + 1):
			y1 = high[neck_a]
			y2 = high[neck_b]
			x1 = neck_a
			x2 = neck_b
			lvl = y1 if x2 == x1 else y1 + (y2 - y1) * (k - x1) / (x2 - x1)
			prev_close = close[k - 1] if k - 1 >= 0 else close[k]
			if prev_close < lvl and close[k] > lvl:
				is_break = True
				break_ts = times.iloc[k]
				entry = close[k]
				atr_val = work_df['atr'].iloc[k]
				sl = min(low[rs], low[ls]) - (atr_val if not np.isnan(atr_val) else 0.0) * 0.2
				if cfg.use_measured_move_target:
					neck_at_hd = y1 if x2 == x1 else y1 + (y2 - y1) * (hd - x1) / (x2 - x1)
					height = max(0.0, neck_at_hd - low[hd])
					tp = entry + height
				else:
					risk = abs(entry - sl)
					tp = entry + risk * cfg.risk_reward_ratio
				# Enforce minimum RR if SL/TP available
				risk = abs(entry - sl) if sl is not None else float('nan')
				reward = abs(tp - entry) if tp is not None else float('nan')
				if not (risk > 0 and reward > 0 and (reward / risk) >= cfg.min_trade_rr):
					is_break = False
					break_ts = None
					entry = None
					sl = None
					tp = None
					continue
				break
		pattern = HeadShouldersPattern(
			symbol=symbol,
			timeframe_name=timeframe_name,
			pattern_type='INV_HS',
			left_idx=ls,
			head_idx=hd,
			right_idx=rs,
			left_ts=times.iloc[ls],
			head_ts=times.iloc[hd],
			right_ts=times.iloc[rs],
			low1_idx=peak1,
			low2_idx=peak2,
			left_price=low[ls],
			head_price=low[hd],
			right_price=low[rs],
			low1_price=high[peak1],
			low2_price=high[peak2],
			neckline_a_idx=neck_a,
			neckline_a_price=high[neck_a],
			neckline_b_idx=neck_b,
			neckline_b_price=high[neck_b],
			detected_at_ts=times.iloc[rs],
			is_breakout=is_break,
			breakout_ts=break_ts,
			entry_price=entry,
			stop_loss=sl,
			take_profit=tp,
		)
		patterns.append(pattern)

	return patterns