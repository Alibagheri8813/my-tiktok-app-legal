import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Dict, Any


class PatternTableUI:
	def __init__(self, root: tk.Tk, on_toggle_trading: Callable[[bool], None]):
		self.root = root
		self.on_toggle_trading = on_toggle_trading
		self.root.title("Head & Shoulders Live Scanner - MT5 Bot")
		self.root.geometry("1200x500")

		self.trading_enabled = tk.BooleanVar(value=True)

		control_frame = ttk.Frame(self.root)
		control_frame.pack(fill=tk.X, padx=8, pady=4)

		toggle_btn = ttk.Checkbutton(control_frame, text="Enable Trading", variable=self.trading_enabled,
									command=self._toggle_trading)
		toggle_btn.pack(side=tk.LEFT)

		refresh_btn = ttk.Button(control_frame, text="Refresh Now", command=self._on_refresh_clicked)
		refresh_btn.pack(side=tk.LEFT, padx=8)

		columns = (
			"id", "symbol", "timeframe", "type", "status",
			"left_at", "head_at", "right_at", "detected_at", "breakout_at",
			"entry", "neckline_a", "neckline_b", "sl", "tp"
		)
		self.tree = ttk.Treeview(self.root, columns=columns, show='headings', height=20)
		for col in columns:
			self.tree.heading(col, text=col.upper())
			# Narrower widths for date columns
			if col in ("left_at", "head_at", "right_at", "detected_at", "breakout_at"):
				self.tree.column(col, width=110, anchor=tk.CENTER)
			else:
				self.tree.column(col, width=100, anchor=tk.CENTER)
		self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

		style = ttk.Style(self.root)
		try:
			style.theme_use('clam')
		except Exception:
			pass

	def _toggle_trading(self):
		self.on_toggle_trading(bool(self.trading_enabled.get()))

	def _on_refresh_clicked(self):
		# Placeholder; controller will bind periodic refresh
		pass

	def update_rows(self, rows: List[Dict[str, Any]]):
		# Preserve selection and scroll position
		selected = self.tree.selection()
		# Clear existing
		for item in self.tree.get_children():
			self.tree.delete(item)
		for row in rows:
			values = [
				row.get('id'), row.get('symbol'), row.get('timeframe'), row.get('type'),
				row.get('status'), row.get('left_at'), row.get('head_at'), row.get('right_at'),
				row.get('detected_at'), row.get('breakout_at'),
				row.get('entry'), row.get('neckline_a'), row.get('neckline_b'),
				row.get('sl'), row.get('tp'),
			]
			self.tree.insert('', tk.END, values=values)
		# Restore selection if possible
		for iid in selected:
			if self.tree.exists(iid):
				self.tree.selection_add(iid)