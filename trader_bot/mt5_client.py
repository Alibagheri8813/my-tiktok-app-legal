import time
import math
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import MetaTrader5 as mt5
import pandas as pd

# ... existing code ...

def timeframe_to_str(timeframe_const: int) -> str:
	return CONST_TO_TIMEFRAME_NAME.get(timeframe_const, str(timeframe_const))

# ... existing code ...

def initialize_mt5(login: Optional[int] = None,
					 password: Optional[str] = None,
					 server: Optional[str] = None,
					 path: Optional[str] = None) -> bool:
	# Read configuration from environment if not provided
	path = path or os.environ.get("MT5_PATH")
	if path and not os.path.exists(path):
		logging.error(f"MT5 path does not exist: {path}")
		return False
	if not mt5.initialize(path=path):
		logging.error(f"MT5 initialize failed: {mt5.last_error()} (path={path})")
		return False
	# Credentials via args or env
	if login is None:
		login_env = os.environ.get("MT5_LOGIN")
		try:
			login = int(login_env) if login_env else None
		except Exception:
			login = None
	if password is None:
		password = os.environ.get("MT5_PASSWORD") or password
	if server is None:
		server = os.environ.get("MT5_SERVER") or server
	if login and password and server:
		authorized = mt5.login(login=login, password=password, server=server)
		if not authorized:
			logging.error(f"MT5 login failed: {mt5.last_error()}")
			return False
	account_info = mt5.account_info()
	if account_info:
		logging.info(f"Connected to MT5 account: {account_info.login}")
	else:
		logging.warning("MT5 connected but no account info available. Ensure the terminal is logged in.")
	return True

# ... existing code ...