#!/usr/bin/env python3
"""Validated market-only technical dashboard.
Install: pip install pandas numpy yfinance requests pandas_market_calendars
Run: python technical_dashboard.py --output dashboard.csv
Optional: --asof YYYY-MM-DD --csv-dir prices --secondary-csv-dir verified_prices
Secondary CSVs are independent vendor exports named TICKER.csv (Date,Open,High,Low,Close,Volume).
Only completed US equity sessions are used. Fail closed if secondary checks unavailable.
"""
import argparse
import datetime as dt
import json
import pathlib
import tempfile
import sys
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

TICKERS = ('SCHG','URA','COPX','PDBC','CEG','VST','CIBR','SMH','SPY')
DIP = {'SCHG','URA','COPX','PDBC','CEG','VST'}
ET = ZoneInfo('America/New_York')
MIN_SESSIONS = 220
OHLC = ['open', 'high', 'low', 'close']
PRICE_TOLERANCE = 0.0075

def last_completed_session(now):
    import pandas_market_calendars as mcal
    cal = mcal.get_calendar('NYSE')
    sessions = cal.schedule(start_date=now.date()-dt.timedelta(days=20),end_date=now.date())
    completed = sessions[sessions['market_close'] <= pd.Timestamp(now.astimezone(dt.timezone.utc))]
    if completed.empty: raise ValueError('No completed NYSE session found')
    return completed.index[-1].date()

def yahoo(ticker):
    import yfinance as yf
    d = yf.Ticker(ticker).history(period='3y',interval='1d',auto_adjust=False,actions=True)
    if d.empty: raise ValueError('Yahoo returned no bars')
    return d,'Yahoo/yfinance'

def stock_analysis(ticker, history_range="1Y"):
    import requests

    asset_type = 's' if ticker in {'CEG', 'VST'} else 'e'
    response = requests.get(
        f'https://api.stockanalysis.com/api/symbol/{asset_type}/{ticker}/history',
        params={'range': history_range},
        headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'},
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get('data')
    if payload.get('status') != 200 or not isinstance(rows, list) or not rows:
        raise ValueError('Stock Analysis returned no usable historical bars')
    bars = pd.DataFrame(rows).rename(
        columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
    ).set_index('t')
    bars.index = pd.to_datetime(bars.index, format='%Y-%m-%d')
    return bars, 'S&P Global via Stock Analysis'


def csv_source(ticker, folder):
    path = pathlib.Path(folder) / (ticker + '.csv')
    return pd.read_csv(path, index_col=0, parse_dates=True), str(path)


def expected_sessions(start, end):
    import pandas_market_calendars as mcal

    return mcal.get_calendar('NYSE').schedule(start_date=start, end_date=end).index


def normalized(raw, cutoff, start=None):
    d = raw.copy()
    d.columns = [str(c).strip().lower().replace(' ', '_') for c in d.columns]
    if 'date' in d.columns:
        d = d.set_index('date')
    index = pd.to_datetime(d.index)
    if index.tz is not None:
        index = index.tz_convert(ET).tz_localize(None)
    d.index = index.normalize()
    d = d.sort_index().loc[:pd.Timestamp(cutoff)]
    if start is not None:
        d = d.loc[start:]
    if d.index.hasnans or d.index.has_duplicates:
        raise ValueError('invalid or duplicate trading dates')
    if len(d) < MIN_SESSIONS:
        raise ValueError(f'{len(d)} sessions; need >= {MIN_SESSIONS}')
    for column in OHLC + ['volume']:
        if column not in d:
            raise ValueError(f'missing {column}')
        d[column] = pd.to_numeric(d[column], errors='coerce')
        invalid = ~np.isfinite(d[column]) | (d[column] <= 0)
        if invalid.any():
            raise ValueError(f'invalid {column} on {d.index[invalid][0].date()}')
    invalid = (d.high < d[OHLC].max(axis=1) - 1e-5)
