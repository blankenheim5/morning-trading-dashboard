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
    invalid = ((d.high < d[OHLC].max(axis=1) - 1e-5)
               | (d.low > d[OHLC].min(axis=1) + 1e-5))
    if invalid.any():
        raise ValueError(f'invalid OHLC relationships on {d.index[invalid][0].date()}')
    return d


def require_sessions(bars, sessions):
    missing = sessions.difference(bars.index)
    extra = bars.index.difference(sessions)
    if len(missing) or len(extra):
        raise ValueError(
            f'session mismatch: {len(missing)} missing, {len(extra)} unexpected; '
            f'first missing {missing[0].date() if len(missing) else "none"}, '
            f'latest bar {bars.index[-1].date()}, expected {sessions[-1].date()}'
        )


def split_factors(primary):
    # Yahoo's OHLC already reflect splits even with auto_adjust=False.
    # Build factors for comparing an independent series quoted on its original basis.
    splits = pd.to_numeric(primary['stock_splits'], errors='coerce')
    if (~np.isfinite(splits) | (splits < 0)).any():
        raise ValueError('invalid Yahoo split metadata')
    ratios = splits.replace(0, 1)
    return ratios.iloc[::-1].cumprod().iloc[::-1].shift(-1, fill_value=1)


def rsi_wilder(close,n=14):
    diff=close.diff();gain=diff.clip(lower=0);loss=-diff.clip(upper=0)
    ag=gain.iloc[1:n+1].mean();al=loss.iloc[1:n+1].mean()
    for g,l in zip(gain.iloc[n+1:],loss.iloc[n+1:]):
        ag=(ag*(n-1)+g)/n;al=(al*(n-1)+l)/n
    return 100 if al==0 and ag>0 else 50 if al==0 else 100-100/(1+ag/al)

def crosscheck(primary, secondary, factors):
    prices = secondary[OHLC]
    deviations = (primary[OHLC] / prices - 1).abs()
    if (deviations > PRICE_TOLERANCE).any().any() and (factors != 1).any():
        # Require one basis for the whole window. Never pick a basis per row.
        adjusted = (primary[OHLC] / prices.div(factors, axis=0) - 1).abs()
        if adjusted.max().max() < deviations.max().max():
            deviations = adjusted
    worst = deviations.stack()
    if worst.max() > PRICE_TOLERANCE:
        date, field = worst.idxmax()
        raise ValueError(f'independent {field} mismatch {worst.max():.4%} on {date.date()}')
    return {
        'secondary_sessions_checked': len(secondary),
        'secondary_max_close_difference_pct': round(float(deviations.close.max() * 100), 6),
        'secondary_max_ohlc_difference_pct': round(float(worst.max() * 100), 6),
    }


def current_open(ticker, now, today, daily, independent):
    """Compare the first Yahoo regular minute with both vendors' daily opens."""
    if today.empty or pd.Timestamp(now) < today.iloc[0].market_open:
        return None, None
    import yfinance as yf

    open_at = today.iloc[0].market_open
    bars = yf.Ticker(ticker).history(period='2d', interval='1m', auto_adjust=False,
                                    prepost=False)
    today_date = pd.Timestamp(now.date())
    daily_index = daily.index.tz_convert(ET).tz_localize(None).normalize()
    opens = daily.loc[daily_index == today_date, 'Open']
    if bars.empty or open_at not in bars.index or len(opens) != 1:
        raise ValueError('current regular-session opening bar is missing')
    opening = float(bars.loc[open_at, 'Open'])
    if today_date not in independent.index:
        raise ValueError('independent current-session open is missing')
    daily_opens = np.array([float(opens.iloc[0]), float(independent.loc[today_date, 'open'])])
    if (not np.isfinite(opening) or opening <= 0 or not np.isfinite(daily_opens).all()
            or (daily_opens <= 0).any() or (abs(opening / daily_opens - 1) > PRICE_TOLERANCE).any()):
        raise ValueError('current regular-session open disagrees with a daily open')
    return round(opening, 4), str(now.date())


def calculate(ticker,d,cutoff,source,secondary_source,checks,opening):
    close=float(d.close.iloc[-1]);row={'ticker':ticker,'asof':str(cutoff),'source':source,'secondary_source':secondary_source,**checks,'bars':len(d),'close':round(close,4),'volume':int(d.volume.iloc[-1]),'rsi14':round(rsi_wilder(d.close),2)}
    for n in (20,50,200):
        sma=float(d.close.iloc[-n:].mean());row[f'sma{n}']=round(sma,4);row[f'above_sma{n}']=bool(close>sma)
    if ticker in DIP:
        h=float(d.high.iloc[-90:].max());row['high90']=round(h,4);row['distance90_pct']=round(100*(close/h-1),2)
        for n,f in ((1,.90),(2,.85),(3,.80)):
            row[f't{n}_price']=round(h*f,4);row[f't{n}_crossed']=bool(close<=h*f)
    row['current_open'], row['current_open_date'] = opening
    return row

def run(args):
    import pandas_market_calendars as mcal

    errors = {}
    results = []
    diagnostics = {}
    cutoff = None
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        now = dt.datetime.now(ET)
        latest = last_completed_session(now)
        cutoff = args.asof or latest
        if cutoff > latest or (cutoff != latest and not args.allow_historical):
            raise ValueError('requested cutoff is not an allowed completed NYSE session')
        sessions = expected_sessions(cutoff - dt.timedelta(days=400), cutoff)[-MIN_SESSIONS:]
        if len(sessions) != MIN_SESSIONS or sessions[-1].date() != cutoff:
            raise ValueError('requested cutoff is not a valid NYSE session')
        today = mcal.get_calendar('NYSE').schedule(start_date=now.date(), end_date=now.date())
        if args.csv_dir and args.secondary_csv_dir:
            raise ValueError('two local CSV inputs cannot establish independent provenance')
        for ticker in TICKERS:
            try:
                raw, source = csv_source(ticker, args.csv_dir) if args.csv_dir else yahoo(ticker)
                primary = normalized(raw, cutoff)
                require_sessions(primary, expected_sessions(primary.index[0], cutoff))
                if args.csv_dir:
                    # The caller must supply split-adjusted, dividend-unadjusted prices.
                    factors = pd.Series(1.0, index=primary.index)
                else:
                    factors = split_factors(primary)
                    if cutoff < latest:
                        dates = raw.index.tz_convert(ET).date
                        if (raw.loc[dates > cutoff, 'Stock Splits'] != 0).any():
                            raise ValueError('historical cutoff precedes a later Yahoo split')
                split_dates = primary.index[factors.ne(factors.shift(1))][1:]
                split_positions = primary.index.get_indexer(split_dates)
                split_sessions = split_dates.union(primary.index[split_positions - 1])
                comparison_sessions = sessions.union(split_sessions)
                if args.secondary_csv_dir:
                    raw_secondary, secondary_source = csv_source(ticker, args.secondary_csv_dir)
                else:
                    history_range = '5Y' if comparison_sessions[0] < sessions[0] or cutoff < latest else '1Y'
                    raw_secondary, secondary_source = stock_analysis(ticker, history_range)
                # Verify calendar coverage before comparing split-boundary bars.
                secondary = normalized(raw_secondary, cutoff, start=sessions[0])
                require_sessions(secondary, sessions)
                if len(comparison_sessions) > len(sessions):
                    secondary = normalized(raw_secondary.loc[raw_secondary.index.isin(comparison_sessions)], cutoff)
                    missing_split_sessions = split_sessions.difference(secondary.index)
                    if len(missing_split_sessions):
                        raise ValueError(f'missing independent split-boundary bar on {missing_split_sessions[0].date()}')
                checks = crosscheck(primary.loc[comparison_sessions], secondary.loc[comparison_sessions],
                                    factors.loc[comparison_sessions])
                checks['secondary_sessions_checked'] = len(sessions)
                checks['split_boundary_sessions_checked'] = len(split_sessions)
                opening = (None, None)
                if args.current_open and cutoff == latest:
                    yahoo_daily = yahoo(ticker)[0] if args.csv_dir else raw
                    open_reference = stock_analysis(ticker)[0] if args.secondary_csv_dir else raw_secondary
                    opening = current_open(ticker, now, today, yahoo_daily, open_reference)
                results.append(calculate(ticker, primary, cutoff, source,
                                         secondary_source, checks, opening))
                diagnostics[ticker] = {'status': 'passed', **checks,
                                       'current_open_checked': opening[0] is not None}
            except Exception as exc:
                errors[ticker] = str(exc)
                diagnostics[ticker] = {'status': 'failed', 'error': str(exc)}
        if errors:
            raise ValueError(f'Only {len(results)}/{len(TICKERS)} independently validated; publication blocked')
        frame = pd.DataFrame(results)
        payload = {
            'asof': str(cutoff), 'status': 'validated', 'tickers': results,
            'validation': {
                'minimum_sessions': MIN_SESSIONS,
                'secondary_price_check': '220 completed sessions and split-boundary bars; every OHLC within 0.75%',
                'method': 'split-adjusted OHLC, dividend-unadjusted close',
                'open_note': 'current_open uses Yahoo regular-session minute checked against both daily feeds',
            },
        }
        with tempfile.TemporaryDirectory(dir=out.parent) as staging:
            csvtmp = pathlib.Path(staging) / 'dashboard.csv'
            jstmp = pathlib.Path(staging) / 'dashboard.json'
            frame.to_csv(csvtmp, index=False)
            jstmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
            csvtmp.replace(out)
            jstmp.replace(out.with_suffix('.json'))
        print('VALIDATED:', len(results), 'tickers, asof', cutoff)
        return 0
    except Exception as exc:
        errors['validation'] = str(exc)
        print('FAILED VALIDATION:', exc, file=sys.stderr)
        return 2
    finally:
        pathlib.Path(str(out) + '.errors.json').write_text(
            json.dumps({'asof': str(cutoff) if cutoff else None, 'errors': errors, 'checks': diagnostics}, indent=2)
        )


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='dashboard.csv');p.add_argument('--asof',type=dt.date.fromisoformat)
    p.add_argument('--csv-dir');p.add_argument('--secondary-csv-dir');p.add_argument('--allow-historical',action='store_true')
    p.add_argument('--current-open',action='store_true',help='Include current regular-session open when available')
    sys.exit(run(p.parse_args()))
