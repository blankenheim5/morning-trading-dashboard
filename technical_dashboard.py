#!/usr/bin/env python3
"""Daily B3 technical dashboard. pip install pandas numpy yfinance requests
Usage: python technical_dashboard.py --output dashboard.csv
Optional: --asof YYYY-MM-DD (completed session cutoff), --csv-dir ./prices (TICKER.csv overrides downloads)
"""
import argparse, datetime as dt, io, json, pathlib, sys, time
import numpy as np
import pandas as pd
TICKERS=['SCHG','URA','COPX','PDBC','CEG','VST','CIBR','SMH','SPY']
DIP=set(['SCHG','URA','COPX','PDBC','CEG','VST'])

def fetch_yahoo(ticker):
    import yfinance as yf
    d=yf.Ticker(ticker).history(period='2y',interval='1d',auto_adjust=False,actions=True)
    if d.empty: raise ValueError('empty Yahoo history')
    return d, 'Yahoo/yfinance'

def fetch_stooq(ticker):
    import requests
    u='https://stooq.com/q/d/l/'
    r=requests.get(u,params={'s':ticker.lower()+'.us','i':'d','d1':(dt.date.today()-dt.timedelta(days=900)).strftime('%Y%m%d')},headers={'User-Agent':'Mozilla/5.0'},timeout=20)
    r.raise_for_status()
    if not r.text.startswith('Date,'): raise ValueError('Stooq returned non-CSV response')
    d=pd.read_csv(io.StringIO(r.text)); d['Date']=pd.to_datetime(d['Date']);d=d.set_index('Date')
    return d, 'Stooq (verify adjustment basis)'

def normalize(df,asof):
    d=df.copy();d.columns=[str(c).strip().lower().replace(' ','_') for c in d.columns]
    if 'date' in d.columns: d=d.set_index('date')
    idx=pd.to_datetime(d.index,utc=True).tz_convert('America/New_York').tz_localize(None) if getattr(pd.to_datetime(d.index),'tz',None) is not None else pd.to_datetime(d.index)
    d.index=idx.normalize();d=d[~d.index.duplicated(keep='last')].sort_index()
    d=d.loc[d.index.date<=asof]
    # Exclude current US session even if market data source has a partial bar.
    now=dt.datetime.now(dt.timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo('America/New_York'))
    if asof>=now.date() and (now.hour<16 or (now.hour==16 and now.minute<15)):
        d=d.loc[d.index.date<now.date()]
    for c in ['open','high','low','close','volume']:
        if c not in d: raise ValueError('missing '+c)
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['open','high','low','close','volume'])
    if len(d)<220:raise ValueError(f'only {len(d)} valid completed bars; need >=220')
    if (d[['open','high','low','close']]<=0).any().any() or (d.volume<0).any():raise ValueError('nonpositive price or negative volume')
    if ((d.high+0.0001<d[['open','close','low']].max(axis=1)) | (d.low-0.0001>d[['open','close','high']].min(axis=1))).any():raise ValueError('inconsistent OHLC bars')
    # Split-adjusted OHLC prices must be on the same basis; do not use dividend-adjusted close with raw highs.
    if 'stock_splits' in d and (d.stock_splits.fillna(0)!=0).any():
        raise ValueError('split in lookback: manually verify adjusted OHLC before signal')
    return d

def wilder_rsi(close,period=14):
    delta=close.diff();gain=delta.clip(lower=0);loss=-delta.clip(upper=0)
    if len(close)<period+1:return float('nan')
    avg_gain=gain.iloc[1:period+1].mean();avg_loss=loss.iloc[1:period+1].mean()
    for g,l in zip(gain.iloc[period+1:],loss.iloc[period+1:]):
        avg_gain=(avg_gain*(period-1)+g)/period
        avg_loss=(avg_loss*(period-1)+l)/period
    if avg_loss==0:return 100.0 if avg_gain>0 else 50.0
    return 100-100/(1+avg_gain/avg_loss)

def calculate(ticker,df,asof,source):
    d=normalize(df,asof);close=float(d.close.iloc[-1]);r={'ticker':ticker,'asof':str(d.index[-1].date()),'source':source,'bars':len(d),'close':round(close,4),'volume':int(d.volume.iloc[-1]),'rsi14':round(wilder_rsi(d.close),2)}
    for n in (20,50,200):
        sma=float(d.close.tail(n).mean());r[f'sma{n}']=round(sma,4);r[f'above_sma{n}']=bool(close>sma)
    if ticker in DIP:
        high=float(d.high.tail(90).max());r['high90']=round(high,4);r['distance90_pct']=round(100*(close/high-1),2)
        for tier,pct in [(1,.90),(2,.85),(3,.80)]:
            r[f't{tier}_price']=round(high*pct,4);r[f't{tier}_crossed']=bool(close<=high*pct)
    return r

def run(asof,output,csv_dir=None):
    results=[];errors={}
    for ticker in TICKERS:
        attempts=[]
        sources=[]
        if csv_dir:sources.append(('local CSV',lambda t=ticker: (pd.read_csv(pathlib.Path(csv_dir)/(t+'.csv'),index_col=0,parse_dates=True),'local CSV')))
        sources.extend([('Yahoo',lambda t=ticker:fetch_yahoo(t)),('Stooq',lambda t=ticker:fetch_stooq(t))])
        for name,fetch in sources:
            try:
                d,source=fetch();results.append(calculate(ticker,d,asof,source));break
            except Exception as exc:attempts.append(f'{name}: {str(exc)[:150]}')
        else:errors[ticker]=attempts
    frame=pd.DataFrame(results)
    if not frame.empty:
        frame.to_csv(output,index=False)
        print(frame.to_string(index=False))
    pathlib.Path(str(output)+'.errors.json').write_text(json.dumps(errors,indent=2))
    print('Validated',len(results),'of',len(TICKERS),'tickers; errors:',json.dumps(errors,indent=2),file=sys.stderr)
    return len(results)==len(TICKERS)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asof',type=dt.date.fromisoformat,default=dt.date.today());p.add_argument('--output',default='dashboard.csv');p.add_argument('--csv-dir');args=p.parse_args()
    sys.exit(0 if run(args.asof,args.output,args.csv_dir) else 2)
