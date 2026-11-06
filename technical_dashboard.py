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
import io
import json
import pathlib
import sys
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

TICKERS = ('SCHG','URA','COPX','PDBC','CEG','VST','CIBR','SMH','SPY')
DIP = {'SCHG','URA','COPX','PDBC','CEG','VST'}
ET = ZoneInfo('America/New_York')

def last_completed_session():
    import pandas_market_calendars as mcal
    now = dt.datetime.now(ET)
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

def stooq(ticker):
    import requests
    r=requests.get('https://stooq.com/q/d/l/',params={'s':ticker.lower()+'.us','i':'d','d1':(dt.date.today()-dt.timedelta(days=1200)).strftime('%Y%m%d')},headers={'User-Agent':'Mozilla/5.0'},timeout=25)
    r.raise_for_status()
    if not r.text.startswith('Date,'): raise ValueError('Stooq did not return daily CSV')
    d=pd.read_csv(io.StringIO(r.text)).set_index('Date')
    return d,'Stooq'

def csv_source(ticker,folder):
    p=pathlib.Path(folder)/(ticker+'.csv')
    return pd.read_csv(p,index_col=0,parse_dates=True),str(p)

def normalized(raw,cutoff):
    d=raw.copy()
    d.columns=[str(c).strip().lower().replace(' ','_') for c in d.columns]
    if 'date' in d.columns: d=d.set_index('date')
    idx=pd.to_datetime(d.index,utc=True) if getattr(pd.to_datetime(d.index),'tz',None) is not None else pd.to_datetime(d.index)
    if getattr(idx,'tz',None) is not None: idx=idx.tz_convert(ET).tz_localize(None)
    d.index=idx.normalize()
    if d.index.has_duplicates: raise ValueError('duplicate trading dates')
    d=d.sort_index().loc[lambda x:x.index.date<=cutoff]
    for c in ('open','high','low','close','volume'):
        if c not in d: raise ValueError('missing '+c)
        d[c]=pd.to_numeric(d[c],errors='coerce')
    if d[['open','high','low','close','volume']].isna().any().any(): raise ValueError('missing or nonnumeric OHLCV')
    if len(d)<220: raise ValueError(f'{len(d)} sessions; need >=220')
    if (d[['open','high','low','close']]<=0).any().any() or (d.volume<0).any(): raise ValueError('invalid price or volume')
    if ((d.high<d[['open','low','close']].max(axis=1)-1e-5)|(d.low>d[['open','high','close']].min(axis=1)+1e-5)).any(): raise ValueError('invalid OHLC relationships')
    return d

def adjust_splits(d):
    """Adjust earlier OHLC for splits, preserving dividend-unadjusted prices and volumes."""
    x=d.copy()
    if 'stock_splits' not in x: return x
    splits=pd.to_numeric(x.stock_splits,errors='coerce').fillna(0)
    factors=splits.replace(0,1).iloc[::-1].cumprod().iloc[::-1].shift(-1,fill_value=1)
    for c in ('open','high','low','close'): x[c]=x[c]/factors
    x['volume']=x['volume']*factors
    return x

def rsi_wilder(close,n=14):
    diff=close.diff();gain=diff.clip(lower=0);loss=-diff.clip(upper=0)
    ag=gain.iloc[1:n+1].mean();al=loss.iloc[1:n+1].mean()
    for g,l in zip(gain.iloc[n+1:],loss.iloc[n+1:]):
        ag=(ag*(n-1)+g)/n;al=(al*(n-1)+l)/n
    return 100 if al==0 and ag>0 else 50 if al==0 else 100-100/(1+ag/al)

def crosscheck(primary,secondary,ticker):
    """Compare last 5 completed session closes against an independent vendor (0.75% tolerance)."""
    common=primary.index.intersection(secondary.index)
    if len(common)<5: raise ValueError('secondary source has fewer than 5 matching sessions')
    common=common[-5:]
    deviations=((primary.loc[common,'close']/secondary.loc[common,'close'])-1).abs()
    if (deviations>0.0075).any(): raise ValueError(f'{ticker} independent close mismatch max {deviations.max():.2%}')
    return round(float(deviations.max()*100),4)

def current_open(ticker,cutoff):
    """Opening price only if today's US regular session has actually opened; otherwise pending."""
    now=dt.datetime.now(ET)
    if now.date()<=cutoff or now.time()<dt.time(9,30): return None,None
    import pandas_market_calendars as mcal
    sched=mcal.get_calendar('NYSE').schedule(start_date=now.date(),end_date=now.date())
    if sched.empty: return None,None
    open_at=sched.iloc[0]['market_open'].tz_convert(ET)
    if pd.Timestamp(now)<open_at: return None,None
    import yfinance as yf
    bars=yf.Ticker(ticker).history(period='2d',interval='1m',auto_adjust=False,prepost=False)
    if bars.empty: return None,None
    index=bars.index.tz_convert(ET)
    todays=bars.loc[index.date==now.date()]
    if todays.empty: return None,None
    first=todays.iloc[0]
    return round(float(first['Open']),4),str(now.date())

def calculate(ticker,d,cutoff,source,secondary_source,discrepancy,with_open):
    close=float(d.close.iloc[-1]);row={'ticker':ticker,'asof':str(cutoff),'source':source,'secondary_source':secondary_source,'secondary_max_close_difference_pct':discrepancy,'bars':len(d),'close':round(close,4),'volume':int(d.volume.iloc[-1]),'rsi14':round(rsi_wilder(d.close),2)}
    for n in (20,50,200):
        sma=float(d.close.iloc[-n:].mean());row[f'sma{n}']=round(sma,4);row[f'above_sma{n}']=bool(close>sma)
    if ticker in DIP:
        h=float(d.high.iloc[-90:].max());row['high90']=round(h,4);row['distance90_pct']=round(100*(close/h-1),2)
        for n,f in ((1,.90),(2,.85),(3,.80)):
            row[f't{n}_price']=round(h*f,4);row[f't{n}_crossed']=bool(close<=h*f)
    row['current_open']=None;row['current_open_date']=None
    if with_open:
        try: row['current_open'],row['current_open_date']=current_open(ticker,cutoff)
        except Exception as exc: row['open_status']='unavailable: '+str(exc)[:100]
    return row

def run(args):
    errors={};results=[]
    try:
        cutoff=args.asof or last_completed_session()
        if cutoff!=last_completed_session() and not args.allow_historical: raise ValueError('requested cutoff is not latest completed NYSE session')
        for ticker in TICKERS:
            failures=[];primary=None;secondary=None
            primary_sources=([('local CSV',lambda:csv_source(ticker,args.csv_dir))] if args.csv_dir else [])+[('Yahoo',lambda:yahoo(ticker)),('Stooq',lambda:stooq(ticker))]
            for name,fn in primary_sources:
                try:
                    raw,source=fn();d=normalized(raw,cutoff)
                    if d.index[-1].date()!=cutoff: raise ValueError(f'stale: latest bar {d.index[-1].date()} vs {cutoff}')
                    primary=adjust_splits(d);break
                except Exception as exc: failures.append(f'{name}: {exc}')
            if primary is None: errors[ticker]=failures;continue
            # Independent verification must not use same provider or uploaded primary CSV.
            checks=([('independent CSV',lambda:csv_source(ticker,args.secondary_csv_dir))] if args.secondary_csv_dir else [])
            checks += ([('Stooq',lambda:stooq(ticker))] if source.startswith('Yahoo') or source.endswith('.csv') else [('Yahoo',lambda:yahoo(ticker))])
            for name,fn in checks:
                try:
                    raw,second_source=fn();s=adjust_splits(normalized(raw,cutoff))
                    if s.index[-1].date()!=cutoff: raise ValueError('secondary data stale')
                    discrepancy=crosscheck(primary,s,ticker);secondary=second_source;break
                except Exception as exc: failures.append(f'{name}: {exc}')
            if secondary is None: errors[ticker]=failures+['Independent verification REQUIRED; ticker excluded'];continue
            try: results.append(calculate(ticker,primary,cutoff,source,secondary,discrepancy,args.current_open))
            except Exception as exc: errors[ticker]=failures+[f'Calculation: {exc}']
        if len(results)!=len(TICKERS): raise ValueError(f'Only {len(results)}/{len(TICKERS)} independently validated; publication blocked')
        frame=pd.DataFrame(results)
        out=pathlib.Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        payload={'asof':str(cutoff),'status':'validated','tickers':results,'validation':{'minimum_sessions':220,'secondary_price_check':'last five closes within 0.75%','method':'split-adjusted OHLC, not dividend-adjusted close','open_note':'current_open is today regular-session open when available; 90-session reference and decline use completed sessions'}}
        # Atomic replacement: only overwrite public-ready files after every check passes.
        csvtmp=out.with_suffix(out.suffix+'.tmp');js=out.with_suffix('.json');jstmp=js.with_suffix('.json.tmp')
        frame.to_csv(csvtmp,index=False);jstmp.write_text(json.dumps(payload,indent=2,allow_nan=False))
        csvtmp.replace(out);jstmp.replace(js)
        print(frame.to_string(index=False));print('VALIDATED:',len(results),'tickers, asof',cutoff)
        return 0
    except Exception as exc:
        errors['validation']=str(exc)
        print('FAILED VALIDATION:',exc,file=sys.stderr)
        return 2
    finally:
        pathlib.Path(str(args.output)+'.errors.json').write_text(json.dumps(errors,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='dashboard.csv');p.add_argument('--asof',type=dt.date.fromisoformat)
    p.add_argument('--csv-dir');p.add_argument('--secondary-csv-dir');p.add_argument('--allow-historical',action='store_true')
    p.add_argument('--current-open',action='store_true',help='Include current regular-session open when available')
    sys.exit(run(p.parse_args()))
