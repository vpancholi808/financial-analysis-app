"""Resume Yahoo Finance screening and persist genuine US company statements in batches."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from build_universe import discover_quotes, extract_company, issuer_key
from finance_models import UNIVERSE_COLUMNS

ROOT=Path(__file__).resolve().parent
def read_json(path, default):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return default
def save_json(path, data):
    path.write_text(json.dumps(data,indent=2)+"\n")

def collect(target=1000,batch_size=90,pause=1.0):
    import yfinance as yf
    records=ROOT/'universe.csv'
    state_path=ROOT/'collection_state.json'
    candidates=ROOT/'candidate_quotes.json'
    state=read_json(state_path,{'cursor':0,'attempted':0,'failures':0})
    d=pd.read_csv(records) if records.exists() else pd.DataFrame(columns=UNIVERSE_COLUMNS)
    if 'ticker' not in d: raise ValueError('Universe CSV must have ticker column')
    d=d.loc[d.ticker.notna() & d.ticker.astype(str).str.upper().ne('TSM')]
    rows=d.to_dict('records')
    seen={str(r['ticker']).upper() for r in rows}
    issuers={issuer_key(r) for r in rows}
    def save():
        pd.DataFrame(rows[:target],columns=UNIVERSE_COLUMNS).to_csv(records,index=False)
        state['verified']=len(rows)
        state['updated_at']=datetime.now(timezone.utc).isoformat(timespec='seconds')
        save_json(state_path,state)
    quotes=read_json(candidates,[])
    if not quotes:
        try:
            quotes=discover_quotes(yf,target,buffer=1000)
            quotes=[{'symbol':q.get('symbol',''),'intradaymarketcap':q.get('intradaymarketcap'),
                     'exchange':q.get('exchange'),'quoteType':q.get('quoteType')} for q in quotes if q.get('symbol')]
            if not quotes: raise RuntimeError('Yahoo screener returned zero symbols')
            save_json(candidates,quotes)
        except Exception as e:
            state['last_error']=str(e)
            save()
            raise
    cursor=int(state.get('cursor',0))
    attempted=0
    failed=0
    accepted=0
    while cursor<len(quotes) and attempted<batch_size and len(rows)<target:
        q=quotes[cursor]
        cursor+=1
        sym=str(q.get('symbol','')).strip().upper()
        if not sym or sym=='TSM' or sym in seen: continue
        attempted+=1
        try:
            record=extract_company(yf.Ticker(sym),q)
            if record and issuer_key(record) not in issuers:
                rows.append(record)
                seen.add(sym)
                issuers.add(issuer_key(record))
                accepted+=1
                print('ACCEPT',sym,len(rows),flush=True)
            else:
                failed+=1
                print('SKIP',sym,flush=True)
        except Exception as e:
            failed+=1
            print('ERROR',sym,str(e),flush=True)
        state.update(cursor=cursor,attempted=int(state.get('attempted',0))+1)
        if attempted%15==0: save()
        if pause: time.sleep(pause)
    state['failures']=int(state.get('failures',0))+failed
    state['accepted_last_batch']=accepted
    state['failures_last_batch']=failed
    state['status']='complete' if len(rows)>=target else 'candidate_pool_exhausted' if cursor>=len(quotes) else 'collecting'
    state['last_error']='Yahoo did not return any eligible statements in this batch' if attempted and accepted==0 else ''
    save()
    print(json.dumps(state,indent=2))
    if attempted and accepted==0:
        raise RuntimeError(state['last_error'])
    if state['status']=='candidate_pool_exhausted' and len(rows)<500:
        raise RuntimeError('Too few valid issuers. Broaden the candidate universe.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--target',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=90)
    p.add_argument('--pause',type=float,default=1.0)
    a=p.parse_args()
    if a.target<500 or a.batch_size<1 or a.pause<0:
        p.error('target>=500, batch-size>=1 and pause>=0 required')
    collect(a.target,a.batch_size,a.pause)
