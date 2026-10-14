from pathlib import Path
from urllib.request import urlopen
import hashlib
import pandas as pd

base=Path('outputs/pl-match-predictor/data')
for start in range(2022,2026):
    end=start+1
    code=f'{start%100:02d}{end%100:02d}'
    local=base/f'epl_{start}_{end}.csv'
    url=f'https://www.football-data.co.uk/mmz4281/{code}/E0.csv'
    with urlopen(url, timeout=25) as response:
        current=response.read()
    online=pd.read_csv(pd.io.common.BytesIO(current))
    saved=pd.read_csv(local)
    def results(df):
        cols=[c for c in ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'] if c in df.columns]
        x=df[cols].dropna(subset=['Date','HomeTeam','AwayTeam','FTR']).copy()
        x['Date']=pd.to_datetime(x.Date, dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
        return x
    a,b=results(online),results(saved)
    common=[c for c in ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'] if c in a.columns and c in b.columns]
    aa=set(map(tuple,a[common].astype(str).to_numpy()))
    bb=set(map(tuple,b[common].astype(str).to_numpy()))
    print(f'{start}/{end}: online bytes={len(current)}, local bytes={local.stat().st_size}, online matches={len(a)}, local matches={len(b)}, exact_match_rows={aa==bb}, online_only={len(aa-bb)}, local_only={len(bb-aa)}, online_sha256={hashlib.sha256(current).hexdigest()}, local_sha256={hashlib.sha256(local.read_bytes()).hexdigest()}')
