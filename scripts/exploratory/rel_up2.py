import pandas as pd, numpy as np
from buildup.fingerprint import split_half
d=pd.read_parquet('out10/decisions_style.parquet'); p2=pd.read_parquet('out10/_p2.parquet')
o=pd.read_csv('out10/option_values.csv',usecols=['decision_id','xpass_completion'])
mr={m:i for i,m in enumerate(sorted(d.match_id.unique()))}
def sh(df,col,w=None,mh=15,within=True):
    df=df[df[col].notna()].assign(half=df.match_id.map(mr)%2)
    df['_v']=df[col]-df.groupby('pos_group')[col].transform('mean') if within else df[col]
    df['_w']=1.0 if w is None else df[w]
    g=df.groupby(['player','half']).apply(lambda x: pd.Series({'m':np.average(x._v,weights=x._w) if x._w.sum()>0 else np.nan,'n':len(x)}),include_groups=False).unstack()
    g=g[(g[('n',0)]>=mh)&(g[('n',1)]>=mh)]
    r=g[('m',0)].corr(g[('m',1)]); return round(r,3), round(2*max(r,0)/(1+max(r,0)),2), len(g)
def team(df,col):
    return split_half(df[df[col].notna()],col,'team_shortname',mr,1,within=None)['split_half_r']
s=o.groupby('decision_id').xpass_completion.agg(['min','max','std'])
d=d.merge(s,left_on='decision_id',right_index=True)
d['spread']=d['max']-d['min']
m=d.n_options>=2
print('risk >=2 unweighted',sh(d[m],'risk'))
print('risk weighted by spread',sh(d[m],'risk','spread'))
print('risk weighted by spread^2',sh(d[m].assign(s2=d.spread**2),'risk','s2'))
print('risk spread>0.1',sh(d[m&(d.spread>0.1)],'risk'), team(d[m&(d.spread>0.1)],'risk'))
print('risk spread>0.2',sh(d[m&(d.spread>0.2)],'risk'), team(d[m&(d.spread>0.2)],'risk'))
print('risk/spread, spread>0.1',sh(d[m&(d.spread>0.1)].assign(x=d.risk/d.spread),'x'))
d['absr']=1-d.xpass_completion
print('abs >=2 weighted spread',sh(d[m],'absr','spread'))
# composite at player level: per-half z of two
print('--- CREATOR')
for c in ['possession_epv_delta_for','opt_ahead_gain','n_opponents_bypassed','separation_gain']:
    print(c, sh(p2,c), team(p2,c))
for c in ['possession_epv_delta_for']:
    for lo,hi in [(-0.02,0.02),(-0.05,0.05),(0,0.05)]:
        print(c,'clip',lo,hi, sh(p2.assign(x=p2[c].clip(lo,hi)),'x'))
    print(c,'carries only',sh(p2[p2.carry==1],c), team(p2[p2.carry==1],c))
    print(c,'>0 share',sh(p2.assign(x=(p2[c]>0.005).astype(float)),'x'))
    print(c,'build/create',sh(p2[p2.team_in_possession_phase_type.isin(['build_up','create'])],c))
print('opt_ahead carries only',sh(p2[p2.carry==1],'opt_ahead_gain'))
print(p2[['possession_epv_delta_for','on_ball_value_created','opt_ahead_gain','n_opponents_bypassed','duration','carry']].corr().round(2))
