import pandas as pd, numpy as np
from buildup.fingerprint import split_half
p=pd.read_parquet('out10/_p2.parquet')
mr={m:i for i,m in enumerate(sorted(p.match_id.unique()))}
def rep(n,df,c):
    df=df[df[c].notna()]
    r=split_half(df,c,'player',mr,15);t=split_half(df,c,'team_shortname',mr,1,within=None)
    print(f"{n:40s} r={r['split_half_r']:.3f} rel={r['full_sample_reliability']:.2f} n={r['n']} team={t['split_half_r']:.2f}")
c='possession_epv_delta_for'
ns=p[p.end_type!='shot']
rep('epv delta, no shots',ns,c)
lo,hi=p[c].quantile([.01,.99])
rep('epv delta winsor 1/99',p.assign(x=p[c].clip(lo,hi)),'x')
rep('epv delta winsor 1/99 no shots',ns.assign(x=ns[c].clip(lo,hi)),'x')
lo5,hi5=p[c].quantile([.025,.975])
rep('epv delta winsor 2.5/97.5',p.assign(x=p[c].clip(lo5,hi5)),'x')
rep('epv delta, no shots no losses',ns[ns.end_type!='possession_loss'],c)
rep('epv delta, carries no shots',ns[ns.carry==1],c)
rep('epv delta pass-ending only',p[p.end_type=='pass'],c)
for th in [0.002,0.005,0.01]:
    rep(f'share epv delta>{th} (no shots)',ns.assign(x=(ns[c]>th).astype(float)),'x')
rep('share epv delta>0.005 carries no shots',ns[ns.carry==1].assign(x=(ns[c]>0.005).astype(float)),'x')
rep('opt_ahead_gain',p,'opt_ahead_gain')
rep('opt_ahead_gain>0 share',p.assign(x=(p.opt_ahead_gain>0).astype(float)),'x')
rep('opt_ahead_gain clip -2..2',p.assign(x=p.opt_ahead_gain.clip(-2,2)),'x')
rep('opt_ahead_gain >=2s possessions',p[p.duration>=2],'opt_ahead_gain')
