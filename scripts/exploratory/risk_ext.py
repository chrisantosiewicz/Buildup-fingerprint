import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from buildup.fingerprint import split_half
d=pd.read_parquet("out10/decisions_style.parquet")
o=pd.read_csv('out10/option_values.csv',usecols=['decision_id','xpass_completion'])
s=o.groupby('decision_id').xpass_completion.agg(['min','max']); d=d.merge(s,left_on='decision_id',right_index=True)
d['spread']=d['max']-d['min']; m2=d.n_options>=2
ms=d.loc[m2,'spread'].mean()
d['risk_w']=np.where(m2,d.risk*d.spread/ms,np.nan)
mr={m:i for i,m in enumerate(sorted(d.match_id.unique()))}
r=split_half(d[m2],'risk_w','player',mr,15);t=split_half(d[m2],'risk_w','team_shortname',mr,1,within=None)
print('risk_w transform',r,t['split_half_r'])
A=pd.read_csv("dl/opendata/data/aggregates/aus1league_passingaggregates_20242025.csv").dropna(subset=["pass_count_attempted_total"])
A["difficult_share"]=A.pass_count_difficultpass_attempted_p30tip/A.pass_count_attempted_p30tip
A=A.groupby("player_id").apply(lambda g: pd.Series({c: np.average(g[c].fillna(g[c].mean()),weights=g.performance_count) for c in ['pass_avgxpass_attempted','difficult_share']}|{'games':g.performance_count.sum()}))
A=A[A.games>=8]
def agg(df,col,w=None):
    df=df[df[col].notna()]
    if w is None: return df.groupby('passer_id')[col].mean()
    return df.groupby('passer_id').apply(lambda g: np.average(g[col],weights=g[w]) if g[w].sum()>0 else np.nan)
ours=pd.DataFrame({'n':d.groupby('passer_id').size(),'pos':d.groupby('passer_id').pos_group.agg(lambda s:s.mode().iloc[0]),
 'risk_all':agg(d,'risk'),'risk_2':agg(d[m2],'risk'),'risk_spreadw':agg(d[m2],'risk','spread'),'risk_w':agg(d,'risk_w')})
m=ours.join(A,how='inner'); m=m[m.n>=40]
for a in ['risk_all','risk_2','risk_spreadw','risk_w']:
    for b in ['pass_avgxpass_attempted','difficult_share']:
        mm=m[[a,b,'pos']].dropna(); w=mm[[a,b]].sub(mm.groupby('pos')[[a,b]].transform('mean'))
        print(f"{a:13s} vs {b:25s} within-pos rho={spearmanr(w[a],w[b]).correlation:+.2f} n={len(mm)}")
