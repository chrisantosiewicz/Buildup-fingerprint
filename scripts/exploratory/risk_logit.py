import pandas as pd, numpy as np
from scipy.optimize import minimize, minimize_scalar
o=pd.read_csv('out10/option_values.csv')
d=pd.read_parquet('out10/decisions_style.parquet')[['decision_id','player','pos_group','team_shortname']]
o=o.drop(columns=[c for c in ['player','pos_group','team_shortname'] if c in o]).merge(d,on='decision_id')
o=o[o.groupby('decision_id').decision_id.transform('size')>=2].copy()
feats=['xpass_completion','xthreat','P','pass_dx','V_cost']
X=o[feats].fillna(0).values; X=(X-X.mean(0))/X.std(0)
o[[f+'_z' for f in feats]]=X
gid=o.decision_id.values; codes,gi=np.unique(gid,return_inverse=True)
y=o.chosen.values.astype(float)
def nll(b,X=X):
    u=X@b; m=pd.Series(u).groupby(gi).transform('max').values
    e=np.exp(u-m); s=np.bincount(gi,e); lp=u-m-np.log(s[gi]); return -(y*lp).sum()
res=minimize(nll,np.zeros(len(feats)),method='BFGS'); b=res.x; print(dict(zip(feats,b.round(3))))
o['u0']=X@b; o['xz']=o.xpass_completion_z
mr={m:i for i,m in enumerate(sorted(o.match_id.unique()))}; o['half']=o.match_id.map(mr)%2
def slope(g,lam):
    gi2=pd.factorize(g.decision_id)[0]; u0=g.u0.values; x=g.xz.values; yy=g.chosen.values
    def f(t):
        u=u0+t*x; m=pd.Series(u).groupby(gi2).transform('max').values; e=np.exp(u-m); s=np.bincount(gi2,e)
        return -(yy*(u-m-np.log(s[gi2]))).sum()+lam*t*t
    return minimize_scalar(f,bounds=(-5,5),method='bounded').x
for lam in [0,2,5,10]:
    rows=[]
    for (pl,h),g in o.groupby(['player','half']):
        nd=g.decision_id.nunique()
        if nd<15: continue
        rows.append((pl,h,g.pos_group.iloc[0],g.team_shortname.iloc[0],-slope(g,lam),nd))
    r=pd.DataFrame(rows,columns=['player','half','pos','team','risk','n'])
    r['w']=r.risk-r.groupby('pos').risk.transform('mean')
    w=r.pivot(index='player',columns='half',values='w').dropna()
    rr=w[0].corr(w[1]); print('lam',lam,'n',len(w),'r',round(rr,3),'rel',round(2*max(rr,0)/(1+max(rr,0)),2))
