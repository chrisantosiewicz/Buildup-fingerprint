import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from buildup.fingerprint import METRICS, _rows
dec=pd.read_parquet("out8/decisions_style.parquet"); poss=pd.read_parquet("out8/possessions_style.parquet")
fp=pd.read_csv("out8/player_fingerprint.csv",index_col=0)
style=[m for m in METRICS if m.kind=="style" and m.key!="back_under_pressure"]   # too sparse per half
mr={m:i for i,m in enumerate(sorted(dec.match_id.unique()))}
def half_means(h):
    cols={}
    for m in style:
        r=_rows(m,dec,poss); r=r[r.match_id.map(mr)%2==h]
        cols[m.key]=r.groupby("player")[m.column].mean()
    return pd.DataFrame(cols)
H0,H1=half_means(0),half_means(1)
keys=[m.key for m in style]
for pos in ["CB","FB","MF","AM","FW"]:
    pl=fp[fp.pos_group==pos].index
    X=fp.loc[pl,keys]; mu,sd=X.mean(),X.std(); Z=(X-mu)/sd
    a=H0.reindex(pl).dropna(); b=H1.reindex(pl).dropna(); both=a.index.intersection(b.index)
    for k in [2,3]:
        km=KMeans(k,n_init=20,random_state=0).fit(Z)
        sil=silhouette_score(Z,km.labels_)
        la=km.predict(((a.loc[both]-mu)/sd)); lb=km.predict(((b.loc[both]-mu)/sd))
        print(pos,len(pl),"k",k,"silhouette",round(sil,2),"split-half role agreement ARI",round(adjusted_rand_score(la,lb),2),"same role",round((la==lb).mean(),2), "n",len(both))
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
print("--- pairwise style-distance agreement between halves (all outfield, standardised within position)")
pl=fp.index
a=H0.reindex(pl); b=H1.reindex(pl)
n0=dec[dec.match_id.map(mr)%2==0].groupby("player").size(); n1=dec[dec.match_id.map(mr)%2==1].groupby("player").size()
ok=[p for p in pl if n0.get(p,0)>=25 and n1.get(p,0)>=25 and a.loc[p].notna().all() and b.loc[p].notna().all()]
pos=fp.loc[ok,"pos_group"]
def zs(D): return D.groupby(pos).transform(lambda c:(c-c.mean())/c.std()).fillna(0)
A,B=zs(a.loc[ok,keys]),zs(b.loc[ok,keys])
print("players",len(ok),"Spearman of pairwise distances", round(spearmanr(pdist(A),pdist(B)).correlation,3))
# nearest-neighbour hit rate: is even-half NN among odd-half top-5?
from sklearn.metrics import pairwise_distances
DA,DB=pairwise_distances(A),pairwise_distances(B); np.fill_diagonal(DA,np.inf); np.fill_diagonal(DB,np.inf)
hits=np.mean([DB[i].argmin() in np.argsort(DA[i])[:5] for i in range(len(ok))])
print("NN in even half is in odd-half top 5:",round(hits,2),"chance:",round(5/(len(ok)-1),2))
