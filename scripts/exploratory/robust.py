import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from buildup.value import add_option_value, decisions
o=pd.read_csv("out10/option_values.csv",low_memory=False)
def player_scores(g,k):
    d=decisions(add_option_value(o,gamma=g,kappa=k,completion="completion"))
    d=d[~d.phase.isin(["set_play","disruption"])]
    d["dq_adj"]=d.decision_quality-d.groupby(["pos_group","phase","n_options"]).decision_quality.transform("mean")
    p=d.groupby("player_in_possession_name").agg(n=("V","size"),dq=("dq_adj","mean"),prov=("V_progression","mean"),best=("picked_best","mean"))
    return p[p.n>=40], d.picked_best.mean()
base,_=player_scores(1,1)
for g,k in [(0.5,1),(2,1),(1,0.5),(1,2)]:
    p,pb=player_scores(g,k); j=base.join(p,rsuffix="_x")
    print(f"gamma {g} kappa {k}: picked best {pb:.3f}  rank corr DQ {j.dq.corr(j.dq_x,method='spearman'):.2f}  provider {j.prov.corr(j.prov_x,method='spearman'):.2f}")
