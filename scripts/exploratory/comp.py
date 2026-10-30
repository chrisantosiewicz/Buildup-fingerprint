import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
d=pd.read_parquet("out7/dec_style.parquet")
mr={m:i for i,m in enumerate(sorted(d.match_id.unique()))}; d["half"]=d.match_id.map(mr)%2
# team-adjust: remove team-season mean too (players on dominant teams get easier passes)
for v in ["provided","dq_adj"]:
    d[v+"_p"]=d[v]-d.groupby("pos_group")[v].transform("mean")
    d[v+"_pt"]=d[v+"_p"]-d.groupby("team_shortname")[v+"_p"].transform("mean")
def halves(v):
    a=d.groupby(["player","half"])[v].agg(["mean","size"]).unstack(); return a[(a[("size",0)]>=15)&(a[("size",1)]>=15)]
for v in ["provided_p","provided_pt","dq_adj_p","dq_adj_pt"]:
    a=halves(v); print(v, round(a[("mean",0)].corr(a[("mean",1)]),3), len(a))
a1,a2=halves("provided_p"),halves("dq_adj_p")
z=lambda s:(s-s.mean())/s.std()
c0=z(a1[("mean",0)])+z(a2[("mean",0)]); c1=z(a1[("mean",1)])+z(a2[("mean",1)])
print("composite provider+dq", round(c0.corr(c1),3))
# within-player correlation of provider and dq (do they measure the same thing?)
print("provider vs dq player corr", round(a1[("mean",0)].add(a1[("mean",1)]).corr(a2[("mean",0)].add(a2[("mean",1)])),3))
# team-level style stability
for v in ["risk","forward","long","back","cost_taken","provided","dq_adj","picked_best"]:
    t=d.groupby(["team_shortname","half"])[v].mean().unstack(); print("team",v, round(t[0].corr(t[1]),2))
