import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
o=pd.read_csv("out6/option_values.csv",low_memory=False,usecols=["decision_id","match_id","chosen","V","phase","player_in_possession_name","player_in_possession_position","team_shortname"])
from buildup.value import POSITION_GROUPS
g=o.groupby("decision_id").V
o["V_min"]=g.transform("min"); o["V_max"]=g.transform("max"); o["n"]=g.transform("size")
o["rank_pct"]=g.rank(pct=False,ascending=True)
d=o[o.chosen==True].copy(); d=d[d.n>=2]
d["quality"]=(d.V-d.V_min)/(d.V_max-d.V_min).replace(0,np.nan)
d["rank_score"]=(d.rank_pct-1)/(d.n-1)
d["gap"]=d.V-d.V_max
d["pos"]=d.player_in_possession_position.map(POSITION_GROUPS).fillna("Other")
d["player"]=d.player_in_possession_name
mr={m:i for i,m in enumerate(sorted(d.match_id.unique()))}; d["half"]=d.match_id.map(mr)%2
for ph,df in [("buildup",d[d.phase.isin(["build_up","create"])]),("all",d)]:
    for v in ["gap","quality","rank_score"]:
        df=df.copy(); df[v+"_w"]=df[v]-df.groupby("pos")[v].transform("mean")
        out=[]
        for vv in [v,v+"_w"]:
            a=df.groupby(["player","half"])[vv].agg(["mean","size"]).unstack(); a=a[(a[("size",0)]>=15)&(a[("size",1)]>=15)]
            out.append(round(a[("mean",0)].corr(a[("mean",1)]),3))
        print(ph,v,"raw/within",out,len(a))
print("--- all phases, demeaned by position x phase")
for v in ["quality","rank_score"]:
    df=d.copy(); df["w"]=df[v]-df.groupby(["pos","phase"])[v].transform("mean")
    df["w2"]=df[v]-df.groupby(["pos","phase","n"])[v].transform("mean")
    for vv in ["w","w2"]:
        a=df.groupby(["player","half"])[vv].agg(["mean","size"]).unstack()
        for nm in [15,30]:
            b=a[(a[("size",0)]>=nm)&(a[("size",1)]>=nm)]
            print(v,vv,nm,round(b[("mean",0)].corr(b[("mean",1)]),3),len(b))
