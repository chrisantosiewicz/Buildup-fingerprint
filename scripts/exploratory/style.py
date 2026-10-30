import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings("ignore")
from buildup.value import POSITION_GROUPS
O="out7"; AGG="dl/opendata/data/aggregates/"
o=pd.read_csv(f"{O}/option_values.csv",low_memory=False)
g=o.groupby("decision_id")
for c in ["xpass_completion","pass_dx","xthreat","P","V_cost"]:
    o[c+"_m"]=g[c].transform("mean")
d=pd.read_csv(f"{O}/decisions.csv",low_memory=False)
d=d.merge(o[o.chosen==True][["decision_id"]+[c+"_m" for c in ["xpass_completion","pass_dx","xthreat","P","V_cost"]]],on="decision_id")
d=d[~d.phase.isin(["set_play","disruption"])].copy()
d["player"]=d.player_in_possession_name; d["pid"]=d.passer_id
d["risk"]=d.xpass_m if False else d.xpass_completion_m-d.xpass_completion   # + = riskier than average option
d["forward"]=d.pass_dx-d.pass_dx_m
d["ambition"]=d.xthreat-d.xthreat_m
d["lookahead"]=d.P-d.P_m
d["cost_taken"]=d.V_cost-d.V_cost_m
d["back"]=(d.pass_dx<-2).astype(float)
d["long"]=(d.pass_len>30).astype(float)
d["dq_adj"]=d.decision_quality-d.groupby(["pos_group","phase","n_options"]).decision_quality.transform("mean")
d["provided"]=d.V_progression
d["prog_out"]=d.outcome_prog_10s.astype(float)
hp=d[d.passer_pressure_id>=4].copy()
pp=pd.read_parquet("out5/poss_creator.parquet") if False else None
from buildup.io import load_all_events
ev=load_all_events("dl/opendata/data"); P=ev[ev.event_type=="player_possession"].copy()
P=P[~P.team_in_possession_phase_type.isin(["set_play","disruption"])].copy()
ovc=pd.read_parquet("out4/ovc.parquet")
P=P.merge(ovc,left_on=["match_id","event_id"],right_on=["match_id","poss_event_id"],how="left")
P["player"]=P.player_name; P["pid"]=P.player_id; P["pos_group"]=P.player_position.map(POSITION_GROUPS).fillna("Other")
P["carry"]=(P.carry==True).astype(float); P["carry_dx"]=np.where(P.carry==1,(P.x_end-P.x_start).clip(lower=0),0)
P["created"]=(P.on_ball_value_created>=0.005).astype(float)
P["dur"]=P.duration; P["quick"]=(P.duration<=1.0).astype(float)
P["lost"]=(P.end_type=="possession_loss").astype(float)
P["pressed"]=P.overall_pressure_start_id>=4
Pp=P[P.pressed].copy(); Pp["kept_pressed"]=1-Pp.lost
P["press_escape"]=((P.overall_pressure_start_id>=3)&(P.overall_pressure_end_id<=P.overall_pressure_start_id-2)).astype(float)
mr={m:i for i,m in enumerate(sorted(P.match_id.unique()))}
def stab(df,v,nmin=15):
    df=df.assign(half=df.match_id.map(mr)%2); df["w"]=df[v]-df.groupby("pos_group")[v].transform("mean")
    out={}
    for tag,vv in [("raw",v),("within",'w')]:
        a=df.groupby(["player","half"])[vv].agg(["mean","size"]).unstack(); a=a[(a[("size",0)]>=nmin)&(a[("size",1)]>=nmin)]
        out[tag]=round(float(a[("mean",0)].corr(a[("mean",1)])),3)
    out["n"]=len(a); r=max(out["within"],0); out["full_rel"]=round(2*r/(1+r),2)
    return out
rows={}
for name,df,v,n in [
  ("STYLE risk taken (vs avg option)",d,"risk",15),("STYLE forwardness",d,"forward",15),("STYLE ambition (xThreat)",d,"ambition",15),
  ("STYLE lookahead preference",d,"lookahead",15),("STYLE turnover cost accepted",d,"cost_taken",15),("STYLE back-pass share",d,"back",15),
  ("STYLE long-pass share",d,"long",15),("STYLE back-pass share | high press",hp,"back",8),
  ("STYLE carry share",P,"carry",15),("STYLE carry metres fwd",P,"carry_dx",15),("STYLE tempo: quick release share",P,"quick",15),
  ("STYLE possession duration",P,"dur",15),
  ("QUALITY decision quality (adj)",d,"dq_adj",15),("QUALITY provider",d,"provided",15),("QUALITY creator share",P,"created",15),
  ("QUALITY ball security | pressed",Pp,"kept_pressed",8),("QUALITY press escape",P,"press_escape",15),("OUTCOME progress after pass",d,"prog_out",15)]:
    rows[name]=stab(df,v,n)
for k,v in rows.items(): print(f"{k:40s} {v}")
d.to_parquet(f"{O}/dec_style.parquet"); P.to_parquet(f"{O}/poss_style.parquet")
json.dump(rows,open(f"{O}/style_stability.json","w"),indent=1)
