import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings("ignore")
from buildup.io import load_all_events
from buildup.turnovers import possessions
t=pd.read_parquet("out10/turnovers.parquet"); ev=load_all_events("dl/opendata/data"); p=possessions(ev)
BU=["build_up","create"]
t["bu"]=t.loss_phase_type.isin(BU)
R={}
# 1. share of chances conceded that start from build-up turnovers
g=t.groupby("bu").agg(n=("turnover_id","size"),danger=("dangerous_10s","sum"),shots=("shot_15s","sum"),goals=("goal_15s","sum"))
R["by_origin"]=g.to_dict(orient="index")
# total shots in data, for share
shots_total=int((p.end_type=="shot").sum()); R["shots_total"]=shots_total
# per phase rate
R["danger_rate_by_phase"]=t.groupby("loss_phase_type").agg(n=("turnover_id","size"),danger=("dangerous_10s","mean"),shot=("shot_15s","mean"),goal=("goal_15s","sum")).round(3).to_dict(orient="index")
b=t[t.bu].copy()
# 2. how: lost_by x third
R["bu_by_how"]=b.groupby("lost_by").agg(n=("turnover_id","size"),danger=("dangerous_10s","mean"),shots=("shot_15s","sum"),goals=("goal_15s","sum")).round(3).to_dict(orient="index")
R["bu_by_third"]=b.groupby("loss_third").agg(n=("turnover_id","size"),danger=("dangerous_10s","mean"),shots=("shot_15s","sum"),goals=("goal_15s","sum")).round(3).to_dict(orient="index")
R["bu_by_pressure"]=b.groupby("loss_pressure_id").agg(n=("turnover_id","size"),danger=("dangerous_10s","mean")).round(3).to_dict(orient="index")
R["bu_by_channel"]=b.groupby("loss_channel").agg(n=("turnover_id","size"),danger=("dangerous_10s","mean")).round(3).to_dict(orient="index")
# 3. link failed passes to the decision: was it a risky choice, a bad choice, or bad execution?
d=pd.read_parquet("out10/decisions_style.parquet")
o=pd.read_csv("out10/option_values.csv",low_memory=False,usecols=["decision_id","player_name","V","completion","xthreat","chosen","pass_dx"])
bestopt=o.loc[o.groupby("decision_id").V.idxmax()][["decision_id","player_name","completion","pass_dx"]].rename(columns={"player_name":"best_name","completion":"best_completion","pass_dx":"best_dx"})
safest=o.groupby("decision_id").completion.max().rename("safest_completion")
d=d.merge(bestopt,on="decision_id").merge(safest,left_on="decision_id",right_index=True)
fp=b[b.lost_by.isin(["failed_pass","lost_at_reception"])].merge(d,left_on=["match_id","loss_event_id"],right_on=["match_id","poss_event_id"],how="left",suffixes=("","_d"))
print("linked failed passes",fp.decision_id.notna().sum(),"of",len(fp))
fp=fp[fp.decision_id.notna()].copy()
def cls(r):
    if r.picked_best and r.completion>=0.75: return "right choice, execution failed"
    if r.picked_best: return "right choice, risky by nature"
    if r.safest_completion-r.completion>=0.15: return "safer option ignored"
    return "other option better"
fp["type"]=fp.apply(cls,axis=1)
R["failed_pass_types"]=fp.groupby("type").agg(n=("turnover_id","size"),danger=("dangerous_10s","mean"),shots=("shot_15s","sum"),goals=("goal_15s","sum"),completion=("completion","mean")).round(3).to_dict(orient="index")
fp["dir"]=np.select([fp.pass_dx>2,fp.pass_dx<-2],["forward","backward"],"sideways")
R["failed_pass_direction"]=fp.groupby("dir").agg(n=("turnover_id","size"),danger=("dangerous_10s","mean"),goals=("goal_15s","sum")).round(3).to_dict(orient="index")
R["failed_pass_long"]=fp.groupby(fp.pass_len>30).agg(n=("turnover_id","size"),danger=("dangerous_10s","mean")).round(3).to_dict(orient="index")
# predicted cost vs realised: does V_cost rank which build-up losses get punished?
from sklearn.metrics import roc_auc_score
R["cost_auc_on_bu_failed_passes"]=round(roc_auc_score(fp.dangerous_10s,fp.cost),3)
# 4. teams: build-up turnovers per 100 build-up possessions, danger, goals
pb=p[p.team_in_possession_phase_type.isin(BU)].groupby("team_shortname").size().rename("bu_possessions")
tm=b.groupby("losing_team").agg(bu_turnovers=("turnover_id","size"),danger=("dangerous_10s","sum"),shots=("shot_15s","sum"),goals=("goal_15s","sum"))
tm=tm.join(pb)
matches=p.groupby("team_shortname").match_id.nunique().rename("matches"); tm=tm.join(matches)
tm["turnovers_per100"]=100*tm.bu_turnovers/tm.bu_possessions
tm["danger_per_turnover"]=tm.danger/tm.bu_turnovers
tm["danger_per_match"]=tm.danger/tm.matches
tm["goals_per_match"]=tm.goals/tm.matches
tm=tm.sort_values("danger_per_match",ascending=False)
tm.round(3).to_csv("out10/team_buildup_turnovers.csv")
print(tm.round(2).to_string())
# stability split-half for team metrics
mr={m:i for i,m in enumerate(sorted(p.match_id.unique()))}
def half(k):
    bb=b[b.match_id.map(mr)%2==k]; pp=p[(p.match_id.map(mr)%2==k)&p.team_in_possession_phase_type.isin(BU)]
    x=bb.groupby("losing_team").agg(n=("turnover_id","size"),dz=("dangerous_10s","sum")).join(pp.groupby("team_shortname").size().rename("pos"))
    return pd.DataFrame({"per100":100*x.n/x.pos,"danger_per100":100*x.dz/x.pos})
h0,h1=half(0),half(1)
R["team_split_half"]={c:round(h0[c].corr(h1[c]),2) for c in h0}
# 5. players: who loses it in build-up (raw, with counts) - per 100 build-up possessions
pp=p[p.team_in_possession_phase_type.isin(BU)].groupby(["player_name"]).agg(poss=("event_id","size"),pos=("player_position","first"),team=("team_shortname","first"))
pl=b.groupby("loss_player_name").agg(lost=("turnover_id","size"),danger=("dangerous_10s","sum"),goals=("goal_15s","sum"))
pp=pp.join(pl).fillna({"lost":0,"danger":0,"goals":0}); pp=pp[pp.poss>=60]
pp["lost_per100"]=100*pp.lost/pp.poss; pp["danger_per100"]=100*pp.danger/pp.poss
pp.sort_values("danger_per100",ascending=False).round(2).to_csv("out10/player_buildup_turnovers.csv")
def phalf(k):
    bb=b[b.match_id.map(mr)%2==k]; q=p[(p.match_id.map(mr)%2==k)&p.team_in_possession_phase_type.isin(BU)].groupby("player_name").size().rename("pos")
    x=pd.concat([q,bb.groupby("loss_player_name").size().rename("lost"),bb.groupby("loss_player_name").dangerous_10s.sum().rename("dz")],axis=1).fillna(0)
    x=x[x.pos>=30]; return pd.DataFrame({"lost":x.lost/x.pos,"dz":x.dz/x.pos})
a0,a1=phalf(0),phalf(1); j=a0.join(a1,rsuffix="_1",how="inner")
R["player_split_half"]={"lost_per100":round(j.lost.corr(j.lost_1),2),"danger_per100":round(j.dz.corr(j.dz_1),2),"players":len(j)}
json.dump(R,open("out10/bu_turnovers.json","w"),indent=1,default=str)
for k in ["by_origin","shots_total","bu_by_how","bu_by_third","bu_by_pressure","failed_pass_types","failed_pass_direction","failed_pass_long","cost_auc_on_bu_failed_passes","team_split_half","player_split_half"]: print(k, json.dumps(R[k],default=str))
