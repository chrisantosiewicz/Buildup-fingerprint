import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings("ignore")
from buildup.fingerprint import split_half
d=pd.read_parquet("out11/decisions_style.parquet"); p=pd.read_parquet("out11/possessions_style.parquet")
t=pd.read_parquet("out10/turnovers.parquet")
o=pd.read_csv("out10/option_values.csv",low_memory=False,usecols=["decision_id","V","completion","pass_dx","chosen"])
mr={m:i for i,m in enumerate(sorted(d.match_id.unique()))}
R={}
def sh(df,col,by,mh,within=None):
    return split_half(df[df[col].notna()],col,by,mr,mh,within=within)
# ---- A. failed back passes: pressure + better option
best=o.loc[o.groupby("decision_id").V.idxmax()][["decision_id","completion","pass_dx"]].rename(columns={"completion":"best_c","pass_dx":"best_dx"})
dd=d.merge(best,on="decision_id")
dd["dir"]=np.select([dd.pass_dx>2,dd.pass_dx<-2],["forward","back"],"sideways")
dd["failed"]=dd.poss_pass_outcome.ne("successful") if "poss_pass_outcome" in dd else np.nan
print(dd.poss_pass_outcome.value_counts().head())
tt=t[["match_id","loss_event_id","dangerous_10s","goal_15s"]]
dd=dd.merge(tt,left_on=["match_id","poss_event_id"],right_on=["match_id","loss_event_id"],how="left")
dd["lost"]=dd.dangerous_10s.notna()
dd["hp"]=dd.high_pressure
A={}
for (dr,hp),g in dd[dd.phase.isin(["build_up","create"])].groupby(["dir","hp"]):
    f=g[g.lost]
    A[f"{dr}|{'high' if hp else 'not high'} pressure"]={"passes":len(g),"lost":int(len(f)),"loss_rate":round(len(f)/len(g),3),
      "chance_after_loss":round(f.dangerous_10s.mean(),3) if len(f) else None,
      "lost_with_better_option":round((~f.picked_best.astype(bool)).mean(),3) if len(f) else None,
      "picked_best_share":round(g.picked_best.mean(),3)}
R["A_build_up_passes_by_direction_pressure"]=A
# ---- B. pressure to progress by team
hp=d[d.high_pressure]
R["B_team_high_pressure_progress"]={"split_half":sh(hp,"outcome_prog_10s","team_shortname",1),
   "all_decisions_split_half":sh(d,"outcome_prog_10s","team_shortname",1),
   "table":hp.groupby("team_shortname").agg(n=("decision_id","size"),progress=("outcome_prog_10s","mean"),back=("back","mean"),picked_best=("picked_best","mean")).round(3).sort_values("progress").to_dict(orient="index")}
R["B_player_high_pressure_progress"]=sh(hp.assign(pos_group=hp.pos_group),"outcome_prog_10s","player",8,within="pos_group")
# ---- C. dead-end passes: completed passes whose receiver starts with no option ahead
pp=p.sort_values(["match_id","frame_start"])[["match_id","event_id","frame_start","player_id","team_id","n_passing_options_ahead_at_start","n_passing_options_at_start","best_ev_start","end_type","duration"]].copy()
for c in ["player_id","team_id","n_passing_options_ahead_at_start","n_passing_options_at_start","best_ev_start","end_type","duration","match_id"]:
    pp["next_"+c]=pp[c].shift(-1)
pp=pp[pp.next_match_id==pp.match_id]
c=d[d.poss_pass_outcome.eq("successful")].merge(pp[["event_id","match_id","next_player_id","next_team_id","next_n_passing_options_ahead_at_start","next_n_passing_options_at_start","next_best_ev_start","next_end_type","next_duration"]],left_on=["match_id","poss_event_id"],right_on=["match_id","event_id"],how="left")
c=c[c.next_player_id==c.player_id].copy()
print("completed passes linked to receiver possession",len(c))
c["dead_end"]=(c.next_n_passing_options_ahead_at_start==0).astype(float)
c["dead_end_lowval"]=(c.next_best_ev_start<c.next_best_ev_start.quantile(0.25)).astype(float)
R["C_dead_end"]={"share_no_option_ahead":round(c.dead_end.mean(),3),"share_low_value":0.25,
 "progress_10s_if_dead_end":round(c[c.dead_end==1].outcome_prog_10s.mean(),3),"progress_10s_otherwise":round(c[c.dead_end==0].outcome_prog_10s.mean(),3),
 "conceded_danger_if_dead_end":round(c[c.dead_end==1].outcome_conceded_danger.mean(),3),"conceded_danger_otherwise":round(c[c.dead_end==0].outcome_conceded_danger.mean(),3),
 "receiver_loses_it_if_dead_end":round((c[c.dead_end==1].next_end_type=="possession_loss").mean(),3),"receiver_loses_it_otherwise":round((c[c.dead_end==0].next_end_type=="possession_loss").mean(),3),
 "P_predicts_dead_end_auc":None,
 "team_split_half":sh(c,"dead_end","team_shortname",1),"player_split_half_within_pos":sh(c,"dead_end","player",15,within="pos_group"),
 "team_split_half_lowval":sh(c,"dead_end_lowval","team_shortname",1),"player_split_half_lowval":sh(c,"dead_end_lowval","player",15,within="pos_group")}
from sklearn.metrics import roc_auc_score
R["C_dead_end"]["P_predicts_dead_end_auc"]=round(1-roc_auc_score(c.dead_end,c.P),3)
# dead end by phase and by passer position, and when a better option existed
R["C_dead_end_by_phase"]=c.groupby("phase").dead_end.agg(["mean","size"]).round(3).to_dict(orient="index")
R["C_dead_end_by_pos"]=c.groupby("pos_group").dead_end.agg(["mean","size"]).round(3).to_dict(orient="index")
R["C_dead_end_when_not_best_vs_best"]=c.groupby("picked_best").dead_end.mean().round(3).to_dict()
R["C_team_table"]=c.groupby("team_shortname").agg(n=("decision_id","size"),dead_end=("dead_end","mean")).round(3).sort_values("dead_end").to_dict(orient="index")
c[["match_id","decision_id","team_shortname","player","pos_group","phase","dead_end","dead_end_lowval","outcome_prog_10s","P"]].to_parquet("out11/dead_ends.parquet")
json.dump(R,open("out11/stories.json","w"),indent=1,default=str)
print(json.dumps(R,indent=1,default=str)[:6000])
