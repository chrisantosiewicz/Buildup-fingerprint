import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, roc_auc_score
from buildup.options import OPTION_NUMERIC, OPTION_CATEGORICAL
from buildup.progression import DEFAULT_GBM, _gbm, _X
o=pd.read_parquet("out4/options.parquet"); t=pd.read_parquet("out4/targets.parquet")
o["score_diff"]=o.team_score-o.opponent_team_score
EXTRA=["speed_avg","distance_covered","trajectory_angle","separation_gain","delta_to_last_defensive_line_gain",
       "interplayer_angle","n_opponents_ahead_player_in_possession_pass_moment","passing_option_score","score_diff","minute_start","separation_start"]
tr=o[o.chosen].merge(t[["decision_id","next_best_ev","next_exp_ev","prog_10s","kept_10s"]],on="decision_id")
g=tr.match_id.to_numpy()
def cv(num,target,task):
    p=np.zeros(len(tr)); y=tr[target].astype(float if task=="reg" else int).to_numpy()
    for a,b in GroupKFold(5).split(tr,groups=g):
        m=_gbm(task,DEFAULT_GBM).fit(_X(tr.iloc[a],num),y[a]); X=_X(tr.iloc[b],num)
        p[b]=m.predict(X) if task=="reg" else m.predict_proba(X)[:,1]
    return round(r2_score(y,p),3) if task=="reg" else round(roc_auc_score(y,p),3)
for name,num in [("base",OPTION_NUMERIC),("+extra",OPTION_NUMERIC+EXTRA),("+extra -option_score",OPTION_NUMERIC+[e for e in EXTRA if e!="passing_option_score"])]:
    print(name, "next_best_ev R2", cv(num,"next_best_ev","reg"), "next_exp_ev R2", cv(num,"next_exp_ev","reg"), "prog_10s AUC", cv(num,"prog_10s","clf"), "kept_10s AUC", cv(num,"kept_10s","clf"))
