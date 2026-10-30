import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from buildup.tracking import MatchTracking
DATA="dl/opendata/data"
o=pd.read_parquet("out4/options.parquet")
ch=o[o.chosen & o.xpass_completion.notna()][["decision_id","match_id","pass_frame","period","team_id","player_id","received","separation_end","poss_frame_start"]].copy()
LAGS=[-20,-10,-5,0,5,10]
def run(mid):
    mt=MatchTracking(DATA,mid); rows=[]
    for r in ch[ch.match_id==mid].itertuples():
        row={"decision_id":r.decision_id}
        for L in LAGS:
            s=mt.snapshot(int(r.pass_frame)+L,int(r.team_id),int(r.period))
            if s is None or r.player_id not in set(s.player_id): row[L]=np.nan; continue
            s=s.set_index("player_id"); opp=s[s.team=="opp"]
            row[L]=float(np.hypot(opp.x_rel-s.at[r.player_id,"x_rel"],opp.y_rel-s.at[r.player_id,"y_rel"]).min())
            if L==0:
                row["ball_dist_receiver"]=float(np.hypot(s.ball_x_rel.iloc[0]-s.at[r.player_id,"x_rel"],s.ball_y_rel.iloc[0]-s.at[r.player_id,"y_rel"]))
        rows.append(row)
    return pd.DataFrame(rows)
if __name__=="__main__":
    ids=sorted(ch.match_id.unique())[:8]
    with ProcessPoolExecutor(4) as ex: d=pd.concat(list(ex.map(run,ids)))
    m=ch.merge(d,on="decision_id"); y=(m.received==True).astype(int)
    print("n",len(m)); print("separation_end AUC", round(roc_auc_score(y,m.separation_end.fillna(m.separation_end.median())),3))
    for L in LAGS:
        v=m[L].notna(); print("lag",L/10,"s  nearest opp AUC",round(roc_auc_score(y[v],m.loc[v,L]),3))
    print("ball-to-receiver distance at pass frame by outcome:", m.groupby(y).ball_dist_receiver.median().round(1).to_dict())
