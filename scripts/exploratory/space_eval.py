import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, roc_auc_score, log_loss
from buildup.options import OPTION_NUMERIC
from buildup.progression import DEFAULT_GBM, _gbm, _X
from buildup.space import SPACE_NUMERIC
o=pd.read_parquet("out4/options.parquet"); t=pd.read_parquet("out4/targets.parquet"); sp=pd.read_parquet("out8/option_space_m3.parquet")
o=o.merge(sp.drop(columns="match_id"),on=["decision_id","player_id"],how="left")
print("corr nearest opp vs separation_end", round(o.trk_nearest_opp.corr(o.separation_end),3), "receiver detected", round(o.trk_receiver_detected.mean(),3))
tr=o[o.chosen].merge(t[["decision_id","next_best_ev","prog_10s","kept_10s"]],on="decision_id")
def cv(df,num,target,task):
    g=df.match_id.to_numpy(); p=np.zeros(len(df)); y=df[target].astype(float if task=="reg" else int).to_numpy()
    for a,b in GroupKFold(5).split(df,groups=g):
        m=_gbm(task,DEFAULT_GBM).fit(_X(df.iloc[a],num),y[a]); X=_X(df.iloc[b],num)
        p[b]=m.predict(X) if task=="reg" else m.predict_proba(X)[:,1]
    return (round(r2_score(y,p),3) if task=="reg" else (round(roc_auc_score(y,p),3), round(log_loss(y,np.clip(p,1e-4,1-1e-4)),4)))
for name,num in [("base",OPTION_NUMERIC),("+tracking",OPTION_NUMERIC+SPACE_NUMERIC)]:
    print(name,"progression R2",cv(tr,num,"next_best_ev","reg"),"prog_10s",cv(tr,num,"prog_10s","clf"),"kept_10s",cv(tr,num,"kept_10s","clf"))
# completion: chosen passes, received?
ch=o[o.chosen & o.xpass_completion.notna()].copy(); ch["rec"]=(ch.received==True)
y=ch.rec.astype(int)
print("completion AUC SkillCorner xPass alone", round(roc_auc_score(y,ch.xpass_completion),3), "logloss", round(log_loss(y,ch.xpass_completion.clip(1e-4,1-1e-4)),4))
for name,num in [("xPass+event",OPTION_NUMERIC),("xPass+event+tracking",OPTION_NUMERIC+SPACE_NUMERIC)]:
    print(name,"completion",cv(ch,num,"rec","clf"))
