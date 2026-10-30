import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from buildup.options import OPTION_NUMERIC
from buildup.progression import DEFAULT_GBM, _gbm, _X
from buildup.space import SPACE_NUMERIC
from buildup.value import add_option_value, decisions
o=pd.read_csv("out7/option_values.csv",low_memory=False)
sp=pd.read_parquet("out8/option_space_m3.parquet").drop(columns="match_id")
o=o.merge(sp,on=["decision_id","player_id"],how="left")
NUM=OPTION_NUMERIC+SPACE_NUMERIC
tr=o[o.chosen==True].copy(); y=(tr.received==True).astype(int).to_numpy()
pred=np.zeros(len(o)); oof_ch=np.zeros(len(tr)); mids=o.match_id.to_numpy()
for a,b in GroupKFold(5).split(o,groups=mids):
    test=np.unique(mids[b]); t=tr[~tr.match_id.isin(test)]
    m=_gbm("clf",DEFAULT_GBM).fit(_X(t,NUM),(t.received==True).astype(int))
    pred[b]=m.predict_proba(_X(o.iloc[b],NUM))[:,1]
o["xpass_plus"]=pred
ch=o[o.chosen==True]; yy=(ch.received==True).astype(int)
for c in ["xpass_completion","xpass_plus"]:
    print(c,"AUC",round(roc_auc_score(yy,ch[c]),3),"logloss",round(log_loss(yy,ch[c].clip(1e-4,1-1e-4)),4),"Brier",round(brier_score_loss(yy,ch[c]),4))
# calibration check of xpass_plus on unchosen? can't observe. compare distributions chosen vs not
print("mean xPass SC chosen/unchosen", o.groupby("chosen").xpass_completion.mean().round(3).to_dict(), "plus", o.groupby("chosen").xpass_plus.mean().round(3).to_dict())
dold=pd.read_csv("out7/decisions.csv",low_memory=False,usecols=["decision_id","outcome_prog_10s","outcome_conceded_danger"])
res={}
for name,col in [("skillcorner_xpass","xpass_completion"),("xpass_plus","xpass_plus")]:
    x=o.copy(); x["xpass_completion"]=x[col]
    d=decisions(add_option_value(x,gamma=1.0)).merge(dold,on="decision_id")
    res[name]={"prog_auc":round(roc_auc_score(d.outcome_prog_10s,d.V),3),"danger_auc":round(roc_auc_score(d.outcome_conceded_danger,-d.V),3),
               "picked_best":round(d.picked_best.mean(),3)}
    d.to_csv(f"out8/decisions_{name}.csv",index=False)
    if name=="xpass_plus": x.to_csv("out8/option_values_xpplus.csv",index=False)
print(json.dumps(res,indent=1))
