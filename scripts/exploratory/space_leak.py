import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from buildup.options import OPTION_NUMERIC
from buildup.progression import DEFAULT_GBM, _gbm, _X
from buildup.space import SPACE_NUMERIC
o=pd.read_parquet("out4/options.parquet"); sp=pd.read_parquet("out8/option_space.parquet")
o=o.merge(sp.drop(columns="match_id"),on=["decision_id","player_id"],how="left")
ch=o[o.chosen & o.xpass_completion.notna()].copy(); ch["rec"]=(ch.received==True).astype(int)
print(ch.groupby("rec")[SPACE_NUMERIC].mean().round(2).T)
def cv(num):
    g=ch.match_id.to_numpy(); p=np.zeros(len(ch)); y=ch.rec.to_numpy()
    for a,b in GroupKFold(5).split(ch,groups=g):
        m=_gbm("clf",DEFAULT_GBM).fit(_X(ch.iloc[a],num),y[a]); p[b]=m.predict_proba(_X(ch.iloc[b],num))[:,1]
    return round(roc_auc_score(y,p),3)
base=OPTION_NUMERIC
for f in SPACE_NUMERIC: print("+",f,cv(base+[f]))
# timing check: what does frame_end of possession vs pass event look like? pass_frame - poss_frame_start
print("pass frame minus option frame_end:", (ch.pass_frame-ch.frame_end).describe().round(1).to_dict())
