import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
d=pd.read_parquet("out7/dec_style.parquet"); P=pd.read_parquet("out7/poss_style.parquet")
A=pd.read_csv("dl/opendata/data/aggregates/aus1league_passingaggregates_20242025.csv")
print("agg rows",len(A),"players",A.player_id.nunique(),"minutes median",A.minutes.median(), "matches median", A.performance_count.median())
A["cmp_above_exp"]=A.pass_pct_completed/ (100 if A.pass_pct_completed.max()>1.5 else 1) - A.pass_avgxpass_attempted/(100 if A.pass_avgxpass_attempted.max()>1.5 else 1)
A["linebreak_share"]=A.pass_count_linebreak_attempted_p30tip/A.pass_count_attempted_p30tip
A["dangerous_share"]=A.pass_count_dangerous_attempted_p30tip/A.pass_count_attempted_p30tip
A["long_share"]=A.pass_count_longrange_attempted_p30tip/A.pass_count_attempted_p30tip
A["quick_share"]=A.pass_count_quickpass_attempted_p30tip/A.pass_count_attempted_p30tip
A["difficult_share"]=A.pass_count_difficultpass_attempted_p30tip/A.pass_count_attempted_p30tip
A=A.dropna(subset=["pass_count_attempted_total"])
num=[c for c in A.select_dtypes("number").columns if c not in ("player_id","performance_count")]
A=A.groupby("player_id").apply(lambda g: pd.Series({**{c: np.average(g[c].fillna(g[c].mean()) if g[c].notna().any() else g[c], weights=g.performance_count) for c in num}, "games": g.performance_count.sum()}))
A=A[A.games>=8]
print("season players with 8+ games",len(A), "median games", A.games.median())
ours=d.groupby("pid").agg(n=("risk","size"),pos=("pos_group",lambda s:s.mode().iloc[0]),risk=("risk","mean"),forward=("forward","mean"),
     long=("long","mean"),back=("back","mean"),cost=("cost_taken","mean"),dq=("dq_adj","mean"),provided=("provided","mean"),prog=("prog_out","mean"))
pq=P.groupby("pid").agg(np_=("quick","size"),quick=("quick","mean"),carry=("carry","mean"),created=("created","mean"),dur=("dur","mean"))
m=ours.join(pq).join(A,how="inner"); m=m[m.n>=40]
print("matched players",len(m), "of", (ours.n>=40).sum())
pairs=[("risk","pass_avgxpass_attempted"),("risk","difficult_share"),("forward","linebreak_share"),("long","long_share"),
 ("long","pass_avgdistance"),("quick","quick_share"),("provided","dangerous_share"),("provided","linebreak_share"),
 ("provided","pass_count_shotwithin10s_p30tip"),("dq","cmp_above_exp"),("dq","pass_pct_completed"),("dq","dangerous_share"),("prog","pass_count_shotwithin10s_p30tip"),("cost","pass_avgxpass_attempted")]
for a,b in pairs:
    mm=m[[a,b,'pos']].dropna(); r=spearmanr(mm[a],mm[b]).correlation
    w=mm[[a,b]].sub(mm.groupby("pos")[[a,b]].transform("mean"))
    rw=spearmanr(w[a],w[b]).correlation
    print(f"{a:9s} vs {b:32s} r={r:+.2f} within-pos={rw:+.2f}")
