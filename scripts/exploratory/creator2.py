import sys, pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from buildup.io import load_all_events
ev=load_all_events(sys.argv[1]); ovc=pd.read_parquet(sys.argv[2])
pp=ev[ev.event_type=="player_possession"].copy()
print(pp.team_in_possession_phase_type.value_counts().to_dict())
pp=pp.merge(ovc,left_on=["match_id","event_id"],right_on=["match_id","poss_event_id"],how="left")
mr={m:i for i,m in enumerate(sorted(pp.match_id.unique()))}; pp["half"]=pp.match_id.map(mr)%2; pp["player"]=pp.player_name
pp["carry"]=pp.carry==True
pp["ovc_up"]=(pp.on_ball_value_created>0.002).astype(float)
pp["ovc_up5"]=(pp.on_ball_value_created>0.005).astype(float)
pp["escaped"]=((pp.overall_pressure_end_id<=pp.overall_pressure_start_id-2)&(pp.end_type!="possession_loss")).astype(float)
pp["carry_dx"]=np.where(pp.carry,(pp.x_end-pp.x_start).clip(lower=0),0)
pp["xshot_gain"]=pp.xshot_player_possession_end-pp.xshot_player_possession_start if "xshot_player_possession_end" in pp else np.nan
pp["ahead_up"]=(pp.n_passing_options_ahead_at_end>pp.n_passing_options_ahead_at_start).astype(float)
def r(df,v,nmin=15):
    a=df.groupby(["player","half"])[v].agg(["mean","size"]).unstack(); a=a[(a[("size",0)]>=nmin)&(a[("size",1)]>=nmin)]
    return round(a[("mean",0)].corr(a[("mean",1)]),3), len(a)
for ph,df in [("buildup+create",pp[pp.team_in_possession_phase_type.isin(["build_up","create"])]),
              ("open play",pp[~pp.team_in_possession_phase_type.isin(["set_play","disruption"])])]:
    pr=df[df.overall_pressure_start_id>=3]
    print("==",ph,len(df))
    for name,d,v,n in [("ovc up share",df,"ovc_up",15),("ovc up5 share",df,"ovc_up5",15),("carry share",df,"carry",15),
        ("carry fwd m",df,"carry_dx",15),("ahead up",df,"ahead_up",15),("escape|pressed",pr,"escaped",10),
        ("ovc up|pressed",pr,"ovc_up",10),("xshot gain",df,"xshot_gain",15)]:
        print(f"  {name:16s} {r(d,v,n)}")
