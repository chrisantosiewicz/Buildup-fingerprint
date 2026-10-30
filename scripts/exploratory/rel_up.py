import pandas as pd, numpy as np
from buildup.fingerprint import split_half
d=pd.read_parquet('out10/decisions_style.parquet'); p=pd.read_parquet('out10/possessions_style.parquet')
o=pd.read_csv('out10/option_values.csv',usecols=['decision_id','xpass_completion','completion','pass_dx','xthreat','V','chosen'])
mr={m:i for i,m in enumerate(sorted(d.match_id.unique()))}
def rep(name,df,col,mh=15):
    df=df[df[col].notna()]
    r=split_half(df,col,'player',mr,mh); t=split_half(df,col,'team_shortname',mr,1,within=None)
    print(f"{name:45s} r={r['split_half_r']:.3f} rel={r['full_sample_reliability']:.2f} n={r['n']} team={t['split_half_r']:.2f}")
print('--- RISK')
rep('current',d,'risk')
m=d.n_options>=2
rep('current, >=2 options',d[m],'risk')
rep('tracking risk >=2',d[m].assign(x=d.completion_m-d.completion),'x')
rep('absolute 1-xpass',d.assign(x=1-d.xpass_completion),'x')
rep('absolute 1-xpass >=2',d[m].assign(x=1-d.xpass_completion),'x')
# rank of chosen in decision by xpass
o['rk']=o.groupby('decision_id').xpass_completion.rank(pct=True,method='average',ascending=False)
o['n']=o.groupby('decision_id').xpass_completion.transform('size')
o['minx']=o.groupby('decision_id').xpass_completion.transform('min');o['maxx']=o.groupby('decision_id').xpass_completion.transform('max')
c=o[o.chosen==1][['decision_id','rk','n','minx','maxx']]
dd=d.merge(c,on='decision_id',how='left')
mm=dd.n>=2
rep('not safest option (binary) >=2',dd[mm].assign(x=(dd.xpass_completion<dd.maxx-1e-9).astype(float)),'x')
rep('riskiest option chosen >=2',dd[mm].assign(x=(dd.xpass_completion<=dd.minx+1e-9).astype(float)),'x')
rep('scaled position (max-x)/(max-min) >=2',dd[mm&(dd.maxx-dd.minx>0.05)].assign(x=(dd.maxx-dd.xpass_completion)/(dd.maxx-dd.minx)),'x')
rep('gap to safest (max-x) >=2',dd[mm].assign(x=dd.maxx-dd.xpass_completion),'x')
# situation-adjusted
dd['pb']=dd.passer_pressure_id.fillna(0).clip(0,4)
for nm,col in [('risk',dd.risk),('gap_safe',dd.maxx-dd.xpass_completion)]:
    dd['_y']=col
    typ=dd[mm].groupby(['pos_group','phase','n','pb','third_start'])['_y'].transform('mean')
    rep(f'{nm} situation-adjusted >=2',dd[mm].assign(x=dd.loc[mm,'_y']-typ),'x')
# build-up only
rep('current, build_up+create >=2',d[m&d.phase.isin(['build_up','create'])],'risk')
# composite risk z: gap_safe + forward
print('--- CREATOR')
p2=p[p.on_ball_value_created.notna()].copy()
rep('current created>=0.005',p,'created')
for th in [0.001,0.002,0.003,0.01]:
    rep(f'created>={th}',p2.assign(x=(p2.on_ball_value_created>=th).astype(float)),'x')
rep('continuous clip +-0.02',p2.assign(x=p2.on_ball_value_created.clip(-0.02,0.02)),'x')
rep('continuous clip 0..0.02',p2.assign(x=p2.on_ball_value_created.clip(0,0.02)),'x')
rep('created, carries only',p2[p2.carry==1].assign(x=(p2.on_ball_value_created>=0.005).astype(float)),'x')
rep('created, started under pressure',p2[p2.overall_pressure_start_id.fillna(0)>=3].assign(x=(p2.on_ball_value_created>=0.005).astype(float)),'x')
for col in ['delta_to_last_defensive_line_gain','n_opponents_overtaken','separation_gain','possession_epv_delta_for','forward_momentum','n_opponents_bypassed']:
    if col in p2: rep(col,p2.assign(x=pd.to_numeric(p2[col],errors='coerce').astype(float)),'x')
p2['opt_ahead_gain']=p2.n_passing_options_ahead_at_end-p2.n_passing_options_ahead_at_start
p2['opt_gain']=p2.n_passing_options_at_end-p2.n_passing_options_at_start
p2['xloss_drop']=p2.xloss_player_possession_start-p2.xloss_player_possession_end
p2['xshot_gain']=p2.xshot_player_possession_end-p2.xshot_player_possession_start
p2['press_drop']=p2.overall_pressure_start_id-p2.overall_pressure_end_id
p2['opp_ahead_drop']=p2.n_opponents_ahead_start-p2.n_opponents_ahead_end
for col in ['opt_ahead_gain','opt_gain','xloss_drop','xshot_gain','press_drop','opp_ahead_drop']:
    rep(col,p2,col)
p2.to_parquet('out10/_p2.parquet')
