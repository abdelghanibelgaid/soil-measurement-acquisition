from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

ROOT=Path('data/soil_measurement_paper')
DATA=ROOT/'data'/'ortiz_tomato_covs.csv'
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(DATA)
BASE=['Lat','Long','Irr','Trim','Driv']
SOIL=['pH','OM','P','K']
TARGETS=['ExN','ExP','ExK']
ALPHA=10.0
SCALE=(df[TARGETS].quantile(.75)-df[TARGETS].quantile(.25)).replace(0,1.0)
scale_vec=SCALE.to_numpy(float)

def key_for(bits):
    bits=set(bits)
    s=[x for x in SOIL if x in bits]
    return 'none' if not s else '+'.join(s)

def cv_subset(subset,alpha=ALPHA):
    features=BASE+list(subset)
    X=df[features].to_numpy(float); Y=df[TARGETS].to_numpy(float)
    preds=np.zeros_like(Y,dtype=float)
    for tr,te in LeaveOneOut().split(X):
        m=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=alpha))])
        m.fit(X[tr],Y[tr]); preds[te]=m.predict(X[te])
    per_env=np.mean(np.abs(Y-preds)/scale_vec,axis=1)
    per_target=np.mean(np.abs(Y-preds)/scale_vec,axis=0)
    return preds,per_env,per_target

records=[]; lossstore={}; predstore={}
for r in range(5):
    for subset in itertools.combinations(SOIL,r):
        key=key_for(subset); pred,le,pt=cv_subset(subset)
        predstore[key]=pred; lossstore[key]=le
        records.append({'subset':key,'k':r,'loss_mean':le.mean(),'loss_median':np.median(le),
                        'loss_se':le.std(ddof=1)/np.sqrt(len(le)),
                        'nmae_N':pt[0],'nmae_P':pt[1],'nmae_K':pt[2]})
res=pd.DataFrame(records).sort_values(['k','loss_mean']).reset_index(drop=True)
res.to_csv(OUT/'subset_results.csv',index=False)
loss_map=dict(zip(res.subset,res.loss_mean))

oracle=res.loc[res.groupby('k')['loss_mean'].idxmin()].sort_values('k').copy()
oracle.to_csv(OUT/'oracle_by_k.csv',index=False)

# Sequential greedy under equal unit cost: recompute marginal value after every addition.
chosen=[]; greedy=[]
for k in range(5):
    key=key_for(chosen); cur=loss_map[key]
    greedy.append({'k':k,'subset':key,'loss_mean':cur})
    if k==4: break
    cands=[]
    for j in SOIL:
        if j in chosen: continue
        nk=key_for(chosen+[j]); nl=loss_map[nk]
        cands.append(((cur-nl),j,nl,nk))
    # deterministic tie follows SOIL ordering
    cands=sorted(cands,key=lambda z:(-z[0],SOIL.index(z[1])))
    chosen.append(cands[0][1])
greedy=pd.DataFrame(greedy); greedy.to_csv(OUT/'greedy_equal_cost.csv',index=False)
greedy_seq=[]
for k in range(1,5):
    bits=greedy.loc[greedy.k.eq(k),'subset'].iloc[0].split('+')
    for b in bits:
        if b not in greedy_seq: greedy_seq.append(b)

# Static singleton-VOI ranking.
base=loss_map['none']; singleton=[]
for j in SOIL: singleton.append((base-loss_map[j],j))
static_order=[j for _,j in sorted(singleton,key=lambda x:(-x[0],SOIL.index(x[1])))]
static=[]
for k in range(5):
    key=key_for(static_order[:k]); static.append({'k':k,'subset':key,'loss_mean':loss_map[key]})
static=pd.DataFrame(static); static.to_csv(OUT/'static_marginal.csv',index=False)

# Expected trajectory under random feature order across all 24 permutations.
rr=[]
for k in range(5):
    vals=[]
    for perm in itertools.permutations(SOIL): vals.append(loss_map[key_for(perm[:k])])
    rr.append({'k':k,'loss_mean':np.mean(vals),'loss_sd':np.std(vals,ddof=1)})
pd.DataFrame(rr).to_csv(OUT/'random_order.csv',index=False)

# Current public laboratory bundle menu: UNH April 2026.
# pH only $10; OM $5; field soil test pH+P+K $20; combinations treated additively when listed as add-ons.
menu=[('none',0.0),('OM',5.0),('pH',10.0),('pH+OM',15.0),('pH+P+K',20.0),('pH+OM+P+K',25.0)]
menu_df=pd.DataFrame([{'subset':key_for(k.split('+')) if k!='none' else 'none','cost_usd':c,
                      'loss_mean':loss_map[key_for(k.split('+')) if k!='none' else 'none']} for k,c in menu])
menu_df=menu_df.sort_values('cost_usd'); menu_df.to_csv(OUT/'lab_menu_results.csv',index=False)
budget=[]
for B in range(26):
    f=menu_df[menu_df.cost_usd<=B]
    if len(f):
        z=f.sort_values(['loss_mean','cost_usd']).iloc[0]
        budget.append({'budget_usd':B,'subset':z.subset,'cost_usd':z.cost_usd,'loss_mean':z.loss_mean})
pd.DataFrame(budget).to_csv(OUT/'lab_budget_oracle.csv',index=False)

# Relative-cost robustness: all 4^4 cost vectors and every attainable subset-cost budget.
cost_grid=[0.5,1.0,2.0,4.0]; subsets=[]
for key,l in loss_map.items():
    bits=[] if key=='none' else key.split('+'); subsets.append((key,bits,l))
cr=[]
for costs in itertools.product(cost_grid,repeat=4):
    cd=dict(zip(SOIL,costs))
    infos=[(key,sum(cd[b] for b in bits),l) for key,bits,l in subsets]
    for B in sorted(set(v[1] for v in infos)):
        feasible=[v for v in infos if v[1]<=B+1e-12]
        o=min(feasible,key=lambda v:(v[2],v[1]))
        chosen=[]
        while True:
            cur=loss_map[key_for(chosen)]; spent=sum(cd[j] for j in chosen)
            cand=[]
            for j in SOIL:
                if j in chosen or spent+cd[j]>B+1e-12: continue
                nl=loss_map[key_for(chosen+[j])]
                cand.append(((cur-nl)/cd[j],j,nl))
            if not cand: break
            cand=sorted(cand,key=lambda z:(-z[0],SOIL.index(z[1])))
            chosen.append(cand[0][1])
        gkey=key_for(chosen); gloss=loss_map[gkey]
        cr.append((costs[0],costs[1],costs[2],costs[3],B,o[0],o[2],gkey,gloss,gloss-o[2]))
costres=pd.DataFrame(cr,columns=['cost_pH','cost_OM','cost_P','cost_K','budget','oracle_subset','oracle_loss','greedy_subset','greedy_loss','regret'])
costres.to_csv(OUT/'cost_grid_results.csv',index=False)

# Bootstrap paired intervals on held-out environments; model fits are fixed by LOEO protocol.
rng=np.random.default_rng(20260831)
voi=[]
for j in SOIL:
    d=lossstore['none']-lossstore[j]
    b=np.empty(10000)
    for i in range(len(b)):
        idx=rng.integers(0,len(d),len(d)); b[i]=d[idx].mean()
    lo,hi=np.quantile(b,[.025,.975])
    voi.append({'measurement':j,'mean_voi':d.mean(),'ci_low':lo,'ci_high':hi,'positive_env_fraction':np.mean(d>0)})
voi_df=pd.DataFrame(voi).sort_values('mean_voi',ascending=False); voi_df.to_csv(OUT/'singleton_voi_bootstrap.csv',index=False)

def traj_env(order):
    return np.column_stack([lossstore[key_for(order[:k])] for k in range(5)])
G=traj_env(greedy_seq); S=traj_env(static_order)
# np.trapezoid available in current NumPy
GA=np.trapezoid(G,dx=1,axis=1)/4; SA=np.trapezoid(S,dx=1,axis=1)/4
b=np.empty(10000)
for i in range(len(b)):
    idx=rng.integers(0,len(df),len(df)); b[i]=(SA[idx]-GA[idx]).mean()
ci=np.quantile(b,[.025,.975])

# Ridge-penalty sensitivity.
sens=[]
for alpha in [0.1,1.0,10.0,100.0]:
    lm={}
    for r in range(5):
        for subset in itertools.combinations(SOIL,r):
            _,le,_=cv_subset(subset,alpha=alpha); lm[key_for(subset)]=le.mean()
    b0=lm['none']; ss=sorted([(b0-lm[j],j) for j in SOIL],key=lambda x:(-x[0],SOIL.index(x[1])))
    sens.append({'alpha':alpha,'first_measurement':ss[0][1],'order':' > '.join([j for _,j in ss]),
                 'baseline_loss':b0,'full_loss':lm['pH+OM+P+K']})
pd.DataFrame(sens).to_csv(OUT/'ridge_sensitivity.csv',index=False)

summary={
'n_environments':18,'base_features':BASE,'candidate_measurements':SOIL,'targets':TARGETS,
'ridge_alpha':ALPHA,'target_iqr':{k:float(v) for k,v in SCALE.items()},
'baseline_loss':float(base),'full_loss':float(loss_map['pH+OM+P+K']),
'relative_full_reduction_pct':float(100*(base-loss_map['pH+OM+P+K'])/base),
'oracle_by_k':oracle[['k','subset','loss_mean']].to_dict('records'),
'greedy_equal_cost':greedy.to_dict('records'),'greedy_sequence':greedy_seq,
'static_order':static_order,'singleton_voi':voi_df.to_dict('records'),
'aulc_greedy':float(GA.mean()),'aulc_static':float(SA.mean()),
'aulc_static_minus_greedy':float((SA-GA).mean()),'aulc_diff_ci95':[float(ci[0]),float(ci[1])],
'cost_grid_scenarios':256,'cost_grid_budget_cases':int(len(costres)),
'greedy_exact_oracle_fraction':float(np.mean(costres.regret<=1e-12)),
'greedy_mean_regret':float(costres.regret.mean()),'greedy_max_regret':float(costres.regret.max()),
'unh_menu':menu_df.to_dict('records'),
'ridge_sensitivity':sens,
}
with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
