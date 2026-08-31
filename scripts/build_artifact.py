from __future__ import annotations
import json, hashlib, itertools, math, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path('data/soil_measurement_paper')
RES=ROOT/'results'; FIG=ROOT/'figures'; TAB=ROOT/'tables'; LATEX=ROOT/'latex'; DATA=ROOT/'data'
for d in [RES,FIG,TAB,LATEX]: d.mkdir(parents=True,exist_ok=True)
summary=json.loads((RES/'summary.json').read_text())
subset=pd.read_csv(RES/'subset_results.csv')
oracle=pd.read_csv(RES/'oracle_by_k.csv')
greedy=pd.read_csv(RES/'greedy_equal_cost.csv')
random_order=pd.read_csv(RES/'random_order.csv')
voi=pd.read_csv(RES/'singleton_voi_bootstrap.csv')
menu=pd.read_csv(RES/'lab_menu_results.csv')
costgrid=pd.read_csv(RES/'cost_grid_results.csv')
sens=pd.read_csv(RES/'ridge_sensitivity.csv')

# Exact derived manuscript quantities
baseline=float(summary['baseline_loss']); ph=float(subset.loc[subset.subset.eq('pH'),'loss_mean'].iloc[0]);
phom=float(subset.loc[subset.subset.eq('pH+OM'),'loss_mean'].iloc[0]); full=float(summary['full_loss'])
q={
    'baseline_loss':baseline,
    'ph_loss':ph,
    'ph_improvement_pct':100*(baseline-ph)/baseline,
    'phom_loss':phom,
    'phom_vs_ph_pct':100*(ph-phom)/ph,
    'full_loss':full,
    'full_vs_baseline_improvement_pct':100*(baseline-full)/baseline,
    'full_vs_ph_degradation_pct':100*(full-ph)/ph,
    'greedy_oracle_pct':100*float(summary['greedy_exact_oracle_fraction']),
    'greedy_mean_regret':float(summary['greedy_mean_regret']),
    'greedy_max_regret':float(summary['greedy_max_regret']),
    'cost_budget_cases':int(summary['cost_grid_budget_cases']),
    'pH_voi':float(voi.loc[voi.measurement.eq('pH'),'mean_voi'].iloc[0]),
    'pH_ci_low':float(voi.loc[voi.measurement.eq('pH'),'ci_low'].iloc[0]),
    'pH_ci_high':float(voi.loc[voi.measurement.eq('pH'),'ci_high'].iloc[0]),
    'pH_positive_frac':float(voi.loc[voi.measurement.eq('pH'),'positive_env_fraction'].iloc[0]),
}
(RES/'paper_quantities.json').write_text(json.dumps(q,indent=2))

# Figure 1: decision-loss frontier
fig, ax=plt.subplots(figsize=(6.4,4.0))
ax.plot(oracle['k'],oracle['loss_mean'],marker='o',label='Exact subset oracle')
ax.plot(random_order['k'],random_order['loss_mean'],marker='s',linestyle='--',label='Mean random ordering')
ax.fill_between(random_order['k'], random_order['loss_mean']-random_order['loss_sd'], random_order['loss_mean']+random_order['loss_sd'], alpha=.15)
for _,r in oracle.iterrows():
    ax.annotate(str(r['subset']), (r['k'],r['loss_mean']), xytext=(4,5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Number of acquired soil measurements')
ax.set_ylabel('LOEO normalized NPK action loss')
ax.set_xticks(range(5)); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(FIG/'figure1_frontier.png',dpi=300); fig.savefig(FIG/'figure1_frontier.pdf'); plt.close(fig)

# Figure 2: singleton VOI intervals
voi2=voi.set_index('measurement').loc[['pH','OM','P','K']].reset_index()
y=np.arange(len(voi2)); means=voi2.mean_voi.to_numpy(); lo=means-voi2.ci_low.to_numpy(); hi=voi2.ci_high.to_numpy()-means
fig, ax=plt.subplots(figsize=(6.4,3.6))
ax.errorbar(means,y,xerr=np.vstack([lo,hi]),fmt='o',capsize=4)
ax.axvline(0,linewidth=1)
ax.set_yticks(y,voi2.measurement); ax.invert_yaxis()
ax.set_xlabel('Value of information = reduction in normalized action loss')
ax.set_ylabel('Candidate measurement'); ax.grid(axis='x',alpha=.2)
fig.tight_layout(); fig.savefig(FIG/'figure2_singleton_voi.png',dpi=300); fig.savefig(FIG/'figure2_singleton_voi.pdf'); plt.close(fig)

# Figure 3: operational cost menu (supplementary)
fig, ax=plt.subplots(figsize=(6.4,3.6))
ax.plot(menu.cost_usd,menu.loss_mean,marker='o')
for _,r in menu.iterrows():
    ax.annotate(r.subset,(r.cost_usd,r.loss_mean),xytext=(3,4),textcoords='offset points',fontsize=7)
ax.set_xlabel('Illustrative laboratory bundle price (2026 USD)')
ax.set_ylabel('LOEO normalized NPK action loss'); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(FIG/'figureS1_lab_cost_menu.png',dpi=300); fig.savefig(FIG/'figureS1_lab_cost_menu.pdf'); plt.close(fig)

# Figure 4: greedy regret ECDF
vals=np.sort(costgrid.regret.to_numpy()); ecdf=np.arange(1,len(vals)+1)/len(vals)
fig, ax=plt.subplots(figsize=(6.4,3.6))
ax.plot(vals,ecdf)
ax.axvline(summary['greedy_mean_regret'],linestyle='--',linewidth=1,label='Mean regret')
ax.set_xlabel('Greedy loss minus exact budget-oracle loss')
ax.set_ylabel('Empirical cumulative fraction'); ax.grid(alpha=.2); ax.legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig(FIG/'figureS2_greedy_regret.png',dpi=300); fig.savefig(FIG/'figureS2_greedy_regret.pdf'); plt.close(fig)

# Table 1 CSV and LaTeX
main_rows=[]
for _,r in oracle.iterrows():
    main_rows.append({'Measurements acquired':int(r.k),'Best subset':r.subset,'Normalized action loss':r.loss_mean,
                      'Change vs. no-soil (%)':100*(baseline-r.loss_mean)/baseline})
main=pd.DataFrame(main_rows)
main.to_csv(TAB/'table1_main_results.csv',index=False)

# Bootstrap table
voi.to_csv(TAB/'tableS1_singleton_voi.csv',index=False)
sens.to_csv(TAB/'tableS2_ridge_sensitivity.csv',index=False)
menu.to_csv(TAB/'tableS3_lab_menu.csv',index=False)

# machine-generated LaTeX table snippets
lines=[r'\\begin{tabular}{rllr}',r'\\toprule',r'$k$ & Best subset & Loss & Change vs. baseline \\\\',r'\\midrule']
for _,r in main.iterrows():
    lines.append(f"{int(r['Measurements acquired'])} & {r['Best subset'].replace('+',r' $+$ ')} & {r['Normalized action loss']:.3f} & {r['Change vs. no-soil (%)']:+.2f}\\% \\\\")
lines += [r'\\bottomrule',r'\\end{tabular}']
(TAB/'table1_main_results.tex').write_text('\n'.join(lines))

# Data checksum and provenance
raw=DATA/'ortiz_tomato_covs.csv'; sha=hashlib.sha256(raw.read_bytes()).hexdigest()
prov=f'''# Data provenance\n\nPrimary executable data: `ortiz_tomato_covs.csv`.\n\nSource documentation: agridat `ortiz.tomato.covs`, based on Ortiz, Crossa, Vargas, and Izquierdo (2007), Euphytica 153:119–134.\nRaw public mirror used to reconstruct the exact file: https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/refs/heads/master/csv/agridat/ortiz.tomato.covs.csv\n\nSHA256: `{sha}`\nRows: {len(pd.read_csv(raw))}\n\nThe fertilizer variables ExN, ExP, and ExK are recorded extra nutrient amounts at each environment. They are used as observable historical reference actions, not causal agronomic optima.\n'''
(DATA/'PROVENANCE.md').write_text(prov)

print(json.dumps(q,indent=2))
print('sha256',sha)
