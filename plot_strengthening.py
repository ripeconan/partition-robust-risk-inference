"""Render already computed results; no numerical experiments are run here."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent;D=ROOT/'artifacts/strengthening';O=ROOT/'artifacts/figures'
O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.labelcolor':'#263444','text.color':'#263444'})
colors=['#64748b','#087f8c','#d78039','#25344a']
rs=list(csv.DictReader((D/'real_summary.csv').open()))
methods=[('aggregate_CP','Aggregate counts'),('view_margins_CP','View margins'),('mapped_CP','Full mapping')]
fig,axs=plt.subplots(1,2,figsize=(9,3.65),layout='constrained')
for k,(method,label) in enumerate(methods):
 vals=[next(float(x['mean_count_width']) for x in rs if int(x['q'])==q and float(x['alpha'])==.05 and x['method']==method) for q in range(1,6)]
 axs[0].plot(range(1,6),vals,'o-',color=colors[k],label=label,lw=1.8,ms=4)
 vals=[next(int(x['V']) for x in rs if int(x['q'])==1 and float(x['alpha'])==alpha and x['method']==method) for alpha in [.05,.1,.2,.3]]
 axs[1].plot([5,10,20,30],vals,'o-',color=colors[k],label=label,lw=1.8,ms=4)
axs[0].set(xlabel='Voting threshold q',ylabel='Mean count-set width',xticks=range(1,6),title='(a) Count ambiguity')
axs[1].set(xlabel='Risk budget (%)',ylabel='Resolved violations (out of 120)',xticks=[5,10,20,30],title='(b) Any-frame decisions')
for ax in axs:ax.grid(alpha=.16);ax.set_axisbelow(True)
axs[1].legend(frameon=False,fontsize=9,loc='upper right');fig.savefig(O/'fig10_information_gain.pdf');fig.savefig(O/'fig10_information_gain.png',dpi=180);plt.close(fig)

rs=list(csv.DictReader((D/'recovery_frontier.csv').open()))
fig,axs=plt.subplots(1,2,figsize=(9,3.65),layout='constrained')
for ax,mech,metric,title,frontier in [(axs[0],'sparse','violated','(a) Sparse alarms, p = 0.10',.375),(axs[1],'dense','certified','(b) Dense alarms, p = 0.03',5/6)]:
 for k,n in enumerate([60,300,1000,3000]):
  rr=sorted([x for x in rs if x['mechanism']==mech and int(x['n'])==n],key=lambda x:float(x['recovery_fraction']))
  ax.plot([float(x['recovery_fraction'])*100 for x in rr],[float(x[metric]) for x in rr],marker='o',ms=3.5,lw=1.6,color=colors[k],label=f'N = {n}')
 ax.axvline(frontier*100,color='#8e5050',ls='--',lw=1.2)
 ax.set(xlabel='Recovered item groups (%)',ylabel='Decision probability',title=title,ylim=(-.03,1.03),xticks=[0,25,50,75,100])
 ax.set_title(title,fontsize=10)
 ax.grid(alpha=.16);ax.set_axisbelow(True)
axs[0].legend(frameon=False,fontsize=9,loc='upper left')
fig.savefig(O/'fig11_recovery_frontier.pdf');fig.savefig(O/'fig11_recovery_frontier.png',dpi=180);plt.close(fig)
print('Two result figures rendered.')
