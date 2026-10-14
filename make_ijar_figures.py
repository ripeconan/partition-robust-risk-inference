from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent;M=ROOT/'artifacts/figures';M.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(8,5.4));ax.set(xlim=(0,10),ylim=(0,6.5));ax.axis('off')
boxes=[(.25,4.05,4.1,1.9,'1. OBSERVATIONS','Locked scores and alarms\nFrame count and view cap\nItem count, when known'),(5.4,4.05,4.1,1.9,'2. REPRESENTATION','Attainable item counts\nunder every compatible\nframe-to-item grouping'),(.25,1.15,4.1,1.9,'3. INFERENCE','One-sided binomial bounds\nLower / upper envelopes\nover the attainable set'),(5.4,1.15,4.1,1.9,'4. DECISION','Certified\nUnresolved\nViolated')]
for x,y,w,h,head,body in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',facecolor='#edf3f7',edgecolor='#35596f',linewidth=1.1))
 ax.text(x+w/2,y+h-.30,head,ha='center',va='center',fontsize=13,fontweight='bold',color='#24465c')
 ax.text(x+w/2,y+.70,body,ha='center',va='center',fontsize=12,linespacing=1.5)
for y in (5,2.1):ax.annotate('',xy=(5.2,y),xytext=(4.55,y),arrowprops=dict(arrowstyle='->',color='#24465c',lw=1.5))
ax.annotate('',xy=(2.3,3.2),xytext=(7.4,3.9),arrowprops=dict(arrowstyle='->',color='#24465c',lw=1.5))
ax.text(4.9,.62,'Recovered IDs contract grouping uncertainty.',ha='center',fontsize=12,color='#35596f')
ax.text(4.9,.15,'New independent items address sampling uncertainty.',ha='center',fontsize=12,color='#555555')
fig.tight_layout(pad=.4);fig.savefig(M/'fig0_ijar_uncertainty.pdf',bbox_inches='tight');fig.savefig(M/'fig0_ijar_uncertainty.png',dpi=180,bbox_inches='tight');plt.close(fig)
p=ROOT/'artifacts/ijar_uncertainty/recovery_summary.csv'
if p.exists():
 rows=list(csv.DictReader(p.open()));fig,axs=plt.subplots(1,2,figsize=(8,3.8))
 colors=['#1f6178','#d18632','#536c48','#97637c','#555c91']
 for q,color in zip(range(1,6),colors):
  s=[r for r in rows if int(r['q'])==q];xs=[float(r['fraction'])*100 for r in s]
  axs[0].plot(xs,[float(r['mean_count_width']) for r in s],marker='o',label=f'q = {q}',color=color)
  axs[1].plot(xs,[float(r['mean_violated_cells_per_mask']) for r in s],marker='o',label=f'q = {q}',color=color)
 for a in axs:a.set_xlabel('Recovered item IDs (%)');a.set_xticks([0,25,50,75,100]);a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
 axs[0].set_ylabel('Mean attainable-count width');axs[1].set_ylabel('Mean violated cells (out of 120)');axs[0].legend(frameon=False,ncol=2,fontsize=8)
 axs[0].set_title('(a) Grouping imprecision');axs[1].set_title('(b) Decision resolution')
 fig.tight_layout();fig.savefig(M/'fig9_metadata_recovery.pdf',bbox_inches='tight');fig.savefig(M/'fig9_metadata_recovery.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('IJAR figures generated')
