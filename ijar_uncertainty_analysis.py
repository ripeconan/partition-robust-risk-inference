"""IJAR revision: real-score voting and nested metadata recovery."""
from pathlib import Path
import sys, json, csv, hashlib, time, platform
from collections import defaultdict, Counter
from functools import lru_cache
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from partition_robust import feasible_item_counts
from cp_state import cp_ci
from reproduce_from_scores import load_records, validate_bundle, arrays_for_run, threshold_and_scores, item_ids
OUT=ROOT/'artifacts'/'ijar_uncertainty'

@lru_cache(None)
def bounds(k,n): return cp_ci(int(k),int(n),.05)
def envelope(lo,hi,n):
    l=bounds(lo,n)[0]; u=bounds(hi,n)[1]
    return l,u,('certified' if u<=.05 else 'violated' if l>.05 else 'unresolved')
def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geometry():
    cases=0; triples=0
    # Independent constructive recurrence, no use of the analytic endpoints.
    for r in range(1,7):
      for q in range(1,r+1):
        dp=[set() for _ in range(17)];dp[0].add((0,0,0))
        for f in range(1,17):
          for size in range(1,min(r,f)+1):
            for n,a,k in dp[f-size]:
              for alarms in range(size+1):
                dp[f].add((n+1,a+alarms,k+int(alarms>=q)))
          observed=defaultdict(set)
          for n,a,k in dp[f]: observed[(n,a)].add(k)
          triples+=len(dp[f])
          for n in range((f+r-1)//r,f+1):
            for a in range(f+1):
              expected={k for k,_ in feasible_item_counts(f,a,r,n,q)}
              assert expected==observed[(n,a)],(f,a,n,r,q,expected,observed[(n,a)])
              cases+=1
    return dict(configurations=cases,attainable_triples=triples,mismatches=0,max_frames=16,max_cap=6)

def main():
    t=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    bundle=ROOT/'artifacts'/'score_bundle';records=load_records(bundle/'run_manifest.jsonl')
    validate_bundle(bundle,records)
    chosen=[r for r in records if r['summary']['dataset']=='ad2' and r['summary']['threshold_rule']=='normal_image_quantile' and int(r['summary']['seed']) in range(5)]
    assert len(chosen)==120,len(chosen)
    vote_rows=[]; recovery=[]
    with np.load(bundle/'run_scores.npz',allow_pickle=False) as archive:
      for record in chosen:
        s=record['summary'];arr=arrays_for_run(archive,record['run_id']);tau,scores=threshold_and_scores(s,arr,'s')
        normal=arr['labels_s']==0; ids=item_ids(arr['paths_s'][normal]);alarms=(scores[normal]>tau).astype(int)
        groups={item:alarms[ids==item] for item in sorted(set(ids))}
        assert groups and all(1<=len(v)<=5 for v in groups.values()), (record['semantic_key'], Counter(len(v) for v in groups.values()))
        names=list(groups);counts=np.array([sum(groups[i]) for i in names]);sizes=np.array([len(groups[i]) for i in names]);n=len(names);a=int(sum(counts));f=int(sum(sizes))
        base=dict(run_id=record['run_id'],category=s['unit'],model=s['model'],seed=int(s['seed']),N=n,F=f,A=a)
        for q in range(1,6):
          k=int(sum(counts>=q));pairs=feasible_item_counts(f,a,5,n,q);lo=min(p[0] for p in pairs);hi=max(p[0] for p in pairs)
          assert lo<=k<=hi
          l,u,state=envelope(lo,hi,n);ml,mu,mstate=envelope(k,k,n)
          assert state=='unresolved' or state==mstate
          vote_rows.append(dict(**base,q=q,K=k,K_min=lo,K_max=hi,count_width=hi-lo,lower=l,upper=u,state=state,mapped_lower=ml,mapped_upper=mu,mapped_state=mstate))
          for mask_seed in range(20):
            order=sorted(range(n),key=lambda i:hashlib.sha256(f'{mask_seed}|{s["unit"]}|{names[i]}'.encode()).digest())
            prev_lo,prev_hi=lo,hi
            for fraction in (0,.25,.5,.75,1):
              known=int(fraction*n);selected=order[:known];remaining=order[known:]
              k_known=int(sum(counts[selected]>=q));nr=n-known;ar=int(sum(counts[remaining]))
              if nr:
                residual=feasible_item_counts(int(sum(sizes[remaining])),ar,5,nr,q);rlo=k_known+min(z[0] for z in residual);rhi=k_known+max(z[0] for z in residual)
              else:rlo=rhi=k
              assert prev_lo<=rlo<=k<=rhi<=prev_hi
              prev_lo,prev_hi=rlo,rhi
              rl,ru,rs=envelope(rlo,rhi,n)
              assert rs=='unresolved' or rs==mstate
              recovery.append(dict(**base,q=q,mask_seed=mask_seed,fraction=fraction,known_items=known,K_min=rlo,K_max=rhi,count_width=rhi-rlo,lower=rl,upper=ru,bound_width=ru-rl,state=rs,mapped_state=mstate))
    vote_summary=[]
    for q in range(1,6):
      rows=[r for r in vote_rows if r['q']==q];c=Counter(r['state'] for r in rows);m=Counter(r['mapped_state'] for r in rows)
      vote_summary.append(dict(q=q,cells=len(rows),mean_count_width=float(np.mean([r['count_width'] for r in rows])),robust_C=c['certified'],robust_V=c['violated'],robust_U=c['unresolved'],mapped_C=m['certified'],mapped_V=m['violated'],mapped_U=m['unresolved'],contradictions=0))
    assert vote_summary[0]['robust_V']==64 and vote_summary[0]['mapped_V']==84
    recovery_summary=[]
    for q in range(1,6):
      for fraction in (0,.25,.5,.75,1):
        rows=[r for r in recovery if r['q']==q and r['fraction']==fraction];c=Counter(r['state'] for r in rows)
        recovery_summary.append(dict(q=q,fraction=fraction,rows=len(rows),mean_known_items=float(np.mean([r['known_items'] for r in rows])),mean_count_width=float(np.mean([r['count_width'] for r in rows])),mean_bound_width=float(np.mean([r['bound_width'] for r in rows])),mean_violated_cells_per_mask=c['violated']/20,mean_certified_cells_per_mask=c['certified']/20,mean_unresolved_cells_per_mask=c['unresolved']/20))
    geom=geometry()
    for name,rows in [('vote_cells.csv',vote_rows),('vote_summary.csv',vote_summary),('recovery_cells.csv',recovery),('recovery_summary.csv',recovery_summary)]:write_csv(name,rows)
    info=dict(date='2026-10-14',python=sys.version,platform=platform.platform(),hostname=platform.node(),numpy=np.__version__,scipy=scipy.__version__,elapsed_seconds=time.perf_counter()-t,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),score_bundle_manifest=json.loads((bundle/'score_bundle_manifest.json').read_text()),geometry=geom,vote_summary=vote_summary,recovery_summary=recovery_summary,mask_trials=20,physical_new_items=0)
    (OUT/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({k:info[k] for k in ['hostname','elapsed_seconds','geometry','vote_summary']},indent=2))
if __name__=='__main__':main()
