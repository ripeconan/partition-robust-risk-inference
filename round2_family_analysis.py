"""Frozen full-family sensitivity; no detector fitting or target selection."""
from pathlib import Path
from functools import lru_cache
from collections import Counter
import csv,hashlib,json,math,os,platform,resource,sys,time
import numpy as np
import scipy
from scipy.stats import beta
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/round2_family';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'artifacts/strengthening/real_comparisons.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
@lru_cache(None)
def cp(n,k,d):return (float(beta.ppf(d,k,n-k+1)) if k else 0.,float(beta.ppf(1-d,k+1,n-k)) if k<n else 1.)
def verdict(lo,hi):return 'C' if hi<=.05 else ('V' if lo>.05 else 'U')
t=time.perf_counter();source=list(csv.DictReader(SOURCE.open()))
methods=['aggregate_CP','view_margins_CP','mapped_CP']
base=[x for x in source if float(x['alpha'])==.05 and x['method'] in methods]
assert len(base)==1800
families=[('primary_vote_all_configurations',120,[x for x in base if int(x['q'])==1]),('all_configuration_vote_targets',600,base)]
rows=[];totals=[];members=[]
for family,m,cells in families:
 keys=sorted({(x['run_id'],int(x['q'])) for x in cells});assert len(keys)==m
 for run,q in keys:members.append(dict(family=family,run_id=run,q=q,alpha=.05,family_size=m,delta_cell=.05/m))
write('family_membership.csv',members)
for family,m,cells in families:
 for x in cells:
  n=int(x['n']);lo=int(x['lower_count']);hi=int(x['upper_count']);k=int(x['mapped_count']);assert lo<=k<=hi
  l=cp(n,lo,.05/m)[0];u=cp(n,hi,.05/m)[1];ml,mu=cp(n,k,.05/m);assert l<=ml+1e-12 and u>=mu-1e-12
  state=verdict(l,u);assert state=='U' or state==x['verdict'];assert state=='U' or state==verdict(ml,mu)
  rows.append(dict(family=family,family_size=m,run_id=x['run_id'],category=x['category'],model=x['model'],seed=x['seed'],q=int(x['q']),n=n,method=x['method'],lower_count=lo,upper_count=hi,mapped_count=k,delta_cell=.05/m,lower=l,upper=u,verdict=state,marginal_verdict=x['verdict']))
 for q in sorted({int(x['q']) for x in cells}):
  for method in methods:
   rr=[x for x in rows if x['family']==family and x['q']==q and x['method']==method];c=Counter(x['verdict'] for x in rr)
   totals.append(dict(family=family,family_size=m,q=q,method=method,cells=len(rr),C=c['C'],V=c['V'],U=c['U']))
write('family_membership.csv',members);write('adjusted_cells.csv',rows);write('family_summary.csv',totals)
checks=[]
for n in [1,2,4,8,12,30,60,120,300,1000]:
 for d in [.05,.01,.05/120,.05/600]:
  eps=math.sqrt(math.log(1/d)/(2*n));errors=[]
  for k in range(n+1):
   l,u=cp(n,k,d);errors.append(max(0,k/n-eps-l,u-k/n-eps,l-k/n,k/n-u))
  assert max(errors)<1e-10,(n,d,max(errors))
  checks.append(dict(n=n,delta=d,count_values=n+1,epsilon=eps,max_enclosure_excess=max(errors)))
write('cp_uniform_checks.csv',checks)
report=dict(hostname=platform.node(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,cpu_only=True,threads=os.environ.get('OMP_NUM_THREADS'),source_sha256=sha(SOURCE),script_sha256=sha(Path(__file__)),family_manifest_sha256=sha(OUT/'family_membership.csv'),adjusted_rows=len(rows),family_memberships=len(members),cp_endpoint_cases=sum(x['count_values'] for x in checks),cp_check_failures=0,elapsed_seconds=time.perf_counter()-t,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,prospective_registration=False)
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));print('FAMILY_SUMMARY');print((OUT/'family_summary.csv').read_text())
