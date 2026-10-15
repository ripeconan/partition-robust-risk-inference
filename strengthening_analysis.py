"""Predeclared IJAR strengthening: exact baselines, retained view margins and power.

Formal execution: CPU only on inner, two threads, one persistent SSH connection.
The frozen v1.0.0 inputs and its numerical results are never overwritten.
"""
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
import argparse, csv, hashlib, itertools, json, math, os, platform, resource, sys, time
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from reproduce_from_scores import load_records, validate_bundle, arrays_for_run, threshold_and_scores, item_ids
OUT = ROOT / 'artifacts' / 'strengthening'


def sharp(f, a, n, r, q):
    if q > r:
        return 0, 0
    if n == 0:
        assert f == a == 0
        return 0, 0
    assert n <= f <= r*n and 0 <= a <= f
    if q == 1:
        return (max(1, (a+r-1)//r, n-(f-a)) if a else 0), min(a, n)
    return max(0, (a-(q-1)*n+r-q)//(r-q+1)), min(n, a//q, (f-n)//(q-1))


@lru_cache(None)
def cp_table(n, delta=.05):
    k = np.arange(n+1)
    lo = np.zeros(n+1); hi = np.ones(n+1)
    lo[1:] = beta.ppf(delta, k[1:], n-k[1:]+1)
    hi[:-1] = beta.ppf(1-delta, k[:-1]+1, n-k[:-1])
    return lo, hi


def envelope(lo, hi, n, method='cp'):
    if method == 'hoeffding':
        eps = math.sqrt(math.log(20)/(2*n))
        return max(0, lo/n-eps), min(1, hi/n+eps)
    l, u = cp_table(n)
    return float(l[lo]), float(u[hi])


def verdict(l, u, alpha):
    return 'C' if u <= alpha else 'V' if l > alpha else 'U'


def solve_pair(matrix, rhs, cost, integer=True):
    vals=[]
    for sign in [1, -1]:
        if integer:
            sol=milp(sign*cost, integrality=np.ones(len(cost)),
                     bounds=Bounds(0, np.inf),
                     constraints=LinearConstraint(matrix, rhs, rhs),
                     options={'time_limit': 20, 'mip_rel_gap': 0})
        else:
            sol=linprog(sign*cost, A_eq=matrix, b_eq=rhs, bounds=(0,None),
                        method='highs', options={'time_limit':20})
        if not sol.success:
            raise RuntimeError(f'Optimization failed: {sol.message}')
        vals.append(float(cost @ sol.x))
    if integer:
        assert all(abs(v-round(v)) < 1e-6 for v in vals), vals
        return tuple(int(round(v)) for v in vals)
    # Integer rounding of an outer LP relaxation remains conservative.
    return max(0, math.ceil(vals[0]-1e-7)), math.floor(vals[1]+1e-7)


def generic_count_bounds(f,a,n,r,q,integer=True):
    types=[(s,b) for s in range(1,r+1) for b in range(s+1)]
    matrix=np.array([[1]*len(types),[s for s,b in types],[b for s,b in types]],float)
    cost=np.array([b>=q for s,b in types],float)
    return solve_pair(matrix,np.array([n,f,a],float),cost,integer)


def margin_bounds(margins,n,q):
    r=len(margins)
    if q>r:return 0,0
    patterns=np.array(list(itertools.product([0,1],repeat=r)),int)
    matrix=np.vstack([np.ones(len(patterns)),patterns.T])
    return solve_pair(matrix,np.array([n,*margins]),(patterns.sum(1)>=q).astype(float))


def write_csv(name, rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def finite_geometry(pilot=False):
    """Enumerate real binary matrices, independently of count-table optimization."""
    cases=0
    for r in range(1,4):
      for n in range(1,4):
        observed=defaultdict(set)
        for bits in itertools.product([0,1],repeat=r*n):
            a=np.array(bits).reshape(n,r); margins=tuple(a.sum(0))
            for q in range(1,r+1):
                observed[(margins,q)].add(int((a.sum(1)>=q).sum()))
        for (margins,q),ks in observed.items():
            actual=margin_bounds(margins,n,q)
            assert actual==(min(ks),max(ks)),(n,r,margins,q,actual,ks)
            cases+=1
        if pilot:break
    return {'binary_matrix_margin_cases':cases,'mismatches':0}


def theorem_threshold_checks():
    rows=[]
    for n in [1,2,3,10,30,60,120,300]:
      for r in range(1,6):
       for q in range(1,r+1):
        a=np.arange(r*n+1)
        lo=np.maximum(0,np.ceil((a-(q-1)*n)/(r-q+1)).astype(int))
        hi=np.minimum(n,a//q)
        l,u=cp_table(n)
        for alpha in [.01,.05,.1,.2,.3]:
            cs=np.flatnonzero(u[hi]<=alpha)
            vs=np.flatnonzero(l[lo]>alpha)
            c=int(cs[-1]) if len(cs) else -1
            v=int(vs[0]) if len(vs) else r*n+1
            size_c=float(binom.cdf(math.floor(c/q),n,alpha))
            size_v=float(binom.sf(math.ceil((v-(q-1)*n)/(r-q+1))-1,n,alpha))
            assert size_c <= .05+1e-10 and size_v <= .05+1e-10
            next_c=float(binom.cdf(math.floor((c+1)/q),n,alpha)) if c<r*n else None
            # Largest valid rejection region uses inclusive size calibration.
            # Strict L>alpha can differ at exact binomial-tail ties.
            tails=binom.sf(np.arange(n+1)-1,n,alpha)
            valid=np.flatnonzero(tails<=.05+1e-14)
            critical=int(valid[0]) if len(valid) else n+1
            optimal_v=(q-1)*n+(r-q+1)*(critical-1)+1
            next_v=float(binom.sf(math.ceil((optimal_v-1-(q-1)*n)/(r-q+1))-1,n,alpha)) if optimal_v>0 else None
            assert next_c is None or next_c>.05-1e-10,(n,r,q,alpha,c,next_c)
            assert next_v is None or next_v>.05+1e-12,(n,r,q,alpha,optimal_v,next_v)
            rows.append(dict(n=n,r=r,q=q,alpha=alpha,cert_alarm_max=c,violate_alarm_min=v,
                             cert_worst_size=size_c,violate_worst_size=size_v,
                             optimal_violate_alarm_min=optimal_v,strict_boundary_difference=v-optimal_v,
                             enlarged_cert_size=next_c,enlarged_violate_size=next_v))
    write_csv('threshold_optimality_checks.csv',rows)
    return {'threshold_cases':len(rows),'invalid_enlargements':0,
            'exact_boundary_tie_cases':sum(x['strict_boundary_difference']!=0 for x in rows)}


def real_scores(pilot=False):
    bundle=ROOT/'artifacts/score_bundle'; records=load_records(bundle/'run_manifest.jsonl')
    validate_bundle(bundle,records)
    chosen=[x for x in records if x['summary']['dataset']=='ad2' and
            x['summary']['threshold_rule']=='normal_image_quantile' and int(x['summary']['seed']) in range(5)]
    assert len(chosen)==120
    if pilot: chosen=chosen[:3]
    rows=[]; timing=[]; checks=0; unbalanced=[]
    with np.load(bundle/'run_scores.npz',allow_pickle=False) as archive:
      for idx,record in enumerate(chosen):
        s=record['summary']; arr=arrays_for_run(archive,record['run_id'])
        tau,scores=threshold_and_scores(s,arr,'s'); normal=arr['labels_s']==0
        paths=arr['paths_s'][normal].astype(str); ids=item_ids(paths); z=(scores[normal]>tau).astype(int)
        names=sorted(set(ids)); n=len(names); groups=[z[ids==name] for name in names]
        f=len(z); a=int(z.sum()); lights=[Path(p).stem[len(str(i))+1:] for p,i in zip(paths,ids)]
        view_names=sorted(set(lights)); r_actual=len(view_names)
        complete=all(sorted([l for l,j in zip(lights,ids) if j==name])==view_names for name in names)
        if not complete:unbalanced.append(record['run_id'])
        margins=tuple(int(sum(v for v,l in zip(z,lights) if l==view)) for view in view_names)
        for q in range(1,6):
          k=sum(int(g.sum()>=q) for g in groups)
          t=time.perf_counter(); base=sharp(f,a,n,5,q); closed_time=time.perf_counter()-t
          t=time.perf_counter(); generic=generic_count_bounds(f,a,n,5,q); milp_time=time.perf_counter()-t
          lp=generic_count_bounds(f,a,n,5,q,False)
          assert generic==base,(record['run_id'],q,base,generic)
          assert lp[0]<=base[0]<=k<=base[1]<=lp[1]
          alternatives={'aggregate_CP':base,'generic_MILP_CP':generic,'LP_CP':lp,
                        'aggregate_Hoeffding':base,'mapped_CP':(k,k)}
          if complete:
              design=sharp(f,a,n,r_actual,q); view=margin_bounds(margins,n,q)
              assert base[0]<=design[0]<=view[0]<=k<=view[1]<=design[1]<=base[1]
              alternatives['design_cap_CP']=design;alternatives['view_margins_CP']=view
          timing.append(dict(run_id=record['run_id'],q=q,n=n,f=f,a=a,
                             closed_form_seconds=closed_time,milp_seconds=milp_time))
          for method,(kl,ku) in alternatives.items():
            l,u=envelope(kl,ku,n,'hoeffding' if method=='aggregate_Hoeffding' else 'cp')
            for alpha in [.05,.10,.20,.30]:
              state=verdict(l,u,alpha);ml,mu=envelope(k,k,n)
              if state!='U' and method!='aggregate_Hoeffding':
                  assert state==verdict(ml,mu,alpha)
              rows.append(dict(run_id=record['run_id'],category=s['unit'],model=s['model'],seed=s['seed'],
                               n=n,f=f,a=a,views=r_actual,q=q,alpha=alpha,method=method,
                               lower_count=kl,upper_count=ku,count_width=ku-kl,lower=l,upper=u,
                               envelope_width=u-l,verdict=state,mapped_count=k))
          checks+=1
        if idx%20==0:print('REAL_PROGRESS',idx+1,len(chosen),flush=True)
    write_csv('real_comparisons.csv',rows);write_csv('solver_timings.csv',timing)
    grouped=defaultdict(list)
    for row in rows:grouped[(row['q'],row['alpha'],row['method'])].append(row)
    summary=[]
    for (q,alpha,method),rs in sorted(grouped.items()):
        summary.append(dict(q=q,alpha=alpha,method=method,cells=len(rs),
                            C=sum(x['verdict']=='C' for x in rs),V=sum(x['verdict']=='V' for x in rs),
                            U=sum(x['verdict']=='U' for x in rs),
                            mean_count_width=float(np.mean([x['count_width'] for x in rs])),
                            mean_envelope_width=float(np.mean([x['envelope_width'] for x in rs]))))
    write_csv('real_summary.csv',summary)
    return dict(configurations=len(chosen),vote_checks=checks,rows=len(rows),
                incomplete_view_design_runs=unbalanced,solver_mismatches=0,
                median_closed_seconds=float(np.median([x['closed_form_seconds'] for x in timing])),
                median_milp_seconds=float(np.median([x['milp_seconds'] for x in timing])))


def power_grid(pilot=False):
    """Exact polynomial convolution: no Monte Carlo error or synthetic item inflation."""
    rows=[]; nvals=[30,60,120,300,1000] if not pilot else [30,60]
    for n in nvals:
      for r in [3,5]:
       for q in range(1,r+1):
        a=np.arange(r*n+1)
        lo=np.maximum(0,np.ceil((a-(q-1)*n)/(r-q+1)).astype(int));hi=np.minimum(n,a//q)
        l,u=cp_table(n); hl=np.maximum(0,lo/n-math.sqrt(math.log(20)/(2*n)))
        hu=np.minimum(1,hi/n+math.sqrt(math.log(20)/(2*n)))
        for p in [.01,.03,.05,.06,.10,.20]:
         mapped=binom.pmf(np.arange(n+1),n,p)
         for mechanism in ['sparse','dense','spread']:
          weights=np.zeros(r+1)
          if mechanism=='sparse':weights[0]=1-p;weights[q]=p
          elif mechanism=='dense':weights[q-1]=1-p;weights[r]=p
          else:weights[:q]=(1-p)/q;weights[q:]=p/(r-q+1)
          dist=np.array([1.])
          for _ in range(n):dist=np.convolve(dist,weights)
          assert abs(dist.sum()-1)<1e-10
          for method,lower,upper,prob in [('aggregate_CP',l[lo],u[hi],dist),
                                          ('aggregate_Hoeffding',hl,hu,dist),
                                          ('mapped_CP',l,u,mapped)]:
            for alpha in [.05,.10]:
              cr=float(prob[upper<=alpha].sum());vr=float(prob[lower>alpha].sum())
              assert not (p>alpha and cr>.05+1e-10),(n,r,q,p,mechanism,method,cr)
              assert not (p<=alpha and vr>.05+1e-10),(n,r,q,p,mechanism,method,vr)
              rows.append(dict(n=n,r=r,q=q,p=p,alpha=alpha,mechanism=mechanism,method=method,
                               certified=cr,violated=vr,unresolved=1-cr-vr,
                               upper_coverage=float(prob[upper>=p].sum()),
                               lower_coverage=float(prob[lower<=p].sum())))
      print('POWER_N_COMPLETE',n,flush=True)
    write_csv('exact_power_grid.csv',rows)
    return dict(rows=len(rows),mechanisms=3,n_values=nvals,method='exact convolution, not Monte Carlo')


def recovery_frontier():
    rows=[];r=5;q=1;alpha=.05
    for n in [60,300,1000,3000]:
      l,u=cp_table(n);cs=np.flatnonzero(u<=alpha);vs=np.flatnonzero(l>alpha)
      c=int(cs[-1]) if len(cs) else -1;v=int(vs[0]) if len(vs) else n+1
      for p,mechanism in [(.10,'sparse'),(.03,'dense')]:
       mu=p*(q if mechanism=='sparse' else r)
       lower=max(0,(mu-q+1)/(r-q+1));upper=min(1,mu/q)
       critical=(alpha-lower)/(p-lower) if p>alpha else (upper-alpha)/(upper-p)
       for fraction in [0,.25,.5,.75,.9,1]:
        m=int(fraction*n);nr=n-m;kr=np.arange(m+1);pr=binom.pmf(kr,m,p)
        # Both chosen mechanisms have two-point alarm counts, so exact
        # residual total probabilities follow from a binomial, not simulation.
        ar=np.arange(r*nr+1);dist=np.zeros(len(ar));ks=np.arange(nr+1)
        dist[(q if mechanism=='sparse' else r)*ks]=binom.pmf(ks,nr,p)
        cdf=np.cumsum(dist);lo=np.maximum(0,np.ceil((ar-(q-1)*nr)/(r-q+1)).astype(int))
        hi=np.minimum(nr,ar//q)
        cuts=np.searchsorted(hi,c-kr,side='right')-1
        pc=np.where(cuts>=0,cdf[np.maximum(cuts,0)],0)
        starts=np.searchsorted(lo,v-kr,side='left')
        before=starts-1;pv=np.where(before>=0,1-cdf[np.maximum(before,0)],1)
        cr=float(pr@pc);vr=float(pr@pv)
        rows.append(dict(n=n,p=p,mechanism=mechanism,recovery_fraction=fraction,
                         recovered_items=m,critical_fraction=critical,
                         limit_lower=fraction*p+(1-fraction)*lower,
                         limit_upper=fraction*p+(1-fraction)*upper,
                         certified=cr,violated=vr,unresolved=max(0,1-cr-vr)))
    write_csv('recovery_frontier.csv',rows)
    return dict(rows=len(rows),n_values=[60,300,1000,3000],method='exact binomial summation')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
    report=dict(pilot=args.pilot,hostname=platform.node(),python=sys.version,scipy=scipy.__version__,
                numpy=np.__version__,cpu_only=True,threads=os.environ.get('OMP_NUM_THREADS'),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report['thresholds']=theorem_threshold_checks();print('THRESHOLDS_OK',flush=True)
    report['geometry']=finite_geometry(args.pilot);print('GEOMETRY_OK',flush=True)
    report['real']=real_scores(args.pilot)
    report['power']=power_grid(args.pilot)
    report['recovery_frontier']=recovery_frontier()
    report['elapsed_seconds']=time.perf_counter()-t
    report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
