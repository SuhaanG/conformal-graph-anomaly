"""Non-vacuous finite-design checks; not a replacement for the proof."""
import itertools,json,math
import numpy as np
from allocation_acquisition import D,pvalues,bh,methods

def main():
    rows=[];total_assignments=0;false_events=0
    for n1,n2,nt1,nt2,na1,na2 in [(12,16,1,1,3,2),(9,21,1,2,4,4)]:
        n=n1+n2;N=n+na1+na2
        strata=np.r_[np.zeros(n1),np.ones(n2),np.zeros(na1),np.ones(na2)].astype(int)
        rank=np.arange(N);normal=np.arange(n)
        for alpha in (.1,.3,.7):
            sums={k:0. for k in ('equal_alpha','size_alpha','pooled_stratum')};count=0
            for t1 in itertools.combinations(range(n1),nt1):
                for t2 in itertools.combinations(range(n1,n),nt2):
                    tt=np.array(t1+t2);cal=np.setdiff1d(normal,tt);test=np.r_[tt,np.arange(n,N)]
                    p=np.ones(len(test));eq=np.zeros(len(test),bool);sz=eq.copy()
                    for h in (0,1):
                        mask=strata[test]==h;pp=pvalues(rank,cal[strata[cal]==h],test[mask]);p[mask]=pp
                        eq[mask]=bh(pp,alpha/2);sz[mask]=bh(pp,alpha*mask.sum()/len(test))
                    out={'equal_alpha':eq,'size_alpha':sz,'pooled_stratum':bh(p,alpha)}
                    if alpha==.1:
                        original=methods(rank,cal,test,strata)
                        assert all(np.array_equal(original[k],rr) for k,rr in out.items())
                    for k,rr in out.items():
                        v=int((test[rr]<n).sum());sums[k]+=v/max(rr.sum(),1);false_events+=int(v>0)
                    count+=1
            bound=alpha*(nt1+nt2)/(nt1+nt2+na1+na2)
            fdr={k:v/count for k,v in sums.items()}
            assert all(v<=bound+1e-12 for v in fdr.values()),(fdr,bound)
            rows.append(dict(normal_sizes=[n1,n2],null_test_sizes=[nt1,nt2],alpha=alpha,assignments=count,fdr=fdr,bound=bound))
            total_assignments+=count
    # Non-vacuous unequal-propensity marginal rank check: a designated null test
    # node is uniform among eleven normals, followed by all Bernoulli subsets.
    n=11;rank=np.arange(n);strata=np.arange(n)%2;prob=np.array([.8,.2])[strata]
    thresholds=np.array([.01,.025,.05,.1,.2,.5,.9]);cdf=np.zeros(len(thresholds));mass_total=0.
    for test_id in range(n):
        remaining=np.setdiff1d(np.arange(n),test_id)
        for flags in itertools.product((False,True),repeat=n-1):
            flags=np.array(flags);cal=remaining[flags]
            mass=np.prod(np.where(flags,prob[remaining],1-prob[remaining]))/n
            p=pvalues(rank,cal,np.array([test_id]),1/prob)[0]
            cdf+=mass*(p<=thresholds);mass_total+=mass
    assert abs(mass_total-1)<1e-12
    assert np.all(cdf<=thresholds+1e-12)
    assert false_events>0 and cdf[3]>0
    report=dict(conditional_block_designs=rows,assignments=total_assignments,false_discovery_events_checked=false_events,
        weighted_assignments=n*2**(n-1),weighted_thresholds=thresholds.tolist(),weighted_null_cdf=cdf.tolist())
    (D/'allocation_additional_validation.json').write_text(json.dumps(report,indent=2))
    print('PASSED',total_assignments,'block assignments;',false_events,'false-discovery events;',n*2**(n-1),'weighted assignments')

if __name__=='__main__':main()
