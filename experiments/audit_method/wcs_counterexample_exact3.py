"""Exact FDR of randomized weighted BH and randomized WCS (homogeneous pruning) for the three-node example
in Appendix app:wbhcounter. Randomized p-values are independent uniforms on known intervals given roles;
WCS sizes are deterministic given roles; the pruning coin is integrated exactly over its breakpoints."""
import itertools
from fractions import Fraction as Fr
import numpy as np
from wcs_acquisition import wcs_sizes
import wcs_experiment

s=np.array([1.,2.,0.]);rho=np.array([.3,1.,.3]);w=1/rho;alpha=.5;m=2   # nodes a, b (normals), c (anomaly)

def wcs_cond(test,cal):
    _,S=wcs_sizes(s[cal],w[cal],s[test],w[test],alpha)
    W=w[cal].sum();b=np.array([w[cal][s[cal]>s[j]].sum() for j in test]);wt=w[test]
    q=np.clip((alpha*S/m*(W+wt)-b)/wt,0,1);tot=0.
    for f in itertools.product((0,1),repeat=m):
        f=np.array(f,bool);pr=np.prod(np.where(f,q,1-q))
        if pr==0 or not f.any():continue
        bps=np.unique(np.r_[0,1,[k/x for x in S[f] for k in range(1,m+1) if k/x<1]])
        for lo,hi in zip(bps[:-1],bps[1:]):
            rr=wcs_experiment.prune(f,S,(lo+hi)/2);tot+=pr*(hi-lo)*rr[0]/max(rr.sum(),1)
    return tot

# randomized weighted BH, by hand in exact fractions (derivation in the appendix)
bh_a=Fr(35,100)**2/2+Fr(25,1000)*Fr(65,100);bh_b=Fr(3,10)*1+Fr(7,10)*Fr(1,4)
fdr_bh=(bh_a+bh_b)/2
a=wcs_cond(np.array([0,2]),np.array([1]));b1=wcs_cond(np.array([1,2]),np.array([0]));b0=wcs_cond(np.array([1,2]),np.array([],int))
fdr_wcs=.5*a+.5*(.3*b1+.7*b0)
print('randomized weighted BH',fdr_bh,float(fdr_bh));print('randomized WCS conditional',a,b1,b0,'total',fdr_wcs,Fr(fdr_wcs).limit_denominator(100000))
assert fdr_bh==Fr(221,800) and abs(fdr_wcs-723/3200)<1e-12
