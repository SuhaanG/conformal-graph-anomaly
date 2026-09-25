"""Non-vacuous, frozen-population certificate checks at the primary q/delta."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import beta
import run_certification as rc
D=Path(__file__).resolve().parent

def main():
    rc.REPS=2000;N=4000;order=np.arange(N)
    ends=np.array([25,50,100,200,400,800,1600])
    # Labels are fixed before any audit randomness; clustered placement permits
    # extreme dependence rather than pretending the labels are iid replicates.
    all_normal=np.zeros(N,dtype=int);all_anomaly=np.ones(N,dtype=int)
    useful=all_normal.copy();useful[:800]=1;useful[np.arange(0,800,20)]=0
    boundary=all_normal.copy();boundary[:1600]=1;boundary[np.arange(0,1600,10)]=0
    regimes=[('all_normal',all_normal),('all_anomaly',all_anomaly),('useful_95pct',useful),('boundary_90pct',boundary)]
    rows=[]
    for cid,(name,y) in enumerate(regimes):
        degree=np.where(y==0,100,10) # fixed covariate, never learned from audit labels
        for b in (100,500):
            for mid,method in enumerate(('uniform_candidates','score_strata','score_degree_strata','uniform_graph')):
                rng=np.random.default_rng(np.random.SeedSequence([20260927,cid,b,mid]))
                z=rc.trials(y,degree,order,ends,method,b,rng)
                failures=int(z.certificate_failure.sum())
                upper=float(beta.ppf(.95,failures+1,rc.REPS-failures)) if failures<rc.REPS else 1.
                rows.append(dict(regime=name,budget=b,method=method,repetitions=rc.REPS,
                    failure_rate=failures/rc.REPS,failure_upper95=upper,
                    discoveries=z.discoveries.mean(),true_discoveries=z.true_discoveries.mean(),
                    fdp=z.fdp.mean(),abstain=z.abstain.mean()))
    d=pd.DataFrame(rows);d.to_csv(D/'stress_certificate.csv',index=False)
    assert d.loc[d.regime=='all_normal','discoveries'].eq(0).all()
    assert d.loc[d.regime=='all_anomaly','discoveries'].max()>0
    assert d.loc[d.regime=='useful_95pct','discoveries'].max()>0
    (D/'stress_validation.json').write_text(json.dumps(dict(trials=len(d)*rc.REPS,
        maximum_failure_rate=float(d.failure_rate.max()),maximum_pointwise_upper95=float(d.failure_upper95.max()),
        nonvacuous_positive_controls=True,note='Confidence limits are pointwise Monte Carlo diagnostics; proof supplies simultaneous candidate validity.'),indent=2))
    print(d.to_string(index=False))
if __name__=='__main__':main()
