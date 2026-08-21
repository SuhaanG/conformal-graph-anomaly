import pandas as pd, numpy as np
from scipy.stats import spearmanr
df = pd.read_csv('results/logs/selection_bias_matrix.csv')
cell = df.groupby(['dataset','detector'], as_index=False).agg(
    sdeg=('spearman_score_degree','mean'),
    gamma=('gamma_hat','mean'),
    ks=('ks_uniform','mean'),
    meanp=('mean_p','mean'),
    disc=('n_discoveries','mean'),
    fdr=('realized_fdr','mean'))
def rep(name, d):
    print('--- %s (n=%d) ---' % (name, len(d)))
    for stat in ['meanp','ks','gamma']:
        rho, p = spearmanr(d['sdeg'], d[stat])
        print('    %-7s rho=%+.4f  p=%.4f' % (stat, rho, p))
    print('')
rep('ALL CELLS (headline)', cell)
print('zero-discovery cells: %d of %d' % (int((cell.disc==0).sum()), len(cell)))
print('')
rep('DROP zero-discovery cells', cell[cell.disc>0])
rep('DROP dominant_pygod', cell[cell.detector!='dominant_pygod'])
rep('DROP both', cell[(cell.disc>0)&(cell.detector!='dominant_pygod')])
print('--- within-detector (n=4 datasets each) ---')
for det, g in cell.groupby('detector'):
    rho, p = spearmanr(g['sdeg'], g['gamma'])
    print('    %-16s rho=%+.4f p=%.4f  disc=%s' % (det, rho, p, list(np.round(g.disc,1))))
print('')
print('--- within-dataset (n=5 detectors each) ---')
for ds, g in cell.groupby('dataset'):
    rho, p = spearmanr(g['sdeg'], g['gamma'])
    print('    %-12s rho=%+.4f p=%.4f' % (ds, rho, p))
print('')
print('CELL TABLE')
print(cell.to_string(index=False))
