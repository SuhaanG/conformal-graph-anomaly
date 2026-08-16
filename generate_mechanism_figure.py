"""
generate_mechanism_figure.py

Generates the scatter plot for Section V.E, Detection Power Mechanism.
Shows the relationship between the standardized score separation
statistic (d_prime) and the count of true anomalies whose conformal
p-value falls at or below 0.001, across all 100 trials of the
severity sweep (20 seeds x 5 severity levels). This is the same
data already verified against the correlation values published in
the text, r = 0.749.

Run locally: python3 generate_mechanism_figure.py
Output: mechanism_figure.pdf (vector format, ready for \\includegraphics)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

# Real per-trial data from the severity sweep, 20 seeds at each of 5
# severity levels, the same 100 trials used throughout Section V.E and
# just reverified against the published correlation values.
data = [
(1.851,129),(1.797,39),(1.826,116),(1.688,83),(1.830,130),
(1.904,141),(1.835,134),(1.829,131),(1.788,153),(1.814,84),
(1.778,67),(1.764,50),(1.820,106),(1.779,65),(1.811,141),
(1.711,87),(1.746,116),(1.780,38),(1.744,116),(1.757,104),
(1.784,103),(1.747,77),(1.713,95),(1.767,85),(1.745,103),
(1.825,60),(1.783,98),(1.776,86),(1.747,87),(1.713,86),
(1.799,75),(1.749,70),(1.697,101),(1.669,68),(1.840,64),
(1.781,131),(1.742,53),(1.654,75),(1.720,52),(1.742,162),
(1.654,67),(1.675,57),(1.647,114),(1.676,115),(1.713,99),
(1.688,139),(1.658,103),(1.785,74),(1.714,86),(1.606,48),
(1.682,71),(1.617,67),(1.694,97),(1.713,89),(1.771,72),
(1.618,38),(1.714,5),(1.693,50),(1.700,64),(1.623,81),
(1.475,42),(1.482,34),(1.458,26),(1.478,82),(1.494,43),
(1.455,62),(1.490,61),(1.509,57),(1.437,63),(1.470,70),
(1.430,96),(1.534,40),(1.442,50),(1.489,23),(1.586,83),
(1.499,45),(1.494,39),(1.404,64),(1.439,67),(1.481,22),
(0.939,23),(1.062,10),(0.996,11),(1.000,31),(1.030,16),
(1.030,24),(0.993,18),(1.029,4),(0.961,6),(1.015,24),
(0.946,12),(1.008,18),(0.962,14),(1.005,28),(0.963,2),
(0.990,15),(1.026,15),(0.989,27),(0.952,24),(1.066,14),
]

d_prime = np.array([d[0] for d in data])
n_below = np.array([d[1] for d in data])

r, p = stats.pearsonr(d_prime, n_below)

# fitted regression line for visual reference
slope, intercept = np.polyfit(d_prime, n_below, 1)
x_line = np.linspace(d_prime.min(), d_prime.max(), 100)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(3.5, 2.6))

ax.scatter(d_prime, n_below, s=14, alpha=0.6, color='#1f4e79', edgecolors='none')
ax.plot(x_line, y_line, color='#a83232', linewidth=1.2, linestyle='--',
        label=f'$r = {r:.3f}$')
ax.set_xlabel(r"Score separation ($d'$)")
ax.set_ylabel(r'Anomalies with $p \leq 0.001$')
ax.legend(loc='upper left', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('mechanism_figure.pdf', bbox_inches='tight')
plt.savefig('mechanism_figure.png', dpi=300, bbox_inches='tight')
print(f"Saved mechanism_figure.pdf and mechanism_figure.png, r={r:.3f}, p={p:.2e}")