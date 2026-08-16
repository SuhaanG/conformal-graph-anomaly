"""
generate_baseline_figure.py

Generates the grouped bar chart for Section V.C, Baseline Comparison.
Shows realized FDR for our method versus the ensemble baseline, across
four settings: synthetic contaminated, synthetic adversarial, Amazon
contaminated, Amazon adversarial. Uses the exact numbers already
published in Table baseline-results and Table baseline-real.

Run locally: python3 generate_baseline_figure.py
Output: baseline_figure.pdf (vector format, ready for \\includegraphics)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

# Exact values already published and verified in Table baseline-results
# and Table baseline-real.
settings = ['Synth.\nContam.', 'Synth.\nAdvers.', 'Amazon\nContam.', 'Amazon\nAdvers.']
ours_mean = [0.047, 0.052, 0.059, 0.000]
ours_std =  [0.062, 0.059, 0.040, 0.000]
ens_mean =  [0.043, 0.053, 0.059, 0.000]
ens_std =   [0.062, 0.060, 0.040, 0.000]

x = np.arange(len(settings))
width = 0.35

fig, ax = plt.subplots(figsize=(3.5, 2.6))

ax.bar(x - width/2, ours_mean, width, yerr=ours_std, capsize=3,
       color='#1f4e79', label='Ours', edgecolor='none')
ax.bar(x + width/2, ens_mean, width, yerr=ens_std, capsize=3,
       color='#2e7d4f', label='Ensemble', edgecolor='none')

ax.axhline(y=0.10, color='#a83232', linestyle='--', linewidth=1.0,
           label=r'Nominal $\alpha = 0.10$')

ax.set_ylabel('Realized FDR')
ax.set_xticks(x)
ax.set_xticklabels(settings)
ax.set_ylim(0, 0.16)
ax.legend(loc='upper right', frameon=False, fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('baseline_figure.pdf', bbox_inches='tight')
plt.savefig('baseline_figure.png', dpi=300, bbox_inches='tight')
print("Saved baseline_figure.pdf and baseline_figure.png")