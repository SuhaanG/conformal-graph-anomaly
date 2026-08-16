"""
generate_severity_figure.py

Generates the two-panel severity escalation figure for Section V.A.
Uses the actual per-seed data from the severity sweep experiment
(20 seeds at each of 5 severity levels), the same data reported in
Table synthetic-severity of the paper. Top panel shows realized FDR
against the nominal target line; bottom panel shows power collapsing
at the highest severity level.

Run locally: python3 generate_severity_figure.py
Output: severity_figure.pdf (vector format, ready for \\includegraphics)
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

data = {
    0.002: {
        'fdr':   [0.128, 0.000, 0.111, 0.000, 0.154, 0.098, 0.089, 0.070, 0.137, 0.056,
                  0.021, 0.142, 0.054, 0.000, 0.124, 0.065, 0.079, 0.000, 0.147, 0.149],
        'power': [0.191, 0.000, 0.160, 0.000, 0.183, 0.196, 0.191, 0.177, 0.260, 0.112,
                  0.061, 0.153, 0.141, 0.000, 0.207, 0.116, 0.155, 0.000, 0.217, 0.152],
    },
    0.005: {
        'fdr':   [0.126, 0.061, 0.078, 0.024, 0.088, 0.174, 0.084, 0.055, 0.112, 0.075,
                  0.158, 0.092, 0.142, 0.016, 0.000, 0.138, 0.000, 0.068, 0.021, 0.266],
        'power': [0.157, 0.083, 0.127, 0.109, 0.137, 0.145, 0.131, 0.115, 0.116, 0.115,
                  0.205, 0.092, 0.161, 0.080, 0.000, 0.216, 0.000, 0.055, 0.061, 0.265],
    },
    0.010: {
        'fdr':   [0.000, 0.020, 0.127, 0.124, 0.183, 0.207, 0.134, 0.000, 0.104, 0.020,
                  0.057, 0.000, 0.193, 0.152, 0.019, 0.000, 0.172, 0.103, 0.000, 0.110],
        'power': [0.000, 0.067, 0.175, 0.160, 0.179, 0.240, 0.137, 0.000, 0.115, 0.064,
                  0.067, 0.000, 0.156, 0.119, 0.071, 0.000, 0.231, 0.163, 0.000, 0.108],
    },
    0.020: {
        'fdr':   [0.000, 0.000, 0.000, 0.180, 0.000, 0.041, 0.211, 0.061, 0.100, 0.000,
                  0.207, 0.000, 0.158, 0.000, 0.240, 0.000, 0.000, 0.000, 0.051, 0.000],
        'power': [0.000, 0.000, 0.000, 0.109, 0.000, 0.063, 0.075, 0.061, 0.084, 0.000,
                  0.128, 0.000, 0.064, 0.000, 0.181, 0.000, 0.000, 0.000, 0.075, 0.000],
    },
    0.050: {
        'fdr':   [0.0] * 20,
        'power': [0.0] * 20,
    },
}

severities = sorted(data.keys())
fdr_means = [np.mean(data[s]['fdr']) for s in severities]
fdr_stds = [np.std(data[s]['fdr']) for s in severities]
power_means = [np.mean(data[s]['power']) for s in severities]
power_stds = [np.std(data[s]['power']) for s in severities]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.7), sharex=True)

ax1.errorbar(severities, fdr_means, yerr=fdr_stds, marker='o', markersize=4,
             color='#1f4e79', capsize=3, linewidth=1.2, label='Realized FDR')
ax1.axhline(y=0.10, color='#a83232', linestyle='--', linewidth=1.0, label=r'Nominal $\alpha = 0.10$')
ax1.set_ylabel('Realized FDR')
ax1.set_ylim(-0.02, 0.20)
ax1.legend(loc='upper right', frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.errorbar(severities, power_means, yerr=power_stds, marker='s', markersize=4,
             color='#2e7d4f', capsize=3, linewidth=1.2, label='Power')
ax2.set_ylabel('Power')
ax2.set_xlabel(r'Contamination severity ($p_{an}$)')
ax2.set_ylim(-0.02, 0.25)
ax2.legend(loc='upper right', frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('severity_figure.pdf', bbox_inches='tight')
plt.savefig('severity_figure.png', dpi=300, bbox_inches='tight')
print("Saved severity_figure.pdf and severity_figure.png")