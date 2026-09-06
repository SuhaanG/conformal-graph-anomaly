"""
generate_failure_modes_figure.py

Generates the Figure 1 overview schematic for the TMLR paper: the two
failure modes of calibration on a graph, drawn as two vertical causal
chains side by side. Left: the clean selection rule shifts degree, the
shift carries into scores when the detector depends on degree, null
p-values shrink, and realized FDR inflates (Section 6.1). Right: true
contamination pushes anomalies into the top calibration ranks, raising
the p-value floor and costing power while FDR stays at or below nominal
(Section 6.3). Numbers shown are the published Amazon values.

Run locally: python3 generate_failure_modes_figure.py
Output: failure_modes_figure.pdf (vector format, ready for \\includegraphics)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 9
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'

BLUE = '#1f4e79'
GREEN = '#2e7d4f'
RED = '#a83232'
GRAY = '#666666'
ARROW = '#444444'
FILL_BLUE = '#eef3f8'
FILL_RED = '#f9ecec'
FILL_GREEN = '#ecf5ef'

fig, ax = plt.subplots(figsize=(5.6, 4.3))
ax.set_xlim(-2, 102)
ax.set_ylim(16, 101)
ax.axis('off')

cx = [26.5, 74.5]          # column centers
W, H = 38.0, 9.6           # uniform box width and height
ys = [86.0, 71.5, 57.0, 42.5, 28.0]   # box centers, top to bottom

left_boxes = [
    'Clean rule: keep only nodes\nwith no anomalous neighbors',
    'Calibration degree shifts low\n(7$\\times$ below test normals)',
    'Calibration scores run low',
    'Null p-values shrink: normal\ntest nodes look extreme',
    'FDR inflated: 0.782 realized\nagainst nominal $\\alpha=0.10$',
]
right_boxes = [
    'Anomalies enter calibration\n(5–10% of the draw)',
    'Anomalies occupy the top\ncalibration ranks',
    'P-value floor $1/(n_{\\mathrm{cal}}+1)$ rises',
    'Benjamini-Hochberg\nrejects less often',
    'FDR $\\leq \\alpha$: power falls\ntoward zero',
]
left_refs = ['Sec. 5', 'Sec. 6.1', 'Sec. 4', 'Sec. 4', 'Sec. 6.1']
right_refs = ['Sec. 5', 'Sec. 4', 'Fact 1', 'Sec. 3.1', 'Sec. 6.3']


def draw_chain(x, boxes, refs, outcome_fill, outcome_edge):
    for i, (txt, ref) in enumerate(zip(boxes, refs)):
        last = i == len(boxes) - 1
        face = outcome_fill if last else FILL_BLUE
        edge = outcome_edge if last else BLUE
        lw = 1.2 if last else 0.9
        box = mpatches.FancyBboxPatch(
            (x - W / 2, ys[i] - H / 2), W, H,
            boxstyle='round,pad=0,rounding_size=2.2',
            facecolor=face, edgecolor=edge, linewidth=lw,
            mutation_aspect=0.6)
        ax.add_patch(box)
        ax.text(x, ys[i], txt, ha='center', va='center', fontsize=7.6,
                linespacing=1.3)
        ax.text(x + W / 2 + 1.2, ys[i], ref, ha='left',
                va='center', fontsize=5.5, color=GRAY)
        if i < len(boxes) - 1:
            ax.annotate('', xy=(x, ys[i + 1] + H / 2),
                        xytext=(x, ys[i] - H / 2),
                        arrowprops=dict(arrowstyle='-|>', color=ARROW,
                                        linewidth=0.9, shrinkA=1.5,
                                        shrinkB=1.5))


draw_chain(cx[0], left_boxes, left_refs, FILL_RED, RED)
draw_chain(cx[1], right_boxes, right_refs, FILL_GREEN, GREEN)

# Column headers, two lines so the columns cannot collide.
ax.text(cx[0], 97.8, 'Selection', ha='center', va='center',
        fontsize=8.6, fontweight='bold', color=BLUE)
ax.text(cx[0], 93.6, 'the standard precaution', ha='center', va='center',
        fontsize=7.0, style='italic', color=GRAY)
ax.text(cx[1], 97.8, 'Contamination', ha='center', va='center',
        fontsize=8.6, fontweight='bold', color=BLUE)
ax.text(cx[1], 93.6, 'the standard concern', ha='center', va='center',
        fontsize=7.0, style='italic', color=GRAY)

# The detector-conditional gate on the left chain: the degree shift carries
# into scores only when the score depends on degree.
gate_y = (ys[1] + ys[2]) / 2
ax.annotate('only if score\nincreases with\ndegree ($\\rho=0.92$)',
            xy=(cx[0] - 2.0, gate_y), xytext=(cx[0] - W / 2 - 2.2, gate_y),
            ha='right', va='center', fontsize=6.2, style='italic',
            color=RED,
            arrowprops=dict(arrowstyle='-', color=RED, linewidth=0.7,
                            shrinkA=2, shrinkB=1))

# Visibility tags under the outcome boxes.
ax.text(cx[0], ys[-1] - H / 2 - 2.2, 'silent: output looks normal',
        ha='center', va='top', fontsize=6.4, style='italic', color=RED)
ax.text(cx[1], ys[-1] - H / 2 - 2.2, 'visible: procedure goes quiet',
        ha='center', va='top', fontsize=6.4, style='italic', color=GREEN)

plt.tight_layout()
plt.savefig('failure_modes_figure.pdf', bbox_inches='tight')
plt.savefig('failure_modes_figure.png', dpi=300, bbox_inches='tight')
print('Saved failure_modes_figure.pdf and failure_modes_figure.png')
