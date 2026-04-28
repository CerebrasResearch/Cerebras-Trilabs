import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

fig, ax = plt.subplots(figsize=(10, 6))

label_map = {
    'ref_dbl': 'Reference (FP64)',
    'ref_flt': 'Reference (FP32)',
    't_flt': 'Tungsten (FP32)',
}

# Pair colors: each prefix gets one color, solid vs dashed distinguishes the two
color_map = {
    'ref_dbl': 'tab:blue',
    'ref_flt': 'tab:red',
    't_flt': 'tab:green',
}

# Reverse file order
files = sorted(glob.glob('validation_*.dat'), reverse=True)
for fname in files:
    if 'c_flt' in fname:
        continue
    prefix = os.path.basename(fname).replace('validation_', '').replace('_16x16x16.dat', '')
    label_prefix = label_map.get(prefix, prefix)
    color = color_map.get(prefix, None)
    data = np.loadtxt(fname)
    iters = data[:, 0]
    if iters[0] == 1:
        iters = iters - 1
    cg_norm = data[:, 3]
    true_norm = data[:, 4]
    ax.semilogy(iters, cg_norm, linewidth=2, color=color,
                label=f'{label_prefix} CG-internal norm')
    ax.semilogy(iters, true_norm, '--', linewidth=2, color=color,
                label=f'{label_prefix} true norm')

ax.set_xlim(-1, 31)
ax.set_ylim(1e-26, 1e1)
ax.set_xticks(range(0, 35, 5))
ax.set_yticks([10**i for i in range(0, -30, -5)])
ax.set_xlabel('Iteration', fontsize=21)
ax.set_ylabel('Normalized residual', fontsize=21)
ax.tick_params(axis='both', labelsize=16)
ax.legend(fontsize=17, loc='lower left')
ax.grid(True, which='major', alpha=0.3)
plt.tight_layout()
plt.savefig('validation.pdf', bbox_inches='tight')
print('Saved validation.pdf')
