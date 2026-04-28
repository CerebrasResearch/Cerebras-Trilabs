import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('scaling_gs_z.dat')
x = data[:, 0]
y = data[:, 1]

y_scaled = y * 0.5

spmv_z = np.loadtxt('scaling_spmv_z.dat')

def fit_quality(x, y, coeffs):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    y_pred = np.polyval(coeffs, x)
    nrmse = np.sqrt(np.mean((y - y_pred) ** 2)) / np.mean(y)
    return 1 - nrmse

# Fit the original (non-divided) data, same as scaling_z
gs_coeffs   = np.polyfit(x,            y_scaled,      1)
spmv_coeffs = np.polyfit(spmv_z[:, 0], spmv_z[:, 1], 1)

fq_gs   = fit_quality(x,            y_scaled,      gs_coeffs)
fq_spmv = fit_quality(spmv_z[:, 0], spmv_z[:, 1], spmv_coeffs)

# Smooth hyperbolic curves: (a + b*Z) / Z derived from affine fit
z_gs   = x
z_spmv = spmv_z[:, 0]

z_smooth_gs   = np.linspace(z_gs[0],   z_gs[-1],   500)
z_smooth_spmv = np.linspace(z_spmv[0], z_spmv[-1], 500)
curve_gs_smooth   = (gs_coeffs[1]   + gs_coeffs[0]   * z_smooth_gs)   / z_smooth_gs
curve_spmv_smooth = (spmv_coeffs[1] + spmv_coeffs[0] * z_smooth_spmv) / z_smooth_spmv

fig, ax = plt.subplots(figsize=(10, 6))

# Smooth line, then markers at data points only
ax.plot(z_smooth_gs,   curve_gs_smooth,   '-',  color='tab:blue',  linewidth=2, label='Gauss-Seidel with 16x16 subdomains')
ax.plot(z_smooth_spmv, curve_spmv_smooth, '-',  color='tab:green', linewidth=2, label='SpMV')
ax.plot(z_gs,   (gs_coeffs[1]   + gs_coeffs[0]   * z_gs)   / z_gs,   'o', color='tab:blue',  linewidth=2)
ax.plot(z_spmv, (spmv_coeffs[1] + spmv_coeffs[0] * z_spmv) / z_spmv, 'D', color='tab:green', linewidth=2)

ax.set_xlabel('Z points per PE', fontsize=17)
ax.set_ylabel('Cycles per update', fontsize=17)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_xticks(np.arange(0, max(z_spmv.max(), z_gs.max()) + 32, 32))
ax.tick_params(axis='both', labelsize=17)
ax.legend(fontsize=13, loc='upper right')
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ylo, yhi = ax.get_ylim()
flop_ticks = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
tick_pos = 54 / flop_ticks
visible = tick_pos[(tick_pos >= ylo) & (tick_pos <= yhi)]
visible_labels = 54 / visible
ax2.set_yticks(visible)
ax2.set_yticklabels([f'{v:.2f}' for v in visible_labels])
ax2.set_ylabel('Flops/cycle/PE', fontsize=17)
ax2.tick_params(axis='y', labelsize=17)

plt.tight_layout()
plt.savefig('scaling_z_strong.pdf', bbox_inches='tight')
print('Saved scaling_z_strong.pdf')

def report(name, x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    nrmse = np.sqrt(np.mean((y - y_pred)**2)) / np.mean(y)
    print(f'  {name:42s}  slope={coeffs[0]:.4f}  intercept={coeffs[1]:.1f}  NRMSE={nrmse:.2e}')

print('Linear fit statistics:')
report('GS 16x16 subdomains', x, y_scaled)
report('SpMV', spmv_z[:, 0], spmv_z[:, 1])
