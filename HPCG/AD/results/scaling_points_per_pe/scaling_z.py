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

fig = plt.figure(figsize=(10, 7.5))
gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.15)
ax = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax)

gs_coeffs   = np.polyfit(x, y_scaled, 1)
spmv_coeffs = np.polyfit(spmv_z[:, 0], spmv_z[:, 1], 1)

fq_gs   = fit_quality(x,            y_scaled,      gs_coeffs)
fq_spmv = fit_quality(spmv_z[:, 0], spmv_z[:, 1], spmv_coeffs)

ax.plot(x,            np.polyval(gs_coeffs,   x),            'o-', linewidth=2,              label='Gauss-Seidel, fixed 16x16')
ax.plot(spmv_z[:, 0], np.polyval(spmv_coeffs, spmv_z[:, 0]), 'D-', color='tab:green', linewidth=2, label='SpMV')

# Dashed extensions to y-axis
ax.plot([0, x[0]],            [gs_coeffs[1],   np.polyval(gs_coeffs,   x[0])],            '--', color='tab:blue',  linewidth=1)
ax.plot([0, spmv_z[0, 0]],   [spmv_coeffs[1], np.polyval(spmv_coeffs, spmv_z[0, 0])],   '--', color='tab:green', linewidth=1)

ax.set_ylabel('Cycles', fontsize=17)
ax.set_xlim(left=0)
ax.set_xticks(np.arange(0, max(spmv_z[:, 0].max(), x.max()) + 32, 32))
ax.tick_params(axis='both', labelsize=17)
ax.set_ylim(0, 15000)
ax.legend(fontsize=17, loc='upper left')
ax.grid(True, alpha=0.3)

# Fit error subplot
gs_err = (y_scaled - np.polyval(gs_coeffs, x)) / y_scaled * 100
spmv_err = (spmv_z[:, 1] - np.polyval(spmv_coeffs, spmv_z[:, 0])) / spmv_z[:, 1] * 100
bar_w = 4
ax2.bar(x - bar_w/2,            gs_err,   bar_w, color='tab:blue', edgecolor='black')
ax2.bar(spmv_z[:, 0] + bar_w/2, spmv_err, bar_w, color='tab:green', edgecolor='black')
ax2.set_ylabel('Fit error (%)', fontsize=17)
ax2.set_xlabel('Z (X,Y fixed at 32)', fontsize=17)
ax2.tick_params(axis='both', labelsize=17)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylim(-0.15, 0.15)
ax2.grid(True, alpha=0.3)
plt.setp(ax.get_xticklabels(), visible=False)

plt.tight_layout()
plt.savefig('scaling_z.pdf', bbox_inches='tight')
print('Saved scaling_z.pdf')

def report(name, x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    nrmse = np.sqrt(np.mean((y - y_pred)**2)) / np.mean(y)
    print(f'  {name:42s}  slope={coeffs[0]:.4f}  intercept={coeffs[1]:.1f}  NRMSE={nrmse:.2e}')

print('Linear fit statistics:')
report('GS 16x16 subdomains', x, y_scaled)
report('SpMV', spmv_z[:, 0], spmv_z[:, 1])
