import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

col1, col2, col3 = [], [], []
with open('scaling_gs_xy.dat') as f:
    for line in f:
        parts = line.split()
        col1.append(float(parts[0]))
        col2.append(float(parts[1]))
        if len(parts) >= 3:
            col3.append(float(parts[2]))
        else:
            col3.append(None)

col2_scaled = [v * 0.5 for v in col2]

x3 = [col1[i] for i in range(len(col3)) if col3[i] is not None]
y3 = [col3[i] for i in range(len(col3)) if col3[i] is not None]
y3_scaled = [v * 0.5 for v in y3]

spmv_xy = np.loadtxt('scaling_spmv_xy.dat')

def fit_quality(x, y, coeffs):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    y_pred = np.polyval(coeffs, x)
    nrmse = np.sqrt(np.mean((y - y_pred) ** 2)) / np.mean(y)
    return 1 - nrmse

fig = plt.figure(figsize=(10, 7.5))
gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.15)
ax = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax)

# Fit lines
gs1_coeffs  = np.polyfit(x3, y3_scaled, 1)
gs16_coeffs = np.polyfit(col1, col2_scaled, 1)
spmv_coeffs = np.polyfit(spmv_xy[:, 0], spmv_xy[:, 1], 1)

x3_arr    = np.array(x3)
col1_arr  = np.array(col1)
spmv_x    = spmv_xy[:, 0]

# Fitted lines over data range
fq_gs1  = fit_quality(x3,            y3_scaled,      gs1_coeffs)
fq_gs16 = fit_quality(col1,          col2_scaled,    gs16_coeffs)
fq_spmv = fit_quality(spmv_xy[:, 0], spmv_xy[:, 1], spmv_coeffs)

ax.plot(col1_arr, np.polyval(gs16_coeffs, col1_arr), 'o-', color='tab:blue',   linewidth=2, label='Gauss-Seidel with 16x16 subdomains')
ax.plot(spmv_x,   np.polyval(spmv_coeffs, spmv_x),  'D-', color='tab:green',  linewidth=2, label='SpMV')

# Dashed extensions to y-axis
ax.plot([0, col1_arr[0]], [gs16_coeffs[1], np.polyval(gs16_coeffs, col1_arr[0])], '--', color='tab:blue',   linewidth=1)
ax.plot([0, spmv_x[0]],   [spmv_coeffs[1], np.polyval(spmv_coeffs, spmv_x[0])],  '--', color='tab:green',  linewidth=1)

ax.set_xlim(left=0)
ax.set_ylabel('Cycles', fontsize=17)
ax.tick_params(axis='both', labelsize=17)
ax.legend(fontsize=17)
ax.set_ylim(0, 20000)
ax.grid(True, alpha=0.3)

# Fit error subplot (relative %)
gs16_err = (np.array(col2_scaled) - np.polyval(gs16_coeffs, col1_arr)) / np.array(col2_scaled) * 100
spmv_err = (spmv_xy[:, 1] - np.polyval(spmv_coeffs, spmv_x)) / spmv_xy[:, 1] * 100
bar_w = 4
ax2.bar(col1_arr - bar_w/2, gs16_err, bar_w, color='tab:blue')
ax2.bar(spmv_x + bar_w/2,  spmv_err, bar_w, color='tab:green')
ax2.set_ylabel('Fit error (%)', fontsize=17)
ax2.set_xlabel('X,Y (Z fixed at 192)', fontsize=17)
ax2.tick_params(axis='both', labelsize=17)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylim(-0.3, 0.3)
ax2.grid(True, alpha=0.3)
plt.setp(ax.get_xticklabels(), visible=False)

plt.tight_layout()
plt.savefig('scaling_xy.pdf', bbox_inches='tight')
print('Saved scaling_xy.pdf')

def report(name, x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    nrmse = np.sqrt(np.mean((y - y_pred)**2)) / np.mean(y)
    print(f'  {name:42s}  slope={coeffs[0]:.4f}  intercept={coeffs[1]:.1f}  NRMSE={nrmse:.2e}')

print('Linear fit statistics:')
report('GS 16x16 subdomains', col1, col2_scaled)
report('SpMV', spmv_xy[:, 0], spmv_xy[:, 1])
