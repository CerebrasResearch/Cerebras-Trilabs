import matplotlib.pyplot as plt
import numpy as np

# Data from table (rows=P, cols=L)
P = [1, 2, 3, 4]
rows = {
    1: [57695, 31115, 17684, 10554],
    2: [30991, 16513,  9176,  5351],
    3: [22111, 11850,  6583,  3788],
    4: [18776,  9754,  5323,  3062],
}
# Reorganise by L for plotting
data = {L: [rows[p][L] for p in P] for L in range(4)}

C = 5308416 / 256

plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 1.5})

fig, ax = plt.subplots(figsize=(8, 5))

for L, V_values in data.items():
    U = [C / V / (2 ** L) for V in V_values]
    ax.plot(P, U, marker='o', label=f'L={L}')

ax.set_xlabel('Pipeline stages')
ax.set_ylabel('Flops per cycle per active PE')
ax.legend(title='Level L')
ax.set_xticks(P)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('pipelining_level.pdf')
print("Saved pipelining_level.pdf")
