import sys
import numpy as np

# Check if exactly five arguments (excluding the script name) are provided
if len(sys.argv) != 6:
    print("Usage: python read_tstamp_file.py <file> <X> <Y> <Z> <level>")
    sys.exit(1) 

filename = sys.argv[1]
H = int(sys.argv[2])
V = int(sys.argv[3])
Z = int(sys.argv[4])
L = int(sys.argv[5])

t = np.fromfile(filename, dtype=np.uint16)
nt = 16
npe =  H*V

t = t.reshape([H,V,nt,3])
tt = np.empty([H,V,nt], dtype=np.uint64)
for i1a in range(H):
  for i1b in range(V):
     for i2 in range(nt):
        tt[i1a,i1b,i2] = (t[i1a,i1b,i2,2]<<32) + (t[i1a,i1b,i2,1]<<16) + t[i1a,i1b,i2,0]

F = 1<<L

t1 = tt[::F,::F,2]-tt[::F,::F,1]
t1max = t1.max()
t1mean = t1.mean()
t1min = t1.min()

print("Cycles: Max: ", t1max, "Mean: ", t1mean, "Min: ", t1min)

flops = (26*2 + 2)*H*V*Z/F/F/F 
flops_per_cycle = flops/t1max
flops_per_cycle_per_pe = flops_per_cycle/(npe/F/F)  # Count only active PEs

print("Coarsening factor F: ", F)
print("flops: ", flops)
print("Flops/cycle: ", flops_per_cycle)
print("Flops/cycle/PE: ", flops_per_cycle_per_pe)
print("Cycles/stage: ", 54/flops_per_cycle_per_pe)
