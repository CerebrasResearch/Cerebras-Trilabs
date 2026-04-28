#!/usr/bin/env python3
"""Read dot-product timestamps from tstamp.bin and report cycles per dot."""

import sys
import numpy as np

def main():
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} tstamp.bin H V NDOT")
        sys.exit(1)

    fname = sys.argv[1]
    H = int(sys.argv[2])
    V = int(sys.argv[3])
    NDOT = int(sys.argv[4])

    NTIME = 64  # log2ntime=6 => 64 slots

    # tstamp layout: [H][V][ntime][3] of uint16
    raw = np.fromfile(fname, dtype=np.uint16)
    expected = H * V * NTIME * 3
    if raw.size < expected:
        print(f"Warning: expected {expected} uint16s, got {raw.size}")
        sys.exit(1)

    data = raw[:expected].reshape(V, H, NTIME, 3)

    # Reconstruct 48-bit timestamps: tstamp[m=0] is low 16, [m=1] is mid 16, [m=2] is high 16
    ts = data[:, :, :, 0].astype(np.int64) | \
         (data[:, :, :, 1].astype(np.int64) << 16) | \
         (data[:, :, :, 2].astype(np.int64) << 32)

    # Timestamp sequence: T0, then for each dot: T_before, T_after, then T_end
    # Total stamps = 1 + 2*NDOT + 1
    nstamps = 2 + 2 * NDOT

    print(f"\nDot product timing ({H}x{V} PEs, {NDOT} dots):")
    print(f"{'Phase':<20s} {'Max':>10s} {'Mean':>10s} {'Min':>10s}  cycles")
    print("-" * 60)

    for d in range(NDOT):
        idx_before = 1 + 2 * d
        idx_after = 1 + 2 * d + 1
        cycles = ts[:, :, idx_after] - ts[:, :, idx_before]
        cmax = int(cycles.max())
        cmean = float(cycles.mean())
        cmin = int(cycles.min())
        print(f"  dot {d:<15d} {cmax:>10d} {cmean:>10.1f} {cmin:>10d}")

    # Total: T_end - T0
    total = ts[:, :, nstamps - 1] - ts[:, :, 0]
    tmax = int(total.max())
    tmean = float(total.mean())
    tmin = int(total.min())
    print(f"{'  total':<20s} {tmax:>10d} {tmean:>10.1f} {tmin:>10d}")

    # FLOP estimate: each dot does 2*Z FLOPs (multiply + add) per PE
    # Z is not passed here but we can compute from the cycles
    print(f"\nNote: each dot performs 2*Z FLOPs per PE (Z = vector length)")

if __name__ == "__main__":
    main()
