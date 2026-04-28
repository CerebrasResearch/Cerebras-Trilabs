/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2026 [Redacted for anonymous review]
 * All rights reserved.
 */

socket tx[[dir=out]];
socket s0[[dir=in, id=0]];

xp param Z;
xp param NDOT;

sp zz[2] address(0x7f00) = {0.0,0.0};
iterator zeros = i in [0,∞) j in [0,2) zz[j];

u64    dsd[2];
dsd[0] = s0;
dsd[1] = zeros;

stream e;

// Queue and router access arrays
xp inq_config_regs[8][2]  address(0x7bc0);
xp outq_config_regs[8] address(0x7b80);
xp router_config_regs[24] address(0x7c00);
xp switch_config_regs[r=24][n=4] address(0x7c80);

xp router_dot[2];
xp switch_dot[n=4];

wv const TD = { 0x0000, 0x01c0 };
wv const ADV = { 0x9249, 0x0940 };
sp dot_res[1];

sp a[z=Z]  bank(0);
sp b[z=Z]  bank(2);

// timer
xp const log2ntime=6;
xp const ntime = 1<<log2ntime;
xp const tmask = ntime - 1;
xp tstamp[n=ntime][m=3];
xp timeridx;

xp const tra1 = 0x7b30; // SDR
xp const tra2 = 0x7b34; // SDR
xp timerreg[3] address(tra1);
xp timercontrolreg[1] address(tra2);

let i4 ∈ [0, 4);

function timerinit() {
  xp t;
  t = timercontrolreg[0];
  t <- t | 4;  // Set counter to record every cycle.
  timercontrolreg[0] <- t;
  timeridx <- 0;
}
function recordtime() {
  let i3t ∈ [0, 3);
  xp j;
  j <- timeridx & tmask;
  ∀i3t tstamp[j][i3t] <- timerreg[i3t];
  timeridx <- timeridx + 1;
}

function dot(xp abase, xp bbase) {

  sp av[Z];
  av.base = abase;
  sp bv[Z];
  bv.base = bbase;

  // receive from east; input queue 3
  e = dsd[0];
  xp t4 = inq_config_regs[3][0] & 0xFFE0;
  inq_config_regs[3][0] = t4 | 22;

  // send on out queue 0
  xp const colmask = 0xfe0f; // ~(31<<4)
  xp t = outq_config_regs[0];
  t = t & colmask;
  t = t | (21*16);
  outq_config_regs[0] = t;

  router_config_regs[22] = router_dot[1];  // reset color 22 from td

  sp acc = 0.0;
  ∀i in [0,Z) acc += av[i] * bv[i];
  tx[] ← acc;
  tx[] ← control(ADV);
  dot_res[0] = e[];
}

// main
timerinit();
router_config_regs[21] = router_dot[0];  // set up color 21 (h-reduction)
router_config_regs[22] = 0x8000;          // color 22 (bcast) in teardown
∀i4 in [0,4) switch_config_regs[21][i4] = switch_dot[i4]; // switching on color 21

xp a_base = (xp)(a.base);
xp b_base = (xp)(b.base);

recordtime(); // T0: before all dots

∀iter in [0,NDOT) {
  recordtime(); // before dot
  dot(a_base, b_base);
  recordtime(); // after dot
}

recordtime(); // T_end: after all dots

// task 29
#pragma teardown on
{
}
