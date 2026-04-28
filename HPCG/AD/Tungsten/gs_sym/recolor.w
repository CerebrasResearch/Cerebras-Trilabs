/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2026 [Redacted for anonymous review]
 * All rights reserved.
 */

xp param Y;

sp socket rx;
sp socket tx;

xp param NDOT;

wv const TD = { 0x0000, 0x01c0 };

xp router_bcast[1] address(0x7c16); // dot color 22 (broadcast)
xp router_recolor[1]; // dot color 22 (broadcast)

[[uthread.enable(on)]]
∀iter in [0,NDOT) {
  sp s=0.0;
  ∀i in [0,Y) s += rx[]; 
  tx[] = s;
  tx[] <- control(TD);
}

// task 29
#pragma teardown on
{
  router_bcast[0] = router_recolor[0];
}

