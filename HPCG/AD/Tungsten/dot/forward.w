/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2026 Cerebras Systems Inc.
 * All rights reserved.
 */

xp param X;

sp socket v;
sp socket rx;

xp param NDOT;

wv const ADV = { 0x9249, 0x0940 };

xp router_bcast[1] address(0x7c16); // dot color 22 (broadcast)
xp router_forward[1]; // dot color 22 (broadcast)



[[uthread.enable(on)]]
∀iter in [0,NDOT) {
  sp rx1=0.0;
  ∀i in [0,X) rx1 += rx[];
  v[] ← rx1;
  v[] ← control(ADV);
}

// task 29
#pragma teardown on
{
  router_bcast[0] = router_forward[0];
}

