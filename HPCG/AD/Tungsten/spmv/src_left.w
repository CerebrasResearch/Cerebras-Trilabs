/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2026 [Redacted for anonymous review]
 * All rights reserved.
 */

sp socket tx;
sp socket rx;

xp param NDOT;

wv const ADV = { 0x9249, 0x0940 };
wv const RST = { 0x9249, 0x0980 };
wv const TD = { 0x0000, 0x01c0 };

xp router_src_left[1] address(0x7c16); // dot color 22 (broadcast)
xp router_src_left_content[1]; // dot color 22 (broadcast)

[[uthread.enable(on)]]
∀iter in [0,NDOT) {
  tx[] <- control(RST);
  sp nullmem <- rx[]; 
}

// task 29
#pragma teardown on
{
  router_src_left[0] = router_src_left_content[0];
}

