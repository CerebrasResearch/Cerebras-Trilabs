/*
 * Released under BSD 3-Clause License,
 * Copyright (c) 2025 Cerebras Systems Inc.
 * All rights reserved.
 */

sp socket rx;
sp socket tx; // will be the same as send up or sendright

sp fifomem[n=0x1000];
sp fifo ff = {.mem=fifomem};


sp socket sendup,recvdown,sendright,recvleft,ping;
xp param n,nrot,ncross;
//xp const n = 4;
sp buf1[n=n];
sp buf2[n=n];
//xp const ncross = 3;
//xp const nrot = 1;
let j in [0,n);

main {
  xp icross,irot;
  
  ∀j buf2[j] <- 0.0;
  
  icross <- 0;
  while(icross < ncross) {
    irot <- 0;
    while(irot < nrot) {    // Horizontal rotations
      parallel {
	∀j buf1[j] <- recvleft[];
	∀j sendright[] <- buf2[j] + 1.0;
      }
      irot <+- 1;
    }

    irot <- 0;
    while(irot < nrot) {    // Vertical rotations
      parallel {
	∀j buf2[j] <- recvdown[];
	∀j sendup[] <- buf1[j] + 1.0;
      }
      irot <+- 1;
    }

    icross <+- 1;
  }
  
  /* Fire up compute grid */
  sp tag = 1.0;
  ping[] <- tag;

  // Enter steady state situation
  parallel {
    i in [0,∞) ff[] ← rx[];
    i in [0,∞) tx[] ← ff[];
  }

}
