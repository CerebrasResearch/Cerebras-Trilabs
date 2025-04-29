/*
 * Released under BSD 3-Clause License,
 * Copyright (c) 2025 Cerebras Systems Inc.
 * All rights reserved.
 */

xp param npass;

sp socket rx;
sp socket tx;

sp fifomem[n=0x1000];
sp fifo ff = {.mem=fifomem};

wv rst = {0x9249,0x0980};

main {
  
  let j in [0,npass);
  parallel {
    ∀j ff[] <- rx[];
    ∀j tx[] <- ff[];
  }
  
  tx[] <- control(rst);

  // Enter steady state
  parallel {
    i in [0,∞) ff[] ← rx[];
    i in [0,∞) tx[] ← ff[];
  }
}
