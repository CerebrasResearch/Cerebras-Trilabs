/*
 * Released under BSD 3-Clause License,
 * Copyright (c) 2025 Cerebras Systems Inc.
 * All rights reserved.
 */

xp param ndata,top;
sp socket rx,tx;

wv adv = {0x9249,0x0940};// 0x0000 , 0x0140};
wv rst = {0x9249,0x0980};// 0x0000 , 0x0180};

sp fifomem[n=0x1000];
sp fifo ff = {.mem=fifomem};

let j ∈ [0,ndata);

i in [0,∞) ff[] ← rx[];


/*
forever {
  ∀j tx[] <- ff[];
  if(top > 0)
    tx[] <- control(rst);
  else
    tx[] <- control(adv);
}
*/

main {
  forever {
    ∀j tx[] <- ff[];
    tx.flip;
    if(top > 0) tx[] <- control(rst);
  }
}

