/*
 * Released under BSD 3-Clause License,
 * Copyright (c) 2025 Cerebras Systems Inc.
 * All rights reserved.
 */

xp param ndata,fansize,idx;
sp socket rx,tx;

xp const period = ndata*fansize;
xp const period1 = period-1;
xp const npass = ndata;
xp const npass1 = npass-1;
filter flt[counter] = {
  .limit=period1, // filter period (ndrop+npass) is limit+1
  .maxpass=npass1, // number to let through is maxpass+1
  .state=64, // put in drop state
  .tick=wavelet,
  .socket=rx
};
wv adv = {0x9249,0x0940};// 0x0000 , 0x0140};
wv rst = {0x9249,0x0980};// 0x0000 , 0x0180};


sp fifomem[n=0x1000];
sp fifo ff = {.mem=fifomem};

i in [0,∞) ff[] ← rx[];
i in [0,∞) tx[] ← ff[];

let j ∈ [0,ndata);
let k1 ∈ [idx+1,fansize);

main {
  // rx starts in C>C.
  ∀j ∀k1 rx[] <- 0; // Wind counter to appropriate value
  // Switch rx to (e.g.) D>CU
  rx[] <- control(rst);
}
