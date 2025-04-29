/*
 * Released under BSD 3-Clause License,
 * Copyright (c) 2025 Cerebras Systems Inc.
 * All rights reserved.
 */

xp param preload;
sp socket rx;
sp socket tx;

sp fifomem[n=0x1000];
sp fifo ff = {.mem=fifomem};

main {
  //∀j in [0,preload) ff[] <- 0.0;
  ∀j in [0,preload) tx[] <- 0.0;
  
  parallel {
    i in [0,∞) ff[] ← rx[];
    i in [0,∞) tx[] ← ff[];
  }
}
