/*
 * Released under BSD 3-Clause License,
 * Copyright (c) 2025 Cerebras Systems Inc.
 * All rights reserved.
 */

sp socket rx1,tx1;
sp socket rx2,tx2;

sp fifomem1[n=0x800];
sp fifomem2[n=0x800];

sp fifo ff1 = {.mem=fifomem1};
sp fifo ff2 = {.mem=fifomem2};

i in [0,∞) ff1[] ← rx1[];
i in [0,∞) tx1[] ← ff1[];

i in [0,∞) ff2[] ← rx2[];
i in [0,∞) tx2[] ← ff2[];
