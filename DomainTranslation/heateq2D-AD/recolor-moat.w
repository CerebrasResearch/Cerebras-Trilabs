/*
 * Released under BSD 3-Clause License,
 * Copyright (c) 2025 Cerebras Systems Inc.
 * All rights reserved.
 */

sp socket rx;
sp socket tx;

sp fifomem[n=0x1000];
sp fifo ff = {.mem=fifomem};

i in [0,∞) ff[] ← rx[];
i in [0,∞) tx[] ← ff[];
