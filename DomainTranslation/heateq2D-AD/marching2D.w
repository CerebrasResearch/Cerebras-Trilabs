/*
 * Released under BSD 3-Clause License,
 * Copyright (c) 2025 Cerebras Systems Inc.
 * All rights reserved.
 */

import repfun;

sp socket right,up,down,left,ping;

// Routing plan (interior, computing cores):
//   c1: C>R
//   c2: C>U
//   c3: D>C
//   c4: L>C
//   anti-red:   R>L
//   anti-blue:  R>L
//   anti-green:  U>D
//   anti-yellow: U>D
//
//   Core coord x,y:
//     c1 = (x%2 == 0) ? red : blue
//     c2 = (y%2 == 0) ? green : yellow
//     c3 = (y%2 == 0) ? blue : red
//     c4 = (x%2 == 0) ? yellow : green
//
//   Color swapping moats:
//      left,right: horizontal red <-> anti-red, blue <-> anti-blue
//      top,bottom: vertical green <-> anti-green, yellow <-> anti-yellow
//
//   This means that if I send out the right edge, color swaps, goes
//   all the way back to the left edge, swaps again, and enters first
//   column on the same color it was sent out over the right edge.

xp param n,p;
xp const w = 2*p;
xp const pm1 = p - 1;
xp const pp1 = p + 1;

sp x[n=n+w][m=n+w],y[n=n+w][m=n+w];

struct paramstruct {
  sp origin;
  sp h;  // grid spacing
  sp dt; // time step
  sp D;  // diffusion coefficient
  sp niter,ninner,nsample;
};
struct paramstruct params;

sp a1,a4;
function setweights() {
  sp h,dt,D,hinv,alpha;
  h  <- params.h;
  dt <- params.dt;
  D  <- params.D;

  hinv <- inv2(h);
  alpha <- D*dt*hinv*hinv;
  a1 <- alpha;
  a4 <- 1.0 - 4.0*alpha;
}


function sendrecv(sp xq[n+w][n+w]) {
  let ir ∈ [w,n+w);
  let jr ∈ [n,n+w);
  let iu ∈ [n,n+w);
  let ju ∈ [w,n+w);
  let iul ∈ [n,n+w);
  let jul ∈ [0,w);

  let il ∈ [w,n+w);
  let jl ∈ [0,w);
  let id ∈ [0,w);
  let jd ∈ [w,n+w);
  let idl ∈ [0,w);
  let jdl ∈ [0,w);

  parallel {
    ∀ir ∀jr right[] <- xq[ir][jr];
    ∀iu ∀ju up[] <- xq[iu][ju];
    ∀il ∀jl xq[il][jl] <- left[];
    ∀id ∀jd xq[id][jd] <- down[];
  }

  parallel {
    ∀iul ∀jul up[] <- xq[iul][jul];
    ∀idl ∀jdl xq[idl][jdl] <- down[];
  }
}


function compute(sp xq[n+w][n+w],sp yq[n+w][n+w]) {
  let i ∈ [0,n);
  let j ∈ [0,n);
  let im1 ∈ [p,n+p);
  let jm1 ∈ [p,n+p);
  let ip1 ∈ [p,n+p);
  let jp1 ∈ [p,n+p);

  ∀i ∀j yq[i+w][j+w] <- 0;
  ∀i ∀j yq[i+w][j+w] <+- a4*xq[i+p][j+p];
  ∀i ∀j yq[i+w][j+w] <+- a1*xq[i+p][j+pm1];
  ∀i ∀j yq[i+w][j+w] <+- a1*xq[i+p][j+pp1];
  ∀i ∀j yq[i+w][j+w] <+- a1*xq[i+pm1][j+p];
  ∀i ∀j yq[i+w][j+w] <+- a1*xq[i+pp1][j+p];

  ∀i ∀j xq[i+w][j+w] <- yq[i+w][j+w];
}


xp timeridx;
xp const log2ntime = 9;
xp const ntime = 1<<log2ntime;
xp const tmask = ntime - 1;
xp tstamp[n=ntime][m=3];

/*
xp timerreg_fyn[3] address(0x7f18);
xp timercontrolreg_fyn[1] address(0x7f1e);

xp timerreg_sdr[3] address(0x7b30);
xp timercontrolreg_sdr[1] address(0x7b34);
*/
xp tra1 = 0x7b30;
xp tra2 = 0x7b34;
xp timerreg[3];
xp timercontrolreg[1];

function timerinit() {
  xp t;
  timerreg.base = tra1;
  timercontrolreg.base = tra2;
  
  t = timercontrolreg[0];
  t <- t | 12;
  timercontrolreg[0] <- t;
  timeridx <- 0;
}
function recordtime() {
  let it ∈ [0,3);
  xp j;
  j <- timeridx & tmask;
  ∀it tstamp[j][it] <- timerreg[it];
  timeridx <- timeridx + 1;
}

sp junk[1];
sp junk2[1];
main {
  setweights();

  /* Compute initial conditions */ {
    let ii ∈ [0,n+w);
    let jj ∈ [0,n+w);
    let i1 ∈ [w,w+1);
    let j1 ∈ [w,w+1);
    sp oval;
    oval <- params.origin;
    ∀ii ∀jj x[ii][jj] <- 0.0;
    ∀i1 ∀j1 x[i1][j1] <- 1.0*oval;
  }

  junk2[0] <- ping[];
  
  timerinit();
  
  sp iter,niter,nsample,isample,k,kmax;
  niter <- params.niter;
  nsample <- params.nsample;
  kmax <- params.ninner;
  iter <- 0.0;
  isample <- 0.0;

  k <- 0.0;
  while(iter < niter) {
    k <- 0.0;
    while(k < kmax) {
      sendrecv(x);
      compute(x,y);
      k <+- 1.0;
    }
    isample <+- 1.0;
    if(isample == nsample) {
      recordtime();
      isample <- 0.0;
    }
    iter <+- 1.0;
  }

}
