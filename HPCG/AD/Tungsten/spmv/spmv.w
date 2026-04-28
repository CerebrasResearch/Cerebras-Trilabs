/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2026 Cerebras Systems Inc.
 * All rights reserved.
 */

socket tx_upstr[[dir=out]], tx_downstr[[dir=out]];
socket s0[[dir=in, id=0]], s1[[dir=in, id=1]], s2[[dir=in, id=2]], s3[[dir=in, id=3]], s4[[dir=in, id=4]], s5[[dir=in, id=5]], s6[[dir=in, id=6]], s7[[dir=in, id=7]];

xp param Z;
xp param ZL;
xp param C;
xp param P;
xp param ZP;

xp param NNZ;
xp param BASE;
xp param NNZP;
xp param BASEP;

sp A         [z=NNZ*26] bank(2);
sp diag      [z=NNZ];
sp x         [z=NNZP]; // pre-pad, post-pad by 1
sp ax        [z=NNZ];

// timer
xp const log2ntime=4;
xp const ntime = 1<<log2ntime;
xp const tmask = ntime - 1;
xp tstamp[n=ntime][m=3];
xp timeridx;

xp const tra1 = 0x7b30; // SDR
xp const tra2 = 0x7b34; // SDR
xp timerreg[3] address(tra1);
xp timercontrolreg[1] address(tra2);

sp zz[2] address(0x7f00) = {0.0,0.0};
iterator zeros = i in [0,∞) j in [0,2) zz[j];

u64    dsd[9];
dsd[0] = s0;
dsd[1] = s1;
dsd[2] = s2;
dsd[3] = s3;
dsd[4] = s4;
dsd[5] = s5;
dsd[6] = s6;
dsd[7] = s7;
dsd[8] = zeros;

stream nw;
stream n;
stream ne;
stream e;
stream se;
stream s;
stream sw;
stream w;

sp nullmem[2] = {0.};

// Queue and router access arrays
xp inq_config_regs[8][2]  address(0x7bc0);
xp outq_config_regs[8] address(0x7b80);
xp router_config_regs[24] address(0x7c00); // TODO: is there a more user-frindly option?
xp router_td = 0x8000;

xp switch_config_regs[r=24][n=4] address(0x7c80);

xp router_dot[2];
xp switch_dot[n=4];

wv const TD = { 0x0000, 0x01c0 };
wv const ADV = { 0x9249, 0x0940 };
sp dot_res[1];

xp router_stencil[C][18];
xp boundary_stencil[C][9];
xp colors_stencil[C][9];
xp active_stencil[C];

function init_stencil_td() {
  // set teardown bits for all colors
  ∀ir in [0,18) router_config_regs[ir] = router_td;
}

// when the fabric is in the teardown state
// this fuction sets up a new configuration 
function set_sockets(xp ctx) {

  if (active_stencil[ctx] > 0) { // change socket settings for active tiles only 
    // assign queue colors or zeros dsd to eight input directions

    // set up default color map
    let i8 ∈ [0, 8);
    ∀i8 {
      xp t0 = inq_config_regs[i8][0] & 0xFFE0;
      inq_config_regs[i8][0] = t0 | colors_stencil[ctx][i8];
    }

    // patch the boundary 
    // TODO: loop with an array of streams 
    if ( boundary_stencil[ctx][0] > 0 )
      sw = dsd[8]; // zero
    else 
      sw = dsd[0]; 

    if ( boundary_stencil[ctx][1] > 0 )
      s = dsd[8]; // zero
    else 
      s = dsd[1];

    if ( boundary_stencil[ctx][2] > 0 )
      se = dsd[8]; // zero
    else 
      se = dsd[2];

    if ( boundary_stencil[ctx][3] > 0 )
      w = dsd[8]; // zero
    else 
      w = dsd[3];

    if ( boundary_stencil[ctx][4] > 0 )
      e = dsd[8]; // zero
    else 
      e = dsd[4];

    if ( boundary_stencil[ctx][5] > 0 )
      nw = dsd[8]; // zero
    else 
      nw = dsd[5];

    if ( boundary_stencil[ctx][6] > 0 )
      n = dsd[8]; // zero
    else 
      n = dsd[6]; 

    if ( boundary_stencil[ctx][7] > 0 )
      ne = dsd[8]; // zero
    else 
      ne = dsd[7];

    
    // assign color to the otput queue 
    // TODO: compiler intrinsic set_color for output queue 
    xp const colmask = 0xfe0f; // ~(31<<4)
    xp t = outq_config_regs[0];  // output queue 0 
    t = t & colmask;
    t = t | (colors_stencil[ctx][8]*16);
    outq_config_regs[0] = t;

    t = outq_config_regs[1];  // output queue 1
    t = t & colmask;
    t = t | ((colors_stencil[ctx][8]+9)*16);
    outq_config_regs[1] = t;
  }

  // reset router (and clear teardown bit) for all tiles
  ∀ir in [0,18) router_config_regs[ir] = router_stencil[ctx][ir];
}


let i2 ∈ [0, 2);
let i3 ∈ [0, 3);
let i24z ∈ [0, 24*ZL);
let i2z ∈ [0, 2*ZL);

iterator walkA3_fwd = ∀i24z A[BASE*26+i24z];
iterator walkA2_fwd = ∀i2z A[BASE*26+24*ZL+i2z];

let zp ∈ [0, ZP);
let ip ∈ [0, P);
let inf ∈ [0, ∞);

iterator walkZ2_fwd  = ∀inf ∀zp ∀ip ∀i2 x[BASEP+zp+ip*ZP+2*i2];
iterator walkZ3_fwd  = ∀inf ∀zp ∀ip ∀i3 x[BASEP+zp+ip*ZP+i3];
iterator walkZZ3_fwd = ∀inf ∀zp ∀ip ∀i3 x[BASEP+zp+ip*ZP+1+i3];

// spmv 
iterator walkx_fwd =  ∀inf ∀zp ∀ip x[BASEP+zp+ip*ZP+1];
iterator walkax_fwd =  ∀inf ∀zp ∀ip ax[BASE+zp+ip*ZP];
iterator walkdiag_fwd =  ∀inf ∀zp ∀ip diag[BASE+zp+ip*ZP];

sp tmp[1];
sp acc = 0;

function dot(xp abase, xp bbase) {

  sp av[Z];
  av.base = abase;
  sp bv[Z];
  bv.base = bbase;

  // receive from east; input queue 3
  e = dsd[3];
  xp t4 = inq_config_regs[3][0] & 0xFFE0;
  inq_config_regs[3][0] = t4 | 22;

  // send on out queue 0
  xp const colmask = 0xfe0f; // ~(31<<4)
  xp t = outq_config_regs[0];
  t = t & colmask;
  t = t | (21*16);
  outq_config_regs[0] = t;

  router_config_regs[22] = router_dot[1];  // reset color 22 from td

  sp acc = 0.0;
  ∀i in [0,Z) acc += av[i] * bv[i];
  tx_upstr[] ← acc;
  tx_upstr[] ← control(ADV);
  dot_res[0] = e[];
}

function spmv() {

#pragma uthread enable off
if (active_stencil[0] > 0) {

  // clearing accum0
  sp zz[3] = {0.};
  ∀i3 acc = ∑ zz[i3] * zz[i3];
  //

  ∀ip {
    ∀i3 wv:tx_upstr[] ← x[BASEP+ip*ZP+i3]; // (new,old,old) upstream
  }


  reset walkA3_fwd;
  walkA3_fwd.len = 3;

  reset walkA2_fwd;
  walkA2_fwd.len = 2;

  let z ∈ [0, ZL);

  ∀z ax[BASE+z] = 0;   

  ∀z {

    ∀i3 wv:tx_downstr[] ← walkZ3_fwd[]; // (new,new,old) downstream
    ∀i3 wv:tx_upstr[] ← walkZZ3_fwd[]; // next (new,old,old) upstream

    [[dot.load(off)]]
    ∀i2 acc = ∑ walkZ2_fwd[] *walkA2_fwd[]; //clears the accum
    [[dot.state(on), dot.load(off)]]
    {
        ∀i3 acc = ∑ (sp)(e[]) *walkA3_fwd[];
        ∀i3 acc = ∑ (sp)(nw[])*walkA3_fwd[];
        ∀i3 acc = ∑ (sp)(n[]) *walkA3_fwd[];
        ∀i3 acc = ∑ (sp)(ne[])*walkA3_fwd[];
        ∀i3 acc = ∑ (sp)(sw[])*walkA3_fwd[];
        ∀i3 acc = ∑ (sp)(s[]) *walkA3_fwd[];
        ∀i3 acc = ∑ (sp)(se[])*walkA3_fwd[];
    }
    [[dot.state(on)]]
    ∀i3 acc = ∑ (sp)(w[]) *walkA3_fwd[];
    tmp[0] = walkx_fwd[]*walkdiag_fwd[];
    walkax_fwd[]   = acc + tmp[0];

  }

  ∀ip {
    ∀i3 nullmem[0]  ← (sp)(e[]) + zz[i3];  // discard
    ∀i3 nullmem[0]  ← (sp)(nw[])  + zz[i3];
    ∀i3 nullmem[0]  ← (sp)(n[]) + zz[i3];
    ∀i3 nullmem[0]  ← (sp)(ne[])  + zz[i3];
  }
}
}

function timerinit() {
  xp t;
  t = timercontrolreg[0];
  t <- t | 4;  // Set counter to record every cycle.
  timercontrolreg[0] <- t;
  timeridx <- 0;
  timerreg[0]=0;
}
function recordtime() {
  // Record consecutive calls into an array.
  // wraps around and starts from the beginning
  // when array is full.  let it   [0,3);
  xp j;
  j <- timeridx & tmask;
  ∀i3 tstamp[j][i3] <- timerreg[i3];

  timeridx <- timeridx + 1;
}


// main
init_stencil_td();
timerinit();

let innz ∈ [0, NNZ);
∀innz ax[innz] = 0; 

// set up dot colors (21=h-reduction, 22=broadcast in teardown)
let i4 ∈ [0, 4);
router_config_regs[21] = router_dot[0];
router_config_regs[22] = router_td;
∀i4 switch_config_regs[21][i4] = switch_dot[i4];

// dot barrier (result unused)
xp a_base = (xp)(x.base);
recordtime();
dot(a_base, a_base);
init_stencil_td();
set_sockets(0);
recordtime();
spmv();
recordtime();

// task 29
#pragma teardown on
{
}

