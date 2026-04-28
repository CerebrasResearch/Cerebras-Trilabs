/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2026 [Redacted for anonymous review]
 * All rights reserved.
 */

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

iterator walkA3_fwd = ∀i24z A[i24z];
iterator walkA2_fwd = ∀i2z A[24*ZL+i2z];

let zp ∈ [0, ZP);
let ip ∈ [0, P);
let inf ∈ [0, ∞);

iterator walkZ2_fwd  = ∀inf ∀zp ∀ip ∀i2 x[zp+ip*ZP+2*i2];
iterator walkZ3_fwd  = ∀inf ∀zp ∀ip ∀i3 x[zp+ip*ZP+i3];
iterator walkZZ3_fwd = ∀inf ∀zp ∀ip ∀i3 x[zp+ip*ZP+1+i3];

iterator walkr_fwd =  ∀inf ∀zp ∀ip r[zp+ip*ZP];
iterator walky_fwd =  ∀inf ∀zp ∀ip y[zp+ip*ZP];
iterator walkinvd_fwd =  ∀inf ∀zp ∀ip invd[zp+ip*ZP];

// spmv 
iterator walkx_fwd =  ∀inf ∀zp ∀ip x[zp+ip*ZP+1];
iterator walkax_fwd =  ∀inf ∀zp ∀ip ax[zp+ip*ZP];
iterator walkdiag_fwd =  ∀inf ∀zp ∀ip diag[zp+ip*ZP];

sp tmp[1];
sp acc = 0;

function gs_fwd() {
#pragma uthread enable off
if (active_stencil[0] > 0) {

  // clearing accum0
  sp zz[3] = {0.};
  ∀i3 acc = ∑ zz[i3] * zz[i3];
  //

  ∀ip {
    ∀i3 wv:tx_upstr[] ← x[ip*ZP+i3]; // (new,old,old) upstream
  }


  reset walkA3_fwd;
  walkA3_fwd.len = 3;

  reset walkA2_fwd;
  walkA2_fwd.len = 2;

  let z ∈ [0, ZL);
  ∀z {

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
    tmp[0] = (walkr_fwd[] - acc);
    walky_fwd[]   = tmp[0] * walkinvd_fwd[];
    ∀i3 wv:tx_downstr[] ← walkZ3_fwd[]; // (new,new,old) downstream
    ∀i3 wv:tx_upstr[] ← walkZZ3_fwd[]; // next (new,old,old) upstream

  }

  ∀ip {
    ∀i3 nullmem[0]  ← (sp)(e[]) + zz[i3];  // discard
    ∀i3 nullmem[0]  ← (sp)(nw[])  + zz[i3];
    ∀i3 nullmem[0]  ← (sp)(n[]) + zz[i3];
    ∀i3 nullmem[0]  ← (sp)(ne[])  + zz[i3];
  }
}
}




function spmv() {
#pragma uthread enable off
if (active_stencil[0] > 0) {

  // clearing accum0
  sp zz[3] = {0.};
  ∀i3 acc = ∑ zz[i3] * zz[i3];
  //

  ∀ip {
    ∀i3 wv:tx_upstr[] ← x[ip*ZP+i3]; // (new,old,old) upstream
  }


  reset walkA3_fwd;
  walkA3_fwd.len = 3;

  reset walkA2_fwd;
  walkA2_fwd.len = 2;

  let z ∈ [0, ZL);
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


