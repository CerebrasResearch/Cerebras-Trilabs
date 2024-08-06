/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2024 Cerebras Systems Inc.
 * All rights reserved.
 *
 * This is a Tungsten implementation of the EAM algorithm. 
 */

import eam_header;

/* neighborhood multicast sockets in 4 cardinal directions */
sp socket lr;
sp socket rl;
sp socket du;
sp socket ud;

/* socket for external IO */
sp socket tx;

/* candidate neighborhood radius fabric width and height */
xp param rW;
xp param rH;

/* save frequency and number of checkpoints to take */
xp param nsave;
xp param savefreq;

xp param ninteract; /* maximum possible interactions */
xp param nseg;      /* number of interpolation spline segments */
sp param rcut²;     /* cutoff threshold */
sp param Δx⁻¹;      /* ρ and φ spline bin inverse size */
sp param Δρ̅⁻¹;      /* embedding energy spline bin inverse size */
sp param Δt;        /* time step delta */

sp const m⁻¹ = 1/(mass*amuToInternalMass); /* particle mass */
typedef sp ℝ³[3];

/***********************************
 * Neighborhood multicast constants
 ***********************************/
xp const W          = 2*rW-1;
xp const H          = 2*rH-1;
xp const ncandidate = W*H-1;
xp const lefthalf   = rW;
xp const righthalf  = W - lefthalf;
xp const bothalf    = W*rH-1;
xp const tophalf    = ncandidate - bothalf;

xp const payloadsz  = 4;      // in wavelets
xp const h_limit    = payloadsz*rW - 1;
xp const h_maxpass  = h_limit - payloadsz;
xp const v_limit    = payloadsz*W*rH - 1;
xp const du_maxpass = v_limit - payloadsz;
xp const ud_maxpass = v_limit - payloadsz*W;

filter frl[counter] = {.limit=h_limit, .maxpass=h_maxpass,  .tick=wavelet, .socket=rl};
filter fdu[counter] = {.limit=v_limit, .maxpass=du_maxpass, .tick=wavelet, .socket=du};
filter fud[counter] = {.limit=v_limit, .maxpass=ud_maxpass, .tick=wavelet, .socket=ud};

xp const num_ctrl = 2;
/* Multicast control wavelet sequence: ADV_NOCE, ADV_NOCE, RST_NOCE : ADV_NOCE */
wv mcast_ctrl[n=2] = { {0x0001, 0x2b40}, {0x0000, 0x0140} };

/* Particle state */
ℝ³ r⃗₀;
ℝ³ v⃗₀;
ℝ³ f⃗₀;

/* Struct for neighborhood particle info */
union neighborhood {
    sp full[ncandidate][payloadsz];
    struct {
        sp top[tophalf][payloadsz];
        sp bot[bothalf][payloadsz];
    } half;
};

/* Interpolation tables for ρ, φ, and force */
struct qsegment  ρ_table[c=nseg];
struct lsegment  φ_table[c=nseg];
struct lsegment  f_table[c=nseg];

/* Coefficients of quadratic spline are looked up from
 * interpolation tables for the ij'th particle interaction */
struct data {
    struct qsegment ρ;  /* density contribution from neighboring particle */
    struct lsegment φ;  /* potential contribution with neighbor */
} cf[ncandidate];

/* Set configuration register for fp rounding towards -inf */
xp fpcw[n=1] address(0x7e0c) = { 0x03c2 };

/* Iterators */
xp nhit;
let d  ∈ [0, 3);
let k  ∈ [0, nhit);
let l0 ∈ [0, nsave);
let l1 ∈ [0, savefreq);


function neighborhood_transfer(sp payload[], union neighborhood nhood) {
    union {
        struct {
            sp left[lefthalf][payloadsz];
            sp right[righthalf][payloadsz];
        } half;
        sp full[W][payloadsz];
    } row;

    let s ∈ [0, num_ctrl);
    let l ∈ [0, lefthalf);
    let r ∈ [0, righthalf);
    let w ∈ [0, W);
    let b ∈ [0, bothalf);
    let t ∈ [0, tophalf);
    let p ∈ [0, payloadsz);

    /* Process horizontal transfer */
    parallel {
        {
            ∀p lr[] ← payload[p];
            ∀s lr[] ← control(mcast_ctrl[s]);
        }
        {
            ∀p rl[] ← payload[p];
            ∀s rl[] ← control(mcast_ctrl[s]);
        }
        ∀l ∀p row.half.left[l][p]  ← lr[];
        ∀r ∀p row.half.right[r][p] ← rl[];
    }

    /* Process vertical transfer */
    parallel {
        {
            ∀w ∀p du[] ← row.full[w][p];
            ∀s du[] ← control(mcast_ctrl[s]);
        }
        {
            ∀w ∀p ud[] ← row.full[w][p];
            ∀s ud[] ← control(mcast_ctrl[s]);
        }
        ∀b ∀p nhood.half.bot[b][p] ← du[];
        ∀t ∀p nhood.half.top[t][p] ← ud[];
    }
}

function detect(sp r²[], xp idx[], sp disp[][]) {
    let i ∈ [0, ncandidate);

    union neighborhood candidates;
    neighborhood_transfer(r⃗₀, candidates);

    /* Filter candidates based on distance threshold */
    ∀d ∀i candidates.full[i][d] ← r⃗₀[d] - candidates.full[i][d];

    xp cnt ← 0;
    ∀i {
        sp dist ← 0.;
        ∀d dist ←̟ candidates.full[i][d] * candidates.full[i][d];

        if (dist < rcut²) {
            ∀d disp[cnt][d] ← candidates.full[i][d];
            r²[cnt]  ← dist;
            idx[cnt] ← i;
            cnt      ←̟ 1;
        }
    }
    return cnt;
}

main {
    ∀l0 {
        ∀l1 {
            /* In leapfrog integration, velocity is a half timestep ahead
                * of position, during position update. So vel is initialized
                * at timestep 0.5 and the position is updated first */
            ∀d r⃗₀[d] ←̟ v⃗₀[d] * Δt;

            sp r²[ninteract];   /* squared distance */
            xp idx[ninteract];  /* sparse index of interaction particle */
            ℝ³ r⃗[ninteract];    /* displacement with interaction particle */
            nhit ← detect(r², idx, r⃗);

            sp r⁻¹[ninteract];  /* inverse distance */
            sp r[ninteract];    /* distance */
            invsqrt(nhit, r², r⁻¹);
            ∀k r[k] ← r²[k] * r⁻¹[k];

            /* Interpolate on distance by computing spline segment and offset */
            struct spline_coord x[ninteract];
            get_spline_coord(nhit, Δx⁻¹, r, x);

            /* Lookup values from interpolation table */
            ∀k {
                cf[k].ρ ← ρ_table[x[k].i];
                cf[k].φ ← φ_table[x[k].i];
            }

            sp ρ̅[1];         /* Electronic environment of local particle */
            ρ̅[0] ← 0.0;
            sp ρ[ninteract];    /* ρ(r)  */
            ∀k ρ[k] ← x[k].x̂ * (cf[k].ρ.b + x[k].x̂ * cf[k].ρ.a) + cf[k].ρ.c;
            ∀k ρ̅[0] ←̟ ρ[k];

            sp ρ′[ninteract];   /* ρ'(r)  */
            sp φ′[ninteract];   /* φ'(r)  */
            ∀k ρ′[k] ← (cf[k].ρ.b + x[k].x̂ * cf[k].ρ.a * 2.0) * Δx⁻¹;
            ∀k φ′[k] ← (cf[k].φ.b + x[k].x̂ * cf[k].φ.a * 2.0) * Δx⁻¹;

            /* Interpolate on density environment by computing spline segment and offset */
            struct spline_coord y;
            get_spline_coord(1, Δρ̅⁻¹, ρ̅, y);

            ℝ³ f′;  /* Pad force gradient to ℝ³ for multicast */
            f′[0] ← (f_table[y.i].b + y.x̂ * f_table[y.i].a * 2.0) * Δρ̅⁻¹;

            /* Exchange force gradients with neighborhood */
            union neighborhood forces;
            neighborhood_transfer(f′, forces);

            /* Sum the force acting on particle. Only use neighboring gradients within threshold */
            ∀k f⃗₀[k] ← 0.0;
            sp f[ninteract];
            ∀k f[k] ← φ′[k];
            sp temp ← f′[0];
            ∀k f[k] ←̟ ρ′[k]*temp;
            ∀k {
                sp tempsp ← forces.full[idx[k]][0];
                f[k] ←̟ ρ′[k]*tempsp;
            }
            ∀k f[k] ← f[k] * r⁻¹[k];
            ∀d {
                xp id ← d;
                ∀k f⃗₀[id] ←̟ r⃗[k][id] * f[k];
            }

            /* Update particle velocities */
            ∀d f⃗₀[d] ← f⃗₀[d] * -m⁻¹;
            ∀d v⃗₀[d] ←̟ f⃗₀[d] * Δt;
        }

        /* Send particle position and velocity at current full timestep.
         * Need to subtract half a timestep from velocity due to leapfrog */
        ∀d tx[] ← r⃗₀[d];
        ∀d tx[] ← v⃗₀[d] + f⃗₀[d] * -Δt/2.0;
        tx.flip;
    }
}
