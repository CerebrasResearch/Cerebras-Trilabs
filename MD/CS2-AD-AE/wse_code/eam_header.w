/*
 * Released under BSD 3-Clause License,
 * Copyright (c) 2024 Cerebras Systems Inc.
 * All rights reserved.
 *
 * This is a Tungsten implementation of the EAM algorithm.                 
 */

// Internal mass units are eV * fs^2 / Ang^2
sp const amuToInternalMass = 103.643;
sp const mass = 63.550;
sp const negInvMass = -1.0/(mass*amuToInternalMass);

union wvlt {
    sp sp;
    xp xp[2];
};

/*   Fast inverse square root (Kahan, Ng, 1980s)
 * 
 *   The method uses a few Newton-Raphson iterations starting with a good initial guess. 
 * 
 *   To compute the initial guess, the method reads the binary fp32 representation 
 *   an integer, performs right shift by one and subtracts from a fixed constant to 
 *   approximate dividing by 2 and negating biased exponent.  The resulting integer 
 *   is read back as fp32 to seed the iteration.  It is sufficient to use only top 
 *   16 bit part of the constant, which simplifies WSE implementation
 *
 *   The Newton–Raphson for inverse square root computation is x <- x * (1.5 - 0.5*input*x*x) 
 *   We precompute the constant -0.5*input*x*x and seed the iterations with the 
 *   initial guess define above.  To achieve good accuracy for inverse square root in fp32, 
 *   it is sufficient to run only two iterations. 
 */
function invsqrt(xp len, sp input[], sp output[]) {
    let i ∈ [0, len);

    sp neghalf, inv;
    union wvlt temp;
    xp invhi_magic ← 0x5f37;

    ∀i {
        neghalf ← input[i] * -0.5;

        /* bit hack initial guess */
        temp.sp    ← input[i];
        temp.xp[1] ← temp.xp[1] >> 1;
        temp.xp[1] ← invhi_magic - temp.xp[1];

        inv ← temp.sp;
        inv ← inv*(inv*inv*neghalf + 1.5);
        inv ← inv*(inv*inv*neghalf + 1.5);
        output[i] ← inv;
    }
}

/* segment of quadratic spline */
struct qsegment {
    sp a;
    sp b;
    sp c;
};

/* segment of linear spline */
struct lsegment {
    sp a;
    sp b;
};

/* coordinate space for evaluating splines:
 *   integer index of spline segment and
 *   real offset [0,1) within segment */
struct spline_coord {
    sp x̂;
    xp i;
    xp pad;
};

function get_spline_coord(xp len, sp scale, sp in[], struct spline_coord ret[]) {
    let i ∈ [0, len);

    sp in_fl;

    ∀i {
        ret[i].x̂ ← in[i] * scale;
        ret[i].i ← ret[i].x̂;
        in_fl    ← ret[i].i;
        ret[i].x̂ ← ret[i].x̂ - in_fl;
    }
}
