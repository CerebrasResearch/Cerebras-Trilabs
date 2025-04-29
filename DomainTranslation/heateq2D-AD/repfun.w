/*
 * Released under BSD 3-Clause License,
 * Copyright (c) 2025 Cerebras Systems Inc.
 * All rights reserved.
 */

// Functions for reciprocal square root, inverse, and square root:
//
//   rsqrt(sp x) -> 1/sqrt(x)
//   inv(sp x)   -> 1/x
//   inv2(sp x)  -> 1/x [*]
//   sqrt(sp x)  -> sqrt(x)
//
// They are all based on an approximation of 1/sqrt(x) [except
// inv2]. These functions do not handle denormals, infinities, or nans
// correcly. rsqrt(x) and sqrt(x) produce non-nan values for x < 0.
// There is a check in sqrt(x) so that sqrt(+-0) = 0. inv(+-0) and
// inv2(+-0) return nan.
//
// Over the normal range [ 2^(-126) , 2^128 ) results have a relative
// error of < ~1 ulp. If fma is used where indicated, results have
// maximum accruacy with a relative error of <= 0.5ulp.
//
// [*] inv2(x) computes 1/x and is faster than inv(x). However,
// its range is limited to 2^(-126) <= abs(x) < 2^126, an therefore give
// incorrect results near +-inf.
//
//    Written by Tomas Oppelstrup, February 2025
//

union spbits {
  sp r;
  xp v[2];
};

function rsqrt_est(sp x) {
  // Trick by Kahan et al.: the bit layout of an fp number x is such
  // that if the bits of x is treated as an integer, it is a fair
  // approximation of log2(x)*2^23. If we right shift by one, that's
  // an approxiamtion of 0.5*log2(x)*2^23 = log2(sqrt(x)*2*23. The
  // 'magic' number is to account for the exponent bias, and includes
  // an offset to minimize the maximum relative error of the
  // approximation.
  xp const magic = 0x5f37;
  sp xm2 = -0.5*x;
  xp t;
  union spbits u;
  //u.v[1] <- magic - (u.v[1] >> 1);
  u.r <- x;
  t <- -(u.v[1] >> 1);
  u.v[1] <- magic + t;

  sp y = u.r;
  // After two Newton iterations most of the bits are correct (for
  // fp32).
  y <- y*(y*y*xm2 + 1.5);
  y <- y*(y*y*xm2 + 1.5);
  // Max relative error with exp in [1,254] (whole normal range)
  // is 2^(-16.176) = 1.350e-05.
  return y;
}

function rsqrt(sp x) {
  sp y,t,e;
  y <- rsqrt_est(x);
  t <- x*y;
  e <- x*y - t; // (1)
  sp f1;
  f1 <- y + (0.5*y)*((1.0 - t*y) - e*y);
  //                  ^^^^^^^^^--<-- (2)
  // fma at (1) and (2) will yield correctly rounded result.
  // If you are not using fma, you can set e = 0 and simplify; still
  // don't expand the (1.0 - t*y) parenthesis.
  return f1;
}

function inv(sp x) {
  sp absx,y,s;
  union spbits u;
  xp sgn,t;
  u.r = x;
  sgn <- u.v[1] & 32768; // Save sign bit
  t <- u.v[1] & 32767;   // Remove sign bit
  u.v[1] <- t;
  absx = u.r;
  s <- rsqrt_est(absx);      // Compute 1/sqrt(x) estimate
  y <- s*s;                  // 1/x estimate
  y <- y + y*(1.0 - absx*y); // One Newton iteration 
  //          ^^^^^^^^^^^^^--<-- fma here will get correctly
  //                             rounded result.
  u.r <- y;
  t <- u.v[1] + sgn; // Put original sign back into result
  u.v[1] <- t;
  y <- u.r;
  return y;
}

function inv2(sp x) {
  sp y;
  xp const magic = 32499;
  union spbits u;
  u.r <- x;
  // Same idea as for rsqrt_est, just diffrent (optimized) constant
  // appropriate for 1/x instead of 1/sqrt(x).

  //u.v[1] <- (254*128 - 13) - u.v[1];
  xp t;
  t <- -u.v[1];
  u.v[1] <- magic + t;

  y <- u.r;
  y <- y*(2.0 - x*y);
  y <- y*(2.0 - x*y);  // <==> y + y*(1-xy)

  // relative error in y is < 1.5e-5 < 2^(-16).
  sp f1,f2;
  f1 <- 1.0 - x*y; // fma here will get correcly rounded result
  f2 <- y + y*f1;
  return f2;
  // Correct with exp in [1,251]
}

function sqrt(sp x) {
  sp y,t,e;

  if(x == 0.0) {
    y <- 0.0;
  } else {
    y <- rsqrt_est(x);
    t <- x*y;
    e <- x*y - t; // (1)
    
    y <- t + (e + (0.5*t)*((1.0 - t*y) - e*y));
    //                      ^^^^^^^^^--<-- (2)
    // fma at (1) and (2) will get correctly rounded result.
  }
  return y;
}


/*
 *
 *  // Test program
 *  xp param nmax;
 *  sp xx[nmax];
 *  sp yy[n=nmax][m=5];
 *
 *  main {
 *    let i ∈ [0,nmax);
 *
 *    ∀i { yy[i][0] <- rsqrt_est(xx[i]); }
 *    ∀i { yy[i][1] <- rsqrt(xx[i]); }
 *    ∀i { yy[i][2] <- inv(xx[i]); }
 *    ∀i { yy[i][3] <- inv2(xx[i]); }
 *    ∀i { yy[i][4] <- sqrt(xx[i]); }
 *  }
 *
 */
