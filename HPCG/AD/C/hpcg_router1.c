/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2026 Cerebras Systems Inc.
 * All rights reserved.
 */

/*
 * Standalone version of hpcg_router_03_17.c
 * Compile: gcc -std=c99 -o hpcg_router1 hpcg_router1.c
 * Usage:   ./hpcg_router1 NX=<int> NY=<int> [NGS=<int>] [level=<int>]
 *                         [mode=fwd|bwd|sym|spmv|gs_spmv|tdtest|vcycle1|vcycle2|vcycle3|cg|hpcg]
 *                         [out_split=1|2] [routing_dir=clock|counter|mixed|new]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>

typedef uint16_t u16;
typedef uint32_t u32;

#define BIT(n) (1ULL<<(n))

#define fatalif(pred, s, ...) \
  (void)((pred) ? \
    fflush(0), fprintf(stderr, s "\n", ##__VA_ARGS__), \
    exit(1), 0 : 0)

/* ================================================================
 * Routing bit definitions
 * ================================================================ */

#define IN_C BIT(7)              // 100
#define IN_D (BIT(6) | BIT(5))   // 011
#define IN_U BIT(6)              // 010
#define IN_R BIT(5)              // 001
#define IN_L 0                   // 000

#define IN_MASK (BIT(7) | BIT(6) | BIT(5))
#define NOT_IN  (BIT(7) | BIT(5))

#define OUT_C BIT(4)
#define OUT_D BIT(3)
#define OUT_U BIT(2)
#define OUT_R BIT(1)
#define OUT_L BIT(0)

// Routing pattern: 3x3 stencil grids encoding input source + output destinations
//   Layout:  [0][0]=NE  [0][1]=N   [0][2]=NW
//            [1][0]=E   [1][1]=C   [1][2]=W
//            [2][0]=SE  [2][1]=S   [2][2]=SW
struct routing_pattern {
  u16 full[3][3];  // single-pass stencil
  u16 down[3][3];  // split: first pass (S/SW/SE/W neighbors + center broadcast)
  u16 up[3][3];    // split: second pass (N/NW/NE/E neighbors + center broadcast)
};

// order of 8 inputs and 1 output
enum {
  SRC_SW = 0,
  SRC_S,
  SRC_SE,
  SRC_W,
  SRC_E,
  SRC_NW,
  SRC_N,
  SRC_NE,
  SRC_OUT,
};

enum mode {
  FWD=0,
  BWD,
  SYM,
  SPMV,
  GS_SPMV,
  TDTEST,
  VCYCLE1,
  VCYCLE2,
  VCYCLE3,
  CG,
  HPCG,
};

enum routing_dir {
  CLOCK=0,
  COUNTER,
  MIXED,
  NEW,
};

/* ================================================================
 * Command-line option parsing (standalone replacement)
 * ================================================================ */

struct option {
  int NY, NX, NGS, level, mode, out_split, routing_dir;
};

static struct option opt = {
  .out_split = 1,
};

struct enum_entry { const char *name; int val; };

static const struct enum_entry mode_map[] = {
  {"fwd", FWD}, {"bwd", BWD}, {"sym", SYM}, {"spmv", SPMV},
  {"gs_spmv", GS_SPMV}, {"tdtest", TDTEST},
  {"vcycle1", VCYCLE1}, {"vcycle2", VCYCLE2}, {"vcycle3", VCYCLE3},
  {"cg", CG}, {"hpcg", HPCG}, {NULL, 0}
};

static const struct enum_entry routing_dir_map[] = {
  {"clock", CLOCK}, {"counter", COUNTER}, {"mixed", MIXED}, {"new", NEW}, {NULL, 0}
};

static int parse_enum(const char *val, const struct enum_entry *map, const char *name) {
  for (int i = 0; map[i].name; i++)
    if (strcmp(val, map[i].name) == 0)
      return map[i].val;
  fprintf(stderr, "Unknown value '%s' for option '%s'\n", val, name);
  fprintf(stderr, "Valid values:");
  for (int i = 0; map[i].name; i++)
    fprintf(stderr, " %s", map[i].name);
  fprintf(stderr, "\n");
  exit(1);
}

static void parse_options(int argc, char **argv) {
  int got_NY = 0, got_NX = 0;
  for (int i = 1; i < argc; i++) {
    char *eq = strchr(argv[i], '=');
    if (!eq) {
      fprintf(stderr, "Invalid argument '%s' (expected KEY=VALUE)\n", argv[i]);
      exit(1);
    }
    int klen = (int)(eq - argv[i]);
    char key[128];
    if (klen >= (int)sizeof(key)) klen = (int)sizeof(key) - 1;
    memcpy(key, argv[i], klen);
    key[klen] = '\0';
    const char *val = eq + 1;

    if (strcmp(key, "NY") == 0) { opt.NY = atoi(val); got_NY = 1; }
    else if (strcmp(key, "NX") == 0) { opt.NX = atoi(val); got_NX = 1; }
    else if (strcmp(key, "NGS") == 0) { opt.NGS = atoi(val); }
    else if (strcmp(key, "level") == 0) { opt.level = atoi(val); }
    else if (strcmp(key, "mode") == 0) { opt.mode = parse_enum(val, mode_map, "mode"); }
    else if (strcmp(key, "out_split") == 0) { opt.out_split = atoi(val); }
    else if (strcmp(key, "routing_dir") == 0) { opt.routing_dir = parse_enum(val, routing_dir_map, "routing_dir"); }
    else {
      fprintf(stderr, "Unknown option '%s'\n", key);
      exit(1);
    }
  }
  fatalif(!got_NY, "Required option NY not specified");
  fatalif(!got_NX, "Required option NX not specified");
}

/* ================================================================
 * Routing patterns
 * ================================================================ */

#define P(in, outs) ((u16)(in) | (u16)(outs))

static const struct routing_pattern routing_patterns[] = {
  [CLOCK] = {
    .full = {
      {P(IN_R, OUT_C),            P(IN_U, OUT_C|OUT_L),              P(IN_U, OUT_C)},
      {P(IN_R, OUT_C|OUT_U),      P(IN_C, OUT_L|OUT_R|OUT_U|OUT_D), P(IN_L, OUT_C|OUT_D)},
      {P(IN_D, OUT_C),            P(IN_D, OUT_C|OUT_R),              P(IN_L, OUT_C)},
    },
    .down = {
      {NOT_IN,                    NOT_IN,                            NOT_IN},
      {P(IN_R, OUT_U),            P(IN_C, OUT_L|OUT_R|OUT_U),       P(IN_L, OUT_C)},
      {P(IN_D, OUT_C),            P(IN_D, OUT_C|OUT_R),             P(IN_L, OUT_C)},
    },
    .up = {
      {P(IN_R, OUT_C),            P(IN_U, OUT_C|OUT_L),              P(IN_U, OUT_C)},
      {P(IN_R, OUT_C),            P(IN_C, OUT_L|OUT_R|OUT_D),        P(IN_L, OUT_D)},
      {NOT_IN,                    NOT_IN,                            NOT_IN},
    },
  },
  [COUNTER] = {
    .full = {
      {P(IN_U, OUT_C),            P(IN_U, OUT_C|OUT_R),              P(IN_L, OUT_C)},
      {P(IN_R, OUT_C|OUT_D),      P(IN_C, OUT_L|OUT_R|OUT_U|OUT_D), P(IN_L, OUT_C|OUT_U)},
      {P(IN_R, OUT_C),            P(IN_D, OUT_C|OUT_L),              P(IN_D, OUT_C)},
    },
    .down = {
      {NOT_IN,                    NOT_IN,                            NOT_IN},
      {NOT_IN,                    P(IN_C, OUT_R|OUT_U),              P(IN_L, OUT_C|OUT_U)},
      {P(IN_R, OUT_C),            P(IN_D, OUT_C|OUT_L),             P(IN_D, OUT_C)},
    },
    .up = {
      {P(IN_U, OUT_C),            P(IN_U, OUT_C|OUT_R),              P(IN_L, OUT_C)},
      {P(IN_R, OUT_C|OUT_D),      P(IN_C, OUT_L|OUT_D),              NOT_IN},
      {NOT_IN,                    NOT_IN,                            NOT_IN},
    },
  },
  [MIXED] = {
    .full = {  // same as clockwise
      {P(IN_R, OUT_C),            P(IN_U, OUT_C|OUT_L),              P(IN_U, OUT_C)},
      {P(IN_R, OUT_C|OUT_U),      P(IN_C, OUT_L|OUT_R|OUT_U|OUT_D), P(IN_L, OUT_C|OUT_D)},
      {P(IN_D, OUT_C),            P(IN_D, OUT_C|OUT_R),              P(IN_L, OUT_C)},
    },
    .down = {
      {NOT_IN,                    NOT_IN,                            NOT_IN},
      {NOT_IN,                    P(IN_C, OUT_R|OUT_U),              P(IN_L, OUT_C)},
      {P(IN_R, OUT_C),            P(IN_D, OUT_C|OUT_R|OUT_L),       P(IN_L, OUT_C)},
    },
    .up = {
      {P(IN_R, OUT_C),            P(IN_U, OUT_C|OUT_L|OUT_R),        P(IN_L, OUT_C)},
      {P(IN_R, OUT_C),            P(IN_C, OUT_L|OUT_D),              NOT_IN},
      {NOT_IN,                    NOT_IN,                            NOT_IN},
    },
  },
  [NEW] = {
    .full = {  // same as clockwise
      {P(IN_R, OUT_C),            P(IN_U, OUT_C|OUT_L),              P(IN_U, OUT_C)},
      {P(IN_R, OUT_C|OUT_U),      P(IN_C, OUT_L|OUT_R|OUT_U|OUT_D), P(IN_L, OUT_C|OUT_D)},
      {P(IN_D, OUT_C),            P(IN_D, OUT_C|OUT_R),              P(IN_L, OUT_C)},
    },
    .down = {
      {NOT_IN,                    NOT_IN,                            NOT_IN},
      {P(IN_R, OUT_U),            P(IN_C, OUT_R|OUT_L),              P(IN_L, OUT_C)},
      {P(IN_D, OUT_C|OUT_R),      P(IN_L, OUT_C|OUT_R),              P(IN_L, OUT_C)},
    },
    .up = {
      {P(IN_R, OUT_C|OUT_U),      P(IN_U, OUT_C|OUT_L|OUT_R),        P(IN_L, OUT_C)},
      {P(IN_D, OUT_C),            P(IN_C, OUT_D),                    NOT_IN},
      {NOT_IN,                    NOT_IN,                            NOT_IN},
    },
  },
};

#undef P

// Validate that a routing stencil is self-consistent: for every output direction
// in the 3x3 grid, the adjacent position (if it exists) must either have
// a matching input direction or be NOT_IN.  An output to a neighbor that has
// a different active input (e.g. OUT_L but neighbor has IN_D) would create
// an unroutable link when spacers are inserted.
static void validate_stencil(const char *name, const char *field, const u16 grid[3][3]) {
  // Stencil x-axis is inverted vs grid: col 0=east, col 2=west.
  // Stencil y-axis is also inverted: row 0=north, row 2=south.
  // OUT_R (grid +x) → stencil dc=+1, OUT_L (grid -x) → stencil dc=-1,
  // OUT_U (grid +y) → stencil dr=+1, OUT_D (grid -y) → stencil dr=-1.
  static const struct { u16 out_bit; int dr, dc; u16 expected_in; const char *dir; } checks[] = {
    {OUT_R,  0, +1, IN_L, "R"},
    {OUT_L,  0, -1, IN_R, "L"},
    {OUT_U, +1,  0, IN_D, "U"},
    {OUT_D, -1,  0, IN_U, "D"},
  };
  for (int r=0; r<3; ++r)
  for (int c=0; c<3; ++c) {
    if ((grid[r][c] & IN_MASK) == NOT_IN) continue;
    for (int k=0; k<4; ++k) {
      if (!(grid[r][c] & checks[k].out_bit)) continue;
      int nr = r + checks[k].dr, nc = c + checks[k].dc;
      if (nr < 0 || nr >= 3 || nc < 0 || nc >= 3) continue;
      u16 nb_in = grid[nr][nc] & IN_MASK;
      if (nb_in == NOT_IN) continue;  // neighbor inactive, OK
      fatalif(nb_in != checks[k].expected_in,
        "routing pattern '%s' .%s: [%d][%d] has OUT_%s but [%d][%d] has IN=%d, expected IN=%d",
        name, field, r, c, checks[k].dir, nr, nc, nb_in >> 5, checks[k].expected_in >> 5);
    }
  }
}

static void validate_routing_patterns(void) {
  struct { int id; const char *name; } pats[] = {
    {CLOCK, "clock"}, {COUNTER, "counter"}, {MIXED, "mixed"}, {NEW, "new"},
  };
  for (int i=0; i < (int)(sizeof(pats)/sizeof(pats[0])); ++i) {
    const struct routing_pattern *p = &routing_patterns[pats[i].id];
    validate_stencil(pats[i].name, "full", p->full);
    validate_stencil(pats[i].name, "down", p->down);
    validate_stencil(pats[i].name, "up",   p->up);
  }
}

/* ================================================================
 * Fabric data structures
 * ================================================================ */

struct fabric {
  u16 *router_1d;       // NY*NX*9*out_split
  u16 *colors_1d;       // NY*NX*9
  u16 *boundary_1d;     // NY*NX*9
  u32 *tdmask_1d;       // NY*NX
  u16 *active_tiles_1d; // NY*NX
};

void fabric_init(int NY, int NX, struct fabric *f) {
  f->router_1d       = (u16*)calloc(NY*NX*9*opt.out_split, sizeof(u16));
  f->colors_1d       = (u16*)calloc(NY*NX*9, sizeof(u16));
  f->boundary_1d     = (u16*)calloc(NY*NX*9, sizeof(u16));
  f->tdmask_1d       = (u32*)calloc(NY*NX, sizeof(u32));
  f->active_tiles_1d = (u16*)calloc(NY*NX, sizeof(u16));
}

void fabric_fini(struct fabric *f) {
  free(f->router_1d);
  free(f->colors_1d);
  free(f->boundary_1d);
  free(f->tdmask_1d);
  free(f->active_tiles_1d);
}

// arrays for file dumps
struct file_arrays {
  u16 *router_file, *boundary_file, *colors_file, *active_tiles_file;
  u32 *tdmask_file;
} file_arrays;

void file_arrays_init(int NY, int NX, int context_count) {
  file_arrays.router_file       = (u16*)calloc(NY*NX*context_count*9*opt.out_split, sizeof(u16));
  file_arrays.boundary_file     = (u16*)calloc(NY*NX*context_count*9, sizeof(u16));
  file_arrays.colors_file       = (u16*)calloc(NY*NX*context_count*9, sizeof(u16));
  file_arrays.active_tiles_file = (u16*)calloc(NY*NX*context_count, sizeof(u16));
  file_arrays.tdmask_file       = (u32*)calloc(NY*NX*context_count, sizeof(u32));
}

void file_arrays_fini() {
  free(file_arrays.router_file);
  free(file_arrays.boundary_file);
  free(file_arrays.colors_file);
  free(file_arrays.active_tiles_file);
  free(file_arrays.tdmask_file);
}


// 3x3 circular permutations of the stencil for nine colors
void configure_router_coarse(int NYC, int NXC, struct fabric *fabric_coarse,
                             const struct routing_pattern *pat) {
  int nslots = 9 * opt.out_split;
  u16 (*router)[NXC][nslots] = (void*)(fabric_coarse->router_1d);

  for (int iy=0; iy<NYC; ++iy)
  for (int ix=0; ix<NXC; ++ix)
    for (int col=0; col<nslots; ++col)
      router[iy][ix][col] = NOT_IN;

  for (int col=0; col<9; ++col) {
    int colorx = col % 3;
    int colory = (col / 3) % 3;

    for (int iy=0; iy<NYC; ++iy)
    for (int ix=0; ix<NXC; ++ix) {
      int sy = (iy+3-colory) % 3, sx = (ix+3-colorx) % 3;
      if (opt.out_split == 1) {
        router[iy][ix][col] = pat->full[sy][sx];
      } else {
        router[iy][ix][col]   = pat->up[sy][sx];
        router[iy][ix][col+9] = pat->down[sy][sx];
      }
    }
  }
}


// Map SRC_XX enum to stencil [row][col]:
//   SW=>[2][2], S=>[2][1], SE=>[2][0], W=>[1][2],
//   E=>[1][0], NW=>[0][2], N=>[0][1], NE=>[0][0], OUT=>[1][1]
static const int src_row[] = {2, 2, 2, 1, 1, 0, 0, 0, 1};
static const int src_col[] = {2, 1, 0, 2, 0, 2, 1, 0, 1};

// Compute color assignment: for each tile, which color slot carries each direction.
// The stencil is tiled with 3x3 circular shifts, so the mapping is purely positional.
void configure_colors_coarse(int NYC, int NXC, u16 *colors_coarse_1d) {
  u16 (*colors_coarse)[NXC][9] = (void*)colors_coarse_1d;
  for (int iy=0; iy<NYC; ++iy)
  for (int ix=0; ix<NXC; ++ix) {
    for (int src=0; src<9; ++src) {
      int col = ((iy + 3 - src_row[src]) % 3) * 3 + (ix + 3 - src_col[src]) % 3;
      colors_coarse[iy][ix][src] = col;
    }
    if (opt.out_split == 2) {
      colors_coarse[iy][ix][SRC_SW] += 9;
      colors_coarse[iy][ix][SRC_S]  += 9;
      colors_coarse[iy][ix][SRC_SE] += 9;
      colors_coarse[iy][ix][SRC_W]  += 9;
    }
  }
}

// Remove routing links that cross GS domain boundaries,
// mark corresponding slots in the boundary array.
void fix_boundary_coarse(int NYC, int NXC, int NGSYC, int NGSXC, u16 *router_coarse_1d, u16 *boundary_coarse_1d,
                         const struct routing_pattern *pat) {

  int nslots = 9 * opt.out_split;
  u16 (*router)[NXC][nslots] = (void*)router_coarse_1d;
  u16 (*boundary)[NXC][9] = (void*)boundary_coarse_1d;

// Pass 1: Mark geometric boundary.
// A neighbor is "boundary" when it would be across a GS domain edge.
  for (int iy=0; iy<NYC; ++iy)
  for (int ix=0; ix<NXC; ++ix) {
    for (int ir=0; ir<9; ++ir)
      boundary[iy][ix][ir] = 0;

    int bot   = (iy % NGSYC) == 0;
    int top   = (iy % NGSYC) == NGSYC-1;
    int left  = (ix % NGSXC) == 0;
    int right = (ix % NGSXC) == NGSXC-1;

    if (bot || left)  boundary[iy][ix][SRC_SW] = 1;
    if (bot)          boundary[iy][ix][SRC_S]  = 1;
    if (bot || right) boundary[iy][ix][SRC_SE] = 1;
    if (left)         boundary[iy][ix][SRC_W]  = 1;
    if (right)        boundary[iy][ix][SRC_E]  = 1;
    if (top || left)  boundary[iy][ix][SRC_NW] = 1;
    if (top)          boundary[iy][ix][SRC_N]  = 1;
    if (top || right) boundary[iy][ix][SRC_NE] = 1;
  }

// Pass 2: Sever router links at domain boundaries + cascade orphaned relays.
  for (int iy=0; iy<NYC; ++iy)
  for (int ix=0; ix<NXC; ++ix) {
    int bot   = (iy % NGSYC) == 0;
    int top   = (iy % NGSYC) == NGSYC-1;
    int left  = (ix % NGSXC) == 0;
    int right = (ix % NGSXC) == NGSXC-1;
    if (!bot && !top && !left && !right) continue;

    for (int col=0; col<nslots; ++col) {
      u16 *r = &router[iy][ix][col];
      u16 in_dir = *r & IN_MASK;

      // Sever input that arrives from across a domain boundary
      if ((bot && in_dir == IN_D) || (top && in_dir == IN_U) ||
          (left && in_dir == IN_L) || (right && in_dir == IN_R)) {
        *r = (*r & ~(u16)IN_MASK) | NOT_IN;
      }

      // Sever outputs that would cross a domain boundary
      if (bot)   *r &= ~(u16)OUT_D;
      if (top)   *r &= ~(u16)OUT_U;
      if (left)  *r &= ~(u16)OUT_L;
      if (right) *r &= ~(u16)OUT_R;

      // Cascade: if this entry lost its input, it can't relay anything.
      // Disconnect its remaining outputs and the downstream tiles they feed.
      if ((*r & IN_MASK) != NOT_IN) continue;
      *r &= ~(u16)OUT_C;

      if (*r & OUT_L) {
        assert(ix > 0);
        *r &= ~(u16)OUT_L;
        u16 *nb = &router[iy][ix-1][col];
        *nb = (*nb & ~(u16)IN_MASK) | NOT_IN;
        *nb &= ~(u16)OUT_C;
        if (col < 9 && (pat->up[0][0] & IN_MASK) == IN_R)
          boundary[iy][ix-1][SRC_NE] = 1;
        if (col >= 9 && (pat->down[2][0] & IN_MASK) == IN_R)
          boundary[iy][ix-1][SRC_SE] = 1;
      }
      if (*r & OUT_R) {
        assert(ix < NXC-1);
        *r &= ~(u16)OUT_R;
        u16 *nb = &router[iy][ix+1][col];
        *nb = (*nb & ~(u16)IN_MASK) | NOT_IN;
        *nb &= ~(u16)OUT_C;
        if (col < 9 && (pat->up[0][2] & IN_MASK) == IN_L)
          boundary[iy][ix+1][SRC_NW] = 1;
        if (col >= 9 && (pat->down[2][2] & IN_MASK) == IN_L)
          boundary[iy][ix+1][SRC_SW] = 1;
      }
      if (*r & OUT_D) {
        assert(iy > 0);
        *r &= ~(u16)OUT_D;
        u16 *nb = &router[iy-1][ix][col];
        *nb = (*nb & ~(u16)IN_MASK) | NOT_IN;
        *nb &= ~(u16)OUT_C;
        if (col < 9 && (pat->up[0][0] & IN_MASK) == IN_D)
          boundary[iy-1][ix][SRC_NE] = 1;
        if (col < 9 && (pat->up[0][2] & IN_MASK) == IN_D)
          boundary[iy-1][ix][SRC_NW] = 1;
      }
      if (*r & OUT_U) {
        assert(iy < NYC-1);
        *r &= ~(u16)OUT_U;
        u16 *nb = &router[iy+1][ix][col];
        *nb = (*nb & ~(u16)IN_MASK) | NOT_IN;
        *nb &= ~(u16)OUT_C;
        if (col >= 9 && (pat->down[2][0] & IN_MASK) == IN_U)
          boundary[iy+1][ix][SRC_SE] = 1;
        if (col >= 9 && (pat->down[2][2] & IN_MASK) == IN_U)
          boundary[iy+1][ix][SRC_SW] = 1;
      }
    }
  }

// Pass 3: Reconnect orphaned tiles whose indirect paths were severed by boundaries.
// Iteratively propagate from generators through relay chains: reconnecting one
// tile may enable reconnecting downstream tiles that depend on it.
  for (int reconnect_changed = 1; reconnect_changed; ) {
    reconnect_changed = 0;
    for (int iy=0; iy<NYC; ++iy)
    for (int ix=0; ix<NXC; ++ix) {
      int bot   = (iy % NGSYC) == 0;
      int top   = (iy % NGSYC) == NGSYC-1;
      int left  = (ix % NGSXC) == 0;
      int right = (ix % NGSXC) == NGSXC-1;

      for (int col=0; col<nslots; ++col) {
        u16 *r = &router[iy][ix][col];
        if ((*r & IN_MASK) != NOT_IN) continue;

        // Was this tile originally active in the pattern?
        int cidx = col % 9;
        int colorx = cidx % 3, colory = cidx / 3;
        int sy = (iy + 3 - colory) % 3, sx = (ix + 3 - colorx) % 3;
        u16 orig;
        if (opt.out_split == 1)
          orig = pat->full[sy][sx];
        else
          orig = (col < 9) ? pat->up[sy][sx] : pat->down[sy][sx];
        if ((orig & IN_MASK) == NOT_IN) continue;  // was never active

        // Restore original outputs minus boundary-crossing directions
        u16 orig_outs = orig & 0x1F;
        if (bot)   orig_outs &= ~(u16)OUT_D;
        if (top)   orig_outs &= ~(u16)OUT_U;
        if (left)  orig_outs &= ~(u16)OUT_L;
        if (right) orig_outs &= ~(u16)OUT_R;

        // Look for a neighbor that can feed us (generator or active relay).
        static const struct { int dy, dx; u16 in_dir, out_dir; } dirs[] = {
          { 0, +1, IN_R, OUT_L},
          { 0, -1, IN_L, OUT_R},
          {+1,  0, IN_U, OUT_D},
          {-1,  0, IN_D, OUT_U},
        };

        for (int d = 0; d < 4; ++d) {
          if ((d==0 && right) || (d==1 && left) || (d==2 && top) || (d==3 && bot))
            continue;
          int ny = iy + dirs[d].dy, nx = ix + dirs[d].dx;
          if (ny < 0 || ny >= NYC || nx < 0 || nx >= NXC) continue;
          u16 *nb = &router[ny][nx][col];
          if ((*nb & IN_MASK) == NOT_IN) continue;

          if (*nb & dirs[d].out_dir) {
            // Neighbor already outputs toward us (relay or generator)
            *r = dirs[d].in_dir | orig_outs;
            reconnect_changed = 1;
            break;
          } else if ((*nb & IN_MASK) == IN_C) {
            // Generator without output toward us: add it
            *nb |= dirs[d].out_dir;
            *r = dirs[d].in_dir | orig_outs;
            reconnect_changed = 1;
            break;
          }
        }
      }
    }
  }

// Pass 3b: Clear cascade-set boundary flags for successfully reconnected slots.
  for (int iy=0; iy<NYC; ++iy)
  for (int ix=0; ix<NXC; ++ix) {
    int bot   = (iy % NGSYC) == 0;
    int top   = (iy % NGSYC) == NGSYC-1;
    int left  = (ix % NGSXC) == 0;
    int right = (ix % NGSXC) == NGSXC-1;

    for (int src=0; src<9; ++src) {
      if (boundary[iy][ix][src] == 0) continue;

      // Geometric boundaries must stay — only clear cascade-set flags
      int geo = 0;
      switch(src) {
        case SRC_SW: geo = (bot || left); break;
        case SRC_S:  geo = bot; break;
        case SRC_SE: geo = (bot || right); break;
        case SRC_W:  geo = left; break;
        case SRC_E:  geo = right; break;
        case SRC_NW: geo = (top || left); break;
        case SRC_N:  geo = top; break;
        case SRC_NE: geo = (top || right); break;
      }
      if (geo) continue;

      // Check if the routing slot for this SRC is now connected
      int col = ((iy + 3 - src_row[src]) % 3) * 3 + (ix + 3 - src_col[src]) % 3;
      if (opt.out_split == 2 && (src == SRC_SW || src == SRC_S || src == SRC_SE || src == SRC_W))
        col += 9;

      if ((router[iy][ix][col] & IN_MASK) != NOT_IN)
        boundary[iy][ix][src] = 0;
    }
  }

// Validate: every tile on a domain edge must have its across-edge neighbors marked.
  for (int iy=0; iy<NYC; ++iy)
  for (int ix=0; ix<NXC; ++ix) {
    if ((iy % NGSYC) == 0) {
      assert(boundary[iy][ix][SRC_SW] == 1);
      assert(boundary[iy][ix][SRC_S] == 1);
      assert(boundary[iy][ix][SRC_SE] == 1);
    }
    if ((iy % NGSYC) == NGSYC-1) {
      assert(boundary[iy][ix][SRC_NW] == 1);
      assert(boundary[iy][ix][SRC_N] == 1);
      assert(boundary[iy][ix][SRC_NE] == 1);
    }
    if ((ix % NGSXC) == 0) {
      assert(boundary[iy][ix][SRC_SW] == 1);
      assert(boundary[iy][ix][SRC_W] == 1);
      assert(boundary[iy][ix][SRC_NW] == 1);
    }
    if ((ix % NGSXC) == NGSXC-1) {
      assert(boundary[iy][ix][SRC_SE] == 1);
      assert(boundary[iy][ix][SRC_E] == 1);
      assert(boundary[iy][ix][SRC_NE] == 1);
    }
  }

}


void fabric_configure(int NYC, int NXC, int NGSYC, int NGSXC, struct fabric *fp_coarse,
                      const struct routing_pattern *pat) {
  configure_router_coarse(NYC, NXC, fp_coarse, pat);
  configure_colors_coarse(NYC, NXC, fp_coarse->colors_1d);
  fix_boundary_coarse(NYC, NXC, NGSYC, NGSXC, fp_coarse->router_1d, fp_coarse->boundary_1d, pat);
}

void fabric_extend(int NY, int NX, int level, struct fabric *fp_coarse, struct fabric *fp) {

  int cfactor = (1<<level);
  int spacing = cfactor - 1;

  int NXC = NX/cfactor;
  int NYC = NY/cfactor;

  int nslots = 9 * opt.out_split;
  u16 (*router_coarse)[NXC][nslots] = (void*)(fp_coarse->router_1d);
  u16 (*router)[NX][nslots] = (void*)(fp->router_1d);

  u16 (*colors_coarse)[NXC][9] = (void*)(fp_coarse->colors_1d);   // [NYC][NXC][9]
  u16 (*boundary_coarse)[NXC][9] = (void*)(fp_coarse->boundary_1d);   // [NYC][NXC][9]
  u32 (*tdmask_coarse)[NXC] = (void*)(fp_coarse->tdmask_1d);

  u16 (*colors)[NX][9]  = (void*)(fp->colors_1d);
  u16 (*boundary)[NX][9]  = (void*)(fp->boundary_1d);
  u32 (*tdmask)[NX] = (void*)(fp->tdmask_1d);
  u16 (*active_tiles)[NX] = (void*)(fp->active_tiles_1d);

// initialize
  for (int iy=0; iy<NY; ++iy)
  for (int ix=0; ix<NX; ++ix) {
    tdmask[iy][ix] = 0;
    active_tiles[iy][ix] = 0;
    for (int ir=0; ir<nslots; ++ir)
      router[iy][ix][ir] = NOT_IN;  // no input or output
  }


// populate active tiles
  for (int iy=0; iy<NYC; ++iy)
  for (int ix=0; ix<NXC; ++ix) {
    active_tiles[iy*cfactor][ix*cfactor] = 1;
    tdmask[iy*cfactor][ix*cfactor] = tdmask_coarse[iy][ix];
    for (int ir=0; ir<9; ++ir) {
      colors[iy*cfactor][ix*cfactor][ir] = colors_coarse[iy][ix][ir];
      boundary[iy*cfactor][ix*cfactor][ir] = boundary_coarse[iy][ix][ir];
      if (boundary_coarse[iy][ix][ir] == 0) {
        tdmask[iy*cfactor][ix*cfactor] |= (1<<colors_coarse[iy][ix][ir]);
      }
    }
    if ( (opt.out_split == 2) && (boundary_coarse[iy][ix][8] == 0) )
    { // double up emitting color
      assert(colors_coarse[iy][ix][8] < 9);
      tdmask[iy*cfactor][ix*cfactor] |= (1<<(colors_coarse[iy][ix][8]+9));
    }

    for (int ir=0; ir<nslots; ++ir)
      router[iy*cfactor][ix*cfactor][ir] = router_coarse[iy][ix][ir];
  }

// Horizontal spacers
  for (int iy=0; iy<NYC; ++iy)
  for (int ir=0; ir<nslots; ++ir) {
    for (int ix=0; ix<(NXC-1); ++ix) {
      if (router_coarse[iy][ix][ir] & OUT_R) {
        assert((router_coarse[iy][ix+1][ir] & IN_MASK) == IN_L);
        for (int is=0; is<spacing; ++is) {
          router[iy*cfactor][ix*cfactor+1+is][ir] = IN_L | OUT_R;
          tdmask[iy*cfactor][ix*cfactor+1+is] |= (1<<ir);
        }
      }
      if (router_coarse[iy][ix+1][ir] & OUT_L) {
        assert((router_coarse[iy][ix][ir] & IN_MASK) == IN_R);
        for (int is=0; is<spacing; ++is) {
          router[iy*cfactor][ix*cfactor+1+is][ir] = IN_R | OUT_L;
          tdmask[iy*cfactor][ix*cfactor+1+is] |= (1<<ir);
        }
      }
    }
  }

// Vertical spacers
  for (int ix=0; ix<NXC; ++ix)
  for (int ir=0; ir<nslots; ++ir) {
    for (int iy=0; iy<(NYC-1); ++iy) {
      if (router_coarse[iy][ix][ir] & OUT_U) {
        assert((router_coarse[iy+1][ix][ir] & IN_MASK) == IN_D);
        for (int is=0; is<spacing; ++is) {
          router[iy*cfactor+1+is][ix*cfactor][ir] = IN_D | OUT_U;
          tdmask[iy*cfactor+1+is][ix*cfactor] |= (1<<ir);
        }
      }
      if (router_coarse[iy+1][ix][ir] & OUT_D) {
        assert((router_coarse[iy][ix][ir] & IN_MASK) == IN_U);
        for (int is=0; is<spacing; ++is) {
          router[iy*cfactor+1+is][ix*cfactor][ir] = IN_U | OUT_D;
          tdmask[iy*cfactor+1+is][ix*cfactor] |= (1<<ir);
        }
      }
    }
  }

}


void dot_files(int NY, int NX) {
  FILE *fo;

  u16 c_r = IN_C | OUT_R;   // first position of [C>R, L>R] for upward reductions
  u16 r_cl = IN_R | OUT_C | OUT_L;

  fo = fopen("router_dot.bin", "wb");
  for (int iy=0; iy<NY; ++iy)
  for (int ix=0; ix<NX; ++ix) {
    fwrite(&c_r, sizeof(u16), 1, fo);  // color 21 - horizontal reduction (switched)
    fwrite(&r_cl, sizeof(u16), 1, fo);   // color 22 - broadcast
  }
  fclose(fo);

  u16 s[4] = {0x00a0, 0xa0a0, 0x0000, 0x0000};  // switch configuration registers (no switching)
  u16 sw_l_r = IN_L | OUT_R | BIT(12) | BIT(14);  // D>U, ring, pop on advance

  fo = fopen("switch_dot.bin", "wb");
  for (int iy=0; iy<NY; ++iy)
  for (int ix=0; ix<NX; ++ix) {
    fwrite(&sw_l_r, sizeof(u16), 1, fo);   // color 21, position 0   (L>R)
    fwrite(&(s[1]), sizeof(u16), 3, fo);   // color 21
  }
  fclose(fo);

  u16 d_ul = IN_D | OUT_U | OUT_L;
  fo = fopen("router_dot_edge.bin", "wb");
  for (int iy=0; iy<NY; ++iy) {
    fwrite(&d_ul, sizeof(u16), 1, fo);
  }
  fclose(fo);

  u16 r_c = IN_R | OUT_C;
  fo = fopen("router_dot_src_left.bin", "wb");
  for (int iy=0; iy<NY; ++iy) {
    fwrite(&r_c, sizeof(u16), 1, fo);
  }
  fclose(fo);

  u16 d_c = IN_D | OUT_C;
  fo = fopen("router_dot_src_top.bin", "wb");
  fwrite(&d_c, sizeof(u16), 1, fo);
  fclose(fo);

  u16 c_u = IN_C | OUT_U;
  fo = fopen("router_dot_recolor.bin", "wb");
  fwrite(&c_u, sizeof(u16), 1, fo);
  fclose(fo);

}

void fabric_save(int NY, int NX, struct fabric *fp, int context_count, int context_id) {

  u16 (*router_arr)[NX][context_count][9*opt.out_split] = (void*)(file_arrays.router_file);
  u16 (*boundary_arr)[NX][context_count][9] = (void*)(file_arrays.boundary_file);
  u16 (*colors_arr)[NX][context_count][9] = (void*)(file_arrays.colors_file);
  u16 (*active_tiles_arr)[NX][context_count] = (void*)(file_arrays.active_tiles_file);
  u32 (*tdmask_arr)[NX][context_count] = (void*)(file_arrays.tdmask_file);

  u16 (*router)[NX][9*opt.out_split] = (void*)(fp->router_1d);
  u16 (*colors)[NX][9]  = (void*)(fp->colors_1d);
  u16 (*boundary)[NX][9]  = (void*)(fp->boundary_1d);
  u32 (*tdmask)[NX] = (void*)(fp->tdmask_1d);
  u16 (*active_tiles)[NX] = (void*)(fp->active_tiles_1d);

  for (int iy=0; iy<NY; ++iy)
  for (int ix=0; ix<NX; ++ix) {
    for (int ic=0; ic<9*opt.out_split; ++ic)
      router_arr[iy][ix][context_id][ic] = router[iy][ix][ic];
    for (int ic=0; ic<9; ++ic) {
      boundary_arr[iy][ix][context_id][ic] = boundary[iy][ix][ic];
      colors_arr[iy][ix][context_id][ic] = colors[iy][ix][ic];
    }
    active_tiles_arr[iy][ix][context_id] = active_tiles[iy][ix];
    tdmask_arr[iy][ix][context_id] = tdmask[iy][ix];
  }
}

void fabric_create(int NY, int NX, int NGSY, int NGSX, int level, int fabric_count, int fabric_id,
                   const struct routing_pattern *pat) {

  int coarsening_factor = (1<<level);
  int NXC = NX/coarsening_factor;
  int NYC = NY/coarsening_factor;
  int NGSXC = NGSX/coarsening_factor;
  int NGSYC = NGSY/coarsening_factor;

  struct fabric f_coarse, f_fine;
  fabric_init(NYC, NXC, &f_coarse);
  fabric_configure(NYC, NXC, NGSYC, NGSXC, &f_coarse, pat);
  fabric_init(NY, NX, &f_fine);
  fabric_extend(NY, NX, level, &f_coarse, &f_fine);
  fabric_save(NY, NX, &f_fine, fabric_count, fabric_id);
  fabric_fini(&f_coarse);
  fabric_fini(&f_fine);
}

void fabrics_to_files(int NY, int NX, int context_count) {
  FILE *fo;

  fo = fopen("router_stencil.bin", "wb");
  fwrite(file_arrays.router_file, sizeof(u16), NY*NX*context_count*9*opt.out_split, fo);
  fclose(fo);

  fo = fopen("boundary_stencil.bin", "wb");
  fwrite(file_arrays.boundary_file, sizeof(u16), NY*NX*context_count*9, fo);
  fclose(fo);

  fo = fopen("colors_stencil.bin", "wb");
  fwrite(file_arrays.colors_file, sizeof(u16), NY*NX*context_count*9, fo);
  fclose(fo);

  fo = fopen("active_stencil.bin", "wb");
  fwrite(file_arrays.active_tiles_file, sizeof(u16), NY*NX*context_count, fo);
  fclose(fo);

  fo = fopen("tdmask_stencil.bin", "wb");
  if (opt.out_split == 2)
      fwrite(file_arrays.tdmask_file, sizeof(u32), NY*NX*context_count, fo);
  else {
    for (int i=0; i<NY*NX*context_count; ++i) {
      u16 td = (u16)file_arrays.tdmask_file[i];
      fwrite(&td, sizeof(u16), 1, fo);
    }
  }
  fclose(fo);
}


int main(int argc, char **argv) {
  parse_options(argc, argv);
  validate_routing_patterns();

  int NY = opt.NY;
  int NX = opt.NX;
  int level = opt.level;
  int mode = opt.mode;

  int NGSYC, NGSXC;
  if (opt.NGS) {
    NGSYC = opt.NGS;
    NGSXC = opt.NGS;
  } else {
    NGSYC = NY;
    NGSXC = NX;
  }

  const struct routing_pattern *pat = &routing_patterns[opt.routing_dir];

  int context_count;

  switch (mode) {
    case FWD:
    case BWD:
    {
      context_count = 1;
      file_arrays_init(NY, NX, context_count);
      fabric_create(NY, NX, NGSYC, NGSXC, level, context_count, 0, pat);
      fabrics_to_files(NY, NX, context_count);
      file_arrays_fini();
      break;
    }
    case SPMV:
    {
      context_count = 1;
      file_arrays_init(NY, NX, context_count);
      fabric_create(NY, NX, NY, NX, level, context_count, 0, pat);
      fabrics_to_files(NY, NX, context_count);
      file_arrays_fini();
      break;
    }
    case SYM:
    {
      context_count = 2;
      file_arrays_init(NY, NX, context_count);
      fabric_create(NY, NX, NGSYC, NGSXC, level, context_count, 0, pat);
      fabric_create(NY, NX, NGSYC, NGSXC, level, context_count, 1, pat);
      fabrics_to_files(NY, NX, context_count);
      file_arrays_fini();
      break;
    }
    case GS_SPMV:
    {
      context_count = 2;
      file_arrays_init(NY, NX, context_count);
      fabric_create(NY, NX, NGSYC, NGSXC, level, context_count, 0, pat);
      fabric_create(NY, NX, NY,    NX,    level, context_count, 1, pat);
      fabrics_to_files(NY, NX, context_count);
      file_arrays_fini();
      break;
    }
    case TDTEST:
    {
      context_count = 2;
      file_arrays_init(NY, NX, context_count);
      fabric_create(NY, NX, NGSYC, NGSXC, 1, context_count, 0, pat);
      fabric_create(NY, NX, NGSYC, NGSXC, level, context_count, 1, pat);
      fabrics_to_files(NY, NX, context_count);
      file_arrays_fini();
      break;
    }
    case VCYCLE1:
    case VCYCLE2:
    case VCYCLE3:
    case HPCG:  // TODO: need an extra SpMV context at the beginning
    {
      int mg_levels = (mode == VCYCLE1) ? 1 : (mode == VCYCLE2) ? 2 : 3;
      context_count = 3 * mg_levels + 1;
      file_arrays_init(NY, NX, context_count);
      int ctx = 0;
      for (int l = 0; l < mg_levels; l++) {
        fabric_create(NY, NX, NGSYC, NGSXC, l, context_count, ctx++, pat);  // gs
        fabric_create(NY, NX, NY,    NX,    l, context_count, ctx++, pat);  // spmv
      }
      fabric_create(NY, NX, NGSYC, NGSXC, mg_levels, context_count, ctx++, pat);  // gs deepest
      for (int l = mg_levels - 1; l >= 0; l--)
        fabric_create(NY, NX, NGSYC, NGSXC, l, context_count, ctx++, pat);  // gs ascend
      fabrics_to_files(NY, NX, context_count);
      file_arrays_fini();
      break;
    }
    case CG:
    {
      context_count = 1;
      file_arrays_init(NY, NX, context_count);
      fabric_create(NY, NX, NY, NX, 0, context_count, 0, pat);
      fabrics_to_files(NY, NX, context_count);
      file_arrays_fini();
      break;
    }
    default:
      fatalif(1, "Unknown mode %d", mode);

  }
  printf("context_count = %d\n", context_count);

  dot_files(NY, NX);

  return 0;
}
