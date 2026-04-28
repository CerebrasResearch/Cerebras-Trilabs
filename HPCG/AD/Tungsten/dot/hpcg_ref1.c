/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2026 Cerebras Systems Inc.
 * All rights reserved.
 */

/*
 * hpcg_sim.c - Single-file C implementation of the HPCG benchmark
 *
 * Simulates multiple MPI ranks within a single process.
 * Equivalent computation to the HPCG 3.1 reference implementation.
 *
 * Build (single precision, default):
 *   gcc -O2 -o hpcg_sim hpcg_sim.c -lm
 *
 * Build (double precision):
 *   gcc -O2 -DUSE_DOUBLE -o hpcg_sim hpcg_sim.c -lm
 *
 * Usage: ./hpcg_sim [npx] [npy] [npz] [nx] [ny] [nz] [maxIters] [--halo] [--test MODE] [--init MODE] [--levels N] [--test-level L] [--bin_split]
 *   Defaults: npx=npy=npz=1, nx=ny=nz=16, maxIters=50, no halo exchange
 *   nranks = npx * npy * npz
 *   --halo   enables inter-rank halo exchange for Gauss-Seidel
 *   --test   MODE selects the test (default: hpcg)
 *            hpcg   - full HPCG benchmark (CG + MG preconditioner)
 *            cg     - CG without preconditioner (identity preconditioner)
 *            vcycle - single MG V-cycle
 *            spmv   - sparse matrix-vector product y = A*x
 *            gs_fwd - single forward Gauss-Seidel sweep
 *            gs_bwd - single backward Gauss-Seidel sweep
 *            gs_sym - symmetric Gauss-Seidel (forward + backward sweep)
 *   --init   MODE selects matrix initialization (default: random)
 *            random - 26/-1 stencil with uniform [-1,1] perturbation; b = 1
 *            fixed  - exact 26/-1 stencil; b = A * ones
 *   --perm   MODE selects off-diagonal save order (default: none)
 *            none   - natural stencil order (sz, sy, sx each -1..1)
 *            perm1  - custom permutation, alternating by even/odd gz
 *            perm2  - custom permutation, same for all gz
 *   --levels N sets number of MG coarsening levels (default: 4)
 *   --test-level L selects which MG level to test in gs_fwd/gs_bwd/gs_sym mode (default: 0)
 *   --bin_split  split binary output files by MG level (e.g. A0.bin, A1.bin, ...)
 *
 * Output files contain data for all MG levels concatenated (level 0 first,
 * then level 1, etc).  For coarser levels, global X and Y dimensions remain
 * the full finest-level size (with zeros for inactive cells), while Z has
 * only active values.  Vector x is zero-padded above and below per level.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>

/* Global flag: 0 = no halo exchange (default), 1 = exchange halos */
static int g_do_halo_exchange = 0;

/* Test mode: "hpcg" (default), "gs_fwd", "gs_bwd", or "gs_sym" */
static const char *g_test_mode = "hpcg";

/* Init mode: "random" (default) or "fixed" */
static const char *g_init_mode = "random";

/* Permutation mode for off-diagonal save order: "none" (default) or "perm1" */
static const char *g_perm_mode = "none";

/* Test level for gs_fwd/gs_bwd/gs_sym: which MG level to test (default: 0) */
static int g_test_level = 0;

/* Binary split mode: 0 = single combined file (default), 1 = per-level files */
static int g_bin_split = 0;

/* ================================================================
 * Section 1: Types and Data Structures
 * ================================================================ */

#ifdef USE_DOUBLE
typedef double real_t;
#define REAL_SQRT sqrt
#define REAL_FMT "22.16e"
#define REAL_NAME "double"
#else
typedef float real_t;
#define REAL_SQRT sqrtf
#define REAL_FMT "14.8e"
#define REAL_NAME "float"
#endif

typedef long long global_int_t;
typedef int local_int_t;

#define MAX_NONZEROS_PER_ROW 27

/* Number of MG coarsening levels (overridable via --levels) */
static int g_num_mg_levels = 4;

typedef struct SparseMatrix {
    /* Geometry (embedded) */
    int rank, nranks;
    int nx, ny, nz;
    int npx, npy, npz;
    int ipx, ipy, ipz;
    global_int_t gnx, gny, gnz;
    global_int_t gix0, giy0, giz0;

    /* Matrix data */
    local_int_t nrow;        /* localNumberOfRows = nx*ny*nz */
    local_int_t ncol;        /* nrow + numberOfExternalValues */
    global_int_t totalRows;
    global_int_t totalNnz;
    local_int_t localNnz;

    char *nnzPerRow;                /* [nrow] */
    real_t *values;                 /* [nrow * 27] row-major */
    local_int_t *colInd;            /* [nrow * 27] local column indices */
    real_t *diag;                   /* [nrow] diagonal values */
    global_int_t *localToGlobalMap; /* [nrow] */
    global_int_t *colIndG;          /* [nrow * 27] global col indices (temp) */

    /* Halo communication */
    int num_neighbors;
    int totalToSend;
    int numberOfExternalValues;
    int *neighbors;          /* [num_neighbors] */
    local_int_t *recvLength; /* [num_neighbors] */
    local_int_t *sendLength; /* [num_neighbors] */
    local_int_t *elementsToSend; /* [totalToSend] local row indices */
} SparseMatrix;

typedef struct MGWork {
    int *f2c;       /* [coarse_nrow] fine-to-coarse map */
    real_t *Axf;    /* [fine_nrow] A*x workspace */
    real_t *rc;     /* [coarse_nrow] coarse residual */
    real_t *xc;     /* [coarse_ncol] coarse solution (with halo space) */
} MGWork;

typedef struct MGLevelPtrs {
    real_t **Axf_ptrs; /* [nranks] */
    real_t **rc_ptrs;  /* [nranks] */
    real_t **xc_ptrs;  /* [nranks] */
} MGLevelPtrs;

/* ================================================================
 * Section 2: Timer
 * ================================================================ */

static double mytimer(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

/* ================================================================
 * Section 3: Utility Functions
 * ================================================================ */

static int cmp_global_int(const void *a, const void *b) {
    global_int_t va = *(const global_int_t *)a;
    global_int_t vb = *(const global_int_t *)b;
    return (va > vb) - (va < vb);
}

/* Sort-and-unique for global_int_t array. Returns new count. In-place. */
static int sorted_unique_inplace(global_int_t *arr, int n) {
    if (n <= 0) return 0;
    qsort(arr, n, sizeof(global_int_t), cmp_global_int);
    int out = 1;
    for (int i = 1; i < n; i++) {
        if (arr[i] != arr[out - 1])
            arr[out++] = arr[i];
    }
    return out;
}

/* Comparator for (rank, global_row) pairs */
typedef struct { int rank; global_int_t grow; } RankGrowPair;

static int cmp_rankgrow(const void *a, const void *b) {
    const RankGrowPair *pa = (const RankGrowPair *)a;
    const RankGrowPair *pb = (const RankGrowPair *)b;
    if (pa->rank != pb->rank) return pa->rank - pb->rank;
    return (pa->grow > pb->grow) - (pa->grow < pb->grow);
}

/* Lookup entry: maps global column ID to local halo index */
typedef struct { global_int_t gcol; int halo_idx; } GcolMapEntry;

static int cmp_gcolmap(const void *a, const void *b) {
    const GcolMapEntry *pa = (const GcolMapEntry *)a;
    const GcolMapEntry *pb = (const GcolMapEntry *)b;
    return (pa->gcol > pb->gcol) - (pa->gcol < pb->gcol);
}

static int lookup_halo_idx(const GcolMapEntry *map, int n, global_int_t gcol) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (map[mid].gcol == gcol) return map[mid].halo_idx;
        if (map[mid].gcol < gcol) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* Perturbation table: pre-computed random values in [-1,1] via srand(42)/rand(),
 * generated in global Y,X,Z row order with 27 stencil entries per row in
 * natural order (dz outer, dy middle, dx inner). */
static real_t *g_perturb = NULL;
static long long g_perturb_gnx, g_perturb_gny, g_perturb_gnz;

static void init_perturbation_table(long long gnx, long long gny, long long gnz) {
    g_perturb_gnx = gnx;
    g_perturb_gny = gny;
    g_perturb_gnz = gnz;
    long long total = gnx * gny * gnz * 27;
    g_perturb = (real_t *)malloc((size_t)total * sizeof(real_t));
    srand(42);
    long long idx = 0;
    for (long long gy = 0; gy < gny; gy++)
        for (long long gx = 0; gx < gnx; gx++)
            for (long long gz = 0; gz < gnz; gz++)
                for (int k = 0; k < 27; k++)
                    g_perturb[idx++] =
                        (real_t)((double)rand() / (double)RAND_MAX * 2.0 - 1.0);
}

static real_t get_perturbation(long long gx, long long gy, long long gz,
                               int dz, int dx, int dy) {
    int k = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
    long long row_yxz = gy * g_perturb_gnx * g_perturb_gnz +
                         gx * g_perturb_gnz + gz;
    return g_perturb[row_yxz * 27 + k];
}

static void free_perturbation_table(void) {
    free(g_perturb);
    g_perturb = NULL;
}

/* ================================================================
 * Section 4: Geometry Functions
 * ================================================================ */

static void generate_geometry(int nranks, int rank, int nx, int ny, int nz,
                              int npx, int npy, int npz, SparseMatrix *A) {
    A->rank = rank;
    A->nranks = nranks;
    A->nx = nx;
    A->ny = ny;
    A->nz = nz;
    A->npx = npx;
    A->npy = npy;
    A->npz = npz;
    A->ipx = rank % npx;
    A->ipy = (rank / npx) % npy;
    A->ipz = rank / (npx * npy);
    A->gnx = (global_int_t)npx * nx;
    A->gny = (global_int_t)npy * ny;
    A->gnz = (global_int_t)npz * nz;
    A->gix0 = (global_int_t)A->ipx * nx;
    A->giy0 = (global_int_t)A->ipy * ny;
    A->giz0 = (global_int_t)A->ipz * nz;
}

static int compute_rank_of_row(const SparseMatrix *A, global_int_t index) {
    global_int_t gnx = A->gnx, gny = A->gny;
    global_int_t gz = index / (gnx * gny);
    global_int_t gy = (index - gz * gnx * gny) / gnx;
    global_int_t gx = index % gnx;
    int ipz = (int)(gz / A->nz);
    int ipy = (int)(gy / A->ny);
    int ipx = (int)(gx / A->nx);
    return ipx + ipy * A->npx + ipz * A->npx * A->npy;
}

/* ================================================================
 * Section 5: Problem Generation (27-point stencil)
 * ================================================================ */

static void generate_problem(SparseMatrix *A, real_t *b, real_t *x,
                             real_t *xexact, int skip_matrix) {
    int nx = A->nx, ny = A->ny, nz = A->nz;
    global_int_t gnx = A->gnx, gny = A->gny, gnz = A->gnz;
    global_int_t gix0 = A->gix0, giy0 = A->giy0, giz0 = A->giz0;

    local_int_t nrow = nx * ny * nz;
    A->nrow = nrow;
    A->ncol = nrow;
    A->totalRows = gnx * gny * gnz;

    if (!skip_matrix) {
        A->nnzPerRow = (char *)calloc(nrow, sizeof(char));
        A->values = (real_t *)calloc((size_t)nrow * MAX_NONZEROS_PER_ROW, sizeof(real_t));
        A->colIndG = (global_int_t *)calloc((size_t)nrow * MAX_NONZEROS_PER_ROW, sizeof(global_int_t));
        A->colInd = (local_int_t *)calloc((size_t)nrow * MAX_NONZEROS_PER_ROW, sizeof(local_int_t));
        A->diag = (real_t *)calloc(nrow, sizeof(real_t));
        A->localToGlobalMap = (global_int_t *)calloc(nrow, sizeof(global_int_t));
    }

    local_int_t localNnz = 0;

    for (int iz = 0; iz < nz; iz++) {
        global_int_t giz = giz0 + iz;
        for (int iy = 0; iy < ny; iy++) {
            global_int_t giy = giy0 + iy;
            for (int ix = 0; ix < nx; ix++) {
                global_int_t gix = gix0 + ix;
                local_int_t row = iz * nx * ny + iy * nx + ix;
                global_int_t grow = giz * gnx * gny + giy * gnx + gix;
                if (!skip_matrix)
                    A->localToGlobalMap[row] = grow;

                int nnz = 0;
                real_t *vp = skip_matrix ? NULL : &A->values[row * MAX_NONZEROS_PER_ROW];
                global_int_t *gp = skip_matrix ? NULL : &A->colIndG[row * MAX_NONZEROS_PER_ROW];

                int do_perturb = (strcmp(g_init_mode, "random") == 0) &&
                                 g_perturb != NULL &&
                                 gnx == g_perturb_gnx && gny == g_perturb_gny &&
                                 gnz == g_perturb_gnz;

                real_t row_sum = (real_t)0.0;
                for (int sz = -1; sz <= 1; sz++) {
                    if (giz + sz < 0 || giz + sz >= gnz) continue;
                    for (int sy = -1; sy <= 1; sy++) {
                        if (giy + sy < 0 || giy + sy >= gny) continue;
                        for (int sx = -1; sx <= 1; sx++) {
                            if (gix + sx < 0 || gix + sx >= gnx) continue;
                            global_int_t col = grow + sz * gnx * gny + sy * gnx + sx;
                            real_t p = do_perturb ?
                                get_perturbation(gix, giy, giz, sz, sx, sy) :
                                (real_t)0.0;
                            real_t val;
                            if (col == grow) {
                                val = do_perturb
                                    ? (real_t)(18.0 + 8.0 * (double)p)
                                    : (real_t)26.0;
                                if (!skip_matrix)
                                    A->diag[row] = val;
                            } else {
                                val = do_perturb
                                    ? (real_t)(-0.5 + 0.5 * (double)p)
                                    : (real_t)-1.0;
                            }
                            if (!skip_matrix) {
                                vp[nnz] = val;
                                gp[nnz] = col;
                            }
                            row_sum += val;
                            nnz++;
                        }
                    }
                }
                if (!skip_matrix) {
                    A->nnzPerRow[row] = (char)nnz;
                    localNnz += nnz;
                }

                if (b) b[row] = row_sum;
                if (x) x[row] = (real_t)0.0;
                if (xexact) xexact[row] = (real_t)1.0;
            }
        }
    }

    if (!skip_matrix)
        A->localNnz = localNnz;
}

/* ================================================================
 * Section 6: Halo Setup (C implementation, no std::map)
 * ================================================================ */

static void setup_halo(SparseMatrix *A) {
    local_int_t nrow = A->nrow;
    int myrank = A->rank;

    if (A->nranks == 1) {
        for (local_int_t i = 0; i < nrow; i++) {
            int nnz = A->nnzPerRow[i];
            for (int j = 0; j < nnz; j++) {
                global_int_t gcol = A->colIndG[i * MAX_NONZEROS_PER_ROW + j];
                global_int_t gx = gcol % A->gnx;
                global_int_t gy = (gcol / A->gnx) % A->gny;
                global_int_t gz = gcol / (A->gnx * A->gny);
                int lx = (int)(gx - A->gix0);
                int ly = (int)(gy - A->giy0);
                int lz = (int)(gz - A->giz0);
                A->colInd[i * MAX_NONZEROS_PER_ROW + j] = lz * A->nx * A->ny + ly * A->nx + lx;
            }
        }
        A->ncol = nrow;
        A->numberOfExternalValues = 0;
        A->num_neighbors = 0;
        A->totalToSend = 0;
        A->neighbors = NULL;
        A->recvLength = NULL;
        A->sendLength = NULL;
        A->elementsToSend = NULL;
        free(A->colIndG);
        A->colIndG = NULL;
        return;
    }

    int ext_cap = 1024;
    int ext_count = 0;
    global_int_t *ext_globals = (global_int_t *)malloc(ext_cap * sizeof(global_int_t));

    int pair_cap = 1024;
    int pair_count = 0;
    RankGrowPair *send_pairs = (RankGrowPair *)malloc(pair_cap * sizeof(RankGrowPair));

    for (local_int_t i = 0; i < nrow; i++) {
        int nnz = A->nnzPerRow[i];
        global_int_t my_grow = A->localToGlobalMap[i];
        for (int j = 0; j < nnz; j++) {
            global_int_t gcol = A->colIndG[i * MAX_NONZEROS_PER_ROW + j];
            int owner = compute_rank_of_row(A, gcol);
            if (owner != myrank) {
                if (ext_count >= ext_cap) {
                    ext_cap *= 2;
                    ext_globals = (global_int_t *)realloc(ext_globals, ext_cap * sizeof(global_int_t));
                }
                ext_globals[ext_count++] = gcol;

                if (pair_count >= pair_cap) {
                    pair_cap *= 2;
                    send_pairs = (RankGrowPair *)realloc(send_pairs, pair_cap * sizeof(RankGrowPair));
                }
                send_pairs[pair_count].rank = owner;
                send_pairs[pair_count].grow = my_grow;
                pair_count++;
            }
        }
    }

    int num_ext = sorted_unique_inplace(ext_globals, ext_count);
    A->numberOfExternalValues = num_ext;
    A->ncol = nrow + num_ext;

    typedef struct { int rank; global_int_t gcol; int halo_idx; } RecvEntry;
    RecvEntry *recv_entries = (RecvEntry *)malloc(num_ext * sizeof(RecvEntry));
    for (int i = 0; i < num_ext; i++) {
        recv_entries[i].rank = compute_rank_of_row(A, ext_globals[i]);
        recv_entries[i].gcol = ext_globals[i];
        recv_entries[i].halo_idx = nrow + i;
    }

    qsort(recv_entries, num_ext, sizeof(RecvEntry),
          (int(*)(const void*,const void*))cmp_rankgrow);
    for (int i = 0; i < num_ext; i++) {
        recv_entries[i].halo_idx = nrow + i;
    }

    GcolMapEntry *gcol_map = (GcolMapEntry *)malloc(num_ext * sizeof(GcolMapEntry));
    for (int i = 0; i < num_ext; i++) {
        gcol_map[i].gcol = recv_entries[i].gcol;
        gcol_map[i].halo_idx = recv_entries[i].halo_idx;
    }
    qsort(gcol_map, num_ext, sizeof(GcolMapEntry), cmp_gcolmap);

    int num_neighbors = 0;
    int *nbr_list = (int *)malloc(num_ext * sizeof(int));
    local_int_t *recv_len = (local_int_t *)malloc(num_ext * sizeof(local_int_t));

    if (num_ext > 0) {
        nbr_list[0] = recv_entries[0].rank;
        recv_len[0] = 1;
        num_neighbors = 1;
        for (int i = 1; i < num_ext; i++) {
            if (recv_entries[i].rank != recv_entries[i - 1].rank) {
                nbr_list[num_neighbors] = recv_entries[i].rank;
                recv_len[num_neighbors] = 1;
                num_neighbors++;
            } else {
                recv_len[num_neighbors - 1]++;
            }
        }
    }

    qsort(send_pairs, pair_count, sizeof(RankGrowPair), cmp_rankgrow);
    int unique_pairs = 0;
    for (int i = 0; i < pair_count; i++) {
        if (i == 0 || send_pairs[i].rank != send_pairs[unique_pairs - 1].rank ||
            send_pairs[i].grow != send_pairs[unique_pairs - 1].grow) {
            send_pairs[unique_pairs++] = send_pairs[i];
        }
    }
    pair_count = unique_pairs;

    int totalToSend = pair_count;
    local_int_t *send_len = (local_int_t *)calloc(num_neighbors, sizeof(local_int_t));
    local_int_t *elts_to_send = (local_int_t *)malloc(totalToSend * sizeof(local_int_t));

    int send_idx = 0;
    for (int ni = 0; ni < num_neighbors; ni++) {
        int nbr = nbr_list[ni];
        send_len[ni] = 0;
        while (send_idx < pair_count && send_pairs[send_idx].rank == nbr) {
            global_int_t grow = send_pairs[send_idx].grow;
            global_int_t gx = grow % A->gnx;
            global_int_t gy = (grow / A->gnx) % A->gny;
            global_int_t gz = grow / (A->gnx * A->gny);
            int lx = (int)(gx - A->gix0);
            int ly = (int)(gy - A->giy0);
            int lz = (int)(gz - A->giz0);
            local_int_t local_row = lz * A->nx * A->ny + ly * A->nx + lx;
            elts_to_send[send_idx] = local_row;
            send_len[ni]++;
            send_idx++;
        }
    }

    for (local_int_t i = 0; i < nrow; i++) {
        int nnz = A->nnzPerRow[i];
        for (int j = 0; j < nnz; j++) {
            global_int_t gcol = A->colIndG[i * MAX_NONZEROS_PER_ROW + j];
            int owner = compute_rank_of_row(A, gcol);
            if (owner == myrank) {
                global_int_t gx = gcol % A->gnx;
                global_int_t gy = (gcol / A->gnx) % A->gny;
                global_int_t gz = gcol / (A->gnx * A->gny);
                int lx = (int)(gx - A->gix0);
                int ly = (int)(gy - A->giy0);
                int lz = (int)(gz - A->giz0);
                A->colInd[i * MAX_NONZEROS_PER_ROW + j] = lz * A->nx * A->ny + ly * A->nx + lx;
            } else {
                int halo_idx = lookup_halo_idx(gcol_map, num_ext, gcol);
                assert(halo_idx >= 0);
                A->colInd[i * MAX_NONZEROS_PER_ROW + j] = halo_idx;
            }
        }
    }

    A->num_neighbors = num_neighbors;
    A->neighbors = (int *)malloc(num_neighbors * sizeof(int));
    A->recvLength = (local_int_t *)malloc(num_neighbors * sizeof(local_int_t));
    A->sendLength = (local_int_t *)malloc(num_neighbors * sizeof(local_int_t));
    memcpy(A->neighbors, nbr_list, num_neighbors * sizeof(int));
    memcpy(A->recvLength, recv_len, num_neighbors * sizeof(local_int_t));
    memcpy(A->sendLength, send_len, num_neighbors * sizeof(local_int_t));
    A->totalToSend = totalToSend;
    A->elementsToSend = (local_int_t *)malloc(totalToSend * sizeof(local_int_t));
    memcpy(A->elementsToSend, elts_to_send, totalToSend * sizeof(local_int_t));

    free(ext_globals);
    free(gcol_map);
    free(recv_entries);
    free(send_pairs);
    free(nbr_list);
    free(recv_len);
    free(send_len);
    free(elts_to_send);
    free(A->colIndG);
    A->colIndG = NULL;
}

/* ================================================================
 * Section 7: Generate Coarse Problem
 * ================================================================ */

static void generate_coarse_problem(int nranks,
                                    SparseMatrix *Af,
                                    SparseMatrix *Ac,
                                    MGWork *mgw) {
    for (int r = 0; r < nranks; r++) {
        int nxf = Af[r].nx, nyf = Af[r].ny, nzf = Af[r].nz;
        int nxc = nxf / 2, nyc = nyf / 2, nzc = nzf / 2;
        assert(nxf % 2 == 0 && nyf % 2 == 0 && nzf % 2 == 0);

        local_int_t coarse_nrow = nxc * nyc * nzc;

        mgw[r].f2c = (int *)malloc(coarse_nrow * sizeof(int));
        for (int izc = 0; izc < nzc; izc++) {
            for (int iyc = 0; iyc < nyc; iyc++) {
                for (int ixc = 0; ixc < nxc; ixc++) {
                    int coarse_row = izc * nxc * nyc + iyc * nxc + ixc;
                    int fine_row = (2 * izc) * nxf * nyf + (2 * iyc) * nxf + (2 * ixc);
                    mgw[r].f2c[coarse_row] = fine_row;
                }
            }
        }

        memset(&Ac[r], 0, sizeof(SparseMatrix));
        generate_geometry(nranks, r, nxc, nyc, nzc,
                          Af[r].npx, Af[r].npy, Af[r].npz, &Ac[r]);
        generate_problem(&Ac[r], NULL, NULL, NULL, 0);
        setup_halo(&Ac[r]);

        mgw[r].Axf = (real_t *)calloc(Af[r].nrow, sizeof(real_t));
        mgw[r].rc = (real_t *)calloc(Ac[r].nrow, sizeof(real_t));
        mgw[r].xc = (real_t *)calloc(Ac[r].ncol, sizeof(real_t));
    }
}

/* ================================================================
 * Section 8: Halo Exchange (All Ranks, simulates MPI)
 * ================================================================ */

static void exchange_halo_all(int nranks, SparseMatrix *A, real_t **x) {
    if (nranks == 1) return;

    for (int r = 0; r < nranks; r++) {
        if (A[r].num_neighbors == 0) continue;
        real_t *x_ext = x[r] + A[r].nrow;
        int recv_off = 0;

        for (int ni = 0; ni < A[r].num_neighbors; ni++) {
            int nbr = A[r].neighbors[ni];
            int recv_len = A[r].recvLength[ni];

            int send_off = 0;
            int found = 0;
            for (int nj = 0; nj < A[nbr].num_neighbors; nj++) {
                if (A[nbr].neighbors[nj] == r) {
                    for (int k = 0; k < recv_len; k++) {
                        x_ext[recv_off + k] = x[nbr][A[nbr].elementsToSend[send_off + k]];
                    }
                    found = 1;
                    break;
                }
                send_off += A[nbr].sendLength[nj];
            }
            assert(found);
            recv_off += recv_len;
        }
    }
}

/* Exchange halo data with all non-pure-Z neighbors (same ipx/ipy filtered out).
 * Covers pure X, pure Y, and all diagonal neighbors (XY, XZ, YZ, XYZ). */
static void exchange_halo_xy_all(int nranks, SparseMatrix *A, real_t **x) {
    if (nranks == 1) return;

    for (int r = 0; r < nranks; r++) {
        if (A[r].num_neighbors == 0) continue;
        real_t *x_ext = x[r] + A[r].nrow;
        int recv_off = 0;

        for (int ni = 0; ni < A[r].num_neighbors; ni++) {
            int nbr = A[r].neighbors[ni];
            int recv_len = A[r].recvLength[ni];

            if (A[nbr].ipx != A[r].ipx || A[nbr].ipy != A[r].ipy) {
                int send_off = 0;
                for (int nj = 0; nj < A[nbr].num_neighbors; nj++) {
                    if (A[nbr].neighbors[nj] == r) {
                        for (int k = 0; k < recv_len; k++) {
                            x_ext[recv_off + k] = x[nbr][A[nbr].elementsToSend[send_off + k]];
                        }
                        break;
                    }
                    send_off += A[nbr].sendLength[nj];
                }
            }
            recv_off += recv_len;
        }
    }
}

/* Exchange pure-Z-direction halo data for ranks at a specific ipz layer.
 * Only fills halo slots from neighbors with same ipx/ipy but different ipz. */
static void exchange_halo_z_layer(int nranks, SparseMatrix *A, real_t **x, int target_ipz) {
    if (nranks == 1) return;

    for (int r = 0; r < nranks; r++) {
        if (A[r].ipz != target_ipz) continue;
        if (A[r].num_neighbors == 0) continue;
        real_t *x_ext = x[r] + A[r].nrow;
        int recv_off = 0;

        for (int ni = 0; ni < A[r].num_neighbors; ni++) {
            int nbr = A[r].neighbors[ni];
            int recv_len = A[r].recvLength[ni];

            if (A[nbr].ipx == A[r].ipx && A[nbr].ipy == A[r].ipy &&
                A[nbr].ipz != A[r].ipz) {
                int send_off = 0;
                for (int nj = 0; nj < A[nbr].num_neighbors; nj++) {
                    if (A[nbr].neighbors[nj] == r) {
                        for (int k = 0; k < recv_len; k++) {
                            x_ext[recv_off + k] = x[nbr][A[nbr].elementsToSend[send_off + k]];
                        }
                        break;
                    }
                    send_off += A[nbr].sendLength[nj];
                }
            }
            recv_off += recv_len;
        }
    }
}

/* Zero out halo region [nrow, ncol) of x for every rank. */
static void clear_halo_all(int nranks, SparseMatrix *A, real_t **x) {
    for (int r = 0; r < nranks; r++) {
        local_int_t nrow = A[r].nrow;
        local_int_t ncol = A[r].ncol;
        if (ncol > nrow)
            memset(x[r] + nrow, 0, (ncol - nrow) * sizeof(real_t));
    }
}

/* ================================================================
 * Section 9: Compute Kernels
 * ================================================================ */

/* SpMV: y = A*x for all ranks.
 * Always exchange halos — SpMV needs actual neighbor data regardless of
 * --halo setting.  (--halo only controls GS boundary behaviour.) */
static void spmv_all(int nranks, SparseMatrix *A, real_t **x, real_t **y) {
    exchange_halo_all(nranks, A, x);

    for (int r = 0; r < nranks; r++) {
        local_int_t nrow = A[r].nrow;
        const real_t *vals = A[r].values;
        const local_int_t *cols = A[r].colInd;
        const char *nnz_row = A[r].nnzPerRow;
        const real_t *xv = x[r];
        real_t *yv = y[r];

        for (local_int_t i = 0; i < nrow; i++) {
            real_t sum = (real_t)0.0;
            int nnz = nnz_row[i];
            int base = i * MAX_NONZEROS_PER_ROW;
            for (int j = 0; j < nnz; j++) {
                sum += vals[base + j] * xv[cols[base + j]];
            }
            yv[i] = sum;
        }
    }
}

/* WAXPBY: w = alpha*x + beta*y for all ranks */
static void waxpby_all(int nranks, const local_int_t *nrow_arr,
                       real_t alpha, real_t **x, real_t beta, real_t **y, real_t **w) {
    for (int r = 0; r < nranks; r++) {
        local_int_t n = nrow_arr[r];
        const real_t *xv = x[r];
        const real_t *yv = y[r];
        real_t *wv = w[r];
        if (alpha == (real_t)1.0) {
            for (local_int_t i = 0; i < n; i++)
                wv[i] = xv[i] + beta * yv[i];
        } else if (beta == (real_t)1.0) {
            for (local_int_t i = 0; i < n; i++)
                wv[i] = alpha * xv[i] + yv[i];
        } else {
            for (local_int_t i = 0; i < n; i++)
                wv[i] = alpha * xv[i] + beta * yv[i];
        }
    }
}

/* Dot product with simulated Allreduce(SUM) */
/*
 * Staged dot product matching hardware accumulation order:
 *   1. Each PE accumulates over Z (in real_t/float)  → X*Y partial sums
 *   2. Reduce partial sums in Y (in real_t/float)    → X partial sums
 *   3. Reduce partial sums in X (in real_t/float)    → scalar
 * Data layout: row = iz * nx * ny + iy * nx + ix
 */
static double dot_all(int nranks, const local_int_t *nrow_arr, real_t **x, real_t **y) {
    /* We need geometry; infer from nrow assuming single rank or uniform grid.
     * For multi-rank the caller should pass geometry, but for now this works
     * because all ranks have the same nx,ny,nz and we sum across ranks. */
    real_t global_result = (real_t)0.0;
    for (int r = 0; r < nranks; r++) {
        local_int_t n = nrow_arr[r];
        const real_t *xv = x[r];
        const real_t *yv = y[r];

        /* Simple fallback: flat accumulation in real_t */
        real_t local_sum = (real_t)0.0;
        if (xv == yv) {
            for (local_int_t i = 0; i < n; i++)
                local_sum += xv[i] * xv[i];
        } else {
            for (local_int_t i = 0; i < n; i++)
                local_sum += xv[i] * yv[i];
        }
        global_result += local_sum;
    }
    return (double)global_result;
}

/*
 * Geometry-aware staged dot product for the dot test path.
 * Matches hardware: accumulate Z per PE, reduce Y, reduce X.
 */
static double dot_all_staged(int nranks, SparseMatrix *A, real_t **x, real_t **y) {
    real_t global_result = (real_t)0.0;
    for (int r = 0; r < nranks; r++) {
        int nx = A[r].nx, ny = A[r].ny, nz = A[r].nz;
        const real_t *xv = x[r];
        const real_t *yv = y[r];

        /* Stage 1: accumulate over Z for each (ix, iy) → X*Y partial sums */
        real_t *pe_sums = (real_t *)calloc((size_t)nx * ny, sizeof(real_t));
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                real_t zsum = (real_t)0.0;
                if (xv == yv) {
                    for (int iz = 0; iz < nz; iz++) {
                        local_int_t idx = (local_int_t)iz * nx * ny + iy * nx + ix;
                        zsum += xv[idx] * xv[idx];
                    }
                } else {
                    for (int iz = 0; iz < nz; iz++) {
                        local_int_t idx = (local_int_t)iz * nx * ny + iy * nx + ix;
                        zsum += xv[idx] * yv[idx];
                    }
                }
                pe_sums[iy * nx + ix] = zsum;
            }
        }

        /* Stage 2: reduce in Y → X partial sums */
        real_t *col_sums = (real_t *)calloc(nx, sizeof(real_t));
        for (int ix = 0; ix < nx; ix++) {
            real_t ysum = (real_t)0.0;
            for (int iy = 0; iy < ny; iy++)
                ysum += pe_sums[iy * nx + ix];
            col_sums[ix] = ysum;
        }

        /* Stage 3: reduce in X → scalar */
        real_t rank_sum = (real_t)0.0;
        for (int ix = 0; ix < nx; ix++)
            rank_sum += col_sums[ix];

        global_result += rank_sum;
        free(pe_sums);
        free(col_sums);
    }
    return (double)global_result;
}

/* Forward Gauss-Seidel sweep for all ranks.
 * All Z-ranks proceed concurrently.  Initial Z-halo values are old.
 * After all ranks finish z-planes 0..nz-2, Z halos are refreshed so that
 * the last z-plane (iz=nz-1) sees the newly computed value from ipz+1. */
static void gs_fwd_all(int nranks, SparseMatrix *A, real_t **r, real_t **x) {
    int npz = A[0].npz;

    /* Exchange Z halos with old (current) values */
    if (npz > 1)
        for (int ipz = 0; ipz < npz; ipz++)
            exchange_halo_z_layer(nranks, A, x, ipz);

    /* All ranks: forward sweep over z-planes 0..nz-2 */
    for (int rk = 0; rk < nranks; rk++) {
        local_int_t nrow = A[rk].nrow;
        int nx = A[rk].nx, ny = A[rk].ny;
        local_int_t last_plane = nrow - nx * ny;
        const real_t *vals = A[rk].values;
        const local_int_t *cols = A[rk].colInd;
        const char *nnz_row = A[rk].nnzPerRow;
        const real_t *dg = A[rk].diag;
        const real_t *rv = r[rk];
        real_t *xv = x[rk];

        for (local_int_t i = 0; i < last_plane; i++) {
            real_t sum = rv[i];
            int nnz = nnz_row[i];
            int base = i * MAX_NONZEROS_PER_ROW;
            for (int j = 0; j < nnz; j++)
                sum -= vals[base + j] * xv[cols[base + j]];
            sum += xv[i] * dg[i];
            xv[i] = sum / dg[i];
        }
    }

    /* Refresh Z halos (boundary cells now carry newly computed values) */
    if (npz > 1)
        for (int ipz = 0; ipz < npz; ipz++)
            exchange_halo_z_layer(nranks, A, x, ipz);

    /* All ranks: forward sweep over last z-plane (iz=nz-1) */
    for (int rk = 0; rk < nranks; rk++) {
        local_int_t nrow = A[rk].nrow;
        int nx = A[rk].nx, ny = A[rk].ny;
        local_int_t last_plane = nrow - nx * ny;
        const real_t *vals = A[rk].values;
        const local_int_t *cols = A[rk].colInd;
        const char *nnz_row = A[rk].nnzPerRow;
        const real_t *dg = A[rk].diag;
        const real_t *rv = r[rk];
        real_t *xv = x[rk];

        for (local_int_t i = last_plane; i < nrow; i++) {
            real_t sum = rv[i];
            int nnz = nnz_row[i];
            int base = i * MAX_NONZEROS_PER_ROW;
            for (int j = 0; j < nnz; j++)
                sum -= vals[base + j] * xv[cols[base + j]];
            sum += xv[i] * dg[i];
            xv[i] = sum / dg[i];
        }
    }
}

/* Backward Gauss-Seidel sweep for all ranks.
 * All Z-ranks proceed concurrently.  Initial Z-halo values are old.
 * After all ranks finish z-planes nz-1..1, Z halos are refreshed so that
 * the first z-plane (iz=0) sees the newly computed value from ipz-1.
 * Ranks are traversed in decreasing Z order (high ipz first). */
static void gs_bwd_all(int nranks, SparseMatrix *A, real_t **r, real_t **x) {
    int npz = A[0].npz;

    /* Exchange Z halos with old (current) values */
    if (npz > 1)
        for (int ipz = 0; ipz < npz; ipz++)
            exchange_halo_z_layer(nranks, A, x, ipz);

    /* All ranks: backward sweep over z-planes nz-1..1 (high ipz first) */
    for (int rk = nranks - 1; rk >= 0; rk--) {
        local_int_t nrow = A[rk].nrow;
        int nx = A[rk].nx, ny = A[rk].ny;
        local_int_t first_plane_end = nx * ny;
        const real_t *vals = A[rk].values;
        const local_int_t *cols = A[rk].colInd;
        const char *nnz_row = A[rk].nnzPerRow;
        const real_t *dg = A[rk].diag;
        const real_t *rv = r[rk];
        real_t *xv = x[rk];

        for (local_int_t i = nrow - 1; i >= first_plane_end; i--) {
            real_t sum = rv[i];
            int nnz = nnz_row[i];
            int base = i * MAX_NONZEROS_PER_ROW;
            for (int j = 0; j < nnz; j++)
                sum -= vals[base + j] * xv[cols[base + j]];
            sum += xv[i] * dg[i];
            xv[i] = sum / dg[i];
        }
    }

    /* Refresh Z halos (boundary cells now carry newly computed values) */
    if (npz > 1)
        for (int ipz = 0; ipz < npz; ipz++)
            exchange_halo_z_layer(nranks, A, x, ipz);

    /* All ranks: backward sweep over first z-plane (iz=0, high ipz first) */
    for (int rk = nranks - 1; rk >= 0; rk--) {
        int nx = A[rk].nx, ny = A[rk].ny;
        local_int_t first_plane_end = nx * ny;
        const real_t *vals = A[rk].values;
        const local_int_t *cols = A[rk].colInd;
        const char *nnz_row = A[rk].nnzPerRow;
        const real_t *dg = A[rk].diag;
        const real_t *rv = r[rk];
        real_t *xv = x[rk];

        for (local_int_t i = first_plane_end - 1; i >= 0; i--) {
            real_t sum = rv[i];
            int nnz = nnz_row[i];
            int base = i * MAX_NONZEROS_PER_ROW;
            for (int j = 0; j < nnz; j++)
                sum -= vals[base + j] * xv[cols[base + j]];
            sum += xv[i] * dg[i];
            xv[i] = sum / dg[i];
        }
    }
}

/* Symmetric Gauss-Seidel for all ranks.
 * When --halo is set: exchange XY halos with current values, and Z halos
 *   are exchanged inside gs_fwd_all/gs_bwd_all.
 * When --halo is not set: clear all halo slots to zero so GS always sees
 *   zero boundary (matching hardware behaviour where boundary PEs get zeros). */
static void symgs_all(int nranks, SparseMatrix *A, real_t **r, real_t **x) {
    if (g_do_halo_exchange)
        exchange_halo_xy_all(nranks, A, x);
    else
        clear_halo_all(nranks, A, x);
    gs_fwd_all(nranks, A, r, x);
    gs_bwd_all(nranks, A, r, x);
}

/* ================================================================
 * Section 10: Multigrid Operations
 * ================================================================ */

static void restriction_all(int nranks, MGWork *mgw, real_t **rf,
                            const local_int_t *nc_arr) {
    for (int r = 0; r < nranks; r++) {
        local_int_t nc = nc_arr[r];
        const int *f2c = mgw[r].f2c;
        const real_t *rfv = rf[r];
        const real_t *Axfv = mgw[r].Axf;
        real_t *rcv = mgw[r].rc;
        for (local_int_t i = 0; i < nc; i++) {
            rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
        }
    }
}

static void prolongation_all(int nranks, MGWork *mgw, real_t **xf,
                             const local_int_t *nc_arr) {
    for (int r = 0; r < nranks; r++) {
        local_int_t nc = nc_arr[r];
        const int *f2c = mgw[r].f2c;
        const real_t *xcv = mgw[r].xc;
        real_t *xfv = xf[r];
        for (local_int_t i = 0; i < nc; i++) {
            xfv[f2c[i]] += xcv[i];
        }
    }
}

/* MG V-cycle for all ranks (recursive) */
static void mg_all(int nranks, int level, int numLevels,
                   SparseMatrix *mg_A,
                   MGWork *mg_work,
                   MGLevelPtrs *mg_ptrs,
                   real_t **r, real_t **x) {
    SparseMatrix *A = &mg_A[level * nranks];

    for (int rk = 0; rk < nranks; rk++)
        memset(x[rk], 0, A[rk].ncol * sizeof(real_t));

    if (level < numLevels - 1) {
        MGWork *work = &mg_work[level * nranks];
        MGLevelPtrs *ptrs = &mg_ptrs[level];
        SparseMatrix *Ac = &mg_A[(level + 1) * nranks];

        symgs_all(nranks, A, r, x);

        spmv_all(nranks, A, x, ptrs->Axf_ptrs);

        local_int_t *nc_arr = (local_int_t *)malloc(nranks * sizeof(local_int_t));
        for (int rk = 0; rk < nranks; rk++)
            nc_arr[rk] = Ac[rk].nrow;
        restriction_all(nranks, work, r, nc_arr);

        mg_all(nranks, level + 1, numLevels, mg_A, mg_work, mg_ptrs,
               ptrs->rc_ptrs, ptrs->xc_ptrs);

        prolongation_all(nranks, work, x, nc_arr);
        free(nc_arr);

        symgs_all(nranks, A, r, x);
    } else {
        symgs_all(nranks, A, r, x);
    }
}

/* Allocate and populate MGLevelPtrs array from mg_work. */
static MGLevelPtrs *create_mg_ptrs(int nranks, int numLevels, MGWork *mg_work) {
    if (numLevels <= 1) return NULL;
    MGLevelPtrs *mg_ptrs = (MGLevelPtrs *)malloc((numLevels - 1) * sizeof(MGLevelPtrs));
    for (int level = 0; level < numLevels - 1; level++) {
        mg_ptrs[level].Axf_ptrs = (real_t **)malloc(nranks * sizeof(real_t *));
        mg_ptrs[level].rc_ptrs = (real_t **)malloc(nranks * sizeof(real_t *));
        mg_ptrs[level].xc_ptrs = (real_t **)malloc(nranks * sizeof(real_t *));
        for (int r = 0; r < nranks; r++) {
            mg_ptrs[level].Axf_ptrs[r] = mg_work[level * nranks + r].Axf;
            mg_ptrs[level].rc_ptrs[r] = mg_work[level * nranks + r].rc;
            mg_ptrs[level].xc_ptrs[r] = mg_work[level * nranks + r].xc;
        }
    }
    return mg_ptrs;
}

static void free_mg_ptrs(MGLevelPtrs *mg_ptrs, int numLevels) {
    if (!mg_ptrs) return;
    for (int level = 0; level < numLevels - 1; level++) {
        free(mg_ptrs[level].Axf_ptrs);
        free(mg_ptrs[level].rc_ptrs);
        free(mg_ptrs[level].xc_ptrs);
    }
    free(mg_ptrs);
}

/* ================================================================
 * Section 11: CG Solver (All Ranks)
 * ================================================================ */

static void cg_all(int nranks,
                   SparseMatrix *A,
                   real_t **b,
                   real_t **x,
                   int max_iter,
                   int use_precond,
                   int numLevels, SparseMatrix *mg_A, MGWork *mg_work, MGLevelPtrs *mg_ptrs,
                   real_t **r_vec, real_t **z_vec, real_t **p_vec, real_t **Ap_vec,
                   local_int_t *nrow_arr) {

    double rtz = 0.0, oldrtz = 0.0;
    double alpha = 0.0, beta = 0.0, pAp = 0.0;
    double normr = 0.0;

    /* Copy x to p (p has ncol length for halo) */
    for (int r = 0; r < nranks; r++) {
        memcpy(p_vec[r], x[r], nrow_arr[r] * sizeof(real_t));
        memset(p_vec[r] + nrow_arr[r], 0, (A[r].ncol - nrow_arr[r]) * sizeof(real_t));
    }

    spmv_all(nranks, A, p_vec, Ap_vec);

    waxpby_all(nranks, nrow_arr, (real_t)1.0, b, (real_t)-1.0, Ap_vec, r_vec);

    double rr = dot_all(nranks, nrow_arr, r_vec, r_vec);
    normr = (rr > 0.0) ? sqrt(rr) : 0.0;
    double normr0 = normr;

    printf("Iteration %4d: ||r|| = %" REAL_FMT "  ||r||/||r0|| = %" REAL_FMT "\n",
           0, (double)normr, 1.0);

    for (int k = 1; k <= max_iter; k++) {
        if (use_precond)
            mg_all(nranks, 0, numLevels, mg_A, mg_work, mg_ptrs, r_vec, z_vec);
        else
            for (int r = 0; r < nranks; r++)
                memcpy(z_vec[r], r_vec[r], nrow_arr[r] * sizeof(real_t));

        if (k == 1) {
            for (int r = 0; r < nranks; r++)
                memcpy(p_vec[r], z_vec[r], A[r].ncol * sizeof(real_t));
            rtz = dot_all(nranks, nrow_arr, r_vec, z_vec);
        } else {
            oldrtz = rtz;
            rtz = dot_all(nranks, nrow_arr, r_vec, z_vec);
            beta = rtz / oldrtz;
            waxpby_all(nranks, nrow_arr, (real_t)1.0, z_vec, beta, p_vec, p_vec);
        }

        if (rtz == (real_t)0.0) {
            printf("Iteration %4d: rtz underflow, stopping\n", k);
            break;
        }

        spmv_all(nranks, A, p_vec, Ap_vec);

        pAp = dot_all(nranks, nrow_arr, p_vec, Ap_vec);
        alpha = rtz / pAp;

        waxpby_all(nranks, nrow_arr, (real_t)1.0, x, alpha, p_vec, x);
        waxpby_all(nranks, nrow_arr, (real_t)1.0, r_vec, -alpha, Ap_vec, r_vec);

        rr = dot_all(nranks, nrow_arr, r_vec, r_vec);
        normr = (rr > (real_t)0.0) ? REAL_SQRT(rr) : (real_t)0.0;

        printf("Iteration %4d: ||r|| = %" REAL_FMT "  ||r||/||r0|| = %" REAL_FMT "\n",
               k, (double)normr, (double)(normr / normr0));
        if (normr == (real_t)0.0) break;
    }
}

/* ================================================================
 * Section 12: Binary I/O
 * ================================================================ */

/* Write a raw real_t vector in global Y,X,Z order for all MG levels.
 * v contains data for the level specified by data_level; other levels are zeros.
 * For each (Y,X) column, level Z-values are concatenated: level 0 first, then 1, etc.
 * Coarser levels use full gnx*gny grid with zeros for inactive cells;
 * Z dimension has only active values (gnz >> level). */
static void save_vector(const char *filename, int nranks, int numLevels,
                        SparseMatrix *mg_A, real_t **v, int data_level,
                        int single_level) {
    long long gnx = mg_A[0].gnx, gny = mg_A[0].gny, gnz = mg_A[0].gnz;
    int L_start = (single_level >= 0) ? single_level : 0;
    int L_end   = (single_level >= 0) ? single_level + 1 : numLevels;
    long long total = 0;
    for (int L = L_start; L < L_end; L++)
        total += gnx * gny * (gnz >> L);
    real_t *buf = (real_t *)calloc((size_t)total, sizeof(real_t));
    long long out = 0;

    for (long long gy = 0; gy < gny; gy++) {
        for (long long gx = 0; gx < gnx; gx++) {
            for (int L = L_start; L < L_end; L++) {
                SparseMatrix *AL = &mg_A[L * nranks];
                int scale = 1 << L;
                long long gnz_L = gnz >> L;

                if (L == data_level && gy % scale == 0 && gx % scale == 0) {
                    int lnx = AL[0].nx, lny = AL[0].ny, lnz = AL[0].nz;
                    int npx = AL[0].npx, npy = AL[0].npy;
                    long long cgy = gy / scale, cgx = gx / scale;
                    int ipy = (int)(cgy / lny), ly = (int)(cgy % lny);
                    int ipx = (int)(cgx / lnx), lx = (int)(cgx % lnx);
                    for (long long gz = 0; gz < gnz_L; gz++) {
                        int ipz = (int)(gz / lnz), lz = (int)(gz % lnz);
                        int rank = ipx + ipy * npx + ipz * npx * npy;
                        int local_idx = lz * lnx * lny + ly * lnx + lx;
                        buf[out++] = v[rank][local_idx];
                    }
                } else {
                    out += gnz_L;  /* zeros from calloc */
                }
            }
        }
    }

    FILE *f = fopen(filename, "wb");
    if (f) { fwrite(buf, sizeof(real_t), (size_t)total, f); fclose(f); }
    else fprintf(stderr, "Error: cannot open %s for writing\n", filename);
    free(buf);
}

/*
 * Write a vector with Z-padding in global Y,X,Z order for all MG levels.
 * For each (Y,X) column, level Z-values (with per-level padding) are concatenated.
 * v contains data for the level specified by data_level; other levels are zeros.
 */
static void save_vector_zpad(const char *filename, int nranks, int numLevels,
                              SparseMatrix *mg_A, real_t **v, int data_level,
                              int single_level) {
    long long gnx = mg_A[0].gnx, gny = mg_A[0].gny, gnz = mg_A[0].gnz;
    int L_start = (single_level >= 0) ? single_level : 0;
    int L_end   = (single_level >= 0) ? single_level + 1 : numLevels;
    long long total = 0;
    for (int L = L_start; L < L_end; L++)
        total += gnx * gny * ((gnz >> L) + 2);
    real_t *buf = (real_t *)calloc((size_t)total, sizeof(real_t));
    long long out = 0;

    for (long long gy = 0; gy < gny; gy++) {
        for (long long gx = 0; gx < gnx; gx++) {
            for (int L = L_start; L < L_end; L++) {
                SparseMatrix *AL = &mg_A[L * nranks];
                int scale = 1 << L;
                long long gnz_L = gnz >> L;

                if (L == data_level && gy % scale == 0 && gx % scale == 0) {
                    int lnx = AL[0].nx, lny = AL[0].ny, lnz = AL[0].nz;
                    int npx = AL[0].npx, npy = AL[0].npy;
                    long long cgy = gy / scale, cgx = gx / scale;
                    int ipy = (int)(cgy / lny), ly = (int)(cgy % lny);
                    int ipx = (int)(cgx / lnx), lx = (int)(cgx % lnx);
                    buf[out++] = (real_t)0.0;  /* pad below */
                    for (long long gz = 0; gz < gnz_L; gz++) {
                        int ipz = (int)(gz / lnz), lz = (int)(gz % lnz);
                        int rank = ipx + ipy * npx + ipz * npx * npy;
                        int local_idx = lz * lnx * lny + ly * lnx + lx;
                        buf[out++] = v[rank][local_idx];
                    }
                    buf[out++] = (real_t)0.0;  /* pad above */
                } else {
                    out += (gnz_L + 2);  /* zeros from calloc */
                }
            }
        }
    }

    FILE *f = fopen(filename, "wb");
    if (f) { fwrite(buf, sizeof(real_t), (size_t)total, f); fclose(f); }
    else fprintf(stderr, "Error: cannot open %s for writing\n", filename);
    free(buf);
}

/* Write diagonal values in global Y,X,Z order for all MG levels.
 * For each (Y,X) column, level Z-values are concatenated.
 * Uses actual matrix diagonal at each level. */
static void save_diag(const char *filename, int nranks, int numLevels,
                      SparseMatrix *mg_A, int single_level) {
    long long gnx = mg_A[0].gnx, gny = mg_A[0].gny, gnz = mg_A[0].gnz;
    int L_start = (single_level >= 0) ? single_level : 0;
    int L_end   = (single_level >= 0) ? single_level + 1 : numLevels;
    long long total = 0;
    for (int L = L_start; L < L_end; L++)
        total += gnx * gny * (gnz >> L);
    real_t *buf = (real_t *)calloc((size_t)total, sizeof(real_t));
    long long out = 0;

    for (long long gy = 0; gy < gny; gy++) {
        for (long long gx = 0; gx < gnx; gx++) {
            for (int L = L_start; L < L_end; L++) {
                SparseMatrix *AL = &mg_A[L * nranks];
                int scale = 1 << L;
                long long gnz_L = gnz >> L;

                if (gy % scale == 0 && gx % scale == 0) {
                    int lnx = AL[0].nx, lny = AL[0].ny, lnz = AL[0].nz;
                    int npx = AL[0].npx, npy = AL[0].npy;
                    long long cgy = gy / scale, cgx = gx / scale;
                    int ipy = (int)(cgy / lny), ly = (int)(cgy % lny);
                    int ipx = (int)(cgx / lnx), lx = (int)(cgx % lnx);
                    for (long long gz = 0; gz < gnz_L; gz++) {
                        int ipz = (int)(gz / lnz), lz = (int)(gz % lnz);
                        int rank = ipx + ipy * npx + ipz * npx * npy;
                        int local_idx = lz * lnx * lny + ly * lnx + lx;
                        buf[out++] = AL[rank].diag[local_idx];
                    }
                } else {
                    out += gnz_L;  /* zeros from calloc */
                }
            }
        }
    }

    FILE *f = fopen(filename, "wb");
    if (f) { fwrite(buf, sizeof(real_t), (size_t)total, f); fclose(f); }
    else fprintf(stderr, "Error: cannot open %s for writing\n", filename);
    free(buf);
}

/* Write inverse diagonal values in global Y,X,Z order for all MG levels.
 * For each (Y,X) column, level Z-values are concatenated. */
static void save_invdiag(const char *filename, int nranks, int numLevels,
                          SparseMatrix *mg_A, int single_level) {
    long long gnx = mg_A[0].gnx, gny = mg_A[0].gny, gnz = mg_A[0].gnz;
    int L_start = (single_level >= 0) ? single_level : 0;
    int L_end   = (single_level >= 0) ? single_level + 1 : numLevels;
    long long total = 0;
    for (int L = L_start; L < L_end; L++)
        total += gnx * gny * (gnz >> L);
    real_t *buf = (real_t *)calloc((size_t)total, sizeof(real_t));
    long long out = 0;

    for (long long gy = 0; gy < gny; gy++) {
        for (long long gx = 0; gx < gnx; gx++) {
            for (int L = L_start; L < L_end; L++) {
                SparseMatrix *AL = &mg_A[L * nranks];
                int scale = 1 << L;
                long long gnz_L = gnz >> L;

                if (gy % scale == 0 && gx % scale == 0) {
                    int lnx = AL[0].nx, lny = AL[0].ny, lnz = AL[0].nz;
                    int npx = AL[0].npx, npy = AL[0].npy;
                    long long cgy = gy / scale, cgx = gx / scale;
                    int ipy = (int)(cgy / lny), ly = (int)(cgy % lny);
                    int ipx = (int)(cgx / lnx), lx = (int)(cgx % lnx);
                    for (long long gz = 0; gz < gnz_L; gz++) {
                        int ipz = (int)(gz / lnz), lz = (int)(gz % lnz);
                        int rank = ipx + ipy * npx + ipz * npx * npy;
                        int local_idx = lz * lnx * lny + ly * lnx + lx;
                        buf[out++] = (real_t)1.0 / AL[rank].diag[local_idx];
                    }
                } else {
                    out += gnz_L;  /* zeros from calloc */
                }
            }
        }
    }

    FILE *f = fopen(filename, "wb");
    if (f) { fwrite(buf, sizeof(real_t), (size_t)total, f); fclose(f); }
    else fprintf(stderr, "Error: cannot open %s for writing\n", filename);
    free(buf);
}

/*
 * Write off-diagonal values in global Y,X,Z order for all MG levels,
 * assuming zero-padded domain (27-point stencil at all points).
 * 26 values per row, blocked 24+2 in perm2 mode.
 * Coarser levels have no perturbation (all off-diag = -1.0 for active cells).
 */
static void save_offdiag(const char *filename, int nranks __attribute__((unused)),
                          int numLevels, SparseMatrix *mg_A, int single_level) {
    long long gnx = mg_A[0].gnx, gny = mg_A[0].gny, gnz = mg_A[0].gnz;
    int nz = mg_A[0].nz, npz = mg_A[0].npz;
    int do_perturb = (strcmp(g_init_mode, "random") == 0);
    int perm_mode = 0; /* 0=none, 1=perm1, 2=perm2 */
    if (strcmp(g_perm_mode, "perm1") == 0) perm_mode = 1;
    else if (strcmp(g_perm_mode, "perm2") == 0) perm_mode = 2;

    /* perm1 even-z order */
    static const int p1e[27][3] = {  /* {z, y, x} */
        {0,2,1},{2,2,1},{1,2,1},
        {0,0,2},{2,0,2},{1,0,2},
        {0,1,2},{2,1,2},{1,1,2},
        {0,2,2},{2,2,2},{1,2,2},
        {2,0,0},{1,0,0},{0,0,0},
        {2,1,0},{1,1,0},{0,1,0},
        {2,2,0},{1,2,0},{0,2,0},
        {2,0,1},{1,0,1},{0,0,1},
        {0,1,1},{1,1,1},{2,1,1}
    };
    /* perm1 odd-z order */
    static const int p1o[27][3] = {  /* {z, y, x} */
        {0,2,1},{1,2,1},{2,2,1},
        {0,0,2},{1,0,2},{2,0,2},
        {0,1,2},{1,1,2},{2,1,2},
        {0,2,2},{1,2,2},{2,2,2},
        {2,0,0},{0,0,0},{1,0,0},
        {2,1,0},{0,1,0},{1,1,0},
        {2,2,0},{0,2,0},{1,2,0},
        {2,0,1},{0,0,1},{1,0,1},
        {0,1,1},{1,1,1},{2,1,1}
    };
    /* perm2 order (same for all z) */
    static const int p2[27][3] = {  /* {z, y, x} */
        {0,2,1},{1,2,1},{2,2,1},
        {0,0,2},{1,0,2},{2,0,2},
        {0,1,2},{1,1,2},{2,1,2},
        {0,2,2},{1,2,2},{2,2,2},
        {0,0,0},{1,0,0},{2,0,0},
        {0,1,0},{1,1,0},{2,1,0},
        {0,2,0},{1,2,0},{2,2,0},
        {0,0,1},{1,0,1},{2,0,1},
        {0,1,1},{1,1,1},{2,1,1}
    };
    /* Natural order: sz outer, sy middle, sx inner */
    static const int pn[27][3] = {
        {0,0,0},{0,1,0},{0,2,0},
        {0,0,1},{0,1,1},{0,2,1},
        {0,0,2},{0,1,2},{0,2,2},
        {1,0,0},{1,1,0},{1,2,0},
        {1,0,1},{1,1,1},{1,2,1},
        {1,0,2},{1,1,2},{1,2,2},
        {2,0,0},{2,1,0},{2,2,0},
        {2,0,1},{2,1,1},{2,2,1},
        {2,0,2},{2,1,2},{2,2,2}
    };

    int L_start = (single_level >= 0) ? single_level : 0;
    int L_end   = (single_level >= 0) ? single_level + 1 : numLevels;
    long long count = 0;
    for (int L = L_start; L < L_end; L++)
        count += 26 * gnx * gny * (gnz >> L);

    real_t *buf = (real_t *)calloc((size_t)count, sizeof(real_t));
    long long out = 0;

    for (long long gy = 0; gy < gny; gy++) {
        for (long long gx = 0; gx < gnx; gx++) {
            for (int L = L_start; L < L_end; L++) {
                int scale = 1 << L;
                long long gnz_L = gnz >> L;
                int level_perturb = (L == 0) && do_perturb;

                if (gy % scale != 0 || gx % scale != 0) {
                    out += 26 * gnz_L;  /* zeros from calloc */
                    continue;
                }

                if (perm_mode == 2) {
                    long long nz_L = nz >> L;
                    for (long long lz = 0; lz < nz_L; lz++)
                        for (int ipz = 0; ipz < npz; ipz++) {
                            long long gz = (long long)ipz * nz_L + lz;
                            for (int k = 0; k < 24; k++) {
                                int dz = p2[k][0] - 1, dx = p2[k][1] - 1, dy = p2[k][2] - 1;
                                real_t val = (real_t)-1.0;
                                if (level_perturb) {
                                    real_t p = get_perturbation(gx, gy, gz, dz, dx, dy);
                                    val = (real_t)(-0.5 + 0.5 * (double)p);
                                }
                                buf[out++] = val;
                            }
                        }
                    for (long long lz = 0; lz < nz_L; lz++)
                        for (int ipz = 0; ipz < npz; ipz++) {
                            long long gz = (long long)ipz * nz_L + lz;
                            for (int k = 24; k < 27; k++) {
                                if (p2[k][0] == 1 && p2[k][1] == 1 && p2[k][2] == 1)
                                    continue;
                                int dz = p2[k][0] - 1, dx = p2[k][1] - 1, dy = p2[k][2] - 1;
                                real_t val = (real_t)-1.0;
                                if (level_perturb) {
                                    real_t p = get_perturbation(gx, gy, gz, dz, dx, dy);
                                    val = (real_t)(-0.5 + 0.5 * (double)p);
                                }
                                buf[out++] = val;
                            }
                        }
                } else {
                    long long nz_L = nz >> L;
                    for (long long lz = 0; lz < nz_L; lz++) {
                        for (int ipz = 0; ipz < npz; ipz++) {
                            long long gz = (long long)ipz * nz_L + lz;
                            const int (*perm)[3];
                            if (perm_mode == 1)
                                perm = (gz % 2 == 0) ? p1e : p1o;
                            else
                                perm = pn;
                            for (int k = 0; k < 27; k++) {
                                int dz = perm[k][0] - 1;
                                int dx = perm[k][1] - 1;
                                int dy = perm[k][2] - 1;
                                if (dz == 0 && dx == 0 && dy == 0) continue;
                                real_t val = (real_t)-1.0;
                                if (level_perturb) {
                                    real_t p = get_perturbation(gx, gy, gz, dz, dx, dy);
                                    val = (real_t)(-0.5 + 0.5 * (double)p);
                                }
                                buf[out++] = val;
                            }
                        }
                    }
                }
            }
        }
    }
    FILE *f = fopen(filename, "wb");
    if (f) { fwrite(buf, sizeof(real_t), (size_t)count, f); fclose(f); }
    else fprintf(stderr, "Error: cannot open %s for writing\n", filename);
    free(buf);
}

/*
 * Write full matrix (27 values per row, including diagonal) in global Y,X,Z
 * order with natural stencil ordering (sz outer, sy middle, sx inner)
 * for all MG levels.  Assumes zero-padded domain.
 */
static void save_matrix_full(const char *filename, int nranks __attribute__((unused)),
                              int numLevels, SparseMatrix *mg_A, int single_level) {
    long long gnx = mg_A[0].gnx, gny = mg_A[0].gny, gnz = mg_A[0].gnz;
    int do_perturb = (strcmp(g_init_mode, "random") == 0);
    int L_start = (single_level >= 0) ? single_level : 0;
    int L_end   = (single_level >= 0) ? single_level + 1 : numLevels;

    long long count = 0;
    for (int L = L_start; L < L_end; L++)
        count += 27 * gnx * gny * (gnz >> L);

    real_t *buf = (real_t *)calloc((size_t)count, sizeof(real_t));
    long long out = 0;

    for (long long gy = 0; gy < gny; gy++) {
        for (long long gx = 0; gx < gnx; gx++) {
            for (int L = L_start; L < L_end; L++) {
                int scale = 1 << L;
                long long gnz_L = gnz >> L;
                int level_perturb = (L == 0) && do_perturb;

                if (gy % scale == 0 && gx % scale == 0) {
                    for (long long gz = 0; gz < gnz_L; gz++) {
                        for (int dz = -1; dz <= 1; dz++) {
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    real_t val;
                                    real_t p = level_perturb ? get_perturbation(gx, gy, gz, dz, dx, dy) : (real_t)0.0;
                                    if (dz == 0 && dx == 0 && dy == 0)
                                        val = level_perturb ? (real_t)(18.0 + 8.0 * (double)p) : (real_t)26.0;
                                    else
                                        val = level_perturb ? (real_t)(-0.5 + 0.5 * (double)p) : (real_t)-1.0;
                                    buf[out++] = val;
                                }
                            }
                        }
                    }
                } else {
                    out += 27 * gnz_L;  /* zeros from calloc */
                }
            }
        }
    }
    FILE *f = fopen(filename, "wb");
    if (f) { fwrite(buf, sizeof(real_t), (size_t)count, f); fclose(f); }
    else fprintf(stderr, "Error: cannot open %s for writing\n", filename);
    free(buf);
}

/* ================================================================
 * Section 12b: Split-aware binary I/O wrappers
 *
 * When --bin_split is set, each level gets its own file (e.g. A0.bin).
 * Per-level files use the *finest*-level gnx*gny for X and Y (with zeros
 * for inactive cells at coarser levels) but only gnz>>L for the Z
 * dimension, matching the layout of the combined multi-level files.
 * ================================================================ */

/* Generate per-level filename: "A.bin" + level 2 -> "A2.bin" */
static void level_filename(char *out, size_t sz, const char *base, int level) {
    const char *dot = strrchr(base, '.');
    if (dot) {
        int plen = (int)(dot - base);
        snprintf(out, sz, "%.*s%d%s", plen, base, level, dot);
    } else {
        snprintf(out, sz, "%s%d", base, level);
    }
}

/* --- Dispatch wrappers: combined file or per-level split --- */

static void do_save_offdiag(const char *filename, int nranks, int numLevels, SparseMatrix *mg_A) {
    if (!g_bin_split) {
        save_offdiag(filename, nranks, numLevels, mg_A, -1);
        return;
    }
    char fname[256];
    for (int L = 0; L < numLevels; L++) {
        level_filename(fname, sizeof(fname), filename, L);
        save_offdiag(fname, nranks, numLevels, mg_A, L);
    }
}

static void do_save_matrix_full(const char *filename, int nranks, int numLevels, SparseMatrix *mg_A) {
    if (!g_bin_split) {
        save_matrix_full(filename, nranks, numLevels, mg_A, -1);
        return;
    }
    char fname[256];
    for (int L = 0; L < numLevels; L++) {
        level_filename(fname, sizeof(fname), filename, L);
        save_matrix_full(fname, nranks, numLevels, mg_A, L);
    }
}

static void do_save_diag(const char *filename, int nranks, int numLevels, SparseMatrix *mg_A) {
    if (!g_bin_split) {
        save_diag(filename, nranks, numLevels, mg_A, -1);
        return;
    }
    char fname[256];
    for (int L = 0; L < numLevels; L++) {
        level_filename(fname, sizeof(fname), filename, L);
        save_diag(fname, nranks, numLevels, mg_A, L);
    }
}

static void do_save_invdiag(const char *filename, int nranks, int numLevels, SparseMatrix *mg_A) {
    if (!g_bin_split) {
        save_invdiag(filename, nranks, numLevels, mg_A, -1);
        return;
    }
    char fname[256];
    for (int L = 0; L < numLevels; L++) {
        level_filename(fname, sizeof(fname), filename, L);
        save_invdiag(fname, nranks, numLevels, mg_A, L);
    }
}

static void do_save_vector(const char *filename, int nranks, int numLevels,
                           SparseMatrix *mg_A, real_t **v, int data_level) {
    if (!g_bin_split) {
        save_vector(filename, nranks, numLevels, mg_A, v, data_level, -1);
        return;
    }
    char fname[256];
    for (int L = 0; L < numLevels; L++) {
        level_filename(fname, sizeof(fname), filename, L);
        save_vector(fname, nranks, numLevels, mg_A, v, data_level, L);
    }
}

static void do_save_vector_zpad(const char *filename, int nranks, int numLevels,
                                SparseMatrix *mg_A, real_t **v, int data_level) {
    if (!g_bin_split) {
        save_vector_zpad(filename, nranks, numLevels, mg_A, v, data_level, -1);
        return;
    }
    char fname[256];
    for (int L = 0; L < numLevels; L++) {
        level_filename(fname, sizeof(fname), filename, L);
        save_vector_zpad(fname, nranks, numLevels, mg_A, v, data_level, L);
    }
}

/* ================================================================
 * Section 13: Main Driver
 * ================================================================ */

static void free_matrix(SparseMatrix *A) {
    free(A->nnzPerRow);
    free(A->values);
    free(A->colInd);
    free(A->diag);
    free(A->localToGlobalMap);
    free(A->colIndG);
    free(A->neighbors);
    free(A->recvLength);
    free(A->sendLength);
    free(A->elementsToSend);
}

int main(int argc, char **argv) {
    int npx = 1, npy = 1, npz = 1;
    int nx = 16, ny = 16, nz = 16;
    int maxIters = 50;

    {
        int pos = 0;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--halo") == 0) { g_do_halo_exchange = 1; continue; }
            if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) { g_test_mode = argv[++i]; continue; }
            if (strcmp(argv[i], "--init") == 0 && i + 1 < argc) { g_init_mode = argv[++i]; continue; }
            if (strcmp(argv[i], "--perm") == 0 && i + 1 < argc) { g_perm_mode = argv[++i]; continue; }
            if (strcmp(argv[i], "--levels") == 0 && i + 1 < argc) { g_num_mg_levels = atoi(argv[++i]); continue; }
            if (strcmp(argv[i], "--test-level") == 0 && i + 1 < argc) { g_test_level = atoi(argv[++i]); continue; }
            if (strcmp(argv[i], "--bin_split") == 0) { g_bin_split = 1; continue; }
            pos++;
            if (pos == 1) npx = atoi(argv[i]);
            else if (pos == 2) npy = atoi(argv[i]);
            else if (pos == 3) npz = atoi(argv[i]);
            else if (pos == 4) nx = atoi(argv[i]);
            else if (pos == 5) ny = atoi(argv[i]);
            else if (pos == 6) nz = atoi(argv[i]);
            else if (pos == 7) maxIters = atoi(argv[i]);
        }
    }

    if (npx < 1) npx = 1;
    if (npy < 1) npy = 1;
    if (npz < 1) npz = 1;
    int nranks = npx * npy * npz;
    if (nx < 1 || ny < 1 || nz < 1) {
        fprintf(stderr, "Error: nx, ny, nz must be >= 1\n");
        return 1;
    }
    if (maxIters < 1) maxIters = 1;
    if (g_num_mg_levels < 1) g_num_mg_levels = 1;

    if (strcmp(g_test_mode, "hpcg") != 0 && strcmp(g_test_mode, "cg") != 0 &&
        strcmp(g_test_mode, "vcycle") != 0 &&
        strcmp(g_test_mode, "spmv") != 0 &&
        strcmp(g_test_mode, "dot") != 0 &&
        strcmp(g_test_mode, "gs_fwd") != 0 && strcmp(g_test_mode, "gs_bwd") != 0 &&
        strcmp(g_test_mode, "gs_sym") != 0) {
        fprintf(stderr, "Error: unknown test mode '%s' (use hpcg, cg, vcycle, spmv, dot, gs_fwd, gs_bwd, or gs_sym)\n", g_test_mode);
        return 1;
    }
    if (strcmp(g_init_mode, "random") != 0 && strcmp(g_init_mode, "fixed") != 0) {
        fprintf(stderr, "Error: unknown init mode '%s' (use random or fixed)\n", g_init_mode);
        return 1;
    }
    if (strcmp(g_perm_mode, "none") != 0 && strcmp(g_perm_mode, "perm1") != 0 &&
        strcmp(g_perm_mode, "perm2") != 0) {
        fprintf(stderr, "Error: unknown perm mode '%s' (use none, perm1, or perm2)\n", g_perm_mode);
        return 1;
    }
    if (g_num_mg_levels > 1) {
        int divisor = 1 << (g_num_mg_levels - 1);
        if (nx % divisor != 0 || ny % divisor != 0 || nz % divisor != 0) {
            fprintf(stderr, "Error: nx, ny, nz must be divisible by %d for %d MG levels\n",
                    divisor, g_num_mg_levels);
            return 1;
        }
    }
    if (g_test_level < 0 || g_test_level >= g_num_mg_levels) {
        fprintf(stderr, "Error: --test-level %d must be in [0, %d) (number of levels)\n",
                g_test_level, g_num_mg_levels);
        return 1;
    }

    printf("HPCG Reference Computation\n");
    printf("  Test mode: %s\n", g_test_mode);
    printf("  Init mode: %s\n", g_init_mode);
    printf("  Perm mode: %s\n", g_perm_mode);
    printf("  Precision: %s\n", REAL_NAME);
    printf("  Simulated ranks: %d (%dx%dx%d process grid)\n", nranks, npx, npy, npz);
    printf("  Local grid per rank: %dx%dx%d\n", nx, ny, nz);
    printf("  Global grid: %lldx%lldx%lld\n",
           (long long)npx * nx, (long long)npy * ny, (long long)npz * nz);
    printf("  MG levels: %d\n", g_num_mg_levels);
    if (g_bin_split)
        printf("  Binary output: split by level\n");
    if (strcmp(g_test_mode, "hpcg") == 0 || strcmp(g_test_mode, "cg") == 0 ||
        strcmp(g_test_mode, "vcycle") == 0) {
        printf("  CG iterations: %d\n", maxIters);
    }
    printf("  Halo exchange: %s\n", g_do_halo_exchange ? "enabled" : "disabled");
    printf("\n");

    /* ----- Common setup: finest-level matrix, b, x ----- */

    SparseMatrix *A = (SparseMatrix *)calloc(nranks, sizeof(SparseMatrix));

    real_t **b_arr = (real_t **)malloc(nranks * sizeof(real_t *));
    real_t **x_arr = (real_t **)malloc(nranks * sizeof(real_t *));
    real_t **xexact_arr = (real_t **)malloc(nranks * sizeof(real_t *));

    double setup_time = mytimer();

    if (strcmp(g_init_mode, "random") == 0)
        init_perturbation_table((long long)npx * nx, (long long)npy * ny,
                                (long long)npz * nz);

    printf("Setting up problem...\n");
    for (int r = 0; r < nranks; r++) {
        generate_geometry(nranks, r, nx, ny, nz, npx, npy, npz, &A[r]);
        local_int_t nrow = nx * ny * nz;
        b_arr[r] = (real_t *)calloc(nrow, sizeof(real_t));
        x_arr[r] = (real_t *)calloc(nrow, sizeof(real_t));
        xexact_arr[r] = (real_t *)calloc(nrow, sizeof(real_t));
        int dot_mode = (strcmp(g_test_mode, "dot") == 0);
        generate_problem(&A[r], b_arr[r], x_arr[r], xexact_arr[r], dot_mode);
        if (!dot_mode)
            setup_halo(&A[r]);
    }

    {
        global_int_t totalNnz = 0;
        for (int r = 0; r < nranks; r++)
            totalNnz += A[r].localNnz;
        for (int r = 0; r < nranks; r++)
            A[r].totalNnz = totalNnz;
    }

    /* Reallocate x to have ncol (includes halo space) */
    for (int r = 0; r < nranks; r++) {
        if (A[r].ncol > A[r].nrow) {
            real_t *new_x = (real_t *)calloc(A[r].ncol, sizeof(real_t));
            memcpy(new_x, x_arr[r], A[r].nrow * sizeof(real_t));
            free(x_arr[r]);
            x_arr[r] = new_x;
        }
    }

    setup_time = mytimer() - setup_time;
    printf("  Setup time: %.3f seconds\n", setup_time);

    /* ----- Dispatch by test mode ----- */

    /* ----- Build MG hierarchy (both modes need it for multi-level output) ----- */

    int numLevels = g_num_mg_levels;

    SparseMatrix *mg_A = (SparseMatrix *)calloc(numLevels * nranks, sizeof(SparseMatrix));
    MGWork *mg_work = NULL;
    if (numLevels > 1)
        mg_work = (MGWork *)calloc((numLevels - 1) * nranks, sizeof(MGWork));

    /* Copy finest level into mg_A[0..nranks-1] */
    memcpy(mg_A, A, nranks * sizeof(SparseMatrix));

    for (int level = 0; level < numLevels - 1; level++) {
        SparseMatrix *Af = &mg_A[level * nranks];
        SparseMatrix *Ac = &mg_A[(level + 1) * nranks];
        MGWork *work = &mg_work[level * nranks];
        generate_coarse_problem(nranks, Af, Ac, work);
    }

    if (strcmp(g_test_mode, "dot") == 0) {
        /*
         * Dot product test:
         *   Generate two random vectors a and b, compute dot(a,b),
         *   save a.bin, b.bin, dot_out.bin.
         */
        printf("\nDot product test (Z=%d per PE, %d ranks)...\n", nz, nranks);

        /* Allocate per-rank vectors */
        real_t **a_vec = (real_t **)malloc(nranks * sizeof(real_t *));
        real_t **b_vec = (real_t **)malloc(nranks * sizeof(real_t *));
        local_int_t *nrow_arr = (local_int_t *)malloc(nranks * sizeof(local_int_t));
        for (int r = 0; r < nranks; r++) {
            local_int_t nrow_r = A[r].nrow;
            nrow_arr[r] = nrow_r;
            a_vec[r] = (real_t *)malloc(nrow_r * sizeof(real_t));
            b_vec[r] = (real_t *)malloc(nrow_r * sizeof(real_t));
            /* Fill vectors */
            for (local_int_t i = 0; i < nrow_r; i++) {
                if (strcmp(g_init_mode, "fixed") == 0) {
                    a_vec[r][i] = (real_t)1.0;
                    b_vec[r][i] = (real_t)1.0;
                } else {
                    a_vec[r][i] = b_arr[r][i];     /* use RHS as random source */
                    b_vec[r][i] = xexact_arr[r][i]; /* use xexact as random source */
                }
            }
        }

        /* Compute reference dot product (staged to match hardware) */
        double dot_result_d = dot_all_staged(nranks, A, a_vec, b_vec);
        real_t dot_result = (real_t)dot_result_d;
        printf("  dot(a,b) = %.10e (double: %.10e)\n", (double)dot_result, dot_result_d);

        /* Save a.bin and b.bin in Y,X,Z order (no padding) */
        do_save_vector("a.bin", nranks, numLevels, mg_A, a_vec, 0);
        do_save_vector("b.bin", nranks, numLevels, mg_A, b_vec, 0);

        /* Save scalar result replicated for each PE (nx*ny copies) */
        {
            int num_pes = nx * ny;
            FILE *f = fopen("dot_out.bin", "wb");
            if (f) {
                for (int i = 0; i < num_pes; i++)
                    fwrite(&dot_result, sizeof(real_t), 1, f);
                fclose(f);
            }
        }
        printf("  Saved a.bin, b.bin, dot_out.bin\n");

        for (int r = 0; r < nranks; r++) {
            free(a_vec[r]);
            free(b_vec[r]);
        }
        free(a_vec); free(b_vec); free(nrow_arr);

    } else if (strcmp(g_test_mode, "spmv") == 0) {
        /*
         * SpMV test at the specified MG level:
         *   inputs:  x = xexact (all 1s), with halo space
         *   outputs: y = A*x
         */
        int test_level = g_test_level;
        SparseMatrix *A_test = &mg_A[test_level * nranks];
        real_t **x_test, **y_test;
        int own_test_vecs = 0;

        if (test_level == 0) {
            x_test = x_arr;
            /* Set x = xexact (all 1s) */
            for (int r = 0; r < nranks; r++) {
                for (local_int_t i = 0; i < A_test[r].nrow; i++)
                    x_test[r][i] = (real_t)1.0;
                for (local_int_t i = A_test[r].nrow; i < A_test[r].ncol; i++)
                    x_test[r][i] = (real_t)0.0;
            }
        } else {
            own_test_vecs = 1;
            x_test = (real_t **)malloc(nranks * sizeof(real_t *));
            for (int r = 0; r < nranks; r++) {
                x_test[r] = (real_t *)calloc(A_test[r].ncol, sizeof(real_t));
                for (local_int_t i = 0; i < A_test[r].nrow; i++)
                    x_test[r][i] = (real_t)1.0;
            }
        }

        y_test = (real_t **)malloc(nranks * sizeof(real_t *));
        for (int r = 0; r < nranks; r++)
            y_test[r] = (real_t *)calloc(A_test[r].nrow, sizeof(real_t));

        do_save_offdiag("A.bin", nranks, numLevels, mg_A);
        do_save_matrix_full("A_orig.bin", nranks, numLevels, mg_A);
        do_save_diag("diag.bin", nranks, numLevels, mg_A);
        do_save_invdiag("invdiag.bin", nranks, numLevels, mg_A);
        do_save_vector_zpad("x.bin", nranks, numLevels, mg_A, x_test, test_level);
        if (g_bin_split)
            printf("  Saved A[0..%d].bin, A_orig[0..%d].bin, diag[0..%d].bin, invdiag[0..%d].bin, x[0..%d].bin\n",
                   numLevels-1, numLevels-1, numLevels-1, numLevels-1, numLevels-1);
        else
            printf("  Saved A.bin, A_orig.bin, diag.bin, invdiag.bin, x.bin\n");

        printf("\nRunning SpMV y = A*x (level %d)...\n", test_level);
        spmv_all(nranks, A_test, x_test, y_test);

        do_save_vector("y.bin", nranks, numLevels, mg_A, y_test, test_level);
        if (g_bin_split)
            printf("Saved y[0..%d].bin\n", numLevels-1);
        else
            printf("Saved y.bin\n");

        for (int r = 0; r < nranks; r++)
            free(y_test[r]);
        free(y_test);

        if (own_test_vecs) {
            for (int r = 0; r < nranks; r++)
                free(x_test[r]);
            free(x_test);
        }

    } else if (strcmp(g_test_mode, "gs_fwd") == 0 ||
               strcmp(g_test_mode, "gs_bwd") == 0 ||
               strcmp(g_test_mode, "gs_sym") == 0) {
        /*
         * Gauss-Seidel test at the specified MG level:
         *   gs_fwd: forward sweep only
         *   gs_bwd: backward sweep only
         *   gs_sym: forward + backward sweep
         */
        int test_level = g_test_level;
        SparseMatrix *A_test = &mg_A[test_level * nranks];
        real_t **b_test, **x_test;
        int own_test_vecs = 0;

        if (test_level == 0) {
            b_test = b_arr;
            x_test = x_arr;
        } else {
            own_test_vecs = 1;
            b_test = (real_t **)malloc(nranks * sizeof(real_t *));
            x_test = (real_t **)malloc(nranks * sizeof(real_t *));
            for (int r = 0; r < nranks; r++) {
                local_int_t nrow_t = A_test[r].nrow;
                b_test[r] = (real_t *)calloc(nrow_t, sizeof(real_t));
                x_test[r] = (real_t *)calloc(A_test[r].ncol, sizeof(real_t));
                for (local_int_t i = 0; i < nrow_t; i++)
                    b_test[r][i] = (real_t)(26.0 - (double)(A_test[r].nnzPerRow[i] - 1));
            }
        }

        do_save_offdiag("A.bin", nranks, numLevels, mg_A);
        do_save_matrix_full("A_orig.bin", nranks, numLevels, mg_A);
        do_save_diag("diag.bin", nranks, numLevels, mg_A);
        do_save_invdiag("invdiag.bin", nranks, numLevels, mg_A);
        do_save_vector_zpad("x.bin", nranks, numLevels, mg_A, x_test, test_level);
        do_save_vector("r.bin", nranks, numLevels, mg_A, b_test, test_level);
        if (g_bin_split)
            printf("  Saved A[0..%d].bin, A_orig[0..%d].bin, diag[0..%d].bin, invdiag[0..%d].bin, x[0..%d].bin, r[0..%d].bin\n",
                   numLevels-1, numLevels-1, numLevels-1, numLevels-1, numLevels-1, numLevels-1);
        else
            printf("  Saved A.bin, A_orig.bin, diag.bin, invdiag.bin, x.bin, r.bin\n");

        printf("\nRunning %s Gauss-Seidel (level %d)...\n", g_test_mode, test_level);
        if (g_do_halo_exchange)
            exchange_halo_xy_all(nranks, A_test, x_test);
        else
            clear_halo_all(nranks, A_test, x_test);
        if (strcmp(g_test_mode, "gs_bwd") != 0)
            gs_fwd_all(nranks, A_test, b_test, x_test);
        if (strcmp(g_test_mode, "gs_fwd") != 0)
            gs_bwd_all(nranks, A_test, b_test, x_test);

        do_save_vector_zpad("x_out.bin", nranks, numLevels, mg_A, x_test, test_level);
        if (g_bin_split)
            printf("Saved x_out[0..%d].bin\n", numLevels-1);
        else
            printf("Saved x_out.bin\n");

        if (own_test_vecs) {
            for (int r = 0; r < nranks; r++) {
                free(b_test[r]);
                free(x_test[r]);
            }
            free(b_test);
            free(x_test);
        }

    } else if (strcmp(g_test_mode, "vcycle") == 0) {
        /*
         * Multigrid V-cycle test:
         *   inputs:  r = b (RHS), x = 0 (with halo space)
         *   outputs: x after one V-cycle
         */
        do_save_offdiag("A.bin", nranks, numLevels, mg_A);
        do_save_matrix_full("A_orig.bin", nranks, numLevels, mg_A);
        do_save_diag("diag.bin", nranks, numLevels, mg_A);
        do_save_invdiag("invdiag.bin", nranks, numLevels, mg_A);
        do_save_vector("r.bin", nranks, numLevels, mg_A, b_arr, 0);

        /* Zero x before saving initial state */
        for (int r = 0; r < nranks; r++)
            memset(x_arr[r], 0, mg_A[r].ncol * sizeof(real_t));
        do_save_vector_zpad("x.bin", nranks, numLevels, mg_A, x_arr, 0);

        if (g_bin_split)
            printf("  Saved A[0..%d].bin, A_orig[0..%d].bin, diag[0..%d].bin, invdiag[0..%d].bin, r[0..%d].bin, x[0..%d].bin\n",
                   numLevels-1, numLevels-1, numLevels-1, numLevels-1, numLevels-1, numLevels-1);
        else
            printf("  Saved A.bin, A_orig.bin, diag.bin, invdiag.bin, r.bin, x.bin\n");

        /* Set up MGLevelPtrs */
        MGLevelPtrs *mg_ptrs = create_mg_ptrs(nranks, numLevels, mg_work);

        printf("\nRunning multigrid V-cycle (%d levels)...\n", numLevels);
        mg_all(nranks, 0, numLevels, mg_A, mg_work, mg_ptrs, b_arr, x_arr);

        do_save_vector_zpad("x_out.bin", nranks, numLevels, mg_A, x_arr, 0);
        if (g_bin_split)
            printf("Saved x_out[0..%d].bin\n", numLevels-1);
        else
            printf("Saved x_out.bin\n");

        free_mg_ptrs(mg_ptrs, numLevels);

    } else if (strcmp(g_test_mode, "cg") == 0) {
        /* ----- CG solver (no preconditioner) ----- */

        do_save_diag("diag.bin", nranks, numLevels, mg_A);
        do_save_offdiag("A.bin", nranks, numLevels, mg_A);
        do_save_matrix_full("A_orig.bin", nranks, numLevels, mg_A);
        do_save_vector("r.bin", nranks, numLevels, mg_A, b_arr, 0);
        if (g_bin_split)
            printf("  Saved diag[0..%d].bin, A[0..%d].bin, A_orig[0..%d].bin, r[0..%d].bin\n",
                   numLevels-1, numLevels-1, numLevels-1, numLevels-1);
        else
            printf("  Saved diag.bin, A.bin, A_orig.bin, r.bin\n");

        real_t **r_vec = (real_t **)malloc(nranks * sizeof(real_t *));
        real_t **z_vec = (real_t **)malloc(nranks * sizeof(real_t *));
        real_t **p_vec = (real_t **)malloc(nranks * sizeof(real_t *));
        real_t **Ap_vec = (real_t **)malloc(nranks * sizeof(real_t *));
        local_int_t *nrow_arr = (local_int_t *)malloc(nranks * sizeof(local_int_t));

        for (int r = 0; r < nranks; r++) {
            local_int_t nrow_r = mg_A[r].nrow;
            local_int_t ncol_r = mg_A[r].ncol;
            nrow_arr[r] = nrow_r;
            r_vec[r] = (real_t *)calloc(nrow_r, sizeof(real_t));
            z_vec[r] = (real_t *)calloc(ncol_r, sizeof(real_t));
            p_vec[r] = (real_t *)calloc(ncol_r, sizeof(real_t));
            Ap_vec[r] = (real_t *)calloc(nrow_r, sizeof(real_t));
        }

        printf("\nCG solve (no preconditioner, %d iterations):\n", maxIters);

        for (int r = 0; r < nranks; r++)
            memset(x_arr[r], 0, mg_A[r].ncol * sizeof(real_t));

        cg_all(nranks, mg_A, b_arr, x_arr, maxIters,
               0 /* use_precond */, numLevels, mg_A, mg_work, NULL,
               r_vec, z_vec, p_vec, Ap_vec,
               nrow_arr);

        do_save_vector_zpad("x.bin", nranks, numLevels, mg_A, x_arr, 0);
        if (g_bin_split)
            printf("\nSaved x[0..%d].bin\n", numLevels-1);
        else
            printf("\nSaved x.bin\n");

        for (int r = 0; r < nranks; r++) {
            free(r_vec[r]);
            free(z_vec[r]);
            free(p_vec[r]);
            free(Ap_vec[r]);
        }
        free(r_vec); free(z_vec); free(p_vec); free(Ap_vec);
        free(nrow_arr);

    } else {
        /* ----- HPCG benchmark (CG + MG preconditioner) ----- */

        /* Save per-level input data for hardware */
        do_save_diag("diag.bin", nranks, numLevels, mg_A);
        do_save_invdiag("invdiag.bin", nranks, numLevels, mg_A);
        do_save_offdiag("A.bin", nranks, numLevels, mg_A);
        do_save_matrix_full("A_orig.bin", nranks, numLevels, mg_A);

        /* Save initial x = zeros (z-padded) per level */
        for (int r = 0; r < nranks; r++)
            memset(x_arr[r], 0, mg_A[r].ncol * sizeof(real_t));
        if (g_bin_split) {
            for (int L = 0; L < numLevels; L++) {
                char fname[256];
                level_filename(fname, sizeof(fname), "x.bin", L);
                save_vector_zpad(fname, nranks, numLevels, mg_A, x_arr, 0, L);
            }
        } else {
            do_save_vector_zpad("x.bin", nranks, numLevels, mg_A, x_arr, 0);
        }

        /* Save initial r: level 0 = b (RHS), coarser levels = zeros.
         * Level 0 is saved as r_in.bin; levels 1+ as r{L}.bin. */
        if (g_bin_split) {
            save_vector("r_in.bin", nranks, numLevels, mg_A, b_arr, 0, 0);
            for (int L = 1; L < numLevels; L++) {
                char fname[256];
                level_filename(fname, sizeof(fname), "r.bin", L);
                save_vector(fname, nranks, numLevels, mg_A, b_arr, -1, L);
            }
        } else {
            do_save_vector("b.bin", nranks, numLevels, mg_A, b_arr, 0);
        }

        if (g_bin_split)
            printf("  Saved diag[0..%d].bin, invdiag[0..%d].bin, A[0..%d].bin, A_orig[0..%d].bin, x[0..%d].bin, r_in.bin, r[1..%d].bin\n",
                   numLevels-1, numLevels-1, numLevels-1, numLevels-1,
                   numLevels-1, numLevels-1);
        else
            printf("  Saved diag.bin, invdiag.bin, A.bin, A_orig.bin, x.bin, b.bin\n");

        /* Set up MG hierarchy pointers */
        MGLevelPtrs *mg_ptrs = create_mg_ptrs(nranks, numLevels, mg_work);

        real_t **r_vec = (real_t **)malloc(nranks * sizeof(real_t *));
        real_t **z_vec = (real_t **)malloc(nranks * sizeof(real_t *));
        real_t **p_vec = (real_t **)malloc(nranks * sizeof(real_t *));
        real_t **Ap_vec = (real_t **)malloc(nranks * sizeof(real_t *));
        local_int_t *nrow_arr = (local_int_t *)malloc(nranks * sizeof(local_int_t));

        for (int r = 0; r < nranks; r++) {
            local_int_t nrow_r = mg_A[r].nrow;
            local_int_t ncol_r = mg_A[r].ncol;
            nrow_arr[r] = nrow_r;
            r_vec[r] = (real_t *)calloc(nrow_r, sizeof(real_t));
            z_vec[r] = (real_t *)calloc(ncol_r, sizeof(real_t));
            p_vec[r] = (real_t *)calloc(ncol_r, sizeof(real_t));
            Ap_vec[r] = (real_t *)calloc(nrow_r, sizeof(real_t));
        }

        printf("\nCG solve (MG preconditioner, %d iterations, %d levels):\n",
               maxIters, numLevels);

        for (int r = 0; r < nranks; r++)
            memset(x_arr[r], 0, mg_A[r].ncol * sizeof(real_t));

        cg_all(nranks, mg_A, b_arr, x_arr, maxIters,
               1 /* use_precond */, numLevels, mg_A, mg_work, mg_ptrs,
               r_vec, z_vec, p_vec, Ap_vec,
               nrow_arr);

        /* Save CG solution */
        if (g_bin_split) {
            save_vector_zpad("out_xx.bin", nranks, numLevels, mg_A, x_arr, 0, 0);
            printf("\nSaved out_xx.bin\n");
        } else {
            do_save_vector_zpad("x.bin", nranks, numLevels, mg_A, x_arr, 0);
            printf("\nSaved x.bin\n");
        }

        /* Cleanup CG resources */
        for (int r = 0; r < nranks; r++) {
            free(r_vec[r]);
            free(z_vec[r]);
            free(p_vec[r]);
            free(Ap_vec[r]);
        }
        free(r_vec); free(z_vec); free(p_vec); free(Ap_vec);
        free(nrow_arr);

        free_mg_ptrs(mg_ptrs, numLevels);
    }

    /* Cleanup MG hierarchy */
    if (mg_work) {
        for (int level = 0; level < numLevels - 1; level++) {
            for (int r = 0; r < nranks; r++) {
                free(mg_work[level * nranks + r].f2c);
                free(mg_work[level * nranks + r].Axf);
                free(mg_work[level * nranks + r].rc);
                free(mg_work[level * nranks + r].xc);
            }
        }
        free(mg_work);
    }
    for (int level = 1; level < numLevels; level++)
        for (int r = 0; r < nranks; r++)
            free_matrix(&mg_A[level * nranks + r]);
    free(mg_A);

    free_perturbation_table();

    /* Common cleanup */
    for (int r = 0; r < nranks; r++) {
        free(b_arr[r]);
        free(x_arr[r]);
        free(xexact_arr[r]);
        free_matrix(&A[r]);
    }
    free(b_arr); free(x_arr); free(xexact_arr);
    free(A);

    return 0;
}
