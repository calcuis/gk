// The matmuls and fused attention.
//
// These are separated from the other kernels because they are the only ones
// whose shape is chosen for speed rather than for clarity. Everything else in
// this backend is one thread per output element; that is the wrong shape here,
// because a matmul's output element is a whole reduction over k and a weight
// row read once per output element is the entire memory traffic of inference.
//
// So the unit of work is a block per output element (or per small group of
// them), the reduction is a block reduction, and the weight row is read once
// and used for several activation columns. The quantized formats are decoded
// on the fly rather than expanded into a temporary: a 4096x4096 q4_K weight is
// 9 MB packed and 64 MB as f32, and materialising that per matmul would spend
// more bandwidth than the decode costs arithmetic.
//
// The tensor-core paths came last and in that order: the clear implementation
// first, checked against the CPU, and the fast one after, checked against the
// clear one. There are two of them - one for nvfp4 through the integer
// instruction, one for f16 - and both are gated so that any shape, type or
// device they do not cover falls through to the float tile, which still
// computes the same thing.

#include "gk_cuda_ops.cuh"

#include <float.h>

#define GK_CU_MM_BLOCK 128

// Blocks to cover n items, uncapped: the activation quantizer indexes a flat
// range and needs the whole grid, not gk_cu_blocks' first 65535 of it.
static __host__ __forceinline__ unsigned gk_cu_blocks_1d(int64_t n, int block) {
    return (unsigned) ((n + block - 1) / block);
}

// Output rows below which the integer path is not worth taking. It costs one
// extra launch to quantize the activations, and that is amortized over the
// rows: at a few thousand it disappears, at a couple of hundred the matmul is
// too small to fill the device either way and the extra launch is most of the
// difference. Measured, the crossover is a few hundred rows; attn_k and attn_v
// are the shapes that sit below it.
#define GK_CU_MM_Q8_MIN_ROWS 512

// Whether the integer dot path applies: the format has one, the row cuts into
// whole 32-element groups so a group never straddles two of them, and there
// are enough rows to pay for the quantize pass.
static __host__ __forceinline__ bool gk_cuda_mm_q8_supported(int type, int64_t k_len,
                                                             int64_t n_rows) {
    if (k_len % 32 != 0 || n_rows < GK_CU_MM_Q8_MIN_ROWS) {
        return false;
    }
    switch (type) {
        case GK_TYPE_Q4_0: case GK_TYPE_Q4_1:
        case GK_TYPE_Q8_0: case GK_TYPE_Q4_K:
            return true;
        default:
            return false;
    }
}

// Activation columns one pass over a weight row serves. Each column costs a
// register accumulator and a shared-memory slot in the reduction; four is
// where the reuse stops paying for the occupancy it costs.
#define GK_CU_MM_NC 4

// Consecutive weight elements a lane takes at a time. Every quantized block
// size in the format set - 32, 64, 128, 256 - is a multiple of the largest
// value here, so an aligned run always sits inside one block and that block's
// header is decoded once for the whole run rather than once per element.
//
// The trade is against coalescing: a longer run spreads a warp's reads further
// apart, and where the header is cheap that costs more than it saves. Four is
// where the two balance for every format measured on an Ada part except q6_K,
// whose 210-byte block comes out ahead with no run at all - 0.18 ms against
// 0.21. The type is already a template parameter, so saying so per format
// costs nothing.
//
// These are measured, not derived. bench-cuda's decoder-cost group is the
// measurement to retune them against: one matmul shape in every format, where
// the only thing that varies between rows is the decode.
#define GK_CU_MM_RUN_MAX 4

template <int TYPE>
static __device__ __host__ __forceinline__ int gk_cu_mm_run() {
    return TYPE == GKT_Q6_K ? 1 : GK_CU_MM_RUN_MAX;
}

// The tiled path below. A block computes a TILE_M x TILE_N patch of the
// result, marching over k in TILE_K slices staged in shared memory, with each
// of its 16x16 threads holding a 4x4 patch in registers.
//
// The point is the decode. In the mat-vec kernel a weight element is decoded
// once and used GK_CU_MM_NC times; here it is decoded once and used TILE_N
// times, and an activation element once and used TILE_M times. For a diffusion
// transformer - hundreds of tokens per matmul rather than the one token
// generation has - that difference is the whole cost of the pass.
#define GK_CU_MM_TILE_M 64
#define GK_CU_MM_TILE_N 64
#define GK_CU_MM_TILE_K 16
#define GK_CU_MM_TILE_T 4   // results per thread per axis (TILE_M / 16)

// Below this many columns the tile is mostly padding and the mat-vec kernel,
// which splits k across the block instead, keeps the device busier.
#define GK_CU_MM_TILE_MIN_N 24

// A block reduction over `NC` accumulators at once. Written out rather than
// calling the single-value reduction NC times so the barriers are shared.
template <int NC>
static __device__ __forceinline__ void gk_cu_reduce_n(float (&acc)[NC], float * shared) {
    const int n_warps = blockDim.x / GK_WARP_SIZE;
    const int lane    = threadIdx.x % GK_WARP_SIZE;
    const int warp    = threadIdx.x / GK_WARP_SIZE;

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = gk_cu_warp_sum(acc[j]);
    }

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            shared[j * n_warps + warp] = acc[j];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float total = 0.0f;
            for (int w = 0; w < n_warps; ++w) {
                total += shared[j * n_warps + w];
            }
            acc[j] = total;
        }
    }
}

// --------------------------------------------------------------------------
// dst[i0,i1,i2,i3] = dot( a_row(i0, i2/r2, i3/r3), b_row(i1,i2,i3) )
//
// `a` is the weight, read along its fastest dimension, which is what lets a
// quantized weight be used without transposing it. The higher dimensions of
// `a` broadcast onto `b`'s: grouped-query attention has fewer key heads than
// query heads and each key head serves a group.
// --------------------------------------------------------------------------

// A *warp* owns an output row, not a block.
//
// The block-per-row shape this replaced spent most of a short row's time in
// its own reduction: a 1536-wide q4_0 row is 864 packed bytes, which 128
// threads read in a few instructions each and then pay two barriers and a
// shared-memory round trip to add up. At 42 GB/s against a card that does 183
// the bottleneck was never the decode.
//
// A warp needs no barriers and no shared memory at all - the reduction is
// shuffles - and a block of several warps retires several rows for one launch.
// The activations are read by every warp in the block at the same k, so they
// stay resident in L1 rather than being staged by hand; staging them through
// shared memory was tried and was slower than not.
//
// Within a warp, a lane takes gk_cu_mm_run<ATYPE>() consecutive elements at a
// time, aligned so the run never crosses a quantized block's boundary. That is
// what lets the block be found once and its header decoded once for the whole
// run instead of once per element.
template <int NC, int ATYPE>
static __global__ void gk_cu_k_mul_mat(gk_tview a, gk_tview b, gk_tview_mut d,
                                       int64_t k_len, int64_t r2, int64_t r3) {
    const int lane    = threadIdx.x % GK_WARP_SIZE;
    const int warp    = threadIdx.x / GK_WARP_SIZE;
    const int n_warps = blockDim.x / GK_WARP_SIZE;

    const int64_t i0  = (int64_t) blockIdx.x * n_warps + warp; // weight row
    const int64_t c0  = (int64_t) blockIdx.y * NC;             // first column
    const int64_t i23 = blockIdx.z;

    if (i0 >= d.ne[0]) {
        return; // whole-warp uniform, so the shuffles below stay well formed
    }

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = 0.0f;
    }

    const char * a_row = gk_cu_row(a, i0, a2, a3);

    const int run = gk_cu_mm_run<ATYPE>();

    for (int64_t k0 = (int64_t) lane * run; k0 < k_len;
         k0 += GK_WARP_SIZE * run) {

        const uint8_t * blk = NULL;
        int j0 = 0;
        if (gk_cu_is_blocked<ATYPE>()) {
            // k0 is a multiple of the run and every block size is a multiple
            // of it too, so the whole run lives in this one block.
            const int blck = gk_cu_blck_size(ATYPE);
            const int tsz  = gk_cu_type_size(ATYPE);
            blk = (const uint8_t *) a_row + (k0 / blck) * tsz;
            j0  = (int) (k0 % blck);
        }

        // The run's weights, decoded together, so the block's header is paid
        // for once instead of once per element. Only for the formats where
        // that is worth an array to hold them; the rest decode in place below.
        const bool use_run = gk_cu_is_blocked<ATYPE>() && gk_cu_has_run_path<ATYPE>();

        float wv[GK_CU_MM_RUN_MAX];
        if (use_run) {
            const int64_t left = k_len - k0;
            gk_cu_blk_run_t<ATYPE, GK_CU_MM_RUN_MAX>(
                blk, j0, (int) (left < run ? left : run), wv);
        }

#pragma unroll
        for (int e = 0; e < GK_CU_MM_RUN_MAX; ++e) {
            if (e >= run) {
                break;
            }
            const int64_t kk = k0 + e;
            if (kk >= k_len) {
                break;
            }

            // the reuse this kernel exists for: one decode of the weight
            // element, NC multiplies against it
            const float av = use_run ? wv[e]
                : gk_cu_is_blocked<ATYPE>() ? gk_cu_blk_elem_t<ATYPE>(blk, j0 + e)
                : gk_cu_get_t<ATYPE>(a, kk, i0, a2, a3);

#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const int64_t col = c0 + j;
                if (col < d.ne[1]) {
                    acc[j] += av * gk_cu_get(b, kk, col, i2, i3);
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = gk_cu_warp_sum(acc[j]);
    }

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int64_t col = c0 + j;
            if (col < d.ne[1]) {
                gk_cu_set(d, i0, col, i2, i3, acc[j]);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The integer path: the same mat-vec, dotted as 8-bit integers.
//
// Two kernels. The first quantizes the activation columns once per matmul -
// cheap, because there are at most a couple of dozen of them against thousands
// of weight rows. The second is the mat-vec, shaped exactly like the float one
// above (a warp per output row, no barriers, shuffles for the reduction) but
// with the inner loop replaced by an integer dot.
// --------------------------------------------------------------------------

// One thread per 32-element group. The absolute maximum sets the scale, and
// the sum of the codes is carried alongside because the asymmetric formats
// need it.
static __global__ void gk_cu_k_quantize_act(gk_tview b, gk_cu_q8blk * out,
                                            int64_t n_grp, int64_t n_cols, int64_t total) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }

    const int64_t g   = t % n_grp;
    const int64_t col = (t / n_grp) % n_cols;
    const int64_t i23 = t / (n_grp * n_cols);

    const int64_t i2 = i23 % b.ne[2];
    const int64_t i3 = i23 / b.ne[2];

    float v[32];
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 32; ++e) {
        v[e] = gk_cu_get(b, g * 32 + e, col, i2, i3);
        amax = fmaxf(amax, fabsf(v[e]));
    }

    gk_cu_q8blk blk;
    blk.d = amax / 127.0f;

    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;

    int sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int packed = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int q = (int) rintf(v[i * 4 + e] * inv);
            sum += q;
            packed |= (q & 0xff) << (8 * e);
        }
        blk.q[i] = packed;
    }
    blk.s = (float) sum;

    out[t] = blk;
}

template <int NC, int ATYPE>
static __global__ void gk_cu_k_mul_mat_q8(gk_tview a, gk_tview_mut d,
                                          const gk_cu_q8blk * aq, int64_t n_grp,
                                          int64_t r2, int64_t r3) {
    const int lane    = threadIdx.x % GK_WARP_SIZE;
    const int warp    = threadIdx.x / GK_WARP_SIZE;
    const int n_warps = blockDim.x / GK_WARP_SIZE;

    const int64_t i0  = (int64_t) blockIdx.x * n_warps + warp;
    const int64_t c0  = (int64_t) blockIdx.y * NC;
    const int64_t i23 = blockIdx.z;

    if (i0 >= d.ne[0]) {
        return; // whole-warp uniform, so the shuffles below stay well formed
    }

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = 0.0f;
    }

    const uint8_t * a_row = (const uint8_t *) gk_cu_row(a, i0, a2, a3);

    // The quantized activations for this (i2, i3) slice, one run of groups per
    // column.
    const gk_cu_q8blk * aq23 = aq + i23 * d.ne[1] * n_grp;

    for (int64_t g = lane; g < n_grp; g += GK_WARP_SIZE) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int64_t col = c0 + j;
            if (col < d.ne[1]) {
                acc[j] += gk_cu_vecdot32<ATYPE>(a_row, g, aq23[col * n_grp + g]);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = gk_cu_warp_sum(acc[j]);
    }

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int64_t col = c0 + j;
            if (col < d.ne[1]) {
                gk_cu_set(d, i0, col, i2, i3, acc[j]);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The tensor-core path, for nvfp4.
//
// A pilot rather than a general kernel: one format, to find out what
// `mma.sync` is worth here before the same is done for the rest.
//
// nvfp4 is the awkward case to start from and that is deliberate - it is the
// format the diffusion models use. Its scale changes every 16 elements, and an
// integer mma accumulates in s32, so the accumulator has to be drained to
// float and rescaled every 16 of k. That is one `m16n8k16` (2048 multiply-
// accumulates in a single warp instruction) against roughly eight instructions
// of rescaling, where a format with a scale per 32 could use `m16n8k32` and
// halve the rescaling per unit of work. So this measures the *floor* of what
// tensor cores give, not the ceiling.
//
// What makes it fit at all: nvfp4's values are `sub_scale * e2m1[code]`, and
// every entry of the e2m1 codebook is a small integer, so the codebook lookup
// can be folded into the staging - once per weight row per 16 of k, reused
// across the whole column tile - and what reaches the tensor core is plain
// int8. There is no offset term, so unlike q4_1 or q4_K this needs nothing
// from the activation block but its codes and its scale.
// --------------------------------------------------------------------------

#if !defined(GK_USE_HIP) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define GK_CU_HAVE_MMA 1
#endif

// D += A*B, for a 16x8 tile of s32 over 16 of k, warp-wide.
//
// The fragment layouts are the ones PTX fixes for this shape: with
// `group = lane/4` and `tig = lane%4`, a thread holds A rows `group` and
// `group+8` at columns `4*tig..+3`, B column `group` at rows `4*tig..+3`, and
// D at rows `group`/`group+8`, columns `2*tig` and `2*tig+1`.
static __device__ __forceinline__ void gk_cu_mma_s8(int (&d)[4], const int (&a)[2], int b) {
#if defined(GK_CU_HAVE_MMA)
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
#else
    // No such instruction on this target. The host gate below is what keeps
    // this kernel from being launched; this branch exists so a build for an
    // older part still compiles.
    (void) a; (void) b; (void) d;
#endif
}

// Four consecutive codes of a packed nibble word, through the e2m1 codebook
// and into four int8 lanes. `shift` picks the low or high nibble of each byte.
static __device__ __forceinline__ int gk_cu_e2m1_quad(int w, int shift) {
    int out = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int code = (w >> (8 * e + shift)) & 0xf;
        out |= ((int) gk_cu_e2m1_values[code] & 0xff) << (8 * e);
    }
    return out;
}

#define GK_CU_MMA_TILE_M 64   // four warps of sixteen rows
#define GK_CU_MMA_TILE_N 64   // mma column tiles of eight
#define GK_CU_MMA_NCT    (GK_CU_MMA_TILE_N / 8)
#define GK_CU_MMA_K      32   // one activation block; two mma windows of 16

static __global__ void gk_cu_k_mul_mat_mma_nvfp4(gk_tview a, gk_tview_mut d,
                                                 const gk_cu_q8blk * aq, int64_t n_grp,
                                                 int64_t r2, int64_t r3) {
    // Staged as ints rather than bytes so that the fragment reads below are
    // plain aligned loads: word `h*4 + tig` is exactly the four codes a lane
    // wants for half `h`.
    __shared__ int   As[GK_CU_MMA_TILE_M][8];
    __shared__ int   Bs[GK_CU_MMA_TILE_N][8];
    __shared__ float Ws[GK_CU_MMA_TILE_M][2];
    __shared__ float Ad[GK_CU_MMA_TILE_N];

    const int lane  = threadIdx.x % GK_WARP_SIZE;
    const int warp  = threadIdx.x / GK_WARP_SIZE;
    const int group = lane / 4;
    const int tig   = lane % 4;

    const int64_t m0  = (int64_t) blockIdx.x * GK_CU_MMA_TILE_M;
    const int64_t n0  = (int64_t) blockIdx.y * GK_CU_MMA_TILE_N;
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    const gk_cu_q8blk * aq23 = aq + i23 * n_cols * n_grp;

    float acc[GK_CU_MMA_NCT][4];
#pragma unroll
    for (int ct = 0; ct < GK_CU_MMA_NCT; ++ct) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[ct][i] = 0.0f;
        }
    }

    for (int64_t g = 0; g < n_grp; ++g) {
        const int64_t kk0 = g * GK_CU_MMA_K;

        // Staging: sixty-four threads take a weight row each, thirty-two take
        // an activation column each.
        if (threadIdx.x < GK_CU_MMA_TILE_M) {
            const int     r = threadIdx.x;
            const int64_t m = m0 + r;

            if (m < n_rows) {
                // one 64-element nvfp4 block holds four sub-scales; this
                // k-step covers two of them
                const uint8_t * blk =
                    (const uint8_t *) gk_cu_row(a, m, a2, a3) + (kk0 / 64) * 36;
                const int sub0 = (int) ((kk0 % 64) / 16);

#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint8_t * qs = blk + 4 + (sub0 + h) * 8;

                    const int w0 = gk_cu_int_b2(qs, 0); // sub-block elements 0..3, 8..11
                    const int w1 = gk_cu_int_b2(qs, 1); // and 4..7, 12..15

                    As[r][h * 4 + 0] = gk_cu_e2m1_quad(w0, 0);
                    As[r][h * 4 + 1] = gk_cu_e2m1_quad(w1, 0);
                    As[r][h * 4 + 2] = gk_cu_e2m1_quad(w0, 4);
                    As[r][h * 4 + 3] = gk_cu_e2m1_quad(w1, 4);

                    Ws[r][h] = gk_cu_ue4m3(blk[sub0 + h]);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    As[r][i] = 0;
                }
                Ws[r][0] = 0.0f;
                Ws[r][1] = 0.0f;
            }
        } else if (threadIdx.x < GK_CU_MMA_TILE_M + GK_CU_MMA_TILE_N) {
            const int     c = (int) threadIdx.x - GK_CU_MMA_TILE_M;
            const int64_t n = n0 + c;

            if (n < n_cols) {
                const gk_cu_q8blk & ab = aq23[n * n_grp + g];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Bs[c][i] = ab.q[i];
                }
                Ad[c] = ab.d;
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Bs[c][i] = 0;
                }
                Ad[c] = 0.0f;
            }
        }

        __syncthreads();

        // Two mma windows, one per sub-scale. The accumulator is drained to
        // float between them because the scale it would be multiplied by
        // changes - which is the whole cost of this format on tensor cores.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int af[2];
            af[0] = As[warp * 16 + group    ][h * 4 + tig];
            af[1] = As[warp * 16 + group + 8][h * 4 + tig];

            const float ws_lo = Ws[warp * 16 + group    ][h];
            const float ws_hi = Ws[warp * 16 + group + 8][h];

#pragma unroll
            for (int ct = 0; ct < GK_CU_MMA_NCT; ++ct) {
                int df[4] = { 0, 0, 0, 0 };

                gk_cu_mma_s8(df, af, Bs[ct * 8 + group][h * 4 + tig]);

                const float ad0 = Ad[ct * 8 + tig * 2 + 0];
                const float ad1 = Ad[ct * 8 + tig * 2 + 1];

                acc[ct][0] += ws_lo * ad0 * (float) df[0];
                acc[ct][1] += ws_lo * ad1 * (float) df[1];
                acc[ct][2] += ws_hi * ad0 * (float) df[2];
                acc[ct][3] += ws_hi * ad1 * (float) df[3];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int ct = 0; ct < GK_CU_MMA_NCT; ++ct) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t m = m0 + warp * 16 + group + (i >= 2 ? 8 : 0);
            const int64_t n = n0 + ct * 8 + tig * 2 + (i & 1);

            if (m < n_rows && n < n_cols) {
                gk_cu_set(d, m, n, i2, i3, acc[ct][i]);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The tensor-core path, for f16 activations.
//
// The nvfp4 pilot above answered what `mma.sync` is worth; this is the same
// instruction where the shapes that dominate a diffusion graph actually live.
// Every convolution in a UNet or a VAE lowers to `im2col` plus a matmul whose
// src0 is that f16 im2col buffer - so this one kernel is most of the time in
// both, and until it existed all of it ran on the float tile below at roughly
// a tenth of what the part can do.
//
// Three things make this simpler than the nvfp4 case:
//
//   * The operands are already f16, so nothing is decoded on the way to the
//     fragment and nothing is rescaled inside the k loop. The accumulator is
//     f32 and stays in the tensor core across the whole reduction, which is
//     what the integer path could not do.
//
//   * gk's matmul is `dst[m,n] = sum_k a[k,m] * b[k,n]` with both operands
//     k-major, and `mma.row.col` wants exactly that: A row-major with k along
//     the row, B column-major with k along the column. So the fragments are
//     the operands' own layout, not a transpose of it.
//
//   * src1 is read through the runtime accessor rather than a template. It is
//     touched once per tile and used TILE_M times, so what it costs is a
//     rounding error on the total, and not templating it keeps this to two
//     instantiations instead of two per weight format.
//
// The tile is sized by memory, not by arithmetic. At these shapes the GEMM is
// bandwidth-bound - src0 is an im2col buffer of tens of megabytes and gets
// re-read once per column tile - so what sets the ceiling is TILE_N, and 128
// is where a 512-channel VAE convolution stops being limited by DRAM. TILE_M
// only picks how many blocks there are, and drops to 64 when there are not
// enough rows to fill 128.
// --------------------------------------------------------------------------

// D += A*B for a 16x8 tile of f32 over 16 of k, warp-wide, f16 operands. The
// fragment geometry is the s8 instruction's at half the k per register: with
// `group = lane/4` and `tig = lane%4`, a lane holds A rows `group` and
// `group+8` at columns `2*tig..+1` and `2*tig+8..+9`, B column `group` at
// those same rows, and D at rows `group`/`group+8`, columns `2*tig` and
// `2*tig+1` - the layout the epilogue below and the nvfp4 kernel's share.
static __device__ __forceinline__ void gk_cu_mma_f16(float (&d)[4], const int (&a)[4],
                                                     const int (&b)[2]) {
#if defined(GK_CU_HAVE_MMA)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    // No such instruction on this target; the host gate keeps the kernel from
    // being launched. This branch is here so the build still compiles.
    (void) a; (void) b; (void) d;
#endif
}

#define GK_CU_MMA_F16_K   32   // k staged per pass: two mma windows of sixteen
#define GK_CU_MMA_F16_WM  32   // rows a warp owns: two mma row tiles
#define GK_CU_MMA_F16_WN  64   // columns a warp owns: eight mma column tiles
#define GK_CU_MMA_F16_WNT (GK_CU_MMA_F16_WN / 8)
#define GK_CU_MMA_F16_WARPS_N 2
#define GK_CU_MMA_F16_TILE_N  (GK_CU_MMA_F16_WARPS_N * GK_CU_MMA_F16_WN)

// The staged rows are padded past the k they hold. A fragment read has the
// eight lanes of a quadrant on eight different rows at the same k, so an
// unpadded row of 32 halves - 8 banks apart - would put four of those eight on
// the banks of another four. At a stride of 40 halves the eight land on eight
// disjoint quads and a warp's fragment read is conflict-free.
#define GK_CU_MMA_F16_SK  (GK_CU_MMA_F16_K + 8)

// Columns below which the tile is mostly padding and the paths below, which
// split k across a block instead of tiling n, keep the device busier.
#define GK_CU_MMA_F16_MIN_N 32

// The shortest piece of k worth giving a block of its own. Below this the
// block spends more of itself on its prologue and its share of the combine
// pass than on the reduction it was split off to do.
#define GK_CU_MMA_F16_SPLIT_MIN_K 1024

// Warps a shape has to put in flight before splitting k stops paying: one
// wide-tile block's worth per multiprocessor.
#define GK_CU_MMA_F16_SPLIT_WARPS 8

// Eight elements of a float-typed operand, widened to half and packed into one
// 16-byte word - the unit the tile is staged in.
//
// Staging one element per instruction is what the float tile does, and there
// it is affordable because each staged element feeds 16 multiply-accumulates.
// Here it feeds 128, so an instruction spent per element is an instruction
// spent against a tensor-core instruction rather than against sixteen FFMAs,
// and it dominates. Whether the eight can be read as one word is decided on
// the host, once per launch, and passed in.
static __device__ __forceinline__ int4 gk_cu_pack8_half(const char * p, int type) {
    if (type == GKT_F16) {
        return *(const int4 *) p;
    }

    __align__(16) __half h[8];

    if (type == GKT_F32) {
        const float4 lo = *(const float4 *) p;
        const float4 hi = *(const float4 *) (p + 16);

        h[0] = __float2half(lo.x); h[1] = __float2half(lo.y);
        h[2] = __float2half(lo.z); h[3] = __float2half(lo.w);
        h[4] = __float2half(hi.x); h[5] = __float2half(hi.y);
        h[6] = __float2half(hi.z); h[7] = __float2half(hi.w);
    } else {
        const int4       w = *(const int4 *) p;
        const uint16_t * u = (const uint16_t *) &w;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            h[i] = __float2half(gk_cu_bf2f(u[i]));
        }
    }

    return *(const int4 *) h;
}

// Whether an operand can be staged eight at a time: contiguous along k, every
// row start 16-byte aligned, and a k that a run of eight either fits inside or
// starts past the end of, so the guard stays a comparison rather than a
// per-element mask.
static __host__ __forceinline__ bool gk_cuda_mma_f16_vec(const struct gk_tensor * t, int64_t k_len) {
    if ((size_t) t->nb[0] != gk_type_size(t->type) || k_len % 8 != 0) {
        return false;
    }
    const uintptr_t bits = (uintptr_t) t->data | (uintptr_t) t->nb[1]
                         | (uintptr_t) t->nb[2] | (uintptr_t) t->nb[3];
    return bits % 16 == 0;
}

template <int WARPS_M>
static __global__ __launch_bounds__(WARPS_M * GK_CU_MMA_F16_WARPS_N * GK_WARP_SIZE, 1)
void gk_cu_k_mul_mat_mma_f16(gk_tview a, gk_tview b, gk_tview_mut d,
                             int64_t k_len, int64_t r2, int64_t r3,
                             bool a_vec, bool b_vec,
                             float * part, int64_t k_split, int64_t n_23) {
    const int TILE_M = WARPS_M * GK_CU_MMA_F16_WM;

    __shared__ __half As[2][WARPS_M * GK_CU_MMA_F16_WM][GK_CU_MMA_F16_SK];
    __shared__ __half Bs[2][GK_CU_MMA_F16_TILE_N]     [GK_CU_MMA_F16_SK];

    const int tid    = (int) threadIdx.x;
    const int lane   = tid % GK_WARP_SIZE;
    const int warp   = tid / GK_WARP_SIZE;
    const int warp_m = warp / GK_CU_MMA_F16_WARPS_N;
    const int warp_n = warp % GK_CU_MMA_F16_WARPS_N;
    const int group  = lane / 4;
    const int tig    = lane % 4;

    const int n_threads = WARPS_M * GK_CU_MMA_F16_WARPS_N * GK_WARP_SIZE;

    const int64_t m0  = (int64_t) blockIdx.x * TILE_M;
    const int64_t n0  = (int64_t) blockIdx.y * GK_CU_MMA_F16_TILE_N;

    // z carries the slice and, when k is split, which piece of k this block
    // reduces. A split block owns the same output patch as its siblings and
    // differs only in the range it sums, so it cannot write the destination -
    // it writes its own plane of `part` and the pass below adds them up.
    const int64_t i23   = blockIdx.z % n_23;
    const int64_t split = blockIdx.z / n_23;

    const int64_t k_beg = split * k_split;
    const int64_t k_end = part != NULL && k_beg + k_split < k_len ? k_beg + k_split : k_len;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    float acc[2][GK_CU_MMA_F16_WNT][4];
#pragma unroll
    for (int wt = 0; wt < 2; ++wt) {
#pragma unroll
        for (int ct = 0; ct < GK_CU_MMA_F16_WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[wt][ct][i] = 0.0f;
            }
        }
    }

    // A thread's share of one k-slice, held in registers between being read
    // from memory and being written to the tile. Both are known at compile
    // time: the slice is a fixed size and so is the block.
    int4 pa[TILE_M * (GK_CU_MMA_F16_K / 8) / (WARPS_M * GK_CU_MMA_F16_WARPS_N * GK_WARP_SIZE)];
    int4 pb[GK_CU_MMA_F16_TILE_N * (GK_CU_MMA_F16_K / 8) / (WARPS_M * GK_CU_MMA_F16_WARPS_N * GK_WARP_SIZE)];

    const int n_runs_a = (int) (sizeof(pa) / sizeof(pa[0]));
    const int n_runs_b = (int) (sizeof(pb) / sizeof(pb[0]));

    // Reading a k-slice into those registers. A run of eight per thread:
    // consecutive threads take consecutive k of one row, which is the
    // direction both operands are contiguous in, so a run is one 16-byte
    // load. The scalar arms are what an operand that is permuted, misaligned
    // or ragged in k falls to - the same answer, an instruction at a time.
#define GK_CU_MMA_F16_LOAD(k0)                                                       \
    do {                                                                             \
        _Pragma("unroll")                                                            \
        for (int i = 0; i < n_runs_a; ++i) {                                         \
            const int     e  = tid + i * n_threads;                                  \
            const int     r  = e / (GK_CU_MMA_F16_K / 8);                            \
            const int     kr = e % (GK_CU_MMA_F16_K / 8);                            \
            const int64_t m  = m0 + r;                                               \
            const int64_t k  = (k0) + kr * 8;                                        \
                                                                                     \
            pa[i] = make_int4(0, 0, 0, 0);                                           \
                                                                                     \
            if (m < n_rows) {                                                        \
                const char * row = gk_cu_row(a, m, a2, a3);                          \
                if (a_vec && k + 8 <= k_end) {                                       \
                    pa[i] = *(const int4 *) (row + k * a.nb[0]);                     \
                } else {                                                             \
                    __align__(16) __half h[8];                                       \
                    _Pragma("unroll")                                                \
                    for (int j = 0; j < 8; ++j) {                                    \
                        h[j] = (k + j < k_end)                                       \
                             ? *(const __half *) (row + (k + j) * a.nb[0])           \
                             : __float2half(0.0f);                                   \
                    }                                                                \
                    pa[i] = *(const int4 *) h;                                       \
                }                                                                    \
            }                                                                        \
        }                                                                            \
                                                                                     \
        _Pragma("unroll")                                                            \
        for (int i = 0; i < n_runs_b; ++i) {                                         \
            const int     e  = tid + i * n_threads;                                  \
            const int     c  = e / (GK_CU_MMA_F16_K / 8);                            \
            const int     kr = e % (GK_CU_MMA_F16_K / 8);                            \
            const int64_t n  = n0 + c;                                               \
            const int64_t k  = (k0) + kr * 8;                                        \
                                                                                     \
            pb[i] = make_int4(0, 0, 0, 0);                                           \
                                                                                     \
            if (n < n_cols) {                                                        \
                if (b_vec && k + 8 <= k_end) {                                       \
                    pb[i] = gk_cu_pack8_half(gk_cu_row(b, n, i2, i3) + k * b.nb[0],  \
                                             b.type);                                \
                } else {                                                             \
                    __align__(16) __half h[8];                                       \
                    _Pragma("unroll")                                                \
                    for (int j = 0; j < 8; ++j) {                                    \
                        h[j] = __float2half((k + j < k_end)                          \
                                            ? gk_cu_get(b, k + j, n, i2, i3)         \
                                            : 0.0f);                                 \
                    }                                                                \
                    pb[i] = *(const int4 *) h;                                       \
                }                                                                    \
            }                                                                        \
        }                                                                            \
    } while (0)

#define GK_CU_MMA_F16_STORE(buf)                                                     \
    do {                                                                             \
        _Pragma("unroll")                                                            \
        for (int i = 0; i < n_runs_a; ++i) {                                         \
            const int e = tid + i * n_threads;                                       \
            *(int4 *) &As[buf][e / (GK_CU_MMA_F16_K / 8)]                            \
                            [(e % (GK_CU_MMA_F16_K / 8)) * 8] = pa[i];               \
        }                                                                            \
        _Pragma("unroll")                                                            \
        for (int i = 0; i < n_runs_b; ++i) {                                         \
            const int e = tid + i * n_threads;                                       \
            *(int4 *) &Bs[buf][e / (GK_CU_MMA_F16_K / 8)]                            \
                            [(e % (GK_CU_MMA_F16_K / 8)) * 8] = pb[i];               \
        }                                                                            \
    } while (0)

    // The tile is double-buffered and the next slice is read while the current
    // one is being multiplied. That ordering is the point: a global read costs
    // several hundred cycles and a slice's worth of mma covers it, where a
    // kernel that loads and then waits at a barrier pays the two in series.
    // What it costs is one more copy of the tile in shared memory and the
    // registers to hold a slice in flight.
    GK_CU_MMA_F16_LOAD(k_beg);
    GK_CU_MMA_F16_STORE(0);
    __syncthreads();

    int buf = 0;

    for (int64_t k0 = k_beg; k0 < k_end; k0 += GK_CU_MMA_F16_K) {
        const bool more = k0 + GK_CU_MMA_F16_K < k_end;

        if (more) {
            GK_CU_MMA_F16_LOAD(k0 + GK_CU_MMA_F16_K);
        }

        // Two mma windows over the staged k. The A fragments are read once
        // and used for every column tile - that reuse is the whole point of
        // giving a warp 64 columns rather than 8.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int kk = h * 16;

            int af[2][4];
#pragma unroll
            for (int wt = 0; wt < 2; ++wt) {
                const int r_lo = warp_m * GK_CU_MMA_F16_WM + wt * 16 + group;
                const int r_hi = r_lo + 8;

                af[wt][0] = *(const int *) &As[buf][r_lo][kk + 2 * tig];
                af[wt][1] = *(const int *) &As[buf][r_hi][kk + 2 * tig];
                af[wt][2] = *(const int *) &As[buf][r_lo][kk + 8 + 2 * tig];
                af[wt][3] = *(const int *) &As[buf][r_hi][kk + 8 + 2 * tig];
            }

#pragma unroll
            for (int ct = 0; ct < GK_CU_MMA_F16_WNT; ++ct) {
                const int c = warp_n * GK_CU_MMA_F16_WN + ct * 8 + group;

                int bf[2];
                bf[0] = *(const int *) &Bs[buf][c][kk + 2 * tig];
                bf[1] = *(const int *) &Bs[buf][c][kk + 8 + 2 * tig];

#pragma unroll
                for (int wt = 0; wt < 2; ++wt) {
                    gk_cu_mma_f16(acc[wt][ct], af[wt], bf);
                }
            }
        }

        // Into the other half, which nothing is reading, so the only barrier
        // in the loop is the one that publishes it.
        if (more) {
            GK_CU_MMA_F16_STORE(1 - buf);
        }

        __syncthreads();
        buf = 1 - buf;
    }

#undef GK_CU_MMA_F16_LOAD
#undef GK_CU_MMA_F16_STORE

#pragma unroll
    for (int wt = 0; wt < 2; ++wt) {
#pragma unroll
        for (int ct = 0; ct < GK_CU_MMA_F16_WNT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t m = m0 + warp_m * GK_CU_MMA_F16_WM + wt * 16
                                + group + (i >= 2 ? 8 : 0);
                const int64_t n = n0 + warp_n * GK_CU_MMA_F16_WN + ct * 8
                                + tig * 2 + (i & 1);

                if (m < n_rows && n < n_cols) {
                    if (part != NULL) {
                        part[((split * n_23 + i23) * n_cols + n) * n_rows + m] = acc[wt][ct][i];
                    } else {
                        gk_cu_set(d, m, n, i2, i3, acc[wt][ct][i]);
                    }
                }
            }
        }
    }
}

// Adding the split blocks' planes back together. Summed in slice order rather
// than by whichever block finished first, so a split matmul returns the same
// bits every run - which an atomic accumulation into the destination, the
// other way to write this, would not.
static __global__ void gk_cu_k_mma_f16_combine(const float * part, gk_tview_mut d,
                                               int64_t n_splits, int64_t n_23,
                                               int64_t n_rows, int64_t n_cols) {
    const int64_t n_out = n_rows * n_cols * n_23;

    for (int64_t e = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
         e < n_out; e += (int64_t) gridDim.x * blockDim.x) {
        const int64_t m   = e % n_rows;
        const int64_t n   = (e / n_rows) % n_cols;
        const int64_t i23 = e / (n_rows * n_cols);

        float sum = 0.0f;
        for (int64_t s = 0; s < n_splits; ++s) {
            sum += part[((s * n_23 + i23) * n_cols + n) * n_rows + m];
        }

        gk_cu_set(d, m, n, i23 % d.ne[2], i23 / d.ne[2], sum);
    }
}

// Whether this matmul goes to the tensor core. The instruction is Ampere and
// later and has no HIP spelling here, the operand has to already be f16 for
// the fragments to be its own bytes, and a caller that asked for f32 precision
// asked for the float path - the accumulator is f32 either way, but the
// products are not.
static __host__ __forceinline__ bool gk_cuda_mma_f16_supported(const struct gk_tensor * dst,
                                                               const struct gk_cuda_scratch * s) {
#if defined(GK_USE_HIP)
    GK_UNUSED(dst);
    GK_UNUSED(s);
    return false;
#else
    if (s == NULL || s->cc < 80) {
        return false;
    }
    if ((int) dst->src[0]->type != GK_TYPE_F16) {
        return false;
    }
    if (gk_get_op_params_i32(dst, 0) == GK_PREC_F32) {
        return false;
    }
    return dst->ne[1] >= GK_CU_MMA_F16_MIN_N;
#endif
}

// --------------------------------------------------------------------------
// The tiled path, dotted as integers.
//
// The float tiled kernel below already decodes each weight element once and
// uses it TILE_N times, so decoding is not what it spends its time on -
// measured, f16 with no decode at all and q4_K are within 14% of each other
// there. What it spends its time on is the multiply-accumulate itself, one
// FFMA per element pair.
//
// So the win here is not avoiding the decode, it is `__dp4a`: four
// multiply-accumulates in one instruction instead of one. The weights are
// staged as their integer codes rather than as floats, which also makes the
// tile a quarter of the size in shared memory, and each group's two scaling
// constants are applied once per 32 elements instead of once per element.
//
// A block owns a TILE_M x TILE_N patch and marches over k a group at a time.
// --------------------------------------------------------------------------

#define GK_CU_MMQ_TILE_M 64
#define GK_CU_MMQ_TILE_N 64
#define GK_CU_MMQ_T      4   // results per thread per axis (TILE / 16)

template <int ATYPE>
static __global__ void gk_cu_k_mul_mat_tiled_q8(gk_tview a, gk_tview_mut d,
                                                const gk_cu_q8blk * aq, int64_t n_grp,
                                                int64_t r2, int64_t r3) {
    // Transposed - [element word][row] - so that a warp walking rows or
    // columns for a fixed word reads consecutive shared addresses.
    __shared__ int   Wc[8][GK_CU_MMQ_TILE_M];
    __shared__ int   Ac[8][GK_CU_MMQ_TILE_N];
    __shared__ float Wscale[GK_CU_MMQ_TILE_M];
    __shared__ float Woff  [GK_CU_MMQ_TILE_M];
    __shared__ float Ad    [GK_CU_MMQ_TILE_N];
    __shared__ float Asum  [GK_CU_MMQ_TILE_N];

    const int tx  = threadIdx.x;          // 0..15, column group
    const int ty  = threadIdx.y;          // 0..15, row group
    const int tid = ty * 16 + tx;         // 0..255

    const int64_t m0  = (int64_t) blockIdx.x * GK_CU_MMQ_TILE_M;
    const int64_t n0  = (int64_t) blockIdx.y * GK_CU_MMQ_TILE_N;
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    const gk_cu_q8blk * aq23 = aq + i23 * n_cols * n_grp;

    float acc[GK_CU_MMQ_T][GK_CU_MMQ_T];
#pragma unroll
    for (int i = 0; i < GK_CU_MMQ_T; ++i) {
#pragma unroll
        for (int j = 0; j < GK_CU_MMQ_T; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t g = 0; g < n_grp; ++g) {
        // Staging. The first 64 threads take a weight row each, the next 64 an
        // activation column each; a row or column past the end stages zeros so
        // that the arithmetic below needs no bounds test.
        if (tid < GK_CU_MMQ_TILE_M) {
            const int64_t m = m0 + tid;

            int   codes[8];
            float sc = 0.0f, off = 0.0f;

            if (m < n_rows) {
                gk_cu_wblk32<ATYPE>((const uint8_t *) gk_cu_row(a, m, a2, a3),
                                    g, codes, sc, off);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    codes[i] = 0;
                }
            }

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                Wc[i][tid] = codes[i];
            }
            Wscale[tid] = sc;
            Woff  [tid] = off;
        } else if (tid < GK_CU_MMQ_TILE_M + GK_CU_MMQ_TILE_N) {
            const int     slot = tid - GK_CU_MMQ_TILE_M;
            const int64_t n    = n0 + slot;

            if (n < n_cols) {
                const gk_cu_q8blk & ab = aq23[n * n_grp + g];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Ac[i][slot] = ab.q[i];
                }
                Ad  [slot] = ab.d;
                Asum[slot] = ab.s;
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Ac[i][slot] = 0;
                }
                Ad  [slot] = 0.0f;
                Asum[slot] = 0.0f;
            }
        }

        __syncthreads();

        // The integer dot: eight instructions per output pair, against the
        // thirty-two a float pass would take.
        int sumi[GK_CU_MMQ_T][GK_CU_MMQ_T];
#pragma unroll
        for (int i = 0; i < GK_CU_MMQ_T; ++i) {
#pragma unroll
            for (int j = 0; j < GK_CU_MMQ_T; ++j) {
                sumi[i][j] = 0;
            }
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            int wv[GK_CU_MMQ_T];
            int av[GK_CU_MMQ_T];
#pragma unroll
            for (int i = 0; i < GK_CU_MMQ_T; ++i) {
                wv[i] = Wc[e][ty * GK_CU_MMQ_T + i];
            }
#pragma unroll
            for (int j = 0; j < GK_CU_MMQ_T; ++j) {
                av[j] = Ac[e][tx * GK_CU_MMQ_T + j];
            }
#pragma unroll
            for (int i = 0; i < GK_CU_MMQ_T; ++i) {
#pragma unroll
                for (int j = 0; j < GK_CU_MMQ_T; ++j) {
                    sumi[i][j] = gk_cu_dp4a(wv[i], av[j], sumi[i][j]);
                }
            }
        }

        // The scales, once per group rather than once per element.
#pragma unroll
        for (int i = 0; i < GK_CU_MMQ_T; ++i) {
            const int   mi = ty * GK_CU_MMQ_T + i;
            const float ws = Wscale[mi];
            const float wo = Woff[mi];
#pragma unroll
            for (int j = 0; j < GK_CU_MMQ_T; ++j) {
                const int nj = tx * GK_CU_MMQ_T + j;
                acc[i][j] += Ad[nj] * (ws * (float) sumi[i][j] + wo * Asum[nj]);
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < GK_CU_MMQ_T; ++i) {
        const int64_t m = m0 + ty * GK_CU_MMQ_T + i;
        if (m >= n_rows) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < GK_CU_MMQ_T; ++j) {
            const int64_t n = n0 + tx * GK_CU_MMQ_T + j;
            if (n < n_cols) {
                gk_cu_set(d, m, n, i2, i3, acc[i][j]);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The tiled path: same result as gk_cu_k_mul_mat, arranged for reuse.
//
// Each block owns a TILE_M x TILE_N patch of `d` and walks the whole of k,
// which is the opposite of the mat-vec kernel's split-k. That trade is what
// buys the reuse: k stays in one block, so a staged tile serves every output
// in the patch and no cross-block reduction is needed.
// --------------------------------------------------------------------------

template <int ATYPE>
static __global__ void gk_cu_k_mul_mat_tiled(gk_tview a, gk_tview b, gk_tview_mut d,
                                             int64_t k_len, int64_t r2, int64_t r3) {
    __shared__ float As[GK_CU_MM_TILE_K][GK_CU_MM_TILE_M];
    __shared__ float Bs[GK_CU_MM_TILE_K][GK_CU_MM_TILE_N];

    const int tx  = threadIdx.x;              // 0..15, column group
    const int ty  = threadIdx.y;              // 0..15, row group
    const int tid = ty * 16 + tx;             // 0..255

    const int64_t m0  = (int64_t) blockIdx.x * GK_CU_MM_TILE_M;  // first weight row
    const int64_t n0  = (int64_t) blockIdx.y * GK_CU_MM_TILE_N;  // first activation column
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    const int64_t n_rows = d.ne[0];
    const int64_t n_cols = d.ne[1];

    float acc[GK_CU_MM_TILE_T][GK_CU_MM_TILE_T];
#pragma unroll
    for (int i = 0; i < GK_CU_MM_TILE_T; ++i) {
#pragma unroll
        for (int j = 0; j < GK_CU_MM_TILE_T; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < k_len; k0 += GK_CU_MM_TILE_K) {
        // Stage both tiles. The index split puts consecutive threads on
        // consecutive k, which is the direction both operands are contiguous
        // in - a quantized row is packed along k, and a permuted activation
        // still has its k stride in nb[0].
#pragma unroll
        for (int e = tid; e < GK_CU_MM_TILE_M * GK_CU_MM_TILE_K; e += 256) {
            const int     mm = e / GK_CU_MM_TILE_K;
            const int     kk = e % GK_CU_MM_TILE_K;
            const int64_t k  = k0 + kk;
            const int64_t m  = m0 + mm;
            As[kk][mm] = (k < k_len && m < n_rows) ? gk_cu_get_t<ATYPE>(a, k, m, a2, a3) : 0.0f;
        }
#pragma unroll
        for (int e = tid; e < GK_CU_MM_TILE_N * GK_CU_MM_TILE_K; e += 256) {
            const int     nn = e / GK_CU_MM_TILE_K;
            const int     kk = e % GK_CU_MM_TILE_K;
            const int64_t k  = k0 + kk;
            const int64_t n  = n0 + nn;
            Bs[kk][nn] = (k < k_len && n < n_cols) ? gk_cu_get(b, k, n, i2, i3) : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < GK_CU_MM_TILE_K; ++kk) {
            float av[GK_CU_MM_TILE_T];
            float bv[GK_CU_MM_TILE_T];
#pragma unroll
            for (int i = 0; i < GK_CU_MM_TILE_T; ++i) {
                av[i] = As[kk][ty * GK_CU_MM_TILE_T + i];
            }
#pragma unroll
            for (int j = 0; j < GK_CU_MM_TILE_T; ++j) {
                bv[j] = Bs[kk][tx * GK_CU_MM_TILE_T + j];
            }
#pragma unroll
            for (int i = 0; i < GK_CU_MM_TILE_T; ++i) {
#pragma unroll
                for (int j = 0; j < GK_CU_MM_TILE_T; ++j) {
                    acc[i][j] += av[i] * bv[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < GK_CU_MM_TILE_T; ++i) {
        const int64_t m = m0 + ty * GK_CU_MM_TILE_T + i;
        if (m >= n_rows) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < GK_CU_MM_TILE_T; ++j) {
            const int64_t n = n0 + tx * GK_CU_MM_TILE_T + j;
            if (n < n_cols) {
                gk_cu_set(d, m, n, i2, i3, acc[i][j]);
            }
        }
    }
}

void gk_cuda_mul_mat(gkStream_t stream, struct gk_cuda_scratch * scratch,
                     struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    const int64_t k_len = src0->ne[0];
    const int64_t r2    = src1->ne[2] / src0->ne[2];
    const int64_t r3    = src1->ne[3] / src0->ne[3];

    const int n_warps = GK_CU_MM_BLOCK / GK_WARP_SIZE;

    // Enough columns for a tile to be mostly real work: reuse across the tile
    // beats splitting k across the block, by a wide margin on a batch.
    if (dst->ne[1] >= GK_CU_MM_TILE_MIN_N) {
        // The tensor-core pilot. nvfp4 only, and only where the instruction
        // exists: `mma.sync` with integer operands is Ampere and later. A
        // 64-element nvfp4 block and a 32-element activation block both have
        // to divide k, so 64 does.
        if ((int) src0->type == GK_TYPE_NVFP4 && k_len % 64 == 0 &&
            scratch != NULL && scratch->cc >= 80) {
            const int64_t n_grp  = k_len / 32;
            const int64_t n_cols = dst->ne[1];
            const int64_t n_23   = dst->ne[2] * dst->ne[3];
            const int64_t n_blk  = n_grp * n_cols * n_23;

            gk_cu_q8blk * aq = (gk_cu_q8blk *) gk_cu_scratch_get(
                scratch, (size_t) n_blk * sizeof(gk_cu_q8blk), stream);

            if (aq != NULL) {
                gk_cu_k_quantize_act<<<gk_cu_blocks_1d(n_blk, GK_CUDA_BLOCK),
                                       GK_CUDA_BLOCK, 0, stream>>>(
                    gk_cu_view(src1), aq, n_grp, n_cols, n_blk);

                dim3 mgrid;
                mgrid.x = (unsigned) ((dst->ne[0] + GK_CU_MMA_TILE_M - 1) / GK_CU_MMA_TILE_M);
                mgrid.y = (unsigned) ((n_cols     + GK_CU_MMA_TILE_N - 1) / GK_CU_MMA_TILE_N);
                mgrid.z = (unsigned) n_23;

                gk_cu_k_mul_mat_mma_nvfp4<<<mgrid, 128, 0, stream>>>(
                    gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3);
                return;
            }
        }

        // The f16 tensor-core tile - the convolution path of every diffusion
        // model, and the one shape class where the float tile below is an
        // order of magnitude off what the part can do.
        if (gk_cuda_mma_f16_supported(dst, scratch)) {
            // Rows only pick how many blocks there are, so a shape with too
            // few of them for the wide tile takes the narrow one rather than
            // launching half-empty blocks.
            const int warps_m = dst->ne[0] >= 4 * GK_CU_MMA_F16_WM ? 4 : 2;
            const int tile_m  = warps_m * GK_CU_MMA_F16_WM;

            const int64_t n_23   = dst->ne[2] * dst->ne[3];
            const int64_t n_rows = dst->ne[0];
            const int64_t n_cols = dst->ne[1];

            const int64_t grid_m = (n_rows + tile_m - 1) / tile_m;
            const int64_t grid_n = (n_cols + GK_CU_MMA_F16_TILE_N - 1) / GK_CU_MMA_F16_TILE_N;

            // A tile this wide buys its reuse by making blocks scarce, and an
            // output too small to cover the device leaves multiprocessors
            // standing still no matter how much arithmetic each block does.
            // A UNet's deepest levels are exactly that shape - 64 pixels
            // against 1280 channels is ten blocks on a twenty multiprocessor
            // part - and the only axis left to cut is k, because it is the one
            // dimension the output does not have.
            //
            // What decides it is warps rather than blocks, and the difference
            // is not pedantic: the narrow tile has half the warps of the wide
            // one, so counting blocks says the same thing about a shape that
            // fills the device and one that half-fills it. Measured, 20 blocks
            // of the wide tile want no split and splitting them costs half the
            // throughput, while 10 blocks of the narrow tile - the same block
            // count, half the warps - gain three times over.
            //
            // A split costs a plane of f32 per piece and a pass to add them
            // up, so it is also capped by k being long enough that a piece of
            // it is still a reasonable unit of work.
            const int64_t n_warps = warps_m * GK_CU_MMA_F16_WARPS_N;
            const int64_t want    = (int64_t) scratch->n_sm * GK_CU_MMA_F16_SPLIT_WARPS;
            const int64_t have    = grid_m * grid_n * n_23 * n_warps;

            int64_t n_splits = 1;

            if (have < want) {
                n_splits = (want + have - 1) / have;
                if (n_splits > k_len / GK_CU_MMA_F16_SPLIT_MIN_K) {
                    n_splits = k_len / GK_CU_MMA_F16_SPLIT_MIN_K;
                }
                if (n_splits < 1) {
                    n_splits = 1;
                }
            }

            // Rounded up to the staged k so that every split starts on a
            // boundary the vectorized staging can still read from.
            const int64_t k_split =
                ((k_len + n_splits - 1) / n_splits + GK_CU_MMA_F16_K - 1)
                / GK_CU_MMA_F16_K * GK_CU_MMA_F16_K;

            n_splits = (k_len + k_split - 1) / k_split;

            float * part = NULL;
            if (n_splits > 1) {
                part = (float *) gk_cu_scratch_get(
                    scratch, (size_t) n_splits * n_23 * n_cols * n_rows * sizeof(float), stream);
                if (part == NULL) {
                    n_splits = 1;   // no room; one block per patch still works
                }
            }

            // Still short of warps with k already cut up as far as it goes:
            // the output is simply too small for a tile this wide, and the
            // paths below - narrower tiles, k split across a block rather than
            // across blocks - keep more of the device busy.
            if (have * n_splits >= want / 2) {
                const bool a_vec = gk_cuda_mma_f16_vec(src0, k_len);
                const bool b_vec = gk_cuda_mma_f16_vec(src1, k_len);

                dim3 fgrid;
                fgrid.x = (unsigned) grid_m;
                fgrid.y = (unsigned) grid_n;
                fgrid.z = (unsigned) (n_23 * n_splits);

                if (warps_m == 4) {
                    gk_cu_k_mul_mat_mma_f16<4><<<fgrid, 4 * GK_CU_MMA_F16_WARPS_N * GK_WARP_SIZE, 0, stream>>>(
                        gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst),
                        k_len, r2, r3, a_vec, b_vec,
                        n_splits > 1 ? part : NULL, k_split, n_23);
                } else {
                    gk_cu_k_mul_mat_mma_f16<2><<<fgrid, 2 * GK_CU_MMA_F16_WARPS_N * GK_WARP_SIZE, 0, stream>>>(
                        gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst),
                        k_len, r2, r3, a_vec, b_vec,
                        n_splits > 1 ? part : NULL, k_split, n_23);
                }

                if (n_splits > 1) {
                    const int64_t n_out = n_rows * n_cols * n_23;
                    gk_cu_k_mma_f16_combine<<<gk_cu_blocks_1d(n_out, GK_CUDA_BLOCK),
                                              GK_CUDA_BLOCK, 0, stream>>>(
                        part, gk_cu_view_mut(dst), n_splits, n_23, n_rows, n_cols);
                }
                return;
            }
        }

        // The integer tile, where the format has one. Same quantized
        // activations as the mat-vec path, so the same scratch and the same
        // one-off quantize pass; a batch amortizes it even better.
        if (gk_cuda_mm_q8_supported((int) src0->type, k_len, dst->ne[0]) &&
            scratch != NULL) {
            const int64_t n_grp  = k_len / 32;
            const int64_t n_cols = dst->ne[1];
            const int64_t n_23   = dst->ne[2] * dst->ne[3];
            const int64_t n_blk  = n_grp * n_cols * n_23;

            gk_cu_q8blk * aq = (gk_cu_q8blk *) gk_cu_scratch_get(
                scratch, (size_t) n_blk * sizeof(gk_cu_q8blk), stream);

            if (aq != NULL) {
                gk_cu_k_quantize_act<<<gk_cu_blocks_1d(n_blk, GK_CUDA_BLOCK),
                                       GK_CUDA_BLOCK, 0, stream>>>(
                    gk_cu_view(src1), aq, n_grp, n_cols, n_blk);

                dim3 qgrid;
                qgrid.x = (unsigned) ((dst->ne[0] + GK_CU_MMQ_TILE_M - 1) / GK_CU_MMQ_TILE_M);
                qgrid.y = (unsigned) ((n_cols     + GK_CU_MMQ_TILE_N - 1) / GK_CU_MMQ_TILE_N);
                qgrid.z = (unsigned) n_23;

#define GK_CU_LAUNCH_TILED_Q8(T)                                           \
                gk_cu_k_mul_mat_tiled_q8<T><<<qgrid, dim3(16, 16), 0, stream>>>( \
                    gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3)

                GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_TILED_Q8);
#undef GK_CU_LAUNCH_TILED_Q8
                return;
            }
        }

        dim3 tgrid;
        tgrid.x = (unsigned) ((dst->ne[0] + GK_CU_MM_TILE_M - 1) / GK_CU_MM_TILE_M);
        tgrid.y = (unsigned) ((dst->ne[1] + GK_CU_MM_TILE_N - 1) / GK_CU_MM_TILE_N);
        tgrid.z = (unsigned) (dst->ne[2] * dst->ne[3]);

#define GK_CU_LAUNCH_TILED(T)                                              \
        gk_cu_k_mul_mat_tiled<T><<<tgrid, dim3(16, 16), 0, stream>>>(      \
            gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst), k_len, r2, r3)

        GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_TILED);
#undef GK_CU_LAUNCH_TILED
        return;
    }

    // A warp per output row, so a block of GK_CU_MM_BLOCK threads retires
    // n_warps of them and the grid is that much shorter.
    dim3 grid;
    grid.x = (unsigned) ((dst->ne[0] + n_warps - 1) / n_warps);
    grid.z = (unsigned) (dst->ne[2] * dst->ne[3]);

    // The integer path, where the format has one and the row divides into
    // whole 32-element groups. It quantizes the activations first, which needs
    // somewhere to put them; without that scratch it simply does not run and
    // the float path below gives the same answer more slowly.
    if (gk_cuda_mm_q8_supported((int) src0->type, k_len, dst->ne[0]) && scratch != NULL) {
        const int64_t n_grp  = k_len / 32;
        const int64_t n_cols = dst->ne[1];
        const int64_t n_23   = dst->ne[2] * dst->ne[3];
        const int64_t n_blk  = n_grp * n_cols * n_23;

        gk_cu_q8blk * aq = (gk_cu_q8blk *) gk_cu_scratch_get(
            scratch, (size_t) n_blk * sizeof(gk_cu_q8blk), stream);

        if (aq != NULL) {
            gk_cu_k_quantize_act<<<gk_cu_blocks_1d(n_blk, GK_CUDA_BLOCK),
                                   GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src1), aq, n_grp, n_cols, n_blk);

            if (n_cols < GK_CU_MM_NC) {
                grid.y = (unsigned) n_cols;

#define GK_CU_LAUNCH_Q8_1(T)                                               \
                gk_cu_k_mul_mat_q8<1, T><<<grid, GK_CU_MM_BLOCK, 0, stream>>>( \
                    gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3)

                GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_Q8_1);
#undef GK_CU_LAUNCH_Q8_1
            } else {
                grid.y = (unsigned) ((n_cols + GK_CU_MM_NC - 1) / GK_CU_MM_NC);

#define GK_CU_LAUNCH_Q8_N(T)                                               \
                gk_cu_k_mul_mat_q8<GK_CU_MM_NC, T><<<grid, GK_CU_MM_BLOCK, 0, stream>>>( \
                    gk_cu_view(src0), gk_cu_view_mut(dst), aq, n_grp, r2, r3)

                GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_Q8_N);
#undef GK_CU_LAUNCH_Q8_N
            }
            return;
        }
    }

    // One column at a time when there is only one - the decode case, every
    // token of generation - and blocks of NC once a batch makes the reuse
    // real. Below NC columns the extra accumulators would sit idle.
    if (dst->ne[1] < GK_CU_MM_NC) {
        grid.y = (unsigned) dst->ne[1];

#define GK_CU_LAUNCH_MV1(T)                                                \
        gk_cu_k_mul_mat<1, T><<<grid, GK_CU_MM_BLOCK, 0, stream>>>(            \
            gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst), k_len, r2, r3)

        GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_MV1);
#undef GK_CU_LAUNCH_MV1
    } else {
        grid.y = (unsigned) ((dst->ne[1] + GK_CU_MM_NC - 1) / GK_CU_MM_NC);

#define GK_CU_LAUNCH_MVN(T)                                                \
        gk_cu_k_mul_mat<GK_CU_MM_NC, T><<<grid, GK_CU_MM_BLOCK, 0, stream>>>(  \
            gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst), k_len, r2, r3)

        GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_MVN);
#undef GK_CU_LAUNCH_MVN
    }
}

// --------------------------------------------------------------------------
// mixture of experts
//
// `ids` holds, per token, which experts it routes to; every (token, slot) pair
// picks one expert's rows out of `as`. One block per output element again, but
// the expert index is read per block rather than per element.
// --------------------------------------------------------------------------

template <int ATYPE>
static __global__ void gk_cu_k_mul_mat_id(gk_tview as, gk_tview b, gk_tview ids,
                                          gk_tview_mut d, int64_t k_len) {
    extern __shared__ float shared[];

    const int64_t i0 = blockIdx.x; // row within the expert
    const int64_t is = blockIdx.y; // which of the token's expert slots
    const int64_t it = blockIdx.z; // which token

    const int32_t expert = *(const int32_t *) (ids.data + it * ids.nb[1] + is * ids.nb[0]);

    float acc[1] = { 0.0f };

    for (int64_t kk = threadIdx.x; kk < k_len; kk += blockDim.x) {
        acc[0] += gk_cu_get_t<ATYPE>(as, kk, i0, expert, 0) * gk_cu_get(b, kk, 0, it, 0);
    }

    gk_cu_reduce_n<1>(acc, shared);

    if (threadIdx.x == 0) {
        gk_cu_set(d, i0, is, it, 0, acc[0]);
    }
}

void gk_cuda_mul_mat_id(gkStream_t stream, struct gk_tensor * dst) {
    const struct gk_tensor * as  = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];
    const struct gk_tensor * ids = dst->src[2];

    dim3 grid;
    grid.x = (unsigned) dst->ne[0];
    grid.y = (unsigned) dst->ne[1];
    grid.z = (unsigned) dst->ne[2];

    const int n_warps = GK_CU_MM_BLOCK / GK_WARP_SIZE;

#define GK_CU_LAUNCH_ID(T)                                                 \
    gk_cu_k_mul_mat_id<T><<<grid, GK_CU_MM_BLOCK,                          \
                            n_warps * sizeof(float), stream>>>(            \
        gk_cu_view(as), gk_cu_view(src1), gk_cu_view(ids), gk_cu_view_mut(dst), as->ne[0])

    GK_CU_MM_DISPATCH((int) as->type, GK_CU_LAUNCH_ID);
#undef GK_CU_LAUNCH_ID
}

// --------------------------------------------------------------------------
// fused attention
//
// One pass over the keys with an online softmax: keep the running maximum M
// and normaliser S, and rescale the value accumulator whenever the maximum
// moves. The accumulator is f32 whatever the value type is, which is what the
// CPU pass does and what keeps a graph split across devices from drifting.
//
// The query row and the value accumulator live in shared memory, which is what
// bounds the head sizes this kernel accepts; larger ones fall back to the CPU
// rather than being silently truncated.
//
// There are two ways to spread that pass over the device, and which one is
// right depends entirely on the batch.
//
// A prompt pass has a query row per token per head - thousands of them - so
// one block each already fills the card, and each block walking the whole
// cache is the cheapest thing to do: the query row is read once and the cache
// is read once.
//
// Generation has one token. Eight heads, one query row each, is eight blocks,
// and a card with twenty multiprocessors runs them in a single wave with most
// of itself idle - and then that handful of blocks walks the entire cache one
// position at a time. Measured on a 4050, that path ran at 0.8 GB/s and lost
// to the CPU by a factor of three.
//
// So when the row count alone will not fill the device, the cache is cut into
// slices and a block takes one. Each produces a partial result: its own
// accumulator, its own M, its own S, each relative to the slice it saw. A
// second pass merges them, which is possible because the online softmax
// composes - a partial from a slice is exactly what that slice would have
// contributed had it been seen first, and rescaling it to a common maximum is
// the same arithmetic the single-block path already does whenever its running
// maximum moves.
//
// Both paths run the same accumulation loop, below, told apart only by whether
// it was handed somewhere to put partials.
// --------------------------------------------------------------------------

#define GK_CU_FA_BLOCK   128

// Splitting only pays once there is enough cache to divide; below this the
// launch and the merge cost more than the parallelism is worth.
#define GK_CU_FA_MIN_SPLIT_KV 256

// What the split is aiming at: enough blocks to give every multiprocessor
// several, and slices short enough that no block is left walking a long tail
// alone. Both are targets rather than limits - the clamps below decide.
#define GK_CU_FA_BLOCKS_PER_SM  8
#define GK_CU_FA_TARGET_SLICE 256

// A slice shorter than this is mostly launch overhead, and more splits than
// this stop paying for the merge they add.
#define GK_CU_FA_MIN_SLICE  64
#define GK_CU_FA_MAX_SPLIT  64

// --------------------------------------------------------------------------
// The tiled path: a block of query rows against the cache a tile at a time.
//
// The kernel below walks the cache one position per iteration, and for a
// generation batch that is fine - the cache is short and the split across
// blocks covers it. For a diffusion transformer it is ruinous. Every token
// attends to every other, so a 4096-token layer has 65536 query rows and each
// one of them re-reads the whole of K and V: about 64 GB of requests per
// attention layer, and a block reduction with two barriers at every one of the
// 4096 steps. Measured, 1.15 seconds for a single layer.
//
// Two changes fix it, and they are the two FlashAttention makes:
//
//   * A block owns several query rows, not one, and stages each K/V tile into
//     shared memory once for all of them. That is where the factor of eight in
//     memory traffic comes from.
//
//   * A lane owns a whole key position rather than a slice of one. The dot
//     product for that key is then entirely within the lane - no reduction at
//     all - and the only cross-lane work left is the softmax's maximum and
//     sum, which happen once per tile of 32 keys instead of once per key.
//
// The output accumulator is distributed the other way, a lane per value
// dimension, so the probabilities have to cross between the two layouts. That
// is what the small per-warp `Ps` array is for.
// --------------------------------------------------------------------------

#define GK_CU_FAT_WARPS   8    // warps per block
#define GK_CU_FAT_QR      4    // query rows per warp
#define GK_CU_FAT_QROWS   (GK_CU_FAT_WARPS * GK_CU_FAT_QR)
#define GK_CU_FAT_BC      32   // key positions per tile, one per lane
#define GK_CU_FAT_MAX_D   128  // head width the shared tiles are budgeted for
#define GK_CU_FAT_DV_LANE (GK_CU_FAT_MAX_D / GK_WARP_SIZE)

static __global__ void gk_cu_k_flash_attn_tiled(gk_tview q, gk_tview k, gk_tview v,
                                                gk_tview mask, bool has_mask,
                                                const float * sinks, gk_tview_mut d,
                                                float scale, float max_bias,
                                                float logit_softcap, int64_t n_head_log2,
                                                int64_t rk2, int64_t rk3,
                                                int64_t rv2, int64_t rv3) {
    extern __shared__ float fat_smem[];

    const int64_t DK   = k.ne[0];
    const int64_t DV   = v.ne[0];
    const int64_t n_kv = k.ne[1];

    // Ks and Vs are padded by one so that a lane walking a key's row and a lane
    // walking a value's column both stride across banks rather than into one.
    float * Ks = fat_smem;
    float * Vs = Ks + GK_CU_FAT_BC * (DK + 1);
    float * Qs = Vs + GK_CU_FAT_BC * (DV + 1);
    float * Ps = Qs + GK_CU_FAT_QROWS * DK;

    const int lane = threadIdx.x % GK_WARP_SIZE;
    const int warp = threadIdx.x / GK_WARP_SIZE;

    const int64_t q1_0 = (int64_t) blockIdx.x * GK_CU_FAT_QROWS + warp * GK_CU_FAT_QR;
    const int64_t iq2  = blockIdx.y;
    const int64_t iq3  = blockIdx.z;

    const int64_t ik2 = iq2 / rk2, ik3 = iq3 / rk3;
    const int64_t iv2 = iq2 / rv2, iv3 = iq3 / rv3;

    const float slope = gk_cu_alibi_slope(max_bias, iq2, n_head_log2);

    bool live[GK_CU_FAT_QR];
#pragma unroll
    for (int r = 0; r < GK_CU_FAT_QR; ++r) {
        live[r] = q1_0 + r < q.ne[1];
        if (live[r]) {
            for (int64_t i = lane; i < DK; i += GK_WARP_SIZE) {
                Qs[(warp * GK_CU_FAT_QR + r) * DK + i] = gk_cu_get(q, i, q1_0 + r, iq2, iq3);
            }
        }
    }

    float M[GK_CU_FAT_QR];
    float S[GK_CU_FAT_QR];
    float O[GK_CU_FAT_QR][GK_CU_FAT_DV_LANE];
#pragma unroll
    for (int r = 0; r < GK_CU_FAT_QR; ++r) {
        M[r] = -INFINITY;
        S[r] = 0.0f;
#pragma unroll
        for (int t = 0; t < GK_CU_FAT_DV_LANE; ++t) {
            O[r][t] = 0.0f;
        }
    }

    for (int64_t c0 = 0; c0 < n_kv; c0 += GK_CU_FAT_BC) {
        // The whole block stages the tile and every warp reads it back. Four
        // query rows per warp is what makes this pay: the tile is staged once
        // for thirty-two of them, and each staged element is then used four
        // times in the arithmetic below rather than once.
        for (int64_t e = threadIdx.x; e < GK_CU_FAT_BC * DK; e += blockDim.x) {
            const int64_t c = e / DK, i = e % DK;
            Ks[c * (DK + 1) + i] = c0 + c < n_kv ? gk_cu_get(k, i, c0 + c, ik2, ik3) : 0.0f;
        }
        for (int64_t e = threadIdx.x; e < GK_CU_FAT_BC * DV; e += blockDim.x) {
            const int64_t c = e / DV, i = e % DV;
            Vs[c * (DV + 1) + i] = c0 + c < n_kv ? gk_cu_get(v, i, c0 + c, iv2, iv3) : 0.0f;
        }
        __syncthreads();

        // This lane's key. The dot products never leave the lane, and the key
        // is read from shared once for all four query rows.
        const int64_t ic = c0 + lane;

        float sc[GK_CU_FAT_QR];
#pragma unroll
        for (int r = 0; r < GK_CU_FAT_QR; ++r) {
            sc[r] = 0.0f;
        }

        if (ic < n_kv) {
            for (int64_t i = 0; i < DK; ++i) {
                const float kv = Ks[lane * (DK + 1) + i];
#pragma unroll
                for (int r = 0; r < GK_CU_FAT_QR; ++r) {
                    sc[r] += Qs[(warp * GK_CU_FAT_QR + r) * DK + i] * kv;
                }
            }
        }

        float corr[GK_CU_FAT_QR];

#pragma unroll
        for (int r = 0; r < GK_CU_FAT_QR; ++r) {
            float s = -INFINITY;

            if (ic < n_kv && live[r]) {
                float mv = 0.0f;
                bool  skip = false;
                if (has_mask) {
                    mv = slope * gk_cu_get(mask, ic, q1_0 + r,
                                           iq2 % mask.ne[2], iq3 % mask.ne[3]);
                    skip = mv == -INFINITY;
                }
                if (!skip) {
                    s = sc[r] * scale;
                    if (logit_softcap != 0.0f) {
                        s = logit_softcap * tanhf(s);
                    }
                    s += mv;
                }
            }

            // The online softmax, once for the tile rather than once per key.
            const float m_tile = gk_cu_warp_max(s);
            const float m_new  = fmaxf(M[r], m_tile);

            // A tile that was entirely masked leaves the running maximum where
            // it was; expf(-inf - -inf) is not a number, so it is not asked.
            corr[r] = m_new == -INFINITY ? 1.0f : expf(M[r] - m_new);
            const float p = s == -INFINITY || m_new == -INFINITY
                          ? 0.0f : expf(s - m_new);

            S[r] = S[r] * corr[r] + gk_cu_warp_sum(p);
            M[r] = m_new;

            // The probabilities are held one per key, the accumulator one per
            // value dimension; this is where the two layouts meet.
            Ps[(warp * GK_CU_FAT_QR + r) * GK_CU_FAT_BC + lane] = p;
        }
        __syncwarp();

#pragma unroll
        for (int t = 0; t < GK_CU_FAT_DV_LANE; ++t) {
            const int64_t dd = lane + (int64_t) t * GK_WARP_SIZE;
            if (dd >= DV) {
                continue;
            }

            float a[GK_CU_FAT_QR];
#pragma unroll
            for (int r = 0; r < GK_CU_FAT_QR; ++r) {
                a[r] = 0.0f;
            }

            // One value read serves all four rows, which is the other half of
            // what the four-row warp buys.
            for (int c = 0; c < GK_CU_FAT_BC; ++c) {
                const float vv = Vs[c * (DV + 1) + dd];
#pragma unroll
                for (int r = 0; r < GK_CU_FAT_QR; ++r) {
                    a[r] += Ps[(warp * GK_CU_FAT_QR + r) * GK_CU_FAT_BC + c] * vv;
                }
            }

#pragma unroll
            for (int r = 0; r < GK_CU_FAT_QR; ++r) {
                O[r][t] = O[r][t] * corr[r] + a[r];
            }
        }
        __syncwarp();

        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < GK_CU_FAT_QR; ++r) {
        if (!live[r]) {
            continue;
        }

        // the sink is one more virtual position, with a logit but no value row
        if (sinks != NULL) {
            const float sv = sinks[iq2];
            if (sv > M[r]) {
                const float c = M[r] == -INFINITY ? 0.0f : expf(M[r] - sv);
                M[r] = sv;
                S[r] = S[r] * c + 1.0f;
#pragma unroll
                for (int t = 0; t < GK_CU_FAT_DV_LANE; ++t) {
                    O[r][t] *= c;
                }
            } else {
                S[r] += expf(sv - M[r]);
            }
        }

        const float inv = S[r] == 0.0f ? 0.0f : 1.0f / S[r];

#pragma unroll
        for (int t = 0; t < GK_CU_FAT_DV_LANE; ++t) {
            const int64_t dd = lane + (int64_t) t * GK_WARP_SIZE;
            if (dd < DV) {
                gk_cu_set(d, dd, iq2, q1_0 + r, iq3, O[r][t] * inv);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The tensor-core path.
//
// The kernel above is the same algorithm as this one and runs at about a
// tenth of the speed, for a reason that is not the algorithm. Its two inner
// loops each read five values out of shared memory to feed four FFMAs - one
// K element and four Q elements, then one V element and four probabilities -
// and shared memory issues one warp-wide load per multiprocessor-cycle where
// the FFMA pipe issues four. So five loads cost five cycles to feed one
// cycle of arithmetic, and the kernel spends five sixths of itself waiting on
// the memory it is reading operands out of. Measured at the shape a 64x64
// UNet layer runs - 4096 queries against 4096 keys, eight heads, d_head 40 -
// that is 934 GFLOP/s where the same card's f16 GEMM does 12 to 16 TFLOP/s.
//
// The fix is to stop moving operands one element at a time. `mma.sync`
// consumes a whole 16x16-by-16x8 tile per instruction from registers, so the
// ratio inverts: a warp reads its operands once into fragments and gets 2048
// multiply-accumulates out of them.
//
// Three things make the mapping work out better here than a GEMM's:
//
//   * Q is read once, into registers, and stays there for the whole cache.
//     It is the operand a query block reuses across every key tile, so it
//     never belongs in shared memory at all - which is also what frees the
//     room for K and V to be staged at a 64-key tile instead of 32.
//
//   * The S fragment that comes out of the first mma is bit-for-bit the A
//     fragment the second one wants. `mma.m16n8k16` leaves a lane holding D
//     rows `group`/`group+8` at columns `2*tig`/`2*tig+1`, and wants A rows
//     `group`/`group+8` at columns `2*tig`/`2*tig+1` of each 16-wide window.
//     Those are the same elements in the same lanes, so the probabilities go
//     from the softmax straight into the next instruction as registers. No
//     staging, no barrier, and no shared memory for P.
//
//   * The softmax's row reductions shrink from a warp to a quadrant. A row of
//     S lives in the four lanes that share a `group`, so the maximum and the
//     sum are two shuffles over the low two lane bits instead of the five a
//     whole-warp reduction costs.
//
// What the tile costs in exchange is that the products are f16. The
// accumulator is f32 across the whole reduction, and K and V arrive f16 from
// the caller already, so what is new is Q and the probabilities being rounded
// before they are multiplied - the same trade the f16 GEMM makes, and the
// reason the host gate below insists on f16 K and V rather than converting
// an f32 cache down to reach this path.
// --------------------------------------------------------------------------

#define GK_CU_FAM_WARPS   4                                  // warps per block
#define GK_CU_FAM_BR      (GK_CU_FAM_WARPS * 16)             // query rows a block owns
#define GK_CU_FAM_THREADS (GK_CU_FAM_WARPS * GK_WARP_SIZE)

// Key positions per tile. A wide head already spends its registers on the
// output accumulator - a lane holds DV/2 floats of it - and the S tile is
// what it can give back: halving the tile halves S and the probability
// fragments both, and halves the staged K and V with them, which is the
// difference between two blocks resident per multiprocessor and five.
// Narrow heads have the registers to spare and would rather sync half as
// often.
#define GK_CU_FAM_BC(D)   ((D) <= 80 ? 64 : 32)
#define GK_CU_FAM_NTC(D)  (GK_CU_FAM_BC(D) /  8)             // 8-wide S tiles a warp holds
#define GK_CU_FAM_NKW(D)  (GK_CU_FAM_BC(D) / 16)             // 16-wide PV windows

// The head width is a template parameter because the accumulator it sizes is
// registers, which have to be counted at compile time. The buckets are the
// widths that exist: 40 and 80 and 160 are SD's three levels, 64 and 128 are
// what a language model and a diffusion transformer run. A head is rounded up
// to the next bucket and the slack is zero-padded, so 40 costs a 48-wide
// reduction and nothing else.
#define GK_CU_FAM_NTK(D)  (((D) + 15) / 16)                  // 16-wide k windows
#define GK_CU_FAM_NTV(D)  (((D) +  7) /  8)                  // 8-wide value tiles

// Staged rows are padded past the width they hold, for the same reason the
// GEMM's are: a fragment read puts the eight lanes of a quadrant on eight
// different rows at one k, and eight rows an exact power of two apart would
// land on one bank. Every bucket's padded stride is a multiple of 8 halves
// past a multiple of 16, which spreads those eight over eight disjoint quads.
#define GK_CU_FAM_SK(D)   (GK_CU_FAM_NTK(D) * 16 + 8)
#define GK_CU_FAM_SC(D)   (GK_CU_FAM_BC(D) + 8)

// Two floats into the half2 an mma fragment register is. The low half is the
// lower k, which is the order both operands are read out of shared memory in.
static __device__ __forceinline__ int gk_cu_pack2_half(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *(const int *) &h;
}

// Eight f16 out of a row, as one 16-byte word, with the tail of a head that is
// not a multiple of eight zero-filled.
//
// Staging is why this matters. A tile is 64 keys of K and 64 of V, so a block
// moves several thousand elements into shared memory per iteration and issues
// a few hundred mma against them - which means an instruction spent per staged
// element costs several times what the arithmetic does, and the kernel becomes
// a memcpy with a tensor core attached. Whether the eight can be taken as one
// word is decided on the host, once per launch.
static __device__ __forceinline__ int4 gk_cu_fam_run8(const gk_tview & t, bool vec,
                                                      int64_t i0, int64_t i1,
                                                      int64_t i2, int64_t i3,
                                                      int64_t n0) {
    const char * row = gk_cu_row(t, i1, i2, i3);

    if (vec && i0 + 8 <= n0) {
        return *(const int4 *) (row + i0 * 2);
    }

    __align__(16) __half h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        h[j] = i0 + j < n0 ? *(const __half *) (row + (i0 + j) * t.nb[0])
                           : __float2half(0.0f);
    }
    return *(const int4 *) h;
}

template <int D_PAD>
static __global__ __launch_bounds__(GK_CU_FAM_THREADS)
void gk_cu_k_flash_attn_mma(gk_tview q, gk_tview k, gk_tview v,
                            gk_tview mask, bool has_mask,
                            const float * sinks, gk_tview_mut d,
                            float scale, float max_bias,
                            float logit_softcap, int64_t n_head_log2,
                            int64_t rk2, int64_t rk3,
                            int64_t rv2, int64_t rv3,
                            bool k_vec, bool v_vec) {
    const int NT_K = GK_CU_FAM_NTK(D_PAD);
    const int NT_V = GK_CU_FAM_NTV(D_PAD);

    __shared__ __half Ks[GK_CU_FAM_BC(D_PAD)]          [GK_CU_FAM_SK(D_PAD)];
    __shared__ __half Vt[GK_CU_FAM_NTV(D_PAD)*8][GK_CU_FAM_SC(D_PAD)];

    const int64_t DK   = k.ne[0];
    const int64_t DV   = v.ne[0];
    const int64_t n_kv = k.ne[1];
    const int64_t n_q  = q.ne[1];

    const int tid   = (int) threadIdx.x;
    const int lane  = tid % GK_WARP_SIZE;
    const int warp  = tid / GK_WARP_SIZE;
    const int group = lane / 4;
    const int tig   = lane % 4;

    const int64_t iq2 = blockIdx.y;
    const int64_t iq3 = blockIdx.z;

    const int64_t ik2 = iq2 / rk2, ik3 = iq3 / rk3;
    const int64_t iv2 = iq2 / rv2, iv3 = iq3 / rv3;

    const float slope = gk_cu_alibi_slope(max_bias, iq2, n_head_log2);

    // The two query rows this lane owns, in the warp's 16-row tile.
    const int64_t q1_0 = (int64_t) blockIdx.x * GK_CU_FAM_BR + warp * 16;
    const int64_t row0 = q1_0 + group;
    const int64_t row1 = q1_0 + group + 8;

    const bool live0 = row0 < n_q;
    const bool live1 = row1 < n_q;

    // Q, once, as fragments. Rows past the end and k past the head read zero,
    // which contributes nothing to a dot product - so the ragged tail needs no
    // branch anywhere below this point.
    int qf[GK_CU_FAM_NTK(D_PAD)][4];
#pragma unroll
    for (int kt = 0; kt < NT_K; ++kt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t r  = (j & 1) ? row1 : row0;
            const int64_t k0 = kt * 16 + (j >= 2 ? 8 : 0) + 2 * tig;

            const float a = r < n_q && k0     < DK ? gk_cu_get(q, k0,     r, iq2, iq3) : 0.0f;
            const float b = r < n_q && k0 + 1 < DK ? gk_cu_get(q, k0 + 1, r, iq2, iq3) : 0.0f;

            qf[kt][j] = gk_cu_pack2_half(a, b);
        }
    }

    float acc[GK_CU_FAM_NTV(D_PAD)][4];
#pragma unroll
    for (int ct = 0; ct < NT_V; ++ct) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[ct][i] = 0.0f;
        }
    }

    // The running softmax, one pair per lane: row0 in slot 0, row1 in slot 1.
    // Both are replicated across the four lanes of a quadrant, because that is
    // what the reductions below leave behind.
    float M[2] = { -INFINITY, -INFINITY };
    float S[2] = { 0.0f, 0.0f };

    for (int64_t c0 = 0; c0 < n_kv; c0 += GK_CU_FAM_BC(D_PAD)) {
        // K as [key][k] and V transposed to [value dim][key], which is the
        // layout `mma.row.col` wants for the B operand of each of the two
        // products. V's transpose is paid here, on the staging store, rather
        // than in the inner loop - the read side of it is what runs 64 times
        // per tile in the arithmetic below, and the write side once.
        //
        // Both are staged eight elements to the instruction. The divisors are
        // template constants, so the index arithmetic is shifts rather than
        // the 64-bit division a runtime divisor would compile to.
#pragma unroll
        for (int e = tid; e < GK_CU_FAM_BC(D_PAD) * GK_CU_FAM_NTK(D_PAD) * 2;
             e += GK_CU_FAM_THREADS) {
            const int     c  = e / (NT_K * 2);
            const int     i  = (e - c * (NT_K * 2)) * 8;
            const int64_t ic = c0 + c;

            *(int4 *) &Ks[c][i] = ic < n_kv
                ? gk_cu_fam_run8(k, k_vec, i, ic, ik2, ik3, DK)
                : make_int4(0, 0, 0, 0);
        }

#pragma unroll
        for (int e = tid; e < GK_CU_FAM_BC(D_PAD) * GK_CU_FAM_NTV(D_PAD);
             e += GK_CU_FAM_THREADS) {
            const int     c  = e / NT_V;
            const int     i  = (e - c * NT_V) * 8;
            const int64_t ic = c0 + c;

            const int4 w = ic < n_kv
                ? gk_cu_fam_run8(v, v_vec, i, ic, iv2, iv3, DV)
                : make_int4(0, 0, 0, 0);

            // The one scattered write in the kernel: eight value dimensions of
            // one key, which land a row apart in the transposed tile.
            const __half * h = (const __half *) &w;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                Vt[i + j][c] = h[j];
            }
        }
        __syncthreads();

        // S = Q K^T, for this warp's 16 query rows against all 64 keys.
        float s[GK_CU_FAM_NTC(D_PAD)][4];
#pragma unroll
        for (int nt = 0; nt < GK_CU_FAM_NTC(D_PAD); ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[nt][i] = 0.0f;
            }
        }
#pragma unroll
        for (int kt = 0; kt < NT_K; ++kt) {
#pragma unroll
            for (int nt = 0; nt < GK_CU_FAM_NTC(D_PAD); ++nt) {
                int bf[2];
                bf[0] = *(const int *) &Ks[nt * 8 + group][kt * 16 + 2 * tig];
                bf[1] = *(const int *) &Ks[nt * 8 + group][kt * 16 + 8 + 2 * tig];

                gk_cu_mma_f16(s[nt], qf[kt], bf);
            }
        }

        // Scale, softcap and mask, and mark everything that is not a real
        // (query, key) pair as -inf so the softmax gives it no weight. A key
        // past the end of the cache was staged as zero, and a zero logit is
        // not a zero probability - this is the guard that matters.
        //
        // Only the last tile of the cache can be ragged, and whether a query
        // row exists does not change across the cache at all, so both bounds
        // are decided once rather than per element. Without that, the 32
        // elements a lane holds each carry two comparisons against a 64-bit
        // extent into what is otherwise a multiply and an exponential.
        const bool whole = c0 + GK_CU_FAM_BC(D_PAD) <= n_kv;

#pragma unroll
        for (int nt = 0; nt < GK_CU_FAM_NTC(D_PAD); ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t r  = (i >= 2) ? row1 : row0;
                const int64_t ic = c0 + nt * 8 + 2 * tig + (i & 1);

                float x = -INFINITY;
                if (((i >= 2) ? live1 : live0) && (whole || ic < n_kv)) {
                    x = s[nt][i] * scale;
                    if (logit_softcap != 0.0f) {
                        x = logit_softcap * tanhf(x);
                    }
                    if (has_mask) {
                        const float mv = slope * gk_cu_get(mask, ic, r,
                                                           iq2 % mask.ne[2],
                                                           iq3 % mask.ne[3]);
                        x = mv == -INFINITY ? -INFINITY : x + mv;
                    }
                }
                s[nt][i] = x;
            }
        }

        // The row maximum. A row lives in the four lanes that share a group,
        // so the reduction is over the low two lane bits and nothing else.
        float m[2] = { -INFINITY, -INFINITY };
#pragma unroll
        for (int nt = 0; nt < GK_CU_FAM_NTC(D_PAD); ++nt) {
            m[0] = fmaxf(m[0], fmaxf(s[nt][0], s[nt][1]));
            m[1] = fmaxf(m[1], fmaxf(s[nt][2], s[nt][3]));
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 1));
            m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 2));
        }

        float corr[2];
        float sum[2] = { 0.0f, 0.0f };
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float m_new = fmaxf(M[r], m[r]);
            // A tile that was entirely masked leaves the running maximum
            // where it was; expf(-inf - -inf) is not a number, so it is not
            // asked.
            corr[r] = m_new == -INFINITY ? 1.0f : expf(M[r] - m_new);
            M[r]    = m_new;
        }

#pragma unroll
        for (int nt = 0; nt < GK_CU_FAM_NTC(D_PAD); ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int   r = (i >= 2) ? 1 : 0;
                const float p = s[nt][i] == -INFINITY || M[r] == -INFINITY
                              ? 0.0f : expf(s[nt][i] - M[r]);
                s[nt][i] = p;
                sum[r]  += p;
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 1);
            sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 2);
            S[r]    = S[r] * corr[r] + sum[r];
        }

        // O = O * corr + P V. The probabilities are already in the lanes and
        // the registers the A operand wants, so they are packed rather than
        // staged: window `kw` covers keys `kw*16 .. +15`, which is exactly the
        // pair of 8-wide S tiles `2*kw` and `2*kw+1`.
        int pf[GK_CU_FAM_NKW(D_PAD)][4];
#pragma unroll
        for (int kw = 0; kw < GK_CU_FAM_NKW(D_PAD); ++kw) {
            pf[kw][0] = gk_cu_pack2_half(s[2 * kw + 0][0], s[2 * kw + 0][1]);
            pf[kw][1] = gk_cu_pack2_half(s[2 * kw + 0][2], s[2 * kw + 0][3]);
            pf[kw][2] = gk_cu_pack2_half(s[2 * kw + 1][0], s[2 * kw + 1][1]);
            pf[kw][3] = gk_cu_pack2_half(s[2 * kw + 1][2], s[2 * kw + 1][3]);
        }

#pragma unroll
        for (int ct = 0; ct < NT_V; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[ct][i] *= corr[(i >= 2) ? 1 : 0];
            }
#pragma unroll
            for (int kw = 0; kw < GK_CU_FAM_NKW(D_PAD); ++kw) {
                int bf[2];
                bf[0] = *(const int *) &Vt[ct * 8 + group][kw * 16 + 2 * tig];
                bf[1] = *(const int *) &Vt[ct * 8 + group][kw * 16 + 8 + 2 * tig];

                gk_cu_mma_f16(acc[ct], pf[kw], bf);
            }
        }

        // The tile is read by every warp, so it cannot be overwritten until
        // every warp is done with it.
        __syncthreads();
    }

    // The sink is one more virtual position, with a logit but no value row.
    if (sinks != NULL) {
        const float sv = sinks[iq2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (sv > M[r]) {
                const float c = M[r] == -INFINITY ? 0.0f : expf(M[r] - sv);
                M[r] = sv;
                S[r] = S[r] * c + 1.0f;
#pragma unroll
                for (int ct = 0; ct < NT_V; ++ct) {
                    acc[ct][2 * r + 0] *= c;
                    acc[ct][2 * r + 1] *= c;
                }
            } else {
                S[r] += expf(sv - M[r]);
            }
        }
    }

    const float inv[2] = { S[0] == 0.0f ? 0.0f : 1.0f / S[0],
                           S[1] == 0.0f ? 0.0f : 1.0f / S[1] };

#pragma unroll
    for (int ct = 0; ct < NT_V; ++ct) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t r  = (i >= 2) ? row1 : row0;
            const int64_t dv = ct * 8 + 2 * tig + (i & 1);

            if (r < n_q && dv < DV) {
                gk_cu_set(d, dv, iq2, r, iq3, acc[ct][i] * inv[(i >= 2) ? 1 : 0]);
            }
        }
    }
}

// Which instantiation a head goes to, or zero for one that has none. Both
// widths have to fit: the k windows are sized from the bucket and so are the
// value tiles, so a head that is narrow in V and wide in K - which is what
// multi-head latent attention is - has no bucket here and takes the float
// path instead of a bucket that would silently drop the rest of K.
// Whether a cache can be staged eight elements at a time: packed along the
// head, and every row start 16-byte aligned so the run is one word. A cache
// that is neither still works, an element at a time, through the scalar arm of
// gk_cu_fam_run8 - which is what a permuted or offset view of one falls to.
static __host__ __forceinline__ bool gk_cuda_fam_vec(const struct gk_tensor * t) {
    if ((size_t) t->nb[0] != gk_type_size(t->type)) {
        return false;
    }
    const uintptr_t bits = (uintptr_t) t->data | (uintptr_t) t->nb[1]
                         | (uintptr_t) t->nb[2] | (uintptr_t) t->nb[3];
    return bits % 16 == 0;
}

static __host__ __forceinline__ int gk_cuda_fam_bucket(int64_t dk, int64_t dv) {
#if defined(GK_USE_HIP)
    GK_UNUSED(dk);
    GK_UNUSED(dv);
    return 0;   // `mma.sync` is PTX; there is no HIP spelling of it here
#else
    const int64_t w = dk > dv ? dk : dv;

    if (w <=  40) { return  40; }
    if (w <=  64) { return  64; }
    if (w <=  80) { return  80; }
    if (w <= 128) { return 128; }
    if (w <= 160) { return 160; }

    return 0;
#endif
}

// `part_vkq` and `part_ms`, when given, are where a slice leaves its partial
// result instead of writing an answer: n_split accumulators of DV floats, and
// n_split (M, S) pairs, indexed by row then slice. When they are NULL this is
// the whole-cache path and the tail below finishes the softmax itself.
static __global__ void gk_cu_k_flash_attn(gk_tview q, gk_tview k, gk_tview v,
                                          gk_tview mask, bool has_mask,
                                          const float * sinks, gk_tview_mut d,
                                          float scale, float max_bias, float logit_softcap,
                                          int64_t n_head_log2,
                                          int64_t rk2, int64_t rk3, int64_t rv2, int64_t rv3,
                                          float * part_vkq, float * part_ms, int n_split) {
    __shared__ float sq[GK_CUDA_FA_MAX_DK];
    __shared__ float vkq[GK_CUDA_FA_MAX_DV];
    __shared__ float reduce[GK_CU_FA_BLOCK / GK_WARP_SIZE];
    __shared__ float s_shared;
    __shared__ float ms_shared;
    __shared__ float vs_shared;

    const int64_t DK = k.ne[0];
    const int64_t DV = v.ne[0];

    const int64_t ir  = blockIdx.x;
    const int64_t iq1 = ir % q.ne[1];
    const int64_t iq2 = (ir / q.ne[1]) % q.ne[2];
    const int64_t iq3 = ir / (q.ne[1] * q.ne[2]);

    // The slice of the cache this block owns. Unsplit, that is all of it.
    const int64_t n_kv = k.ne[1];
    const int64_t per  = (n_kv + n_split - 1) / n_split;
    const int64_t ic0  = (int64_t) blockIdx.y * per;
    const int64_t ic1  = ic0 + per < n_kv ? ic0 + per : n_kv;

    // A slice past the end of the cache - the last one, when the split does
    // not divide evenly. It still has to leave a partial behind, because the
    // merge reads every slot rather than checking which were written.
    if (ic0 >= n_kv) {
        if (part_ms != NULL && threadIdx.x == 0) {
            const int64_t slot = (ir * n_split + blockIdx.y) * 2;
            part_ms[slot + 0] = -INFINITY;
            part_ms[slot + 1] = 0.0f;
        }
        return;
    }

    const int64_t ik2 = iq2 / rk2, ik3 = iq3 / rk3;
    const int64_t iv2 = iq2 / rv2, iv3 = iq3 / rv3;

    const float slope = gk_cu_alibi_slope(max_bias, iq2, n_head_log2);

    for (int64_t i = threadIdx.x; i < DK; i += blockDim.x) {
        sq[i] = gk_cu_get(q, i, iq1, iq2, iq3);
    }
    for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
        vkq[i] = 0.0f;
    }
    __syncthreads();

    float M = -INFINITY;
    float S = 0.0f;

    for (int64_t ic = ic0; ic < ic1; ++ic) {
        // A fully masked position contributes nothing and is skipped before
        // the dot product rather than after it.
        float mv = 0.0f;
        if (has_mask) {
            mv = slope * gk_cu_get(mask, ic, iq1, iq2 % mask.ne[2], iq3 % mask.ne[3]);
            if (mv == -INFINITY) {
                continue;
            }
        }

        float part = 0.0f;
        for (int64_t i = threadIdx.x; i < DK; i += blockDim.x) {
            part += sq[i] * gk_cu_get(k, i, ic, ik2, ik3);
        }
        const float dot = gk_cu_block_sum(part, reduce);

        if (threadIdx.x == 0) {
            float s = dot * scale;
            if (logit_softcap != 0.0f) {
                s = logit_softcap * tanhf(s);
            }
            s += mv;
            s_shared = s;
        }
        __syncthreads();

        const float s = s_shared;

        // The rescale: if this logit is the new maximum, everything already
        // accumulated is scaled down to match it; otherwise this value is
        // scaled down to the running maximum.
        float ms = 1.0f;
        float vs = 1.0f;
        if (s > M) {
            ms = expf(M - s);
            M  = s;
            S  = S * ms + 1.0f;
        } else {
            vs = expf(s - M);
            S += vs;
        }

        if (threadIdx.x == 0) {
            ms_shared = ms;
            vs_shared = vs;
        }
        __syncthreads();

        const float ms_b = ms_shared;
        const float vs_b = vs_shared;

        for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
            vkq[i] = vkq[i] * ms_b + gk_cu_get(v, i, ic, iv2, iv3) * vs_b;
        }
        __syncthreads();
    }

    // A slice hands its accumulator over unnormalised and unsunk. Dividing by
    // its own S here would be wrong - S is only this slice's share of the
    // total - and the sink belongs to the row rather than to any one slice, so
    // applying it per slice would count it n_split times. Both are the merge's
    // job.
    if (part_vkq != NULL) {
        const int64_t slot = ir * n_split + blockIdx.y;

        for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
            part_vkq[slot * DV + i] = vkq[i];
        }
        if (threadIdx.x == 0) {
            part_ms[slot * 2 + 0] = M;
            part_ms[slot * 2 + 1] = S;
        }
        return;
    }

    // the sink is one more virtual position, with a logit but no value row
    if (sinks != NULL) {
        const float s = sinks[iq2];
        if (s > M) {
            const float ms = expf(M - s);
            M = s;
            S = S * ms + 1.0f;

            for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
                vkq[i] *= ms;
            }
            __syncthreads();
        } else {
            S += expf(s - M);
        }
    }

    const float inv = S == 0.0f ? 0.0f : 1.0f / S;

    // heads and batch swap on the way out: dst is [DV, n_head, n_batch, ne3]
    for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
        gk_cu_set(d, i, iq2, iq1, iq3, vkq[i] * inv);
    }
}

// Merges the slices of one query row.
//
// Each slice arrived with its accumulator relative to its own maximum, so the
// merge picks the maximum of those maxima and rescales every slice to it - the
// same `exp(M_slice - M)` factor the accumulation loop applies to itself
// whenever its running maximum moves. The sink, which belongs to the row and
// not to any slice, joins here as one more virtual position.
//
// The slices are walked in index order rather than reduced in a tree, so the
// sum is the same sum every run. That costs a little parallelism at n_split of
// 64 and buys back the property that two runs of one prompt agree.
static __global__ void gk_cu_k_flash_attn_combine(const float * part_vkq,
                                                  const float * part_ms,
                                                  const float * sinks, gk_tview_mut d,
                                                  int64_t DV, int n_split,
                                                  int64_t nq1, int64_t nq2) {
    const int64_t ir  = blockIdx.x;
    const int64_t iq1 = ir % nq1;
    const int64_t iq2 = (ir / nq1) % nq2;
    const int64_t iq3 = ir / (nq1 * nq2);

    const float * ms = part_ms + ir * n_split * 2;

    // The common maximum. A slice that saw nothing - every position masked, or
    // a slice off the end of the cache - reports S of zero and is skipped
    // rather than allowed to drag the maximum to -inf.
    float M = -INFINITY;
    for (int s = 0; s < n_split; ++s) {
        if (ms[s * 2 + 1] > 0.0f && ms[s * 2 + 0] > M) {
            M = ms[s * 2 + 0];
        }
    }
    if (sinks != NULL && sinks[iq2] > M) {
        M = sinks[iq2];
    }

    // Every thread works this out for itself. It is n_split adds against a
    // block-wide reduction and two barriers, and n_split is at most 64.
    float S = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        if (ms[s * 2 + 1] > 0.0f) {
            S += ms[s * 2 + 1] * expf(ms[s * 2 + 0] - M);
        }
    }
    if (sinks != NULL) {
        S += expf(sinks[iq2] - M);
    }

    const float inv = S == 0.0f ? 0.0f : 1.0f / S;

    for (int64_t i = threadIdx.x; i < DV; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_split; ++s) {
            if (ms[s * 2 + 1] > 0.0f) {
                acc += part_vkq[((ir * n_split) + s) * DV + i] * expf(ms[s * 2 + 0] - M);
            }
        }
        gk_cu_set(d, i, iq2, iq1, iq3, acc * inv);
    }
}

// How many slices to cut the cache into, or 1 to walk it whole. Zero means the
// split was wanted but its scratch could not be had, which the caller turns
// back into the unsplit path rather than a failure.
static int gk_cu_fa_n_split(int n_sm, int64_t rows, int64_t n_kv) {
    if (n_sm <= 0 || n_kv < GK_CU_FA_MIN_SPLIT_KV) {
        return 1;
    }

    // The row count alone already fills the device: splitting would add a
    // merge and buy nothing. This is every prompt pass.
    const int64_t want = (int64_t) n_sm * GK_CU_FA_BLOCKS_PER_SM;
    if (rows >= want) {
        return 1;
    }

    int64_t n_split = (want + rows - 1) / rows;

    // A long cache wants cutting further than the block count asks for: ten
    // blocks each walking eight hundred positions is still ten long walks.
    const int64_t by_length = (n_kv + GK_CU_FA_TARGET_SLICE - 1) / GK_CU_FA_TARGET_SLICE;
    if (by_length > n_split) {
        n_split = by_length;
    }

    const int64_t by_kv = n_kv / GK_CU_FA_MIN_SLICE;
    if (n_split > by_kv) {
        n_split = by_kv;
    }
    if (n_split > GK_CU_FA_MAX_SPLIT) {
        n_split = GK_CU_FA_MAX_SPLIT;
    }

    return n_split < 1 ? 1 : (int) n_split;
}

void gk_cuda_flash_attn(gkStream_t stream, struct gk_cuda_scratch * scratch,
                        struct gk_tensor * dst) {
    const struct gk_tensor * q     = dst->src[0];
    const struct gk_tensor * k     = dst->src[1];
    const struct gk_tensor * v     = dst->src[2];
    const struct gk_tensor * mask  = dst->src[3];
    const struct gk_tensor * sinks = dst->src[4];

    float scale               = gk_get_op_params_f32(dst, 0);
    const float max_bias      = gk_get_op_params_f32(dst, 1);
    const float logit_softcap = gk_get_op_params_f32(dst, 2);

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    int64_t n_head_log2 = 1;
    while (n_head_log2 * 2 <= q->ne[2]) {
        n_head_log2 *= 2;
    }

    const int64_t rows = q->ne[1] * q->ne[2] * q->ne[3];
    const int64_t DK   = k->ne[0];
    const int64_t DV   = v->ne[0];
    const int64_t n_kv = k->ne[1];

    // The tensor-core path, when the head fits one of the widths its
    // accumulator is compiled for and the cache is already f16. A cache that
    // is not gets the float kernel rather than a conversion: reaching the
    // tensor core by rounding an operand the caller chose to keep in f32
    // would be trading their precision for our speed.
    //
    // Query rows are the other condition. A block owns 64 of them and does
    // not split the cache, so a handful of rows against a long cache - which
    // is generation - would leave the card idle. That shape is what the split
    // path below is for.
    const int fam_d = gk_cuda_fam_bucket(DK, DV);

    if (fam_d != 0 && q->ne[1] >= 16 &&
        (int) k->type == GK_TYPE_F16 && (int) v->type == GK_TYPE_F16 &&
        scratch != NULL && scratch->cc >= 80) {

        dim3 mgrid;
        mgrid.x = (unsigned) ((q->ne[1] + GK_CU_FAM_BR - 1) / GK_CU_FAM_BR);
        mgrid.y = (unsigned) q->ne[2];
        mgrid.z = (unsigned) q->ne[3];

#define GK_CU_FAM_LAUNCH(D)                                                        \
        gk_cu_k_flash_attn_mma<D><<<mgrid, GK_CU_FAM_THREADS, 0, stream>>>(         \
            gk_cu_view(q), gk_cu_view(k), gk_cu_view(v),                            \
            mask ? gk_cu_view(mask) : gk_cu_view(q), mask != NULL,                  \
            sinks ? (const float *) sinks->data : NULL,                             \
            gk_cu_view_mut(dst), scale, max_bias, logit_softcap, n_head_log2,       \
            q->ne[2] / k->ne[2], q->ne[3] / k->ne[3],                               \
            q->ne[2] / v->ne[2], q->ne[3] / v->ne[3],                               \
            gk_cuda_fam_vec(k), gk_cuda_fam_vec(v))

        switch (fam_d) {
            case  40: GK_CU_FAM_LAUNCH( 40); return;
            case  64: GK_CU_FAM_LAUNCH( 64); return;
            case  80: GK_CU_FAM_LAUNCH( 80); return;
            case 128: GK_CU_FAM_LAUNCH(128); return;
            default:  GK_CU_FAM_LAUNCH(160); return;
        }

#undef GK_CU_FAM_LAUNCH
    }

    // The tiled path, when there are enough query rows to fill a block's warps
    // and the head is narrow enough for two tiles of it to sit in shared
    // memory. A single query row would leave seven of eight warps idle, so
    // generation keeps the split path below instead.
    const size_t fat_smem = ((size_t) GK_CU_FAT_BC * (DK + 1)
                           + (size_t) GK_CU_FAT_BC * (DV + 1)
                           + (size_t) GK_CU_FAT_QROWS * DK
                           + (size_t) GK_CU_FAT_QROWS * GK_CU_FAT_BC) * sizeof(float);

    // The tiled path needs its whole working set in shared memory, and how much
    // a block may have is a property of the device, not a constant: 48 KB
    // without asking, and on Ampere and later most of the multiprocessor's
    // store on request. A head of 128 - which is what a diffusion transformer
    // usually runs - needs 52 KB, so the request is not optional.
    //
    // Asking for more than the device will give is a launch failure, not a
    // slow kernel, so the limit is checked rather than assumed.
    // DV is bounded separately from the memory: a lane holds the accumulator
    // for value dimensions `lane, lane+32, ...`, and that array is sized at
    // compile time. A wider DV would fit in shared memory long before it fit
    // in the registers, and would quietly drop every dimension past the end.
    if (q->ne[1] >= GK_CU_FAT_QR && DV <= GK_CU_FAT_MAX_D && scratch != NULL &&
        fat_smem <= (size_t) scratch->smem_max) {

        if (fat_smem > 48u * 1024u) {
            // Raising the cap is per-kernel and sticky, so it is done once.
            static bool raised = false;
            if (!raised) {
                raised = true;
                GK_CUDA_CHECK(gkFuncSetAttribute(
                    (const void *) gk_cu_k_flash_attn_tiled,
                    gkFuncAttributeMaxDynamicSharedMemorySize, scratch->smem_max));
            }
        }

        const size_t smem = fat_smem;

        dim3 tgrid;
        tgrid.x = (unsigned) ((q->ne[1] + GK_CU_FAT_QROWS - 1) / GK_CU_FAT_QROWS);
        tgrid.y = (unsigned) q->ne[2];
        tgrid.z = (unsigned) q->ne[3];

        gk_cu_k_flash_attn_tiled<<<tgrid, GK_CU_FAT_WARPS * GK_WARP_SIZE, smem, stream>>>(
            gk_cu_view(q), gk_cu_view(k), gk_cu_view(v),
            mask ? gk_cu_view(mask) : gk_cu_view(q), mask != NULL,
            sinks ? (const float *) sinks->data : NULL,
            gk_cu_view_mut(dst), scale, max_bias, logit_softcap, n_head_log2,
            q->ne[2] / k->ne[2], q->ne[3] / k->ne[3],
            q->ne[2] / v->ne[2], q->ne[3] / v->ne[3]);
        return;
    }

    int n_split = gk_cu_fa_n_split(scratch != NULL ? scratch->n_sm : 0, rows, n_kv);

    float * part_vkq = NULL;
    float * part_ms  = NULL;

    if (n_split > 1) {
        // The accumulators and the (M, S) pairs share one allocation, the
        // pairs after the accumulators, so a grow moves one buffer.
        const size_t n_vkq   = (size_t) rows * n_split * DV;
        const size_t n_ms    = (size_t) rows * n_split * 2;
        const size_t needed  = (n_vkq + n_ms) * sizeof(float);

        float * buf = (float *) gk_cu_scratch_get(scratch, needed, stream);
        if (buf != NULL) {
            part_vkq = buf;
            part_ms  = buf + n_vkq;
        } else {
            // No room for the partials. The whole-cache path needs none and
            // gives the same answer, so it is the fallback rather than an
            // error - slower is better than refusing a graph the scheduler
            // has already placed here.
            n_split = 1;
        }
    }

    dim3 grid;
    grid.x = (unsigned) rows;
    grid.y = (unsigned) n_split;
    grid.z = 1;

    gk_cu_k_flash_attn<<<grid, GK_CU_FA_BLOCK, 0, stream>>>(
        gk_cu_view(q), gk_cu_view(k), gk_cu_view(v),
        mask ? gk_cu_view(mask) : gk_cu_view(q), mask != NULL,
        sinks ? (const float *) sinks->data : NULL,
        gk_cu_view_mut(dst), scale, max_bias, logit_softcap, n_head_log2,
        q->ne[2] / k->ne[2], q->ne[3] / k->ne[3],
        q->ne[2] / v->ne[2], q->ne[3] / v->ne[3],
        part_vkq, part_ms, n_split);

    if (n_split > 1) {
        // One block per row again, and the merge is the only thing that writes
        // the destination. Same stream, so it cannot start before the slices
        // that feed it have finished.
        const int block = DV < GK_CU_FA_BLOCK ? (int) DV : GK_CU_FA_BLOCK;

        gk_cu_k_flash_attn_combine<<<(int) rows, block, 0, stream>>>(
            part_vkq, part_ms,
            sinks ? (const float *) sinks->data : NULL,
            gk_cu_view_mut(dst), DV, n_split, q->ne[1], q->ne[2]);
    }
}
