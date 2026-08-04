#pragma once

// What every kernel in this directory shares: how a tensor is described to the
// device, how an element is read out of one, and the two reductions.
//
// The description is a flat struct rather than the gk_tensor the host holds,
// because a kernel needs only the numbers - a pointer, four extents, four byte
// strides and a type - and copying them into the launch is cheaper than
// chasing host pointers that mean nothing on the device.
//
// Strides are carried and honoured everywhere. It would be simpler to demand
// contiguous operands and make the caller materialise anything else, and it
// would also mean a `gk_cont` before half the ops in a transformer graph: a
// permuted view is the normal shape of attention's operands, not an edge case.

#include "gk_cuda_dequant.cuh"

#include <stdio.h>

// --------------------------------------------------------------------------
// error handling
//
// A failed launch or a failed allocation is not recoverable in any useful
// sense here - the graph is half-run and the device state is unknown - so it
// is reported loudly and the caller is told the graph failed.
// --------------------------------------------------------------------------

#define GK_CUDA_CHECK(expr)                                                     \
    do {                                                                        \
        const gkError_t err_ = (expr);                                          \
        if (err_ != gkSuccess) {                                                \
            fprintf(stderr, "gk %s: %s failed at %s:%d: %s\n",                  \
                    GK_CUDA_BACKEND_NAME, #expr, __FILE__, __LINE__,            \
                    gkGetErrorString(err_));                                    \
        }                                                                       \
    } while (0)

// --------------------------------------------------------------------------
// tensor views
// --------------------------------------------------------------------------

struct gk_tview {
    const char * data;
    int64_t      ne[4];
    int64_t      nb[4];
    int          type;
};

struct gk_tview_mut {
    char *  data;
    int64_t ne[4];
    int64_t nb[4];
    int     type;
};

// Byte address of row (i1,i2,i3).
static __device__ __forceinline__ const char * gk_cu_row(const gk_tview & t,
                                                         int64_t i1, int64_t i2, int64_t i3) {
    return t.data + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3];
}

static __device__ __forceinline__ char * gk_cu_row(const gk_tview_mut & t,
                                                   int64_t i1, int64_t i2, int64_t i3) {
    return t.data + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3];
}

// One element. Float types are read through their stride, which is what makes
// permuted operands work; a quantized row is packed by definition, so its
// elements are found by block arithmetic instead.
static __device__ __forceinline__ float gk_cu_get(const gk_tview & t,
                                                  int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char * row = gk_cu_row(t, i1, i2, i3);

    switch (t.type) {
        case GKT_F32:  return *(const float *)    (row + i0 * t.nb[0]);
        case GKT_F16:  return __half2float(*(const __half *) (row + i0 * t.nb[0]));
        case GKT_BF16: return gk_cu_bf2f(*(const uint16_t *) (row + i0 * t.nb[0]));
        case GKT_I32:  return (float) *(const int32_t *) (row + i0 * t.nb[0]);
        case GKT_I64:  return (float) *(const int64_t *) (row + i0 * t.nb[0]);
        default:       return gk_cu_row_elem(row, t.type, i0);
    }
}

// Reading back what a kernel has just written. Only the float types can be a
// destination, so this needs none of the quantized paths above.
static __device__ __forceinline__ float gk_cu_get_mut(const gk_tview_mut & t,
                                                      int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char * p = gk_cu_row(t, i1, i2, i3) + i0 * t.nb[0];

    switch (t.type) {
        case GKT_F16:  return __half2float(*(const __half *) p);
        case GKT_BF16: return gk_cu_bf2f(*(const uint16_t *) p);
        default:       return *(const float *) p;
    }
}

static __device__ __forceinline__ void gk_cu_set(const gk_tview_mut & t,
                                                 int64_t i0, int64_t i1, int64_t i2, int64_t i3,
                                                 float v) {
    char * p = gk_cu_row(t, i1, i2, i3) + i0 * t.nb[0];

    switch (t.type) {
        case GKT_F32:  *(float *)  p = v; break;
        case GKT_F16:  *(__half *) p = __float2half(v); break;
        case GKT_BF16: {
            // round-to-nearest-even on the way down, the same rule the codec
            // uses, so a value written here and read on the host agree
            uint32_t u;
            memcpy(&u, &v, sizeof(u));
            const uint32_t rounded = u + 0x7fffu + ((u >> 16) & 1u);
            *(uint16_t *) p = (uint16_t) (rounded >> 16);
            break;
        }
        case GKT_I32:  *(int32_t *) p = (int32_t) v; break;
        default: break; // filtered by supports_op
    }
}

// The (i1,i2,i3) of a flat row index, the decomposition every kernel that
// works a row at a time starts with.
static __device__ __forceinline__ void gk_cu_unrow(int64_t ir, const int64_t ne[4],
                                                   int64_t * i1, int64_t * i2, int64_t * i3) {
    *i1 = ir % ne[1];
    *i2 = (ir / ne[1]) % ne[2];
    *i3 = ir / (ne[1] * ne[2]);
}

// --------------------------------------------------------------------------
// reductions
//
// Warp-wide first, then across warps through shared memory. Blocks are always
// a whole number of warps, so the second stage never has a partial warp.
// --------------------------------------------------------------------------

static __device__ __forceinline__ float gk_cu_warp_sum(float x) {
#pragma unroll
    for (int offset = GK_WARP_SIZE / 2; offset > 0; offset >>= 1) {
#if defined(GK_USE_HIP)
        x += __shfl_xor(x, offset, GK_WARP_SIZE);
#else
        x += __shfl_xor_sync(0xffffffff, x, offset, GK_WARP_SIZE);
#endif
    }
    return x;
}

static __device__ __forceinline__ float gk_cu_warp_max(float x) {
#pragma unroll
    for (int offset = GK_WARP_SIZE / 2; offset > 0; offset >>= 1) {
#if defined(GK_USE_HIP)
        x = fmaxf(x, __shfl_xor(x, offset, GK_WARP_SIZE));
#else
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, GK_WARP_SIZE));
#endif
    }
    return x;
}

// Block-wide sum. `scratch` must hold one float per warp in the block.
static __device__ __forceinline__ float gk_cu_block_sum(float x, float * scratch) {
    const int lane = threadIdx.x % GK_WARP_SIZE;
    const int warp = threadIdx.x / GK_WARP_SIZE;
    const int n_warps = (blockDim.x + GK_WARP_SIZE - 1) / GK_WARP_SIZE;

    x = gk_cu_warp_sum(x);

    // An xor reduction leaves every lane holding the warp's answer, so a
    // single-warp block is already done.
    if (n_warps == 1) {
        return x;
    }

    if (lane == 0) {
        scratch[warp] = x;
    }
    __syncthreads();

    float total = 0.0f;
    for (int i = 0; i < n_warps; ++i) {
        total += scratch[i];
    }
    return total;
}

static __device__ __forceinline__ float gk_cu_block_max(float x, float * scratch) {
    const int lane = threadIdx.x % GK_WARP_SIZE;
    const int warp = threadIdx.x / GK_WARP_SIZE;
    const int n_warps = (blockDim.x + GK_WARP_SIZE - 1) / GK_WARP_SIZE;

    x = gk_cu_warp_max(x);

    if (n_warps == 1) {
        return x;
    }

    if (lane == 0) {
        scratch[warp] = x;
    }
    __syncthreads();

    float best = scratch[0];
    for (int i = 1; i < n_warps; ++i) {
        best = fmaxf(best, scratch[i]);
    }
    return best;
}

// The ALiBi slope for head h: the first power-of-two heads step down
// geometrically and any beyond interleave a second sequence. Both branches
// matter - a model with a non-power-of-two head count was trained against
// exactly this assignment.
static __device__ __forceinline__ float gk_cu_alibi_slope(float max_bias, int64_t h,
                                                          int64_t n_head_log2) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float m0 = powf(2.0f, -max_bias / (float) n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float) n_head_log2);

    return h < n_head_log2
        ? powf(m0, (float) (h + 1))
        : powf(m1, (float) (2 * (h - n_head_log2) + 1));
}

// --------------------------------------------------------------------------
// launch geometry
// --------------------------------------------------------------------------

#define GK_CUDA_BLOCK 256

// Grids are capped per dimension; a graph with a very large flat row count
// would exceed the limit, so kernels that use a flat index take a stride loop
// and this only sizes the first pass over it.
static __host__ __forceinline__ int gk_cu_blocks(int64_t n, int block) {
    const int64_t b = (n + block - 1) / block;
    return (int) (b > 65535 ? 65535 : (b < 1 ? 1 : b));
}
