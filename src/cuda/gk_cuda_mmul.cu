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
// What is deliberately absent is a tensor-core path. It would be a large piece
// of vendor-specific code whose correctness is hard to see, and the honest
// order to build in is the one this library has followed everywhere else: the
// clear implementation first, checked against the CPU, and the fast one after,
// checked against the clear one.

#include "gk_cuda_ops.cuh"

#include <float.h>

#define GK_CU_MM_BLOCK 128

// Activation columns one pass over a weight row serves. Each column costs a
// register accumulator and a shared-memory slot in the reduction; four is
// where the reuse stops paying for the occupancy it costs.
#define GK_CU_MM_NC 4

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

template <int NC>
static __global__ void gk_cu_k_mul_mat(gk_tview a, gk_tview b, gk_tview_mut d,
                                       int64_t k_len, int64_t r2, int64_t r3) {
    extern __shared__ float shared[];

    const int64_t i0  = blockIdx.x;                     // weight row
    const int64_t c0  = (int64_t) blockIdx.y * NC;      // first activation column
    const int64_t i23 = blockIdx.z;

    const int64_t i2 = i23 % d.ne[2];
    const int64_t i3 = i23 / d.ne[2];

    const int64_t a2 = i2 / r2;
    const int64_t a3 = i3 / r3;

    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = 0.0f;
    }

    for (int64_t kk = threadIdx.x; kk < k_len; kk += blockDim.x) {
        // the reuse this kernel exists for: one decode of the weight element,
        // NC multiplies against it
        const float av = gk_cu_get(a, kk, i0, a2, a3);

#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int64_t col = c0 + j;
            if (col < d.ne[1]) {
                acc[j] += av * gk_cu_get(b, kk, col, i2, i3);
            }
        }
    }

    gk_cu_reduce_n<NC>(acc, shared);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int64_t col = c0 + j;
            if (col < d.ne[1]) {
                gk_cu_set(d, i0, col, i2, i3, acc[j]);
            }
        }
    }
}

void gk_cuda_mul_mat(gkStream_t stream, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    const int64_t k_len = src0->ne[0];
    const int64_t r2    = src1->ne[2] / src0->ne[2];
    const int64_t r3    = src1->ne[3] / src0->ne[3];

    const int n_warps = GK_CU_MM_BLOCK / GK_WARP_SIZE;

    dim3 grid;
    grid.x = (unsigned) dst->ne[0];
    grid.z = (unsigned) (dst->ne[2] * dst->ne[3]);

    // One column at a time when there is only one - the decode case, every
    // token of generation - and blocks of NC once a batch makes the reuse
    // real. Below NC columns the extra accumulators would sit idle.
    if (dst->ne[1] < GK_CU_MM_NC) {
        grid.y = (unsigned) dst->ne[1];
        gk_cu_k_mul_mat<1><<<grid, GK_CU_MM_BLOCK, n_warps * sizeof(float), stream>>>(
            gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst), k_len, r2, r3);
    } else {
        grid.y = (unsigned) ((dst->ne[1] + GK_CU_MM_NC - 1) / GK_CU_MM_NC);
        gk_cu_k_mul_mat<GK_CU_MM_NC><<<grid, GK_CU_MM_BLOCK,
                                       GK_CU_MM_NC * n_warps * sizeof(float), stream>>>(
            gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst), k_len, r2, r3);
    }
}

// --------------------------------------------------------------------------
// mixture of experts
//
// `ids` holds, per token, which experts it routes to; every (token, slot) pair
// picks one expert's rows out of `as`. One block per output element again, but
// the expert index is read per block rather than per element.
// --------------------------------------------------------------------------

static __global__ void gk_cu_k_mul_mat_id(gk_tview as, gk_tview b, gk_tview ids,
                                          gk_tview_mut d, int64_t k_len) {
    extern __shared__ float shared[];

    const int64_t i0 = blockIdx.x; // row within the expert
    const int64_t is = blockIdx.y; // which of the token's expert slots
    const int64_t it = blockIdx.z; // which token

    const int32_t expert = *(const int32_t *) (ids.data + it * ids.nb[1] + is * ids.nb[0]);

    float acc[1] = { 0.0f };

    for (int64_t kk = threadIdx.x; kk < k_len; kk += blockDim.x) {
        acc[0] += gk_cu_get(as, kk, i0, expert, 0) * gk_cu_get(b, kk, 0, it, 0);
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

    gk_cu_k_mul_mat_id<<<grid, GK_CU_MM_BLOCK, n_warps * sizeof(float), stream>>>(
        gk_cu_view(as), gk_cu_view(src1), gk_cu_view(ids), gk_cu_view_mut(dst), as->ne[0]);
}

// --------------------------------------------------------------------------
// fused attention
//
// One block per query row, one pass over the keys, online softmax: keep the
// running maximum M and normaliser S, and rescale the value accumulator
// whenever the maximum moves. The accumulator is f32 whatever the value type
// is, which is what the CPU pass does and what keeps a graph split across
// devices from drifting.
//
// The query row and the value accumulator live in shared memory, which is what
// bounds the head sizes this kernel accepts; larger ones fall back to the CPU
// rather than being silently truncated.
// --------------------------------------------------------------------------

#define GK_CU_FA_BLOCK   128

static __global__ void gk_cu_k_flash_attn(gk_tview q, gk_tview k, gk_tview v,
                                          gk_tview mask, bool has_mask,
                                          const float * sinks, gk_tview_mut d,
                                          float scale, float max_bias, float logit_softcap,
                                          int64_t n_head_log2,
                                          int64_t rk2, int64_t rk3, int64_t rv2, int64_t rv3) {
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

    for (int64_t ic = 0; ic < k.ne[1]; ++ic) {
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

void gk_cuda_flash_attn(gkStream_t stream, struct gk_tensor * dst) {
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

    gk_cu_k_flash_attn<<<(int) rows, GK_CU_FA_BLOCK, 0, stream>>>(
        gk_cu_view(q), gk_cu_view(k), gk_cu_view(v),
        mask ? gk_cu_view(mask) : gk_cu_view(q), mask != NULL,
        sinks ? (const float *) sinks->data : NULL,
        gk_cu_view_mut(dst), scale, max_bias, logit_softcap, n_head_log2,
        q->ne[2] / k->ne[2], q->ne[3] / k->ne[3],
        q->ne[2] / v->ne[2], q->ne[3] / v->ne[3]);
}
