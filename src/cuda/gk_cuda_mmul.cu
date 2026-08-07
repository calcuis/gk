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

template <int NC, int ATYPE>
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
        const float av = gk_cu_get_t<ATYPE>(a, kk, i0, a2, a3);

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

void gk_cuda_mul_mat(gkStream_t stream, struct gk_tensor * dst) {
    const struct gk_tensor * src0 = dst->src[0];
    const struct gk_tensor * src1 = dst->src[1];

    const int64_t k_len = src0->ne[0];
    const int64_t r2    = src1->ne[2] / src0->ne[2];
    const int64_t r3    = src1->ne[3] / src0->ne[3];

    const int n_warps = GK_CU_MM_BLOCK / GK_WARP_SIZE;

    // Enough columns for a tile to be mostly real work: reuse across the tile
    // beats splitting k across the block, by a wide margin on a batch.
    if (dst->ne[1] >= GK_CU_MM_TILE_MIN_N) {
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

    dim3 grid;
    grid.x = (unsigned) dst->ne[0];
    grid.z = (unsigned) (dst->ne[2] * dst->ne[3]);

    // One column at a time when there is only one - the decode case, every
    // token of generation - and blocks of NC once a batch makes the reuse
    // real. Below NC columns the extra accumulators would sit idle.
    if (dst->ne[1] < GK_CU_MM_NC) {
        grid.y = (unsigned) dst->ne[1];

#define GK_CU_LAUNCH_MV1(T)                                                \
        gk_cu_k_mul_mat<1, T><<<grid, GK_CU_MM_BLOCK,                      \
                                n_warps * sizeof(float), stream>>>(        \
            gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(dst), k_len, r2, r3)

        GK_CU_MM_DISPATCH((int) src0->type, GK_CU_LAUNCH_MV1);
#undef GK_CU_LAUNCH_MV1
    } else {
        grid.y = (unsigned) ((dst->ne[1] + GK_CU_MM_NC - 1) / GK_CU_MM_NC);

#define GK_CU_LAUNCH_MVN(T)                                                \
        gk_cu_k_mul_mat<GK_CU_MM_NC, T><<<grid, GK_CU_MM_BLOCK,            \
                GK_CU_MM_NC * n_warps * sizeof(float), stream>>>(          \
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
    const int64_t DV   = v->ne[0];
    const int64_t n_kv = k->ne[1];

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
