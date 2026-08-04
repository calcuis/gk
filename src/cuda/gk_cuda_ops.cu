// The CUDA/HIP kernels for everything except the matmuls.
//
// These are written to the same rule as the CPU pass they mirror: the CPU
// kernel is the definition of what an op means and this is a second
// implementation of that definition, so where there is a choice, the choice
// that reproduces the CPU's arithmetic wins over the one that runs faster.
// The accumulations that matter - the normalisation statistics, the softmax
// normaliser - are done the same way round and in the same precision, because
// a graph that is scheduled half here and half on the host must not produce
// visibly different numbers depending on where a layer happened to land.
//
// Two shapes cover almost everything:
//
//   elementwise   one thread per destination element, index decomposed from a
//                 flat id. Strides are honoured, so permuted operands work.
//   row-wise      one block per destination row, for the ops with a reduction
//                 along dimension 0 - the norms, the softmaxes, the sums.
//
// The ops that do not fit either - rope's per-position angles, attention's
// online softmax - say so in their own comments.

#include "gk_cuda_ops.cuh"

#include <float.h>

// The op, unary and glu enums are used directly: this file includes gk.h
// through gk_impl.h, and an enumerator is a compile-time constant that device
// code can switch on as happily as host code can. Only the *type* codes are
// mirrored (in gk_cuda_dequant.cuh, which stands alone), and the assertions at
// the bottom of this file hold that mirror against the original.

// --------------------------------------------------------------------------
// activations
//
// Transcribed from gk_compute.c, including the polynomial error function: the
// device has erff(), but it is not the same polynomial, and GELU_ERF differing
// in the last bits between the CPU and the GPU is exactly the kind of thing
// that makes a split graph look subtly broken.
// --------------------------------------------------------------------------

static __device__ __forceinline__ float gk_cu_erf(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    x = fabsf(x);

    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;

    const float t = 1.0f / (1.0f + p * x);
    const float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * expf(-x * x);

    return sign * y;
}

static __device__ __forceinline__ float gk_cu_gelu(float x) {
    const float c = 0.797884560802865f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

static __device__ __forceinline__ float gk_cu_gelu_erf(float x) {
    return 0.5f * x * (1.0f + gk_cu_erf(x * 0.7071067811865475f));
}

static __device__ __forceinline__ float gk_cu_gelu_quick(float x) {
    return x * (1.0f / (1.0f + expf(-1.702f * x)));
}

static __device__ __forceinline__ float gk_cu_silu(float x) {
    return x / (1.0f + expf(-x));
}

static __device__ __forceinline__ float gk_cu_unary(int op, float x,
                                                    float p1, float p2, float p3, float p4) {
    switch (op) {
        case GK_UNARY_OP_ABS:         return fabsf(x);
        case GK_UNARY_OP_SGN:         return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
        case GK_UNARY_OP_NEG:         return -x;
        case GK_UNARY_OP_STEP:        return x > 0.0f ? 1.0f : 0.0f;
        case GK_UNARY_OP_TANH:        return tanhf(x);
        case GK_UNARY_OP_ELU:         return x > 0.0f ? x : expm1f(x);
        case GK_UNARY_OP_RELU:        return x > 0.0f ? x : 0.0f;
        case GK_UNARY_OP_SIGMOID:     return 1.0f / (1.0f + expf(-x));
        case GK_UNARY_OP_GELU:        return gk_cu_gelu(x);
        case GK_UNARY_OP_GELU_QUICK:  return gk_cu_gelu_quick(x);
        case GK_UNARY_OP_GELU_ERF:    return gk_cu_gelu_erf(x);
        case GK_UNARY_OP_SILU:        return gk_cu_silu(x);
        case GK_UNARY_OP_HARDSWISH:   return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
        case GK_UNARY_OP_HARDSIGMOID: return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
        case GK_UNARY_OP_EXP:         return expf(x);
        case GK_UNARY_OP_EXPM1:       return expm1f(x);
        case GK_UNARY_OP_SOFTPLUS:    return x > 20.0f ? x : logf(1.0f + expf(x));
        case GK_UNARY_OP_FLOOR:       return floorf(x);
        case GK_UNARY_OP_CEIL:        return ceilf(x);
        case GK_UNARY_OP_ROUND:       return rintf(x);
        case GK_UNARY_OP_TRUNC:       return truncf(x);
        case GK_UNARY_OP_XIELU: {
            // p1..p4 are alpha_n, alpha_p, beta, eps, already softplus'd where
            // the builder said so
            if (x > 0.0f) {
                return p2 * x * x + p3 * x;
            }
            const float mx = fminf(x, p4);
            return (expm1f(mx) - x) * p1 + p3 * x;
        }
        default:              return x;
    }
}

// --------------------------------------------------------------------------
// elementwise
// --------------------------------------------------------------------------

// The flat destination index, decomposed. Every elementwise kernel below
// starts here, so the decomposition is written once.
struct gk_cu_idx {
    int64_t i0, i1, i2, i3;
};

static __device__ __forceinline__ gk_cu_idx gk_cu_decompose(int64_t k, const int64_t ne[4]) {
    gk_cu_idx x;
    x.i0 = k % ne[0];
    x.i1 = (k / ne[0]) % ne[1];
    x.i2 = (k / (ne[0] * ne[1])) % ne[2];
    x.i3 = k / (ne[0] * ne[1] * ne[2]);
    return x;
}

#define GK_CU_FLAT_LOOP(n) \
    for (int64_t k = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; \
         k < (n); k += (int64_t) gridDim.x * blockDim.x)

// add / sub / mul / div. src1 broadcasts onto src0: every dimension, dimension
// zero included, wraps with a modulo, which is what makes a per-row bias or a
// per-channel scale work without materialising it.
static __global__ void gk_cu_k_binary(gk_tview a, gk_tview b, gk_tview_mut d,
                                      int kind, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        const float va = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        const float vb = gk_cu_get(b, x.i0 % b.ne[0], x.i1 % b.ne[1],
                                      x.i2 % b.ne[2], x.i3 % b.ne[3]);

        float r;
        switch (kind) {
            case 0:  r = va + vb; break;
            case 1:  r = va - vb; break;
            case 2:  r = va * vb; break;
            default: r = va / vb; break;
        }

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, r);
    }
}

static __global__ void gk_cu_k_unary(gk_tview a, gk_tview_mut d, int op,
                                     float p1, float p2, float p3, float p4, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const float v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, gk_cu_unary(op, v, p1, p2, p3, p4));
    }
}

// sqr / sqrt / log / sin / cos, which carry their own op ids rather than
// travelling as unary variants
static __global__ void gk_cu_k_simple(gk_tview a, gk_tview_mut d, int which, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const float v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);

        float r;
        switch (which) {
            case 0:  r = v * v;     break;
            case 1:  r = sqrtf(v);  break;
            case 2:  r = logf(v);   break;
            case 3:  r = sinf(v);   break;
            default: r = cosf(v);   break;
        }

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, r);
    }
}

static __global__ void gk_cu_k_scale(gk_tview a, gk_tview_mut d, float s, float bias, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, gk_cu_get(a, x.i0, x.i1, x.i2, x.i3) * s + bias);
    }
}

static __global__ void gk_cu_k_clamp(gk_tview a, gk_tview_mut d, float lo, float hi, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const float v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, fminf(hi, fmaxf(lo, v)));
    }
}

static __global__ void gk_cu_k_fill(gk_tview_mut d, float c, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, c);
    }
}

static __global__ void gk_cu_k_leaky_relu(gk_tview a, gk_tview_mut d, float slope, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const float v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, v > 0.0f ? v : v * slope);
    }
}

// Gated linear units. With one operand the row splits in half and `swapped`
// says which half is the activated side; with two, src0 is activated and src1
// multiplies. Getting the halves the wrong way round produces plausible and
// wrong output, so the split is spelled out rather than inferred.
static __global__ void gk_cu_k_glu(gk_tview a, gk_tview b, bool has_b, gk_tview_mut d,
                                   int op, bool swapped, float alpha, float limit, int64_t n) {
    const int64_t half = d.ne[0];

    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        float act, mul;
        if (has_b) {
            act = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
            mul = gk_cu_get(b, x.i0, x.i1, x.i2, x.i3);
        } else {
            const int64_t ia = swapped ? x.i0 + half : x.i0;
            const int64_t im = swapped ? x.i0        : x.i0 + half;
            act = gk_cu_get(a, ia, x.i1, x.i2, x.i3);
            mul = gk_cu_get(a, im, x.i1, x.i2, x.i3);
        }

        float r;
        switch (op) {
            case GK_GLU_OP_REGLU:       r = (act > 0.0f ? act : 0.0f) * mul; break;
            case GK_GLU_OP_GEGLU:       r = gk_cu_gelu(act) * mul;           break;
            case GK_GLU_OP_SWIGLU:      r = gk_cu_silu(act) * mul;           break;
            case GK_GLU_OP_GEGLU_ERF:   r = gk_cu_gelu_erf(act) * mul;       break;
            case GK_GLU_OP_GEGLU_QUICK: r = gk_cu_gelu_quick(act) * mul;     break;
            default: {
                // the gpt-oss variant: both operands clamped, a sigmoid with a
                // temperature, and +1 on the gate
                const float xv = fminf(act, limit);
                const float yv = fminf(fmaxf(mul, -limit), limit);
                const float g  = xv / (1.0f + expf(alpha * (-xv)));
                r = g * (yv + 1.0f);
                break;
            }
        }

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, r);
    }
}

// --------------------------------------------------------------------------
// normalisation
//
// One block per row, two passes over it: the statistic, then the scaling.
// The sums are f32 here where the CPU uses f64 for NORM's mean - a block
// reduction over a few thousand elements has a shallow enough tree that the
// difference stays well inside the tolerance the differential tests use, and
// f64 arithmetic on a consumer device costs an order of magnitude.
// --------------------------------------------------------------------------

#define GK_CU_NORM_BLOCK 256

static __global__ void gk_cu_k_norm(gk_tview a, gk_tview_mut d, int kind, float eps) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    const int64_t ir = blockIdx.x;
    int64_t i1, i2, i3;
    gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

    const int64_t n = d.ne[0];

    if (kind == 1) { // NORM: mean first, then the variance about it
        float sum = 0.0f;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            sum += gk_cu_get(a, i, i1, i2, i3);
        }
        const float mean = gk_cu_block_sum(sum, scratch) / (float) n;
        __syncthreads();

        float var = 0.0f;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const float c = gk_cu_get(a, i, i1, i2, i3) - mean;
            var += c * c;
        }
        const float scale = rsqrtf(gk_cu_block_sum(var, scratch) / (float) n + eps);

        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            gk_cu_set(d, i, i1, i2, i3, (gk_cu_get(a, i, i1, i2, i3) - mean) * scale);
        }
        return;
    }

    float sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = gk_cu_get(a, i, i1, i2, i3);
        sumsq += v * v;
    }
    const float total = gk_cu_block_sum(sumsq, scratch);

    // kind 0 is RMS_NORM, kind 2 is L2_NORM: the same sum, different divisor
    const float scale = kind == 0
        ? rsqrtf(total / (float) n + eps)
        : rsqrtf(fmaxf(total, eps));

    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        gk_cu_set(d, i, i1, i2, i3, gk_cu_get(a, i, i1, i2, i3) * scale);
    }
}

// Group norm's statistic spans a group of channels and their whole spatial
// extent, so the unit of work is a group rather than a row.
static __global__ void gk_cu_k_group_norm(gk_tview a, gk_tview_mut d,
                                          int n_groups, float eps) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    const int64_t unit = blockIdx.x;
    const int64_t i3   = unit / n_groups;
    const int64_t g    = unit % n_groups;

    const int64_t per_group = a.ne[2] / n_groups;
    const int64_t c0 = g * per_group;
    const int64_t c1 = c0 + per_group;

    const int64_t n_in_plane = a.ne[0] * a.ne[1];
    const int64_t count = n_in_plane * per_group;

    float sum = 0.0f;
    for (int64_t t = threadIdx.x; t < count; t += blockDim.x) {
        const int64_t i2 = c0 + t / n_in_plane;
        const int64_t r  = t % n_in_plane;
        sum += gk_cu_get(a, r % a.ne[0], r / a.ne[0], i2, i3);
    }
    const float mean = gk_cu_block_sum(sum, scratch) / (float) count;
    __syncthreads();

    float var = 0.0f;
    for (int64_t t = threadIdx.x; t < count; t += blockDim.x) {
        const int64_t i2 = c0 + t / n_in_plane;
        const int64_t r  = t % n_in_plane;
        const float c = gk_cu_get(a, r % a.ne[0], r / a.ne[0], i2, i3) - mean;
        var += c * c;
    }
    const float scale = rsqrtf(gk_cu_block_sum(var, scratch) / (float) count + eps);

    for (int64_t t = threadIdx.x; t < count; t += blockDim.x) {
        const int64_t i2 = c0 + t / n_in_plane;
        const int64_t r  = t % n_in_plane;
        const int64_t i0 = r % a.ne[0];
        const int64_t i1 = r / a.ne[0];
        gk_cu_set(d, i0, i1, i2, i3, (gk_cu_get(a, i0, i1, i2, i3) - mean) * scale);
    }
}

// --------------------------------------------------------------------------
// copies, gathers and scatters
// --------------------------------------------------------------------------

// Same shape: positions line up. Different shape: the copy is defined over the
// flat element order, which is the path a reshape that cannot be a view takes.
static __global__ void gk_cu_k_copy(gk_tview a, gk_tview_mut d, bool same_shape, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        float v;
        if (same_shape) {
            v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        } else {
            const gk_cu_idx s = gk_cu_decompose(k, a.ne);
            v = gk_cu_get(a, s.i0, s.i1, s.i2, s.i3);
        }

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, v);
    }
}

static __global__ void gk_cu_k_get_rows(gk_tview a, gk_tview idx, gk_tview_mut d, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        // the index tensor is [ne1, ne2, ne3] against the destination's
        // (i1,i2,i3), so its dimensions are shifted down by one
        const int64_t r = (int64_t) *(const int32_t *) (idx.data
                + x.i1 * idx.nb[0] + x.i2 * idx.nb[1] + x.i3 * idx.nb[2]);

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, gk_cu_get(a, x.i0, r, x.i2, x.i3));
    }
}

// Writes rows of src0 into the destination at the row indices src1 names -
// how a KV cache is filled. The destination keeps its own type.
static __global__ void gk_cu_k_set_rows(gk_tview b, gk_tview c, gk_tview_mut d,
                                        bool idx_is_64, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, b.ne);

        const char * pi = c.data + x.i1 * c.nb[0]
                        + (x.i2 % c.ne[1]) * c.nb[1] + (x.i3 % c.ne[2]) * c.nb[2];
        const int64_t row = idx_is_64 ? *(const int64_t *) pi : (int64_t) *(const int32_t *) pi;

        gk_cu_set(d, x.i0, row, x.i2, x.i3, gk_cu_get(b, x.i0, x.i1, x.i2, x.i3));
    }
}

static __global__ void gk_cu_k_repeat(gk_tview a, gk_tview_mut d, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                  gk_cu_get(a, x.i0 % a.ne[0], x.i1 % a.ne[1], x.i2 % a.ne[2], x.i3 % a.ne[3]));
    }
}

static __global__ void gk_cu_k_concat(gk_tview a, gk_tview b, gk_tview_mut d,
                                      int dim, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        gk_cu_idx x = gk_cu_decompose(k, d.ne);

        int64_t j0 = x.i0, j1 = x.i1, j2 = x.i2, j3 = x.i3;
        bool second = false;

        switch (dim) {
            case 0: second = x.i0 >= a.ne[0]; if (second) j0 -= a.ne[0]; break;
            case 1: second = x.i1 >= a.ne[1]; if (second) j1 -= a.ne[1]; break;
            case 2: second = x.i2 >= a.ne[2]; if (second) j2 -= a.ne[2]; break;
            default:second = x.i3 >= a.ne[3]; if (second) j3 -= a.ne[3]; break;
        }

        const float v = second ? gk_cu_get(b, j0, j1, j2, j3)
                               : gk_cu_get(a, j0, j1, j2, j3);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, v);
    }
}

// dst[i0,i1,i2] = a[i0,i1,i2] + b[i0, ids[i1,i2]] - the per-row bias shape a
// mixture-of-experts router produces.
static __global__ void gk_cu_k_add_id(gk_tview a, gk_tview b, gk_tview ids,
                                      gk_tview_mut d, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        const int32_t row = *(const int32_t *) (ids.data + x.i1 * ids.nb[0] + x.i2 * ids.nb[1]);

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                  gk_cu_get(a, x.i0, x.i1, x.i2, x.i3) + gk_cu_get(b, x.i0, row, 0, 0));
    }
}

// --------------------------------------------------------------------------
// softmax and masking
// --------------------------------------------------------------------------

#define GK_CU_SOFTMAX_BLOCK 256

static __global__ void gk_cu_k_soft_max(gk_tview a, gk_tview mask, bool has_mask,
                                        const float * sinks, gk_tview_mut d,
                                        float scale, float max_bias, int64_t n_head_log2) {
    __shared__ float scratch[GK_CU_SOFTMAX_BLOCK / GK_WARP_SIZE];

    const int64_t ir = blockIdx.x;
    int64_t i1, i2, i3;
    gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

    const int64_t n = d.ne[0];
    const float slope = gk_cu_alibi_slope(max_bias, i2, n_head_log2);

    float local_max = -INFINITY;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = gk_cu_get(a, i, i1, i2, i3) * scale;
        if (has_mask) {
            v += slope * gk_cu_get(mask, i, i1, i2 % mask.ne[2], i3 % mask.ne[3]);
        }
        local_max = fmaxf(local_max, v);
    }

    float row_max = gk_cu_block_max(local_max, scratch);
    __syncthreads();

    // a sink is one extra virtual logit per head, normalised over but never
    // written out
    const float sink = sinks != NULL ? sinks[i2] : -INFINITY;
    if (sinks != NULL) {
        row_max = fmaxf(row_max, sink);
    }

    // A fully masked row is all -inf; exponentiating that is 0/0. Define it as
    // a uniform zero row, which is what a padded position needs.
    if (!isfinite(row_max)) {
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            gk_cu_set(d, i, i1, i2, i3, 0.0f);
        }
        return;
    }

    float local_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = gk_cu_get(a, i, i1, i2, i3) * scale;
        if (has_mask) {
            v += slope * gk_cu_get(mask, i, i1, i2 % mask.ne[2], i3 % mask.ne[3]);
        }
        local_sum += expf(v - row_max);
    }

    float sum = gk_cu_block_sum(local_sum, scratch);
    if (sinks != NULL) {
        sum += expf(sink - row_max);
    }

    const float inv = 1.0f / sum;

    // The exponentials are recomputed rather than stored and read back. The
    // destination may be f16, and rounding through it before the normalisation
    // would put the answer a visible distance from the CPU's, which normalises
    // in f32 and converts once at the end.
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = gk_cu_get(a, i, i1, i2, i3) * scale;
        if (has_mask) {
            v += slope * gk_cu_get(mask, i, i1, i2 % mask.ne[2], i3 % mask.ne[3]);
        }
        gk_cu_set(d, i, i1, i2, i3, expf(v - row_max) * inv);
    }
}

static __global__ void gk_cu_k_diag_mask(gk_tview a, gk_tview_mut d,
                                         int n_past, float fill, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);
        const float v = gk_cu_get(a, x.i0, x.i1, x.i2, x.i3);
        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3, x.i0 > n_past + x.i1 ? fill : v);
    }
}

// --------------------------------------------------------------------------
// rotary position embedding
//
// One thread per rotated pair. The angle depends only on the position and the
// pair index, so it is recomputed per thread rather than cached per token: the
// two transcendentals are cheaper than the shared memory and the barrier a
// cache would cost, and it keeps every pair independent.
// --------------------------------------------------------------------------

struct gk_cu_rope_params {
    int   n_dims;
    int   mode;
    int   sections[4];
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float corr_dims[2];
    float theta_scale;
    bool  neox;
    bool  mrope;
    bool  vision;
    bool  imrope;
};

static __device__ __forceinline__ float gk_cu_rope_ramp(float low, float high, int i) {
    const float y = ((float) (i / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static __device__ __forceinline__ void gk_cu_rope_angle(float theta_base, const gk_cu_rope_params & p,
                                                        int i, float * cos_out, float * sin_out) {
    float theta  = theta_base * p.freq_scale;
    float mscale = p.attn_factor;

    if (p.ext_factor != 0.0f) {
        const float ramp = gk_cu_rope_ramp(p.corr_dims[0], p.corr_dims[1], i) * p.ext_factor;

        // blend the scaled and unscaled angles across the correction band
        theta = theta * (1.0f - ramp) + theta_base * ramp;

        // YaRN's temperature correction for the entropy a stretched context adds
        mscale *= 1.0f + 0.1f * logf(1.0f / p.freq_scale);
    }

    *cos_out = cosf(theta) * mscale;
    *sin_out = sinf(theta) * mscale;
}

static __global__ void gk_cu_k_rope(gk_tview a, const int32_t * pos, const float * freq_factors,
                                    gk_tview_mut d, gk_cu_rope_params p, int64_t n_pairs) {
    const int64_t n_rows = d.ne[1] * d.ne[2] * d.ne[3];
    const int64_t n_rot  = p.vision ? d.ne[0] : p.n_dims;
    const int64_t pairs_per_row = n_rot / 2;

    GK_CU_FLAT_LOOP(n_pairs) {
        const int64_t ir = k / pairs_per_row;
        const int64_t ip = k % pairs_per_row;
        if (ir >= n_rows) {
            continue;
        }

        int64_t i1, i2, i3;
        gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

        const int64_t i0 = ip * 2;

        // Which angle this pair turns through. A single-axis rope advances one
        // angle geometrically; the multi-axis ropes carry four positions and
        // pick per pair by which section the pair falls in.
        float theta_base;
        if (!p.mrope) {
            theta_base = (float) pos[i2] * powf(p.theta_scale, (float) ip);
        } else {
            const int64_t n_pos = d.ne[2];
            const int sect_dims = p.sections[0] + p.sections[1] + p.sections[2] + p.sections[3];
            const int sec_w = p.sections[1] + p.sections[0];
            const int sec_e = p.sections[2] + sec_w;

            const int sector = (int) (ip % sect_dims);

            int axis;
            if (p.imrope) {
                if (sector % 3 == 1 && sector < 3 * p.sections[1])      axis = 1;
                else if (sector % 3 == 2 && sector < 3 * p.sections[2]) axis = 2;
                else if (sector % 3 == 0 && sector < 3 * p.sections[0]) axis = 0;
                else                                                    axis = 3;
            } else {
                if (sector < p.sections[0])                       axis = 0;
                else if (sector < sec_w)                          axis = 1;
                else if (sector < sec_w + p.sections[2])          axis = 2;
                else                                              axis = 3;
            }

            const float p_axis = (float) pos[i2 + n_pos * axis];

            // The vision rope restarts each axis's angle at its section, so a
            // pair's exponent counts from the start of its section rather than
            // from the start of the row.
            int step = (int) ip;
            if (p.vision) {
                const int base = axis == 0 ? 0
                               : axis == 1 ? p.sections[0]
                               : axis == 2 ? sec_w : sec_e;
                step = (int) ip - base;
            }

            theta_base = p_axis * powf(p.theta_scale, (float) step);
        }

        const float ff = freq_factors != NULL ? freq_factors[ip] : 1.0f;

        float cos_t, sin_t;
        gk_cu_rope_angle(theta_base / ff, p, (int) i0, &cos_t, &sin_t);

        // three pairings share the rotation: normal pairs (2i, 2i+1), neox and
        // the multi-axis ropes pair (i, i + n_dims/2), and the vision rope
        // pairs (i, i + n_dims) across the full width
        const int64_t offset = p.vision ? p.n_dims
                             : (p.neox || p.mrope) ? p.n_dims / 2 : 1;
        const int64_t ic = (p.neox || p.mrope) ? ip : i0;

        const float x0 = gk_cu_get(a, ic, i1, i2, i3);
        const float x1 = gk_cu_get(a, ic + offset, i1, i2, i3);

        gk_cu_set(d, ic,          i1, i2, i3, x0 * cos_t - x1 * sin_t);
        gk_cu_set(d, ic + offset, i1, i2, i3, x0 * sin_t + x1 * cos_t);
    }
}

// Channels at or past n_dims are not rotated and pass through. A separate
// kernel rather than a branch in the one above, so the rotation kernel stays
// one thread per pair with no idle lanes.
static __global__ void gk_cu_k_rope_passthrough(gk_tview a, gk_tview_mut d,
                                                int n_dims, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const int64_t width = d.ne[0] - n_dims;
        const int64_t ir = k / width;
        const int64_t i0 = n_dims + k % width;

        int64_t i1, i2, i3;
        gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

        gk_cu_set(d, i0, i1, i2, i3, gk_cu_get(a, i0, i1, i2, i3));
    }
}

// --------------------------------------------------------------------------
// reductions along a row
// --------------------------------------------------------------------------

static __global__ void gk_cu_k_sum_rows(gk_tview a, gk_tview_mut d, bool mean) {
    __shared__ float scratch[GK_CU_NORM_BLOCK / GK_WARP_SIZE];

    const int64_t ir = blockIdx.x;
    int64_t i1, i2, i3;
    gk_cu_unrow(ir, d.ne, &i1, &i2, &i3);

    float local = 0.0f;
    for (int64_t i = threadIdx.x; i < a.ne[0]; i += blockDim.x) {
        local += gk_cu_get(a, i, i1, i2, i3);
    }

    const float total = gk_cu_block_sum(local, scratch);

    if (threadIdx.x == 0) {
        gk_cu_set(d, 0, i1, i2, i3, mean ? total / (float) a.ne[0] : total);
    }
}

// --------------------------------------------------------------------------
// padding, resampling and the diffusion helpers
// --------------------------------------------------------------------------

static __device__ __forceinline__ int64_t gk_cu_wrap(int64_t i, int64_t n) {
    return ((i % n) + n) % n;
}

static __global__ void gk_cu_k_pad(gk_tview a, gk_tview_mut d,
                                   int lp0, int rp0, int lp1, int rp1,
                                   int lp2, int rp2, int lp3, int rp3,
                                   bool circular, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        if (circular) {
            gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                      gk_cu_get(a, gk_cu_wrap(x.i0 - lp0, a.ne[0]),
                                   gk_cu_wrap(x.i1 - lp1, a.ne[1]),
                                   gk_cu_wrap(x.i2 - lp2, a.ne[2]),
                                   gk_cu_wrap(x.i3 - lp3, a.ne[3])));
            continue;
        }

        const bool inside =
            x.i0 >= lp0 && x.i0 < d.ne[0] - rp0 &&
            x.i1 >= lp1 && x.i1 < d.ne[1] - rp1 &&
            x.i2 >= lp2 && x.i2 < d.ne[2] - rp2 &&
            x.i3 >= lp3 && x.i3 < d.ne[3] - rp3;

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                  inside ? gk_cu_get(a, x.i0 - lp0, x.i1 - lp1, x.i2 - lp2, x.i3 - lp3) : 0.0f);
    }
}

// Nearest and bilinear resampling. The index arithmetic - the truncation, the
// half-pixel offset, the border clamp - is part of the op's meaning: a
// projector was trained against exactly these positions.
static __global__ void gk_cu_k_upscale(gk_tview a, gk_tview_mut d, int mode,
                                       float sf0, float sf1, float sf2, float sf3,
                                       float pixel_offset, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        const gk_cu_idx x = gk_cu_decompose(k, d.ne);

        const int64_t i02 = (int64_t) ((float) x.i2 / sf2);
        const int64_t i03 = (int64_t) ((float) x.i3 / sf3);

        if (mode == 0) { // nearest
            gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                      gk_cu_get(a, (int64_t) ((float) x.i0 / sf0),
                                   (int64_t) ((float) x.i1 / sf1), i02, i03));
            continue;
        }

        const float fy = ((float) x.i1 + pixel_offset) / sf1 - pixel_offset;
        int64_t y0 = (int64_t) floorf(fy);
        int64_t y1 = y0 + 1;
        float dy = fmaxf(0.0f, fminf(fy - (float) y0, 1.0f));
        y0 = max((int64_t) 0, min(y0, a.ne[1] - 1));
        y1 = max((int64_t) 0, min(y1, a.ne[1] - 1));

        const float fx = ((float) x.i0 + pixel_offset) / sf0 - pixel_offset;
        int64_t x0 = (int64_t) floorf(fx);
        int64_t x1 = x0 + 1;
        float dx = fmaxf(0.0f, fminf(fx - (float) x0, 1.0f));
        x0 = max((int64_t) 0, min(x0, a.ne[0] - 1));
        x1 = max((int64_t) 0, min(x1, a.ne[0] - 1));

        const float p00 = gk_cu_get(a, x0, y0, i02, i03);
        const float p10 = gk_cu_get(a, x1, y0, i02, i03);
        const float p01 = gk_cu_get(a, x0, y1, i02, i03);
        const float p11 = gk_cu_get(a, x1, y1, i02, i03);

        gk_cu_set(d, x.i0, x.i1, x.i2, x.i3,
                  p00 * (1 - dx) * (1 - dy) + p10 * dx * (1 - dy)
                + p01 * (1 - dx) * dy       + p11 * dx * dy);
    }
}

static __global__ void gk_cu_k_timestep_embedding(const float * timesteps, gk_tview_mut d,
                                                  int dim, int max_period, int64_t n) {
    const int half = dim / 2;

    GK_CU_FLAT_LOOP(n) {
        const int64_t i = k / half; // which timestep
        const int64_t j = k % half;

        const float t    = timesteps[i];
        const float freq = expf(-logf((float) max_period) * (float) j / (float) half);
        const float arg  = t * freq;

        gk_cu_set(d, j,        i, 0, 0, cosf(arg));
        gk_cu_set(d, j + half, i, 0, 0, sinf(arg));
    }
}

static __global__ void gk_cu_k_arange(float * dst, float start, float step, int64_t n) {
    GK_CU_FLAT_LOOP(n) {
        dst[k] = start + step * (float) k;
    }
}

// im2col: unroll each input patch into a row so a convolution becomes a
// matmul. One thread per output cell.
static __global__ void gk_cu_k_im2col(gk_tview img, gk_tview_mut d,
                                      int64_t IC, int64_t IH, int64_t IW,
                                      int64_t KH, int64_t KW, int64_t OH, int64_t OW,
                                      int64_t ofs_n, int64_t ofs_c, int64_t ofs_h,
                                      int s0, int s1, int p0, int p1, int d0, int d1,
                                      int64_t n) {
    const int64_t patch = IC * KH * KW;

    GK_CU_FLAT_LOOP(n) {
        const int64_t at   = k % patch;
        const int64_t cell = k / patch;

        const int64_t iow = cell % OW;
        const int64_t ioh = (cell / OW) % OH;
        const int64_t in  = cell / (OW * OH);

        const int64_t ikw = at % KW;
        const int64_t ikh = (at / KW) % KH;
        const int64_t iic = at / (KW * KH);

        const int64_t iiw = iow * s0 + ikw * d0 - p0;
        const int64_t iih = ioh * s1 + ikh * d1 - p1;

        float v = 0.0f;
        if (iih >= 0 && iih < IH && iiw >= 0 && iiw < IW) {
            v = *(const float *) (img.data + in * ofs_n + iic * ofs_c
                                  + iih * ofs_h + iiw * img.nb[0]);
        }

        // the destination is contiguous by the builder's contract, so one flat
        // index addresses it
        if (d.type == GKT_F16) {
            ((__half *) d.data)[k] = __float2half(v);
        } else {
            ((float *) d.data)[k] = v;
        }
    }
}

// --------------------------------------------------------------------------
// dispatch
// --------------------------------------------------------------------------

// The type codes in gk_cuda_dequant.cuh are a copy, made because that header
// has to stand on its own in device code. A copy that drifts would decode
// weights as the wrong format and produce noise, so it is held against the
// original here, where both are in scope.
static_assert((int) GK_TYPE_F32  == GKT_F32,  "type enum drifted: F32");
static_assert((int) GK_TYPE_F16  == GKT_F16,  "type enum drifted: F16");
static_assert((int) GK_TYPE_BF16 == GKT_BF16, "type enum drifted: BF16");
static_assert((int) GK_TYPE_Q4_0 == GKT_Q4_0, "type enum drifted: Q4_0");
static_assert((int) GK_TYPE_Q4_1 == GKT_Q4_1, "type enum drifted: Q4_1");
static_assert((int) GK_TYPE_Q5_0 == GKT_Q5_0, "type enum drifted: Q5_0");
static_assert((int) GK_TYPE_Q5_1 == GKT_Q5_1, "type enum drifted: Q5_1");
static_assert((int) GK_TYPE_Q8_0 == GKT_Q8_0, "type enum drifted: Q8_0");
static_assert((int) GK_TYPE_Q2_K == GKT_Q2_K, "type enum drifted: Q2_K");
static_assert((int) GK_TYPE_Q3_K == GKT_Q3_K, "type enum drifted: Q3_K");
static_assert((int) GK_TYPE_Q4_K == GKT_Q4_K, "type enum drifted: Q4_K");
static_assert((int) GK_TYPE_Q5_K == GKT_Q5_K, "type enum drifted: Q5_K");
static_assert((int) GK_TYPE_Q6_K == GKT_Q6_K, "type enum drifted: Q6_K");
static_assert((int) GK_TYPE_IQ4_NL == GKT_IQ4_NL, "type enum drifted: IQ4_NL");
static_assert((int) GK_TYPE_IQ4_XS == GKT_IQ4_XS, "type enum drifted: IQ4_XS");
static_assert((int) GK_TYPE_TQ1_0  == GKT_TQ1_0,  "type enum drifted: TQ1_0");
static_assert((int) GK_TYPE_TQ2_0  == GKT_TQ2_0,  "type enum drifted: TQ2_0");
static_assert((int) GK_TYPE_MXFP4  == GKT_MXFP4,  "type enum drifted: MXFP4");
static_assert((int) GK_TYPE_NVFP4  == GKT_NVFP4,  "type enum drifted: NVFP4");
static_assert((int) GK_TYPE_Q1_0   == GKT_Q1_0,   "type enum drifted: Q1_0");
static_assert((int) GK_TYPE_Q2_0   == GKT_Q2_0,   "type enum drifted: Q2_0");
static_assert((int) GK_TYPE_I32    == GKT_I32,    "type enum drifted: I32");
static_assert((int) GK_TYPE_I64    == GKT_I64,    "type enum drifted: I64");

// The float types a generic kernel can read and write. Quantized destinations
// are not among them: writing one means encoding, and encoding needs the whole
// block, which is a different kernel shape than any of these.
static bool gk_cu_is_float_type(int type) {
    return type == GKT_F32 || type == GKT_F16 || type == GKT_BF16;
}

static bool gk_cu_readable(const struct gk_tensor * t) {
    if (t == NULL) {
        return true;
    }
    const int type = (int) t->type;
    if (gk_cu_is_float_type(type) || type == GKT_I32 || type == GKT_I64) {
        return true;
    }
    // a quantized operand is only readable where it is packed, which is what
    // gk_cu_row_elem assumes
    return gk_cu_type_supported(type) && t->nb[0] == (size_t) gk_cu_type_size(type);
}

bool gk_cuda_supports_op(const struct gk_tensor * op) {
    const struct gk_tensor * s0 = op->src[0];
    const struct gk_tensor * s1 = op->src[1];

    switch ((int) op->op) {
        case GK_OP_NONE: case GK_OP_RESHAPE: case GK_OP_VIEW:
        case GK_OP_PERMUTE: case GK_OP_TRANSPOSE:
            return true;

        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID:
            // the weight may be quantized, the activations may not, and the
            // result is always f32
            return op->type == GKT_F32 && gk_cu_type_supported((int) s0->type) &&
                   gk_cu_readable(s0) && gk_cu_is_float_type((int) s1->type);

        case GK_OP_ADD: case GK_OP_SUB: case GK_OP_MUL: case GK_OP_DIV:
        case GK_OP_ADD_ID:
        case GK_OP_SQR: case GK_OP_SQRT: case GK_OP_LOG: case GK_OP_SIN: case GK_OP_COS:
        case GK_OP_UNARY: case GK_OP_GLU: case GK_OP_LEAKY_RELU:
        case GK_OP_SCALE: case GK_OP_CLAMP: case GK_OP_FILL:
        case GK_OP_NORM: case GK_OP_RMS_NORM: case GK_OP_L2_NORM: case GK_OP_GROUP_NORM:
        case GK_OP_DUP: case GK_OP_CPY: case GK_OP_CONT:
        case GK_OP_GET_ROWS: case GK_OP_REPEAT: case GK_OP_CONCAT:
        case GK_OP_SOFT_MAX: case GK_OP_DIAG_MASK_INF: case GK_OP_DIAG_MASK_ZERO:
        case GK_OP_ROPE: case GK_OP_SUM_ROWS: case GK_OP_MEAN:
        case GK_OP_PAD: case GK_OP_TIMESTEP_EMBEDDING: case GK_OP_ARANGE:
        case GK_OP_FLASH_ATTN_EXT: case GK_OP_IM2COL: case GK_OP_UPSCALE:
        case GK_OP_SET_ROWS:
            break;

        default:
            return false;
    }

    // Destination types: everything above writes through gk_cu_set, except
    // set_rows (whose destination is the cache's own type) and im2col.
    if ((int) op->op == GK_OP_SET_ROWS) {
        return gk_cu_is_float_type((int) op->type) &&
               s0 != NULL && gk_cu_is_float_type((int) s0->type);
    }
    if ((int) op->op == GK_OP_IM2COL) {
        return (op->type == GKT_F32 || op->type == GKT_F16) &&
               s1 != NULL && s1->type == GKT_F32;
    }
    if ((int) op->op == GK_OP_FLASH_ATTN_EXT) {
        // K and V may be quantized; Q and the result are f32. The head widths
        // are bounded because the query row and the value accumulator live in
        // shared memory - a wider head falls back to the CPU rather than
        // being quietly truncated.
        return op->type == GKT_F32 && s0->type == GKT_F32 &&
               gk_cu_type_supported((int) op->src[1]->type) &&
               gk_cu_type_supported((int) op->src[2]->type) &&
               op->src[1]->ne[0] <= GK_CUDA_FA_MAX_DK &&
               op->src[2]->ne[0] <= GK_CUDA_FA_MAX_DV;
    }
    if ((int) op->op == GK_OP_UPSCALE) {
        // only nearest and plain bilinear; the antialiased and bicubic filters
        // stay on the CPU rather than be approximated here
        const int mode_flags = gk_get_op_params_i32(op, 0);
        const int mode = mode_flags & 0xFF;
        if (mode > 1 || (mode_flags & GK_SCALE_FLAG_ANTIALIAS)) {
            return false;
        }
    }

    if (!gk_cu_is_float_type((int) op->type) && (int) op->op != GK_OP_ARANGE) {
        return false;
    }

    for (int i = 0; i < GK_MAX_SRC; ++i) {
        if (!gk_cu_readable(op->src[i])) {
            return false;
        }
    }

    return true;
}

// --------------------------------------------------------------------------

bool gk_cuda_compute_op(gkStream_t stream, struct gk_tensor * node) {
    const int op = (int) node->op;

    struct gk_tensor * src0 = node->src[0];
    struct gk_tensor * src1 = node->src[1];
    struct gk_tensor * src2 = node->src[2];

    const int64_t ne = gk_cu_nelements(node);
    const int     nb = gk_cu_blocks(ne, GK_CUDA_BLOCK);

    switch (op) {
        case GK_OP_NONE: case GK_OP_RESHAPE: case GK_OP_VIEW:
        case GK_OP_PERMUTE: case GK_OP_TRANSPOSE:
            return true; // pure reinterpretations; the memory is already right

        case GK_OP_ADD: case GK_OP_SUB: case GK_OP_MUL: case GK_OP_DIV: {
            const int kind = op == GK_OP_ADD ? 0 : op == GK_OP_SUB ? 1 : op == GK_OP_MUL ? 2 : 3;
            gk_cu_k_binary<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node), kind, ne);
            return true;
        }

        case GK_OP_SQR: case GK_OP_SQRT: case GK_OP_LOG: case GK_OP_SIN: case GK_OP_COS: {
            const int which = op == GK_OP_SQR ? 0 : op == GK_OP_SQRT ? 1 :
                              op == GK_OP_LOG ? 2 : op == GK_OP_SIN ? 3 : 4;
            gk_cu_k_simple<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), which, ne);
            return true;
        }

        case GK_OP_UNARY: {
            const int uop = (int) gk_get_unary_op(node);
            gk_cu_k_unary<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), uop,
                gk_get_op_params_f32(node, 1), gk_get_op_params_f32(node, 2),
                gk_get_op_params_f32(node, 3), gk_get_op_params_f32(node, 4), ne);
            return true;
        }

        case GK_OP_LEAKY_RELU:
            gk_cu_k_leaky_relu<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), gk_get_op_params_f32(node, 0), ne);
            return true;

        case GK_OP_GLU: {
            const int gop = (int) gk_get_glu_op(node);
            const bool swapped = gk_get_op_params_i32(node, 1) != 0;
            gk_cu_k_glu<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), src1 ? gk_cu_view(src1) : gk_cu_view(src0), src1 != NULL,
                gk_cu_view_mut(node), gop, swapped,
                gk_get_op_params_f32(node, 2), gk_get_op_params_f32(node, 3), ne);
            return true;
        }

        case GK_OP_SCALE:
            gk_cu_k_scale<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_f32(node, 0), gk_get_op_params_f32(node, 1), ne);
            return true;

        case GK_OP_CLAMP:
            gk_cu_k_clamp<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_f32(node, 0), gk_get_op_params_f32(node, 1), ne);
            return true;

        case GK_OP_FILL:
            gk_cu_k_fill<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view_mut(node), gk_get_op_params_f32(node, 0), ne);
            return true;

        case GK_OP_NORM: case GK_OP_RMS_NORM: case GK_OP_L2_NORM: {
            const int kind = op == GK_OP_RMS_NORM ? 0 : op == GK_OP_NORM ? 1 : 2;
            const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3];
            gk_cu_k_norm<<<(int) rows, GK_CU_NORM_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), kind, gk_get_op_params_f32(node, 0));
            return true;
        }

        case GK_OP_GROUP_NORM: {
            const int n_groups = gk_get_op_params_i32(node, 0);
            gk_cu_k_group_norm<<<(int) (node->ne[3] * n_groups), GK_CU_NORM_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), n_groups, gk_get_op_params_f32(node, 1));
            return true;
        }

        case GK_OP_DUP: case GK_OP_CPY: case GK_OP_CONT:
            gk_cu_k_copy<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_are_same_shape(src0, node), ne);
            return true;

        case GK_OP_GET_ROWS:
            gk_cu_k_get_rows<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node), ne);
            return true;

        case GK_OP_SET_ROWS: {
            const int64_t n = gk_cu_nelements(src0);
            gk_cu_k_set_rows<<<gk_cu_blocks(n, GK_CUDA_BLOCK), GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node),
                src1->type == GKT_I64, n);
            return true;
        }

        case GK_OP_REPEAT:
            gk_cu_k_repeat<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), ne);
            return true;

        case GK_OP_CONCAT:
            gk_cu_k_concat<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0), ne);
            return true;

        case GK_OP_ADD_ID:
            gk_cu_k_add_id<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view(src1), gk_cu_view(src2),
                gk_cu_view_mut(node), ne);
            return true;

        case GK_OP_SOFT_MAX: {
            const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3];

            int64_t n_head_log2 = 1;
            while (n_head_log2 * 2 <= src0->ne[2]) {
                n_head_log2 *= 2;
            }

            gk_cu_k_soft_max<<<(int) rows, GK_CU_SOFTMAX_BLOCK, 0, stream>>>(
                gk_cu_view(src0),
                src1 ? gk_cu_view(src1) : gk_cu_view(src0), src1 != NULL,
                src2 ? (const float *) src2->data : NULL,
                gk_cu_view_mut(node),
                gk_get_op_params_f32(node, 0), gk_get_op_params_f32(node, 1), n_head_log2);
            return true;
        }

        case GK_OP_DIAG_MASK_INF: case GK_OP_DIAG_MASK_ZERO:
            gk_cu_k_diag_mask<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0),
                op == GK_OP_DIAG_MASK_INF ? -INFINITY : 0.0f, ne);
            return true;

        case GK_OP_ROPE: {
            gk_cu_rope_params p;
            p.n_dims      = gk_get_op_params_i32(node, 1);
            p.mode        = gk_get_op_params_i32(node, 2);
            p.freq_base   = gk_get_op_params_f32(node, 5);
            p.freq_scale  = gk_get_op_params_f32(node, 6);
            p.ext_factor  = gk_get_op_params_f32(node, 7);
            p.attn_factor = gk_get_op_params_f32(node, 8);

            for (int i = 0; i < 4; ++i) {
                p.sections[i] = gk_get_op_params_i32(node, 11 + i);
            }

            gk_rope_corr_dims(p.n_dims, gk_get_op_params_i32(node, 4), p.freq_base,
                              gk_get_op_params_f32(node, 9), gk_get_op_params_f32(node, 10),
                              p.corr_dims);

            p.theta_scale = powf(p.freq_base, -2.0f / (float) p.n_dims);
            p.neox   = (p.mode & GK_ROPE_TYPE_NEOX) != 0;
            p.mrope  = (p.mode & GK_ROPE_TYPE_MROPE) != 0;
            p.vision = p.mode == GK_ROPE_TYPE_VISION;
            p.imrope = p.mode == GK_ROPE_TYPE_IMROPE;

            const int64_t rows  = node->ne[1] * node->ne[2] * node->ne[3];
            const int64_t n_rot = p.vision ? node->ne[0] : p.n_dims;
            const int64_t pairs = rows * (n_rot / 2);

            gk_cu_k_rope<<<gk_cu_blocks(pairs, GK_CUDA_BLOCK), GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), (const int32_t *) src1->data,
                src2 ? (const float *) src2->data : NULL,
                gk_cu_view_mut(node), p, pairs);

            if (!p.vision && node->ne[0] > p.n_dims) {
                const int64_t rest = rows * (node->ne[0] - p.n_dims);
                gk_cu_k_rope_passthrough<<<gk_cu_blocks(rest, GK_CUDA_BLOCK),
                                           GK_CUDA_BLOCK, 0, stream>>>(
                    gk_cu_view(src0), gk_cu_view_mut(node), p.n_dims, rest);
            }
            return true;
        }

        case GK_OP_SUM_ROWS: case GK_OP_MEAN: {
            const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3];
            gk_cu_k_sum_rows<<<(int) rows, GK_CU_NORM_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), op == GK_OP_MEAN);
            return true;
        }

        case GK_OP_PAD:
            gk_cu_k_pad<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node),
                gk_get_op_params_i32(node, 0), gk_get_op_params_i32(node, 1),
                gk_get_op_params_i32(node, 2), gk_get_op_params_i32(node, 3),
                gk_get_op_params_i32(node, 4), gk_get_op_params_i32(node, 5),
                gk_get_op_params_i32(node, 6), gk_get_op_params_i32(node, 7),
                gk_get_op_params_i32(node, 8) != 0, ne);
            return true;

        case GK_OP_UPSCALE: {
            const int mode_flags = gk_get_op_params_i32(node, 0);
            const int mode = mode_flags & 0xFF;

            float sf0 = (float) node->ne[0] / src0->ne[0];
            float sf1 = (float) node->ne[1] / src0->ne[1];
            const float sf2 = (float) node->ne[2] / src0->ne[2];
            const float sf3 = (float) node->ne[3] / src0->ne[3];
            float pixel_offset = 0.5f;

            if (mode_flags & GK_SCALE_FLAG_ALIGN_CORNERS) {
                pixel_offset = 0.0f;
                if (node->ne[0] > 1 && src0->ne[0] > 1) {
                    sf0 = (float) (node->ne[0] - 1) / (src0->ne[0] - 1);
                }
                if (node->ne[1] > 1 && src0->ne[1] > 1) {
                    sf1 = (float) (node->ne[1] - 1) / (src0->ne[1] - 1);
                }
            }

            gk_cu_k_upscale<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src0), gk_cu_view_mut(node), mode,
                sf0, sf1, sf2, sf3, pixel_offset, ne);
            return true;
        }

        case GK_OP_TIMESTEP_EMBEDDING: {
            const int dim = gk_get_op_params_i32(node, 0);
            const int64_t n = node->ne[1] * (dim / 2);
            gk_cu_k_timestep_embedding<<<gk_cu_blocks(n, GK_CUDA_BLOCK),
                                         GK_CUDA_BLOCK, 0, stream>>>(
                (const float *) src0->data, gk_cu_view_mut(node),
                dim, gk_get_op_params_i32(node, 1), n);
            return true;
        }

        case GK_OP_ARANGE:
            gk_cu_k_arange<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                (float *) node->data, gk_get_op_params_f32(node, 0),
                gk_get_op_params_f32(node, 2), ne);
            return true;

        case GK_OP_IM2COL: {
            const int s0 = gk_get_op_params_i32(node, 0);
            const int s1 = gk_get_op_params_i32(node, 1);
            const int p0 = gk_get_op_params_i32(node, 2);
            const int p1 = gk_get_op_params_i32(node, 3);
            const int d0 = gk_get_op_params_i32(node, 4);
            const int d1 = gk_get_op_params_i32(node, 5);
            const bool is_2D = gk_get_op_params_i32(node, 6) != 0;

            const int64_t IC = is_2D ? src1->ne[2] : src1->ne[1];
            const int64_t IH = is_2D ? src1->ne[1] : 1;
            const int64_t IW = src1->ne[0];
            const int64_t KH = is_2D ? src0->ne[1] : 1;
            const int64_t KW = src0->ne[0];
            const int64_t OH = is_2D ? node->ne[2] : 1;
            const int64_t OW = node->ne[1];

            const int64_t ofs_n = (int64_t) (is_2D ? src1->nb[3] : src1->nb[2]);
            const int64_t ofs_c = (int64_t) (is_2D ? src1->nb[2] : src1->nb[1]);
            const int64_t ofs_h = (int64_t) (is_2D ? src1->nb[1] : 0);

            gk_cu_k_im2col<<<nb, GK_CUDA_BLOCK, 0, stream>>>(
                gk_cu_view(src1), gk_cu_view_mut(node),
                IC, IH, IW, KH, KW, OH, OW, ofs_n, ofs_c, ofs_h,
                s0, s1, p0, p1, d0, d1, ne);
            return true;
        }

        case GK_OP_MUL_MAT:
            gk_cuda_mul_mat(stream, node);
            return true;

        case GK_OP_MUL_MAT_ID:
            gk_cuda_mul_mat_id(stream, node);
            return true;

        case GK_OP_FLASH_ATTN_EXT:
            gk_cuda_flash_attn(stream, node);
            return true;

        default:
            return false;
    }
}
