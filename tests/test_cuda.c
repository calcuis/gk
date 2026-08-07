// CUDA/ROCm backend smoke and numerical tests.
//
// The ordinary backend tests intentionally run without a GPU. This one is
// compiled only when the CUDA-family backend is enabled and exercises the
// complete device path: discovery, device allocation, host/device transfers,
// graph execution, synchronization, and result transfer back to the host.
//
// Quantized weights matter here. A float-only smoke test cannot catch a
// device decoder drifting from the on-disk GGUF block layout, which turns an
// otherwise healthy CUDA build into plausible-looking but incorrect output.
// Decoding is therefore checked bit-for-bit. Matmul is checked separately
// with a small numerical tolerance: like ggml CUDA, the device path and the
// CPU SIMD path reduce/quantize activations differently and are not expected
// to be bit-identical.

#include "gk_impl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const enum gk_type g_weight_types[] = {
    GK_TYPE_F32, GK_TYPE_F16, GK_TYPE_BF16,
    GK_TYPE_Q4_0, GK_TYPE_Q4_1, GK_TYPE_Q5_0, GK_TYPE_Q5_1, GK_TYPE_Q8_0,
    GK_TYPE_Q2_K, GK_TYPE_Q3_K, GK_TYPE_Q4_K, GK_TYPE_Q5_K, GK_TYPE_Q6_K,
    GK_TYPE_IQ4_NL, GK_TYPE_IQ4_XS,
    GK_TYPE_TQ1_0, GK_TYPE_TQ2_0,
    GK_TYPE_MXFP4, GK_TYPE_NVFP4, GK_TYPE_Q1_0, GK_TYPE_Q2_0,
};

static float input_value(int i) {
    return sinf((float) i * 0.071f) * 0.5f + cosf((float) i * 0.013f) * 0.25f;
}

static struct gk_tensor * build_graph(struct gk_ctx * ctx, enum gk_type weight_type,
                                      struct gk_tensor ** weight,
                                      struct gk_tensor ** input) {
    const int64_t k = 256;
    const int64_t m = 16;
    const int64_t n = 5;

    *weight = gk_new_tensor_2d(ctx, weight_type, k, m);
    *input  = gk_new_tensor_2d(ctx, GK_TYPE_F32, k, n);

    struct gk_tensor * out = gk_mul_mat(ctx, *weight, *input);
    gk_set_output(out);
    return out;
}

static void encode_weights(enum gk_type type, const float * src, void * dst,
                           int64_t k, int64_t rows) {
    const struct gk_type_traits * traits = gk_get_type_traits(type);
    const size_t row_bytes = gk_row_size(type, k);

    for (int64_t r = 0; r < rows; ++r) {
        traits->from_float(src + r * k, (char *) dst + r * row_bytes, k);
    }
}

static int run_decode_type(gk_backend_t gpu, enum gk_type type) {
    const int64_t k = 256;
    const int64_t rows = 4;

    struct gk_ctx * ctx = gk_init((struct gk_init_params) {
        .mem_size = 1u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * weight = gk_new_tensor_2d(ctx, type, k, rows);
    struct gk_tensor * ids = gk_new_tensor_1d(ctx, GK_TYPE_I32, rows);
    struct gk_tensor * out = gk_get_rows(ctx, weight, ids);
    gk_set_output(out);
    struct gk_cgraph * graph = gk_new_graph(ctx);
    gk_build_forward_expand(graph, out);

    struct gk_gallocr * alloc =
        gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, graph)) {
        fprintf(stderr, "%s: failed to allocate decoder graph\n", gk_type_name(type));
        return 1;
    }

    const int64_t n = k * rows;
    float * src = (float *) malloc((size_t) n * sizeof(float));
    float * expected = (float *) malloc((size_t) n * sizeof(float));
    float * got = (float *) malloc((size_t) n * sizeof(float));
    void * encoded = malloc(gk_nbytes(weight));
    int32_t row_ids[4] = { 0, 1, 2, 3 };
    if (src == NULL || expected == NULL || got == NULL || encoded == NULL) {
        return 1;
    }
    for (int64_t i = 0; i < n; ++i) {
        src[i] = input_value((int) i) * 0.125f;
    }
    encode_weights(type, src, encoded, k, rows);
    const struct gk_type_traits * traits = gk_get_type_traits(type);
    const size_t row_bytes = gk_row_size(type, k);
    for (int64_t r = 0; r < rows; ++r) {
        traits->to_float((const char *) encoded + r * row_bytes, expected + r * k, k);
    }

    gk_backend_tensor_set(weight, encoded, 0, gk_nbytes(weight));
    gk_backend_tensor_set(ids, row_ids, 0, sizeof(row_ids));
    if (gk_backend_graph_compute(gpu, graph) != GK_STATUS_SUCCESS) {
        fprintf(stderr, "%s: decoder graph execution failed\n", gk_type_name(type));
        return 1;
    }
    gk_backend_synchronize(gpu);
    gk_backend_tensor_get(out, got, 0, (size_t) n * sizeof(float));

    int bad = 0;
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float diff = fabsf(got[i] - expected[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (diff != 0.0f) {
            bad++;
        }
    }
    if (bad != 0) {
        printf("  %-8s decoder: max abs error %.8g, %d/%lld mismatches\n",
               gk_type_name(type), max_abs, bad, (long long) n);
    }

    free(encoded);
    free(got);
    free(expected);
    free(src);
    gk_gallocr_free(alloc);
    gk_free(ctx);
    return bad == 0 ? 0 : 1;
}

static int run_type(gk_backend_t gpu, enum gk_type weight_type) {
    struct gk_ctx * gpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 1u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * gpu_w;
    struct gk_tensor * gpu_x;
    struct gk_tensor * gpu_out = build_graph(gpu_ctx, weight_type, &gpu_w, &gpu_x);
    struct gk_cgraph * gpu_graph = gk_new_graph(gpu_ctx);
    gk_build_forward_expand(gpu_graph, gpu_out);

    if (!gk_backend_supports_op(gpu, gpu_out)) {
        fprintf(stderr, "%s: backend unexpectedly rejects the matmul\n",
                gk_type_name(weight_type));
        return 1;
    }

    struct gk_gallocr * gpu_alloc =
        gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (gpu_alloc == NULL || !gk_gallocr_alloc_graph(gpu_alloc, gpu_graph)) {
        fprintf(stderr, "%s: failed to allocate the device graph\n",
                gk_type_name(weight_type));
        return 1;
    }

    const int64_t nw = gk_nelements(gpu_w);
    const int64_t nx = gk_nelements(gpu_x);
    const int64_t no = gk_nelements(gpu_out);
    const size_t encoded_size = gk_nbytes(gpu_w);
    float * w = (float *) malloc((size_t) nw * sizeof(float));
    float * x = (float *) malloc((size_t) nx * sizeof(float));
    void * encoded = malloc(encoded_size);
    float * got = (float *) malloc((size_t) no * sizeof(float));
    float * expected = (float *) malloc((size_t) no * sizeof(float));
    float * decoded = (float *) malloc((size_t) nw * sizeof(float));
    if (w == NULL || x == NULL || encoded == NULL || got == NULL || expected == NULL || decoded == NULL) {
        fprintf(stderr, "%s: host allocation failed\n", gk_type_name(weight_type));
        return 1;
    }
    for (int64_t i = 0; i < nw; ++i) {
        w[i] = input_value((int) i) * 0.125f;
    }
    for (int64_t i = 0; i < nx; ++i) {
        x[i] = input_value((int) (i + nw));
    }
    encode_weights(weight_type, w, encoded, gpu_w->ne[0], gpu_w->ne[1]);
    const struct gk_type_traits * traits = gk_get_type_traits(weight_type);
    const size_t row_bytes = gk_row_size(weight_type, gpu_w->ne[0]);
    for (int64_t r = 0; r < gpu_w->ne[1]; ++r) {
        traits->to_float((const char *) encoded + r * row_bytes,
                         decoded + r * gpu_w->ne[0], gpu_w->ne[0]);
    }

    gk_backend_tensor_set(gpu_w, encoded, 0, encoded_size);
    gk_backend_tensor_set(gpu_x, x, 0, (size_t) nx * sizeof(float));

    if (gk_backend_graph_compute(gpu, gpu_graph) != GK_STATUS_SUCCESS) {
        fprintf(stderr, "%s: device graph execution failed\n", gk_type_name(weight_type));
        return 1;
    }
    gk_backend_synchronize(gpu);
    gk_backend_tensor_get(gpu_out, got, 0, (size_t) no * sizeof(float));

    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 4u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * cpu_w;
    struct gk_tensor * cpu_x;
    struct gk_tensor * cpu_out = build_graph(cpu_ctx, weight_type, &cpu_w, &cpu_x);
    memcpy(cpu_w->data, encoded, encoded_size);
    memcpy(cpu_x->data, x, (size_t) nx * sizeof(float));
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cpu_out);
    if (gk_graph_compute(cpu_graph, 4) != GK_STATUS_SUCCESS) {
        fprintf(stderr, "%s: CPU reference execution failed\n", gk_type_name(weight_type));
        return 1;
    }
    memcpy(expected, cpu_out->data, (size_t) no * sizeof(float));

    int bad = 0;
    float max_abs = 0.0f;
    for (int64_t i = 0; i < no; ++i) {
        const float diff = fabsf(got[i] - expected[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (!(diff <= 5e-3f + 5e-3f * fabsf(expected[i]))) {
            bad++;
        }
    }

    printf("  %-8s %3lld outputs, max abs error %.8g, %d mismatches\n",
           gk_type_name(weight_type), (long long) no, max_abs, bad);
    if (bad != 0) {
        for (int64_t i = 0; i < no && i < 4; ++i) {
            const int64_t row = i % gpu_w->ne[1];
            const int64_t col = i / gpu_w->ne[1];
            float scalar = 0.0f;
            double precise = 0.0;
            for (int64_t kk = 0; kk < gpu_w->ne[0]; ++kk) {
                scalar += decoded[row * gpu_w->ne[0] + kk] *
                          x[col * gpu_w->ne[0] + kk];
                precise += (double) decoded[row * gpu_w->ne[0] + kk] *
                           (double) x[col * gpu_w->ne[0] + kk];
            }
            printf("      [%lld] device %.8g cpu %.8g scalar %.8g double %.8g\n",
                   (long long) i, got[i], expected[i], scalar, (float) precise);
        }
    }

    free(decoded);
    free(expected);
    free(got);
    free(encoded);
    free(x);
    free(w);
    gk_free(cpu_ctx);
    gk_gallocr_free(gpu_alloc);
    gk_free(gpu_ctx);

    return bad == 0 ? 0 : 1;
}

// --------------------------------------------------------------------------
// op parity
//
// The matmul tests above cover the path a language model spends its time in,
// which leaves the ops a multimodal graph reaches for - the vision tower's
// pooling, the audio tower's short convolution and its cyclic shift - checked
// only by whether the whole server produces sensible text. That is too coarse
// to localise anything, and an op the backend silently declines is not a wrong
// answer but a quiet fall back to the CPU, so both facts are asserted here:
// the backend claims the op, and the answer matches the CPU's.
// --------------------------------------------------------------------------

// The op under test, built twice against the same context-creation rules: once
// where the tensors are host memory and once where they are device memory.
typedef struct gk_tensor * (*op_builder)(struct gk_ctx * ctx, struct gk_tensor ** inputs,
                                         int * n_inputs);

static struct gk_tensor * build_roll(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 24, 5, 3);
    *n_in = 1;
    // a negative shift, a positive one and a zero: each takes a different
    // branch through the wrap
    return gk_roll(ctx, in[0], -7, 3, 0, 0);
}

static struct gk_tensor * build_ssm_conv(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t d_conv = 4, n_t = 11, d_inner = 9, n_seqs = 2;
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, d_conv - 1 + n_t, d_inner, n_seqs);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, d_conv, d_inner);
    *n_in = 2;
    return gk_ssm_conv(ctx, in[0], in[1]);
}

static struct gk_tensor * build_pool_2d_max(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 13, 11, 3);
    *n_in = 1;
    return gk_pool_2d(ctx, in[0], GK_OP_POOL_MAX, 3, 3, 2, 2, 1.0f, 1.0f);
}

static struct gk_tensor * build_pool_2d_avg(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 16, 16, 4);
    *n_in = 1;
    // stride equal to the kernel and no padding: the patch-merging shape a
    // vision tower actually uses
    return gk_pool_2d(ctx, in[0], GK_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
}

// The regression this file exists for: a batch that asks for no logits gives
// the output matmul zero columns. There is no launch geometry that means "no
// work", so an empty node has to be recognised before the launch rather than
// rejected by the driver, and the graph it sits in has to still succeed.
static struct gk_tensor * build_empty_mul_mat(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 64, 128);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 64, 0);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

// Padded, strided and dilated all at once, so none of the three index terms
// can be dropped without the comparison noticing.
static struct gk_tensor * build_conv_2d(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 3, 3, 4, 5);  // [KW, KH, IC, OC]
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 14, 12, 4, 2); // [IW, IH, IC, N]
    *n_in = 2;
    return gk_conv_2d_direct(ctx, in[0], in[1], 2, 2, 1, 1, 2, 2);
}

// The half-precision kernel, which also rounds the input through f16 on the
// way in - a separate path in both the CPU pass and the kernel.
static struct gk_tensor * build_conv_2d_f16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 4, 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 14, 12, 4, 2);
    *n_in = 2;
    return gk_conv_2d_direct(ctx, in[0], in[1], 1, 1, 1, 1, 1, 1);
}

// The two selective-scan variants. They differ only in A's first extent, but
// that picks between one decay per head and one per state element, which is a
// whole branch of the kernel each.
static struct gk_tensor * build_ssm_scan_common(struct gk_ctx * ctx, struct gk_tensor ** in,
                                                int * n_in, int64_t a_ne0) {
    const int64_t d_state = 8, head_dim = 4, n_head = 4, n_group = 2;
    const int64_t n_tok = 5, n_seqs = 2;

    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, d_state, head_dim, n_head, n_seqs);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, head_dim, n_head, n_tok, n_seqs);
    in[2] = gk_new_tensor_3d(ctx, GK_TYPE_F32, n_head, n_tok, n_seqs);
    in[3] = gk_new_tensor_2d(ctx, GK_TYPE_F32, a_ne0, n_head);
    in[4] = gk_new_tensor_4d(ctx, GK_TYPE_F32, d_state, n_group, n_tok, n_seqs);
    in[5] = gk_new_tensor_4d(ctx, GK_TYPE_F32, d_state, n_group, n_tok, n_seqs);
    in[6] = gk_new_tensor_1d(ctx, GK_TYPE_I32, n_seqs);
    *n_in = 7;

    return gk_ssm_scan(ctx, in[0], in[1], in[2], in[3], in[4], in[5], in[6]);
}

static struct gk_tensor * build_ssm_scan_m2(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return build_ssm_scan_common(ctx, in, n_in, 1); // Mamba-2
}

static struct gk_tensor * build_ssm_scan_m1(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return build_ssm_scan_common(ctx, in, n_in, 8); // Mamba-1, d_state decays
}


// --------------------------------------------------------------------------
// The depthwise convolution and the linear-attention recurrences.
//
// The convolution is here in both layouts the builders emit: the ordinary
// WHCN one, and the channels-fastest CWHN a permuted input produces, which the
// kernel only gets right by reading through strides rather than assuming a
// packing. The recurrences are shaped like the models that use them - RWKV's
// heads are 64 wide, the delta rule's 128 - and are run over several tokens,
// because a single token would never exercise the state chaining forward.
// --------------------------------------------------------------------------

static struct gk_tensor * build_conv_2d_dw(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 3, 3, 1, 5);  // kernel, one plane per channel
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 13, 11, 5, 2); // image
    *n_in = 2;
    return gk_conv_2d_dw_direct(ctx, in[0], in[1], 1, 1, 1, 1, 1, 1);
}

static struct gk_tensor * build_conv_2d_dw_f16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 1, 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 13, 11, 5, 2);
    *n_in = 2;
    return gk_conv_2d_dw_direct(ctx, in[0], in[1], 1, 1, 1, 1, 1, 1);
}

// Stride 2 and no padding, which is what MobileNetV5's downsampling stages
// ask for and what makes the window arithmetic worth checking separately.
static struct gk_tensor * build_conv_2d_dw_s2(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 1, 4);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 16, 16, 4, 1);
    *n_in = 2;
    return gk_conv_2d_dw_direct(ctx, in[0], in[1], 2, 2, 0, 0, 1, 1);
}

// The channels-fastest layout: the image is built transposed and permuted back,
// so its channel stride is the innermost one.
static struct gk_tensor * build_conv_2d_dw_cwhn(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 3, 3, 1, 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 5, 13, 11, 2); // [C, W, H, N]
    *n_in = 2;

    // src axis 0 (C) becomes axis 2, W becomes 0, H becomes 1
    struct gk_tensor * img = gk_permute(ctx, in[1], 2, 0, 1, 3); // -> [W, H, C, N]
    return gk_conv_2d_dw_direct(ctx, in[0], img, 1, 1, 1, 1, 1, 1);
}

static struct gk_tensor * build_rwkv_wkv6(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t S = 64, H = 3, T = 7, n_seqs = 1;
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // k
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // v
    in[2] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // r
    in[3] = gk_new_tensor_2d(ctx, GK_TYPE_F32, S, H);          // time-mix first
    in[4] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // decay
    in[5] = gk_new_tensor_2d(ctx, GK_TYPE_F32, S * S * H, n_seqs);
    *n_in = 6;
    return gk_rwkv_wkv6(ctx, in[0], in[1], in[2], in[3], in[4], in[5]);
}

// Two sequences, so the state has to be re-read from the input at each
// sequence boundary rather than carried across it.
static struct gk_tensor * build_rwkv_wkv6_seqs(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t S = 64, H = 2, T = 8, n_seqs = 2;
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    in[2] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    in[3] = gk_new_tensor_2d(ctx, GK_TYPE_F32, S, H);
    in[4] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);
    in[5] = gk_new_tensor_2d(ctx, GK_TYPE_F32, S * S * H, n_seqs);
    *n_in = 6;
    return gk_rwkv_wkv6(ctx, in[0], in[1], in[2], in[3], in[4], in[5]);
}

// The decay and the two feedback vectors are scaled down before they reach the
// recurrence. Left at the harness's own fill they are order 1, the state grows
// token over token, and the outputs come out in the tens of thousands - where
// the relative tolerance is tens of thousands of times looser than it looks and
// would wave through a kernel that was merely close. Scaled, the recurrence
// settles and the comparison is worth something: this case lands at 1e-6
// rather than at 2.5.
static struct gk_tensor * build_rwkv_wkv7(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t S = 64, H = 3, T = 7, n_seqs = 1;
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // r
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // w, the decay
    in[2] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // k
    in[3] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // v
    in[4] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // a
    in[5] = gk_new_tensor_3d(ctx, GK_TYPE_F32, S, H, T);       // b
    in[6] = gk_new_tensor_2d(ctx, GK_TYPE_F32, S * S * H, n_seqs);
    *n_in = 7;

    return gk_rwkv_wkv7(ctx, in[0],
                        gk_scale(ctx, in[1], 0.1f),
                        in[2], in[3],
                        gk_scale(ctx, in[4], 0.1f),
                        gk_scale(ctx, in[5], 0.1f),
                        in[6]);
}

static struct gk_tensor * gdn_common(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in,
                                     int64_t S, int64_t H, int64_t T, int64_t n_seqs,
                                     bool kda, int64_t K) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);        // q
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);        // k
    in[2] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, H, T, n_seqs);        // v
    in[3] = gk_new_tensor_4d(ctx, GK_TYPE_F32, kda ? S : 1, H, T, n_seqs); // gate
    in[4] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 1, H, T, n_seqs);        // beta
    in[5] = gk_new_tensor_4d(ctx, GK_TYPE_F32, S, S, H, n_seqs);        // state
    *n_in = 6;
    return gk_gated_delta_net(ctx, in[0], in[1], in[2], in[3], in[4], in[5], K);
}

// The scalar gate: one decay for the whole head, which is Qwen3-Next's form.
static struct gk_tensor * build_gdn_scalar(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return gdn_common(ctx, in, n_in, 32, 3, 6, 1, false, 1);
}

// The per-channel gate, which is KDA's, and the only path with a barrier.
static struct gk_tensor * build_gdn_kda(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return gdn_common(ctx, in, n_in, 32, 3, 6, 1, true, 1);
}

// K above one, so older states are snapshotted as the token loop passes them.
static struct gk_tensor * build_gdn_snapshots(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return gdn_common(ctx, in, n_in, 32, 2, 6, 1, false, 3);
}

static struct gk_tensor * build_gdn_seqs(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return gdn_common(ctx, in, n_in, 32, 2, 5, 2, true, 1);
}

// The padding, reduction and scan kernels. The shapes here are chosen to land
// off every boundary the implementations care about: a row that is not a
// multiple of the scan's chunk, a pad wider than a warp, a reduction row that
// is not a multiple of the block.
static struct gk_tensor * build_pad_reflect_1d(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 97, 5);
    *n_in = 1;
    return gk_pad_reflect_1d(ctx, in[0], 13, 7);
}

// The largest reflection the op allows: a pad one short of the row, which is
// where an off-by-one in the mirror shows up.
static struct gk_tensor * build_pad_reflect_edge(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 40, 3);
    *n_in = 1;
    return gk_pad_reflect_1d(ctx, in[0], 39, 39);
}

static struct gk_tensor * build_argmax(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 1000, 7);
    *n_in = 1;
    return gk_argmax(ctx, in[0]);
}

// A row wider than one block can cover in a pass, so the reduction has to
// combine partial winners rather than just scan.
static struct gk_tensor * build_argmax_wide(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 262144, 2);
    *n_in = 1;
    return gk_argmax(ctx, in[0]);
}

static struct gk_tensor * build_cumsum(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 333, 4, 2);
    *n_in = 1;
    return gk_cumsum(ctx, in[0]);
}

// Several chunks, the last one short: the three-pass scan has to carry a
// prefix across chunk boundaries and stop at the row's real end.
static struct gk_tensor * build_cumsum_wide(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 20011, 3);
    *n_in = 1;
    return gk_cumsum(ctx, in[0]);
}

static struct gk_tensor * build_im2col_3d(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t IC = 2, N = 2;
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 2, IC * 4); // kernel, shape only
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 9, 7, 5, IC * N); // volume
    *n_in = 2;
    return gk_im2col_3d(ctx, in[0], in[1], IC, 1, 1, 1, 0, 0, 0, 1, 1, 1, GK_TYPE_F32);
}

// f16 output, strided and padded, which is the form a video patch embedding
// actually asks for.
static struct gk_tensor * build_im2col_3d_f16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t IC = 3, N = 1;
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 2, 2, 2, IC * 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 11, 9, 6, IC * N);
    *n_in = 2;
    return gk_im2col_3d(ctx, in[0], in[1], IC, 2, 2, 2, 1, 1, 1, 1, 1, 1, GK_TYPE_F16);
}

// --------------------------------------------------------------------------
// The diffusion op set.
//
// A transformer decoder exercises a narrow slice of this backend; an image
// model asks for a different one - normalisations over spatial groups, the
// broadcasts a residual stack builds, the im2col a convolution decomposes to.
// These are the ops the diffusion graphs actually contain, and the geometry is
// theirs: channel counts that are not multiples of the block, a token count
// that is not a multiple of the warp.
// --------------------------------------------------------------------------

static struct gk_tensor * build_norm(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 320, 37, 2);
    *n_in = 1;
    return gk_norm(ctx, in[0], 1e-5f);
}

static struct gk_tensor * build_group_norm(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    // [W, H, C, N] with 32 groups over C, the shape every VAE block uses
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 9, 7, 64, 2);
    *n_in = 1;
    return gk_group_norm(ctx, in[0], 32, 1e-6f);
}

static struct gk_tensor * build_mean(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 133, 5, 3);
    *n_in = 1;
    return gk_mean(ctx, in[0]);
}

static struct gk_tensor * build_repeat(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    // repeat_4d rather than repeat: the shape argument of the two-tensor form
    // is a template, not a graph input, and the harness only fills inputs.
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 64, 1, 3);
    *n_in = 1;
    return gk_repeat_4d(ctx, in[0], 64, 37, 3, 1);
}

static struct gk_tensor * build_concat(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 48, 37, 2);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 48, 11, 2);
    *n_in = 2;
    return gk_concat(ctx, in[0], in[1], 1);
}

static struct gk_tensor * build_timestep_embedding(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_1d(ctx, GK_TYPE_F32, 3);
    *n_in = 1;
    return gk_timestep_embedding(ctx, in[0], 256, 10000);
}

static struct gk_tensor * build_im2col_f16(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    // the destination type a composite conv_2d asks for, which is the route
    // every convolution in these models actually takes
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, 4, 5);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 14, 12, 4, 2);
    *n_in = 2;
    return gk_im2col(ctx, in[0], in[1], 1, 1, 1, 1, 1, 1, true, GK_TYPE_F16);
}

static struct gk_tensor * build_soft_max(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 37, 37, 5);
    *n_in = 1;
    return gk_soft_max(ctx, in[0]);
}

static struct gk_tensor * build_silu(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 133, 7);
    *n_in = 1;
    return gk_silu(ctx, in[0]);
}

static struct gk_tensor * build_gelu(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 133, 7);
    *n_in = 1;
    return gk_gelu(ctx, in[0]);
}

// A residual add where the bias broadcasts over rows - the shape a linear
// layer's bias arrives in, and the one a per-element kernel gets wrong.
static struct gk_tensor * build_add_broadcast(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 320, 37, 2);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 320, 1, 1);
    *n_in = 2;
    return gk_add(ctx, in[0], in[1]);
}

static struct gk_tensor * build_mul_broadcast(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 320, 37, 2);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 320, 1, 2);
    *n_in = 2;
    return gk_mul(ctx, in[0], in[1]);
}

// cont of a permuted view: the attention stacks do this between every
// projection, and it is where a kernel that assumes contiguity shows up.
static struct gk_tensor * build_cont_permuted(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 64, 5, 37, 2);
    *n_in = 1;
    return gk_cont(ctx, gk_permute(ctx, in[0], 0, 2, 1, 3));
}

// A matmul whose activations are a permuted view rather than a packed buffer.
static struct gk_tensor * build_mul_mat_permuted(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 64, 96);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 64, 37, 2);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], gk_cont(ctx, gk_permute(ctx, in[1], 0, 2, 1, 3)));
}

// The tiled matmul path: wide enough to take it, and deliberately not a
// multiple of the tile in either direction.
static struct gk_tensor * build_mul_mat_wide(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 130, 100);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 130, 145);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

// The batched integer tile, which needs all of: a format with an integer dot,
// enough columns to take the tiled path at all, and enough rows to be worth
// quantizing the activations for. Nothing else in this file meets all three -
// build_mul_mat_wide_q is nvfp4 and a hundred rows, so it takes the float
// tile - so these exist to reach it.
//
// The shapes are deliberately off every tile boundary: 64 is the tile in both
// directions, so rows and columns that are not multiples of it are what
// exercise the edge guards.
static struct gk_tensor * mmq_case(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in,
                                   enum gk_type type, int64_t k, int64_t rows, int64_t cols) {
    in[0] = gk_new_tensor_2d(ctx, type, k, rows);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, k, cols);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

static struct gk_tensor * build_mmq_q4_0(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_0, 256, 600, 40);
}

static struct gk_tensor * build_mmq_q4_1(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_1, 256, 1024, 64);
}

static struct gk_tensor * build_mmq_q8_0(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q8_0, 256, 576, 33);
}

static struct gk_tensor * build_mmq_q4_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_K, 512, 700, 70);
}

// One super-block rather than two, so the group-to-super-block indexing never
// has to cross a boundary. If this agrees and the wider one drifts, the drift
// is accumulation, not indexing.
static struct gk_tensor * build_mmq_q4_K_1sb(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_K, 256, 700, 70);
}

// The integer *mat-vec*, which is a different kernel from the tile above and
// needs the opposite shape to reach: enough rows to be worth quantizing the
// activations for, few enough columns to stay off the tiled path.
//
// This is the path lm_head takes, and it went untested the moment the row
// threshold was introduced - run_type's sixteen rows fall below it, so the
// per-weight-type sweep quietly stopped covering the integer kernel it used to.
static struct gk_tensor * build_mmv_q4_0(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_0, 256, 600, 1);
}

static struct gk_tensor * build_mmv_q4_1(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_1, 256, 1024, 2);
}

static struct gk_tensor * build_mmv_q8_0(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q8_0, 256, 576, 1);
}

static struct gk_tensor * build_mmv_q4_K(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_K, 512, 700, 3);
}

// Eight columns: still short of the tiled path, but past the point where the
// mat-vec kernel serves several columns from one pass over a weight row.
static struct gk_tensor * build_mmv_q4_0_nc(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_Q4_0, 256, 600, 8);
}

// A k that is not a whole number of 32-element groups, which the integer path
// declines - so this checks the fall back to the float tile still works.
static struct gk_tensor * build_mmq_ragged_k(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    return mmq_case(ctx, in, n_in, GK_TYPE_F16, 130, 600, 40);
}

// Higher dimensions, so the per-slice indexing of the quantized activations
// has to be right rather than accidentally zero.
static struct gk_tensor * build_mmq_batched(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_Q4_0, 256, 520, 3);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32,  256, 40,  3);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

static struct gk_tensor * build_mul_mat_wide_q(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_NVFP4, 128, 100);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 128, 145);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

// Batched, so the broadcast of the weight's higher dimensions onto the
// activations' is exercised on the tiled path too.
// The same quantized weight through the mat-vec path, as a control: the CPU
// dots a quantized weight against quantized activations and the device decodes
// to float, so the two differ by the activation quantization on *both* paths.
// If this control mismatches too, the tolerance is what is wrong, not the tile.
static struct gk_tensor * build_mul_mat_narrow_q(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_NVFP4, 128, 100);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 128, 8);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

static struct gk_tensor * build_mul_mat_wide_batched(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 96, 70, 2);
    in[1] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 96, 133, 4);
    *n_in = 2;
    return gk_mul_mat(ctx, in[0], in[1]);
}

static struct gk_tensor * build_rope(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_3d(ctx, GK_TYPE_F32, 64, 8, 37);
    in[1] = gk_new_tensor_1d(ctx, GK_TYPE_I32, 37);
    *n_in = 2;
    return gk_rope(ctx, in[0], in[1], 64, 0);
}

static struct gk_tensor * build_get_rows(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 96, 40);
    in[1] = gk_new_tensor_1d(ctx, GK_TYPE_I32, 13);
    *n_in = 2;
    return gk_get_rows(ctx, in[0], in[1]);
}

static struct gk_tensor * build_scale(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, 133, 7);
    *n_in = 1;
    return gk_scale(ctx, in[0], 0.375f);
}

static struct gk_tensor * build_pad(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 9, 7, 5, 2);
    *n_in = 1;
    return gk_pad(ctx, in[0], 3, 2, 1, 0);
}

static struct gk_tensor * build_upscale(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, 9, 7, 5, 2);
    *n_in = 1;
    return gk_upscale(ctx, in[0], 2, GK_SCALE_MODE_NEAREST);
}


// --------------------------------------------------------------------------
// Composite graphs.
//
// Every op above passes on its own, which is not the same as a graph passing.
// A deep chain is where buffer reuse in the allocator, aliasing between a node
// and its source, and any dependence on evaluation order actually show up - a
// single-op test allocates two live tensors and never reuses anything. These
// mirror the two shapes the engine really runs.
// --------------------------------------------------------------------------

static struct gk_tensor * build_vae_stack(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t W = 16, H = 16, C = 64;

    in[0] = gk_new_tensor_4d(ctx, GK_TYPE_F32, W, H, C, 1);
    in[1] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, C, C);
    in[2] = gk_new_tensor_4d(ctx, GK_TYPE_F16, 3, 3, C, C);
    in[3] = gk_new_tensor_1d(ctx, GK_TYPE_F32, C);
    *n_in = 4;

    struct gk_tensor * x = in[0];
    for (int layer = 0; layer < 3; ++layer) {
        struct gk_tensor * h = gk_group_norm(ctx, x, 32, 1e-6f);
        h = gk_silu(ctx, h);
        h = gk_conv_2d(ctx, in[1], h, 1, 1, 1, 1, 1, 1);
        h = gk_add(ctx, h, gk_reshape_4d(ctx, in[3], 1, 1, C, 1));
        h = gk_group_norm(ctx, h, 32, 1e-6f);
        h = gk_silu(ctx, h);
        h = gk_conv_2d(ctx, in[2], h, 1, 1, 1, 1, 1, 1);
        x = gk_add(ctx, x, h);
    }
    return x;
}

static struct gk_tensor * build_transformer_block(struct gk_ctx * ctx, struct gk_tensor ** in, int * n_in) {
    const int64_t D = 128, T = 37, HD = 32, NH = 4;

    in[0] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, T);
    in[1] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[2] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[3] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[4] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D);
    in[5] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D, D * 2);
    in[6] = gk_new_tensor_2d(ctx, GK_TYPE_F32, D * 2, D);
    *n_in = 7;

    struct gk_tensor * x = in[0];
    for (int layer = 0; layer < 2; ++layer) {
        struct gk_tensor * h = gk_norm(ctx, x, 1e-5f);

        struct gk_tensor * q = gk_mul_mat(ctx, in[1], h);
        struct gk_tensor * k = gk_mul_mat(ctx, in[2], h);
        struct gk_tensor * v = gk_mul_mat(ctx, in[3], h);

        q = gk_cont(ctx, gk_permute(ctx, gk_reshape_3d(ctx, q, HD, NH, T), 0, 2, 1, 3));
        k = gk_cont(ctx, gk_permute(ctx, gk_reshape_3d(ctx, k, HD, NH, T), 0, 2, 1, 3));
        v = gk_cont(ctx, gk_permute(ctx, gk_reshape_3d(ctx, v, HD, NH, T), 0, 2, 1, 3));

        struct gk_tensor * att = gk_mul_mat(ctx, k, q);
        att = gk_soft_max_ext(ctx, att, NULL, 1.0f / sqrtf((float) HD), 0.0f);

        struct gk_tensor * o = gk_mul_mat(ctx, gk_cont(ctx, gk_transpose(ctx, v)), att);
        o = gk_cont(ctx, gk_permute(ctx, o, 0, 2, 1, 3));
        o = gk_reshape_2d(ctx, o, D, T);
        o = gk_mul_mat(ctx, in[4], o);

        x = gk_add(ctx, x, o);

        h = gk_norm(ctx, x, 1e-5f);
        h = gk_mul_mat(ctx, in[5], h);
        h = gk_gelu(ctx, h);
        h = gk_mul_mat(ctx, in[6], h);
        x = gk_add(ctx, x, h);
    }
    return x;
}

static int run_op_tol(gk_backend_t gpu, const char * name, op_builder build, float tol);
static int run_op(gk_backend_t gpu, const char * name, op_builder build);

// Most ops are exact to f32 rounding against the CPU and are held to 1e-4. Two
// families legitimately are not, and loosening those here is the difference
// between a suite that catches a regression and one that always prints
// failures:
//
//   * a quantized weight: the CPU dots it against activations it has
//     quantized to 8 bits, the device decodes to float and does not. The
//     device answer is the more accurate one; they differ by the activation
//     quantization, not by a bug.
//   * an f16 intermediate: a composite conv_2d hands its im2col result on as
//     f16 and the CPU keeps it there, while the device widens to f32 to
//     accumulate. Again the device is closer to the truth.
//
// Both bounds are set just above what the difference actually measures, so a
// real regression in either still trips them.
static int run_op(gk_backend_t gpu, const char * name, op_builder build) {
    return run_op_tol(gpu, name, build, 1e-4f);
}

static int run_op_tol(gk_backend_t gpu, const char * name, op_builder build, float tol) {
    struct gk_tensor * gpu_in[GK_MAX_SRC] = { NULL };
    struct gk_tensor * cpu_in[GK_MAX_SRC] = { NULL };
    int n_in = 0;

    struct gk_ctx * gpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 4u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * gpu_out = build(gpu_ctx, gpu_in, &n_in);
    gk_set_output(gpu_out);
    struct gk_cgraph * gpu_graph = gk_new_graph(gpu_ctx);
    gk_build_forward_expand(gpu_graph, gpu_out);

    if (!gk_backend_supports_op(gpu, gpu_out)) {
        printf("  %-14s FAIL: the backend declines the op (it would fall back to the CPU)\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_gallocr * alloc = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, gpu_graph)) {
        printf("  %-14s FAIL: could not allocate the device graph\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    // The same context on the host, filled with the same numbers.
    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 8u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * cpu_out = build(cpu_ctx, cpu_in, &n_in);
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cpu_out);

    int seed = 0;
    for (int i = 0; i < n_in; ++i) {
        const int64_t n     = gk_nelements(cpu_in[i]);
        const size_t bytes  = gk_nbytes(cpu_in[i]);
        void * values = malloc(bytes > 0 ? bytes : 1);
        if (values == NULL) {
            return 1;
        }

        if (cpu_in[i]->type == GK_TYPE_I32) {
            // The only integer input any of these ops takes is ssm_scan's
            // sequence ids, which have to be valid rows of the state cache.
            for (int64_t j = 0; j < n; ++j) {
                ((int32_t *) values)[j] = (int32_t) j;
            }
        } else {
            float * f = (float *) malloc((size_t) (n > 0 ? n : 1) * sizeof(float));
            if (f == NULL) {
                return 1;
            }
            for (int64_t j = 0; j < n; ++j) {
                f[j] = input_value(seed++);
            }
            // A half-precision input is rounded once, here, so both sides read
            // exactly the same bits rather than each rounding its own way.
            if (cpu_in[i]->type == GK_TYPE_F32) {
                memcpy(values, f, (size_t) n * sizeof(float));
            } else {
                gk_get_type_traits(cpu_in[i]->type)->from_float(f, values, n);
            }
            free(f);
        }

        memcpy(cpu_in[i]->data, values, bytes);
        gk_backend_tensor_set(gpu_in[i], values, 0, bytes);
        free(values);
    }

    if (gk_graph_compute(cpu_graph, 4) != GK_STATUS_SUCCESS) {
        printf("  %-14s FAIL: the CPU reference failed\n", name);
        return 1;
    }
    if (gk_backend_graph_compute(gpu, gpu_graph) != GK_STATUS_SUCCESS) {
        printf("  %-14s FAIL: the device graph failed\n", name);
        return 1;
    }
    gk_backend_synchronize(gpu);

    const int64_t no = gk_nelements(gpu_out);
    int   bad     = 0;
    float max_abs = 0.0f;

    if (no > 0) {
        // The output need not be f32 - im2col hands a convolution an f16
        // buffer - so both sides are read as raw bytes and widened through the
        // type's own converter rather than assumed to be floats already.
        const size_t out_bytes = gk_nbytes(gpu_out);
        void  * raw = malloc(out_bytes);
        float * got = (float *) malloc((size_t) no * sizeof(float));
        float * exp_buf = (float *) malloc((size_t) no * sizeof(float));
        if (raw == NULL || got == NULL || exp_buf == NULL) {
            return 1;
        }
        gk_backend_tensor_get(gpu_out, raw, 0, out_bytes);

        if (gpu_out->type == GK_TYPE_F32) {
            memcpy(got, raw, (size_t) no * sizeof(float));
            memcpy(exp_buf, cpu_out->data, (size_t) no * sizeof(float));
        } else {
            const struct gk_type_traits * tr = gk_get_type_traits(gpu_out->type);
            tr->to_float(raw, got, no);
            tr->to_float(cpu_out->data, exp_buf, no);
        }
        free(raw);

        const float * expected = exp_buf;
        for (int64_t i = 0; i < no; ++i) {
            const float diff = fabsf(got[i] - expected[i]);
            if (diff > max_abs) {
                max_abs = diff;
            }
            if (!(diff <= tol + tol * fabsf(expected[i]))) {
                bad++;
            }
        }
        free(got);
        free(exp_buf);
    }

    printf("  %-14s %5lld outputs, max abs error %.8g, %d mismatches%s\n",
           name, (long long) no, max_abs, bad, bad == 0 ? "" : "  FAIL");

    gk_free(cpu_ctx);
    gk_gallocr_free(alloc);
    gk_free(gpu_ctx);

    return bad == 0 ? 0 : 1;
}

// --------------------------------------------------------------------------
// fused attention
//
// This op gets its own harness rather than joining run_op's list, for one
// reason: run_op fills every input from the same generator, and an attention
// mask filled that way holds arbitrary finite numbers. The interesting values
// in a mask are the infinities - a position the kernel must skip entirely -
// and whole regions of them, because the device splits the cache across blocks
// and a block whose entire slice is masked contributes nothing to the merge.
// That path cannot be reached with a mask of ordinary floats.
//
// So the mask is built here, in three shapes: absent, causal, and a suffix
// window that leaves early slices completely masked.
// --------------------------------------------------------------------------

enum fa_mask_mode {
    FA_MASK_NONE = 0,
    FA_MASK_CAUSAL,   // position ic visible to query iq1 if it precedes it
    FA_MASK_SUFFIX,   // only the last quarter of the cache is visible
};

struct fa_shape {
    int64_t n_batch;
    int64_t n_head;
    int64_t n_head_kv;   // fewer than n_head is grouped-query attention
    int64_t n_kv;
    int64_t DK;
    int64_t DV;
    enum fa_mask_mode mask;
    bool    sinks;
};

// Builds the graph into `ctx`, and hands back the inputs so both sides can be
// given identical bytes.
static struct gk_tensor * fa_build(struct gk_ctx * ctx, const struct fa_shape * s,
                                   struct gk_tensor ** q, struct gk_tensor ** k,
                                   struct gk_tensor ** v, struct gk_tensor ** m,
                                   struct gk_tensor ** sk) {
    *q = gk_new_tensor_4d(ctx, GK_TYPE_F32, s->DK, s->n_batch, s->n_head,    1);
    *k = gk_new_tensor_4d(ctx, GK_TYPE_F16, s->DK, s->n_kv,    s->n_head_kv, 1);
    *v = gk_new_tensor_4d(ctx, GK_TYPE_F16, s->DV, s->n_kv,    s->n_head_kv, 1);
    *m = s->mask == FA_MASK_NONE
        ? NULL
        : gk_new_tensor_4d(ctx, GK_TYPE_F16, s->n_kv, s->n_batch, 1, 1);

    struct gk_tensor * out = gk_flash_attn_ext(ctx, *q, *k, *v, *m,
                                               1.0f / sqrtf((float) s->DK), 0.0f, 0.0f);
    *sk = NULL;
    if (s->sinks) {
        *sk = gk_new_tensor_1d(ctx, GK_TYPE_F32, s->n_head);
        gk_flash_attn_ext_add_sinks(out, *sk);
    }
    return out;
}

// The mask, as f16, in whichever shape the case asked for.
static void fa_fill_mask(const struct fa_shape * s, gk_fp16_t * dst) {
    for (int64_t j = 0; j < s->n_batch; ++j) {
        for (int64_t i = 0; i < s->n_kv; ++i) {
            bool visible = true;
            if (s->mask == FA_MASK_CAUSAL) {
                visible = i <= j + (s->n_kv - s->n_batch);
            } else if (s->mask == FA_MASK_SUFFIX) {
                visible = i >= (s->n_kv * 3) / 4;
            }
            dst[j * s->n_kv + i] = gk_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    }
}

static void fa_fill(struct gk_tensor * gpu_t, struct gk_tensor * cpu_t, int * seed) {
    if (gpu_t == NULL) {
        return;
    }
    const int64_t n = gk_nelements(cpu_t);
    const size_t bytes = gk_nbytes(cpu_t);

    float * f = (float *) malloc((size_t) n * sizeof(float));
    void  * raw = malloc(bytes);
    for (int64_t i = 0; i < n; ++i) {
        f[i] = input_value((*seed)++);
    }
    if (cpu_t->type == GK_TYPE_F32) {
        memcpy(raw, f, bytes);
    } else {
        gk_get_type_traits(cpu_t->type)->from_float(f, raw, n);
    }

    memcpy(cpu_t->data, raw, bytes);
    gk_backend_tensor_set(gpu_t, raw, 0, bytes);

    free(raw);
    free(f);
}

static int run_flash_attn(gk_backend_t gpu, const char * name,
                          struct fa_shape s, float tol) {
    struct gk_tensor *gq, *gk_, *gv, *gm, *gs;
    struct gk_tensor *cq, *ck, *cv, *cm, *cs;

    struct gk_ctx * gpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * gpu_out = fa_build(gpu_ctx, &s, &gq, &gk_, &gv, &gm, &gs);
    gk_set_output(gpu_out);
    struct gk_cgraph * gpu_graph = gk_new_graph(gpu_ctx);
    gk_build_forward_expand(gpu_graph, gpu_out);

    if (!gk_backend_supports_op(gpu, gpu_out)) {
        printf("  %-16s FAIL: the backend declines the op\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_gallocr * alloc = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, gpu_graph)) {
        printf("  %-16s FAIL: could not allocate the device graph\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 128u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * cpu_out = fa_build(cpu_ctx, &s, &cq, &ck, &cv, &cm, &cs);
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cpu_out);

    int seed = 0;
    fa_fill(gq,  cq,  &seed);
    fa_fill(gk_, ck,  &seed);
    fa_fill(gv,  cv,  &seed);
    fa_fill(gs,  cs,  &seed);

    if (cm != NULL) {
        const size_t bytes = gk_nbytes(cm);
        gk_fp16_t * mbuf = (gk_fp16_t *) malloc(bytes);
        fa_fill_mask(&s, mbuf);
        memcpy(cm->data, mbuf, bytes);
        gk_backend_tensor_set(gm, mbuf, 0, bytes);
        free(mbuf);
    }

    if (gk_graph_compute(cpu_graph, 4) != GK_STATUS_SUCCESS) {
        printf("  %-16s FAIL: the CPU reference failed\n", name);
        return 1;
    }
    if (gk_backend_graph_compute(gpu, gpu_graph) != GK_STATUS_SUCCESS) {
        printf("  %-16s FAIL: the device graph failed\n", name);
        return 1;
    }
    gk_backend_synchronize(gpu);

    const int64_t no = gk_nelements(gpu_out);
    float * got = (float *) malloc((size_t) no * sizeof(float));
    gk_backend_tensor_get(gpu_out, got, 0, (size_t) no * sizeof(float));
    const float * want = (const float *) cpu_out->data;

    int   bad = 0;
    float max_abs = 0.0f;
    for (int64_t i = 0; i < no; ++i) {
        const float diff = fabsf(got[i] - want[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (!(diff <= tol)) { // catches NaN, which a comparison the other way lets past
            bad++;
        }
    }

    printf("  %-16s %6lld outputs, max abs error %.8g, %d mismatches%s\n",
           name, (long long) no, max_abs, bad, bad == 0 ? "" : "  FAIL");

    free(got);
    gk_free(cpu_ctx);
    gk_gallocr_free(alloc);
    gk_free(gpu_ctx);
    return bad == 0 ? 0 : 1;
}

// --------------------------------------------------------------------------
// top_k
//
// Also its own harness, and for a sharper reason than attention's: top_k's
// result is a set of indices with a fully specified order - descending by
// value, index breaking ties, and then the first two deliberately swapped - so
// the device and the CPU must agree exactly, not approximately. An int
// comparison catches what a tolerance would hide.
//
// The data patterns matter more than the shapes. A row of distinct values
// never exercises the tie-break; a row that is entirely -inf is what a fully
// masked router produces, and is where padding slots would displace real
// indices if their sentinel index were not chosen to lose the tie.
// --------------------------------------------------------------------------

enum tk_data {
    TK_DISTINCT = 0,
    TK_TIES,        // few distinct levels, so most comparisons are ties
    TK_ALL_NEG_INF, // a fully masked row
    TK_SOME_NEG_INF,
};

static void tk_fill(float * p, int64_t n, int64_t rows, enum tk_data mode) {
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < n; ++i) {
            const int64_t at = r * n + i;
            switch (mode) {
                case TK_DISTINCT:
                    p[at] = input_value((int) (at * 7 + 1));
                    break;
                case TK_TIES:
                    // eight levels over the row: every top-k slot is contested
                    p[at] = (float) (((at * 2654435761u) >> 8) % 8u);
                    break;
                case TK_ALL_NEG_INF:
                    p[at] = -INFINITY;
                    break;
                default:
                    p[at] = (i % 3 == 0) ? -INFINITY : input_value((int) (at * 5 + 3));
                    break;
            }
        }
    }
}

static int run_top_k(gk_backend_t gpu, const char * name,
                     int64_t n, int64_t k, int64_t rows, enum tk_data mode) {
    const size_t in_bytes = (size_t) n * rows * sizeof(float);

    struct gk_ctx * gpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * ga  = gk_new_tensor_2d(gpu_ctx, GK_TYPE_F32, n, rows);
    struct gk_tensor * gout = gk_top_k(gpu_ctx, ga, (int) k);
    gk_set_output(gout);
    struct gk_cgraph * gpu_graph = gk_new_graph(gpu_ctx);
    gk_build_forward_expand(gpu_graph, gout);

    if (!gk_backend_supports_op(gpu, gout)) {
        printf("  %-20s FAIL: the backend declines the op\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_gallocr * alloc = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, gpu_graph)) {
        printf("  %-20s FAIL: could not allocate the device graph\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * ca   = gk_new_tensor_2d(cpu_ctx, GK_TYPE_F32, n, rows);
    struct gk_tensor * cout = gk_top_k(cpu_ctx, ca, (int) k);
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cout);

    float * values = (float *) malloc(in_bytes);
    tk_fill(values, n, rows, mode);
    memcpy(ca->data, values, in_bytes);
    gk_backend_tensor_set(ga, values, 0, in_bytes);
    free(values);

    if (gk_graph_compute(cpu_graph, 4) != GK_STATUS_SUCCESS) {
        printf("  %-20s FAIL: the CPU reference failed\n", name);
        return 1;
    }
    if (gk_backend_graph_compute(gpu, gpu_graph) != GK_STATUS_SUCCESS) {
        printf("  %-20s FAIL: the device graph failed\n", name);
        return 1;
    }
    gk_backend_synchronize(gpu);

    const int64_t no = k * rows;
    int32_t * got = (int32_t *) malloc((size_t) no * sizeof(int32_t));
    gk_backend_tensor_get(gout, got, 0, (size_t) no * sizeof(int32_t));
    const int32_t * want = (const int32_t *) cout->data;

    int bad = 0;
    int first_at = -1;
    for (int64_t i = 0; i < no; ++i) {
        if (got[i] != want[i]) {
            if (first_at < 0) {
                first_at = (int) i;
            }
            bad++;
        }
    }

    if (bad == 0) {
        printf("  %-20s %5lld indices, exact\n", name, (long long) no);
    } else {
        printf("  %-20s %5lld indices, %d differ (first at %d: got %d, want %d)  FAIL\n",
               name, (long long) no, bad, first_at, got[first_at], want[first_at]);
    }

    free(got);
    gk_free(cpu_ctx);
    gk_gallocr_free(alloc);
    gk_free(gpu_ctx);
    return bad == 0 ? 0 : 1;
}

// argsort, which shares top_k's data patterns but returns a whole permutation
// rather than a selection - so a wrong tie-break shows up as a transposed pair
// somewhere in the middle rather than as a missing index, and the comparison
// has to cover every slot.
static int run_argsort(gk_backend_t gpu, const char * name,
                       int64_t n, int64_t rows, enum gk_sort_order order,
                       enum tk_data mode) {
    const size_t in_bytes = (size_t) n * rows * sizeof(float);

    struct gk_ctx * gpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 64u << 20, .mem_buffer = NULL, .no_alloc = true,
    });
    struct gk_tensor * ga   = gk_new_tensor_2d(gpu_ctx, GK_TYPE_F32, n, rows);
    struct gk_tensor * gout = gk_argsort(gpu_ctx, ga, order);
    gk_set_output(gout);
    struct gk_cgraph * gpu_graph = gk_new_graph(gpu_ctx);
    gk_build_forward_expand(gpu_graph, gout);

    if (!gk_backend_supports_op(gpu, gout)) {
        printf("  %-20s FAIL: the backend declines the op\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_gallocr * alloc = gk_gallocr_new(gk_backend_get_default_buffer_type(gpu));
    if (alloc == NULL || !gk_gallocr_alloc_graph(alloc, gpu_graph)) {
        printf("  %-20s FAIL: could not allocate the device graph\n", name);
        gk_free(gpu_ctx);
        return 1;
    }

    struct gk_ctx * cpu_ctx = gk_init((struct gk_init_params) {
        .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = false,
    });
    struct gk_tensor * ca   = gk_new_tensor_2d(cpu_ctx, GK_TYPE_F32, n, rows);
    struct gk_tensor * cout = gk_argsort(cpu_ctx, ca, order);
    struct gk_cgraph * cpu_graph = gk_new_graph(cpu_ctx);
    gk_build_forward_expand(cpu_graph, cout);

    float * values = (float *) malloc(in_bytes);
    tk_fill(values, n, rows, mode);
    memcpy(ca->data, values, in_bytes);
    gk_backend_tensor_set(ga, values, 0, in_bytes);
    free(values);

    if (gk_graph_compute(cpu_graph, 4) != GK_STATUS_SUCCESS) {
        printf("  %-20s FAIL: the CPU reference failed\n", name);
        return 1;
    }
    if (gk_backend_graph_compute(gpu, gpu_graph) != GK_STATUS_SUCCESS) {
        printf("  %-20s FAIL: the device graph failed\n", name);
        return 1;
    }
    gk_backend_synchronize(gpu);

    const int64_t no = n * rows;
    int32_t * got = (int32_t *) malloc((size_t) no * sizeof(int32_t));
    gk_backend_tensor_get(gout, got, 0, (size_t) no * sizeof(int32_t));
    const int32_t * want = (const int32_t *) cout->data;

    int bad = 0;
    int first_at = -1;
    for (int64_t i = 0; i < no; ++i) {
        if (got[i] != want[i]) {
            if (first_at < 0) {
                first_at = (int) i;
            }
            bad++;
        }
    }

    if (bad == 0) {
        printf("  %-20s %7lld indices, exact\n", name, (long long) no);
    } else {
        printf("  %-20s %7lld indices, %d differ (first at %d: got %d, want %d)  FAIL\n",
               name, (long long) no, bad, first_at, got[first_at], want[first_at]);
    }

    free(got);
    gk_free(cpu_ctx);
    gk_gallocr_free(alloc);
    gk_free(gpu_ctx);
    return bad == 0 ? 0 : 1;
}

int main(void) {
    gk_device_t device = gk_device_by_type(GK_DEVICE_TYPE_GPU);
    if (device == NULL) {
        fprintf(stderr, "CUDA-family backend was built but no GPU was discovered\n");
        return 1;
    }

    gk_backend_t gpu = gk_device_init_backend(device);
    if (gpu == NULL) {
        fprintf(stderr, "failed to initialize backend for %s\n", gk_device_name(device));
        return 1;
    }

    printf("%s: %s\n", gk_device_name(device), gk_device_description(device));

    int failures = 0;
    const int n_types = (int) (sizeof(g_weight_types) / sizeof(g_weight_types[0]));
    for (int i = 0; i < n_types; ++i) {
        failures += run_decode_type(gpu, g_weight_types[i]);
        failures += run_type(gpu, g_weight_types[i]);
    }

    printf("op parity against the CPU:\n");
    failures += run_op(gpu, "roll",          build_roll);
    failures += run_op(gpu, "ssm_conv",      build_ssm_conv);
    failures += run_op(gpu, "pool_2d max",   build_pool_2d_max);
    failures += run_op(gpu, "pool_2d avg",   build_pool_2d_avg);
    failures += run_op(gpu, "conv_2d",       build_conv_2d);
    failures += run_op(gpu, "conv_2d f16",   build_conv_2d_f16);
    failures += run_op(gpu, "ssm_scan m2",   build_ssm_scan_m2);
    failures += run_op(gpu, "ssm_scan m1",   build_ssm_scan_m1);
    failures += run_op(gpu, "mul_mat empty", build_empty_mul_mat);

    // The recurrences chain a state forward and each token's error feeds the
    // next, so they are held to a looser bound than a stateless op: 1e-4 is
    // above the drift a few tokens of f32 accumulation produce and well below
    // anything a wrong recurrence gives.
    printf("depthwise convolution and recurrences:\n");
    failures += run_op(gpu, "conv_2d_dw",     build_conv_2d_dw);
    failures += run_op(gpu, "conv_2d_dw f16", build_conv_2d_dw_f16);
    failures += run_op(gpu, "conv_2d_dw s2",  build_conv_2d_dw_s2);
    failures += run_op(gpu, "conv_dw cwhn",   build_conv_2d_dw_cwhn);
    failures += run_op(gpu, "rwkv_wkv6",      build_rwkv_wkv6);
    failures += run_op(gpu, "rwkv_wkv6 seqs", build_rwkv_wkv6_seqs);
    failures += run_op(gpu, "rwkv_wkv7",      build_rwkv_wkv7);
    failures += run_op(gpu, "gdn scalar",     build_gdn_scalar);
    failures += run_op(gpu, "gdn kda",        build_gdn_kda);
    failures += run_op(gpu, "gdn snapshots",  build_gdn_snapshots);
    failures += run_op(gpu, "gdn seqs",       build_gdn_seqs);

    printf("padding, reductions and the scan:\n");
    failures += run_op(gpu, "pad_reflect_1d", build_pad_reflect_1d);
    failures += run_op(gpu, "pad_reflect max", build_pad_reflect_edge);
    failures += run_op(gpu, "argmax",         build_argmax);
    failures += run_op(gpu, "argmax wide",    build_argmax_wide);
    failures += run_op(gpu, "im2col_3d",      build_im2col_3d);
    failures += run_op(gpu, "im2col_3d f16",  build_im2col_3d_f16);
    // The scan sums in a different order from the CPU's straight walk - that
    // is what makes it parallel - so it is held to a relative bound rather
    // than to equality. 1e-4 is far above the reassociation and far below a
    // dropped or double-counted chunk.
    failures += run_op(gpu, "cumsum",         build_cumsum);
    failures += run_op(gpu, "cumsum wide",    build_cumsum_wide);

    printf("diffusion op set:\n");
    failures += run_op(gpu, "norm",           build_norm);
    failures += run_op(gpu, "group_norm",     build_group_norm);
    failures += run_op(gpu, "mean",           build_mean);
    failures += run_op(gpu, "repeat",         build_repeat);
    failures += run_op(gpu, "concat",         build_concat);
    failures += run_op(gpu, "timestep_emb",   build_timestep_embedding);
    failures += run_op(gpu, "im2col f16",     build_im2col_f16);
    failures += run_op(gpu, "soft_max",       build_soft_max);
    failures += run_op(gpu, "silu",           build_silu);
    failures += run_op(gpu, "gelu",           build_gelu);
    failures += run_op(gpu, "add broadcast",  build_add_broadcast);
    failures += run_op(gpu, "mul broadcast",  build_mul_broadcast);
    failures += run_op(gpu, "cont permuted",  build_cont_permuted);
    failures += run_op(gpu, "mul_mat permut", build_mul_mat_permuted);
    failures += run_op(gpu, "mul_mat wide",   build_mul_mat_wide);
    failures += run_op_tol(gpu, "mul_mat wide q", build_mul_mat_wide_q, 4e-2f);
    failures += run_op_tol(gpu, "mul_mat narw q", build_mul_mat_narrow_q, 4e-2f);
    failures += run_op(gpu, "mul_mat wide b", build_mul_mat_wide_batched);

    // The batched integer tile. Held to the same bound as the other quantized
    // matmuls: both sides quantize, so they differ by the activation rounding
    // rather than by anything structural.
    failures += run_op_tol(gpu, "mmq q4_0",      build_mmq_q4_0,     4e-2f);
    failures += run_op_tol(gpu, "mmq q4_1",      build_mmq_q4_1,     4e-2f);
    failures += run_op_tol(gpu, "mmq q8_0",      build_mmq_q8_0,     4e-2f);
    // q4_K is held looser than the rest, and for a reason that is about the
    // CPU rather than the device: gk's q4_K vec_dot converts the activation
    // side to Q8_K, which carries one scale per 256 values, while the device
    // carries one per 32. The device is the more accurate of the two and the
    // gap widens with k - 0.040 at one super-block, 0.056 at two - so this
    // bound is set above where that lands rather than where a bug would.
    failures += run_op_tol(gpu, "mmq q4_K 1sb",  build_mmq_q4_K_1sb, 8e-2f);
    failures += run_op_tol(gpu, "mmq q4_K",      build_mmq_q4_K,     8e-2f);
    failures += run_op_tol(gpu, "mmq batched",   build_mmq_batched,  4e-2f);
    failures += run_op_tol(gpu, "mmq ragged k",  build_mmq_ragged_k, 4e-2f);

    failures += run_op_tol(gpu, "mmv q4_0",      build_mmv_q4_0,     4e-2f);
    failures += run_op_tol(gpu, "mmv q4_1",      build_mmv_q4_1,     4e-2f);
    failures += run_op_tol(gpu, "mmv q8_0",      build_mmv_q8_0,     4e-2f);
    failures += run_op_tol(gpu, "mmv q4_K",      build_mmv_q4_K,     8e-2f);
    failures += run_op_tol(gpu, "mmv q4_0 nc",   build_mmv_q4_0_nc,  4e-2f);
    failures += run_op(gpu, "rope",           build_rope);
    failures += run_op(gpu, "get_rows",       build_get_rows);
    failures += run_op(gpu, "scale",          build_scale);
    failures += run_op(gpu, "pad",            build_pad);
    failures += run_op(gpu, "upscale",        build_upscale);

    // Attention, both ways the device can spread it. "prefill" has enough
    // query rows to fill the card, so the cache is walked whole; "decode" has
    // one token, so the cache is cut into slices and merged. The two paths sum
    // in different orders and are held to a looser bound than an exact op -
    // 2e-3 is above what the reordering measures and below anything a real
    // merge bug would produce.
    printf("fused attention:\n");
    {
        const int64_t DK = 64, DV = 64;

        // one block per row: the unsplit path
        failures += run_flash_attn(gpu, "prefill",
            (struct fa_shape) { 256, 4, 4, 256, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);

        // few rows and a long cache: the split path
        failures += run_flash_attn(gpu, "decode",
            (struct fa_shape) { 1, 4, 4, 1024, DK, DV, FA_MASK_NONE, false }, 2e-3f);
        failures += run_flash_attn(gpu, "decode causal",
            (struct fa_shape) { 1, 4, 4, 1024, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);

        // grouped-query: four query heads share one key head
        failures += run_flash_attn(gpu, "decode gqa",
            (struct fa_shape) { 1, 8, 2, 1024, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);

        // the sink belongs to the row, not to a slice, and must be counted
        // once however many slices there are
        failures += run_flash_attn(gpu, "decode sinks",
            (struct fa_shape) { 1, 4, 4, 1024, DK, DV, FA_MASK_CAUSAL, true }, 2e-3f);
        failures += run_flash_attn(gpu, "prefill sinks",
            (struct fa_shape) { 256, 4, 4, 256, DK, DV, FA_MASK_CAUSAL, true }, 2e-3f);

        // only the tail of the cache is visible, so the early slices are
        // entirely masked and contribute nothing - the merge has to skip them
        // rather than let their empty maximum poison the common one
        failures += run_flash_attn(gpu, "decode masked slices",
            (struct fa_shape) { 1, 4, 4, 1024, DK, DV, FA_MASK_SUFFIX, false }, 2e-3f);
        failures += run_flash_attn(gpu, "masked slices+sinks",
            (struct fa_shape) { 1, 4, 4, 1024, DK, DV, FA_MASK_SUFFIX, true }, 2e-3f);

        // a cache the split cannot divide evenly
        failures += run_flash_attn(gpu, "decode ragged",
            (struct fa_shape) { 1, 4, 4, 1000, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);

        // a few tokens rather than one: still short of filling the card, so
        // still split, but with more than one query row per head
        failures += run_flash_attn(gpu, "small batch",
            (struct fa_shape) { 8, 4, 4, 1024, DK, DV, FA_MASK_CAUSAL, false }, 2e-3f);
    }

    // top_k on both sides of the width where one network stops fitting: 4096
    // slots. Below it the row is sorted whole; above it the selection is done
    // in rounds, and the two must give identical indices.
    printf("top_k:\n");
    {
        // the in-network path, which the rounds have to agree with
        failures += run_top_k(gpu, "moe 128 k=8",        128,    8,  17, TK_DISTINCT);
        failures += run_top_k(gpu, "narrow 4096 k=40",   4096,   40,  1, TK_DISTINCT);

        // one past the network's width: the rounds start here
        failures += run_top_k(gpu, "wide 4097 k=40",     4097,   40,  1, TK_DISTINCT);
        failures += run_top_k(gpu, "wide 8192 k=40",     8192,   40,  1, TK_DISTINCT);

        // a real vocabulary row, which is what the old path could not do
        failures += run_top_k(gpu, "vocab 262144 k=40",  262144, 40,  1, TK_DISTINCT);
        failures += run_top_k(gpu, "vocab k=1",          262144,  1,  1, TK_DISTINCT);

        // not a multiple of the chunk, so the last chunk is short and pads
        failures += run_top_k(gpu, "ragged 100000 k=64", 100000, 64,  1, TK_DISTINCT);

        // several rows at once, so the per-row candidate spans have to be
        // indexed apart
        failures += run_top_k(gpu, "wide rows",          20000,  16,  5, TK_DISTINCT);

        // where the tie-break earns its keep
        failures += run_top_k(gpu, "ties narrow",        2048,   32,  3, TK_TIES);
        failures += run_top_k(gpu, "ties wide",          50000,  32,  2, TK_TIES);

        // a fully masked row: every value -inf, so the padding sentinel is the
        // only thing keeping real indices in the answer
        failures += run_top_k(gpu, "all -inf wide",      50000,  24,  2, TK_ALL_NEG_INF);
        failures += run_top_k(gpu, "some -inf wide",     50000,  24,  2, TK_SOME_NEG_INF);

        // a k large enough to need more than one round
        failures += run_top_k(gpu, "big k rounds",       262144, 1024, 1, TK_DISTINCT);
    }

    // argsort across the same boundary. Unlike top_k this returns the whole
    // permutation, so every slot is compared - a tie-break that disagrees with
    // the CPU shows up as a transposed pair in the middle, not as a missing
    // index at the front.
    printf("argsort:\n");
    {
        // the in-network path both directions, as the reference for the rest
        failures += run_argsort(gpu, "narrow desc",   1024,  3, GK_SORT_ORDER_DESC, TK_DISTINCT);
        failures += run_argsort(gpu, "narrow asc",    1024,  3, GK_SORT_ORDER_ASC,  TK_DISTINCT);

        // one past the network's width: out of global memory from here
        failures += run_argsort(gpu, "wide 4097",     4097,  1, GK_SORT_ORDER_DESC, TK_DISTINCT);
        failures += run_argsort(gpu, "wide 8192 asc", 8192,  1, GK_SORT_ORDER_ASC,  TK_DISTINCT);

        // not a power of two, so the padding has to sort past every real slot
        failures += run_argsort(gpu, "ragged 20000",  20000, 2, GK_SORT_ORDER_DESC, TK_DISTINCT);
        failures += run_argsort(gpu, "ragged asc",    20000, 2, GK_SORT_ORDER_ASC,  TK_DISTINCT);

        // where the tie-break decides most of the answer
        failures += run_argsort(gpu, "wide ties",     50000, 2, GK_SORT_ORDER_DESC, TK_TIES);
        failures += run_argsort(gpu, "wide ties asc", 50000, 2, GK_SORT_ORDER_ASC,  TK_TIES);

        // every value the same, so the whole permutation is the tie-break
        failures += run_argsort(gpu, "all -inf",      50000, 2, GK_SORT_ORDER_DESC, TK_ALL_NEG_INF);
        failures += run_argsort(gpu, "some -inf",     50000, 2, GK_SORT_ORDER_ASC,  TK_SOME_NEG_INF);

        // a real vocabulary row, which is the shape the sampler argsorts
        failures += run_argsort(gpu, "vocab 262144",  262144, 1, GK_SORT_ORDER_DESC, TK_DISTINCT);
    }

    printf("composite graphs:\n");
    failures += run_op_tol(gpu, "vae stack",      build_vae_stack, 4e-2f);
    failures += run_op_tol(gpu, "transformer",    build_transformer_block, 4e-3f);

    gk_backend_free(gpu);

    printf("%d failures across %d weight types\n", failures, n_types);
    return failures == 0 ? 0 : 1;
}
