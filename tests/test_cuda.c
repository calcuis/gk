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

static int run_op(gk_backend_t gpu, const char * name, op_builder build) {
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
        float * got = (float *) malloc((size_t) no * sizeof(float));
        if (got == NULL) {
            return 1;
        }
        gk_backend_tensor_get(gpu_out, got, 0, (size_t) no * sizeof(float));

        const float * expected = (const float *) cpu_out->data;
        for (int64_t i = 0; i < no; ++i) {
            const float diff = fabsf(got[i] - expected[i]);
            if (diff > max_abs) {
                max_abs = diff;
            }
            if (!(diff <= 1e-4f + 1e-4f * fabsf(expected[i]))) {
                bad++;
            }
        }
        free(got);
    }

    printf("  %-14s %5lld outputs, max abs error %.8g, %d mismatches%s\n",
           name, (long long) no, max_abs, bad, bad == 0 ? "" : "  FAIL");

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

    gk_backend_free(gpu);

    printf("%d failures across %d weight types\n", failures, n_types);
    return failures == 0 ? 0 : 1;
}
