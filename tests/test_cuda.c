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

    gk_backend_free(gpu);

    printf("%d failures across %d weight types\n", failures, n_types);
    return failures == 0 ? 0 : 1;
}
