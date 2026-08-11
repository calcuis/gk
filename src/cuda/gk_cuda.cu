// The CUDA/HIP backend: devices, memory, and running a graph on them.
//
// The kernels are next door; this file is the part that makes them reachable
// through gk's backend interface - what a buffer of device memory is, how a
// tensor's bytes get in and out of one, and what happens when the scheduler
// hands over a range of nodes.
//
// Three things here are worth stating outright because they are the whole
// difference between this and the CPU backend:
//
//   * Device memory is not host memory. A tensor in it has a `data` pointer
//     that must never be dereferenced on the host, which is why gk's
//     tensor_set/tensor_get exist and why the engine goes through them.
//
//   * Work is queued, not done. Every kernel below is launched on a stream and
//     returns immediately; nothing may read a result before `synchronize`.
//     The scheduler knows this and synchronizes before it copies across a
//     split boundary.
//
//   * One stream per backend, and a backend belongs to one device. Two
//     backends on the same device share its memory and can copy between
//     themselves without going through the host; two on different devices can
//     too, when the driver says the pair can see each other.

#include "gk_cuda_ops.cuh"

#include <stdlib.h>
#include <string.h>

#include <chrono>

// Device memory is handed out in multiples of this. Coalesced loads want their
// rows aligned and the driver's own allocations are far coarser than this
// anyway, so it costs nothing to promise.
#define GK_CUDA_ALIGN 128

// --------------------------------------------------------------------------
// per-device state
//
// Built once, at discovery, and shared by every backend and buffer on that
// device. The buffer types in particular have to be stable addresses: the
// scheduler compares buffer types by pointer to decide what lives where.
// --------------------------------------------------------------------------

struct gk_cuda_device_ctx {
    int  index;
    char name[32];
    char description[256];

    size_t total_memory;
    bool   integrated;
    int    n_sm;         // multiprocessors; a launcher sizing a grid wants it
    int    cc;           // compute capability; the tensor-core path needs 8.0
    int    smem_max;     // shared memory a block may opt in to, past the 48 KB default

    // What gk_device_features hands back. Built once at registration and kept
    // here rather than formatted on demand, because the caller is promised
    // pointers that outlive the call.
    struct gk_feature features[6];
    char              cc_str[16];
    char              n_sm_str[16];
    char              smem_str[16];

    struct gk_backend_buffer_type buft;
    struct gk_backend_buffer_type host_buft;
    struct gk_device              device;
};

static struct gk_cuda_device_ctx g_cuda_devices[GK_CUDA_MAX_DEVICES];
static int  g_cuda_n_devices;
static bool g_cuda_discovered;

// --------------------------------------------------------------------------
// device buffers
// --------------------------------------------------------------------------

struct gk_cuda_buffer_ctx {
    int    device;
    void * base;
    bool   owned; // false for a buffer wrapping memory somebody else allocated
};

static void gk_cuda_buffer_free(gk_backend_buffer_t buffer) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;
    if (ctx == NULL) {
        return;
    }
    if (ctx->owned) {
        GK_CUDA_CHECK(gkSetDevice(ctx->device));
        GK_CUDA_CHECK(gkFree(ctx->base));
    }
    free(ctx);
}

static void * gk_cuda_buffer_get_base(gk_backend_buffer_t buffer) {
    return ((struct gk_cuda_buffer_ctx *) buffer->context)->base;
}

static void gk_cuda_buffer_set_tensor(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                      const void * data, size_t offset, size_t size) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->device));
    // Synchronous by contract: the caller may free `data` the moment this
    // returns, so the copy cannot be left in flight.
    //
    // The wait is on the per-thread stream rather than the legacy default one,
    // and that is the whole point. A plain gkMemcpy is ordered against every
    // other stream on the device, so a model loader filling tensors from
    // several threads has them queue behind one another instead of overlapping.
    // Same contract, same cost for a single thread, several times the
    // throughput for a loader that uses more than one.
    GK_CUDA_CHECK(gkMemcpyAsync((char *) tensor->data + offset, data, size,
                                gkMemcpyHostToDevice, gkStreamPerThread));
    GK_CUDA_CHECK(gkStreamSynchronize(gkStreamPerThread));
}

static void gk_cuda_buffer_get_tensor(gk_backend_buffer_t buffer, const struct gk_tensor * tensor,
                                      void * data, size_t offset, size_t size) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->device));
    GK_CUDA_CHECK(gkMemcpyAsync(data, (const char *) tensor->data + offset, size,
                                gkMemcpyDeviceToHost, gkStreamPerThread));
    GK_CUDA_CHECK(gkStreamSynchronize(gkStreamPerThread));
}

static void gk_cuda_buffer_clear(gk_backend_buffer_t buffer, uint8_t value) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->device));
    GK_CUDA_CHECK(gkMemset(ctx->base, value, buffer->size));
}

static void gk_cuda_buffer_memset_tensor(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                         uint8_t value, size_t offset, size_t size) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->device));
    GK_CUDA_CHECK(gkMemset((char *) tensor->data + offset, value, size));
}

// The device-to-device path. Same device: a plain device copy. Different
// devices: a peer copy, if the driver has told us the pair can see each other.
// Anything else returns false and the caller stages it through the host.
static bool gk_cuda_buffer_cpy_tensor(gk_backend_buffer_t buffer, const struct gk_tensor * src,
                                      struct gk_tensor * dst) {
    struct gk_cuda_buffer_ctx * dctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    if (src->buffer == NULL) {
        return false;
    }

    const gk_backend_buffer_type_t src_buft = gk_backend_buffer_get_type(src->buffer);

    // Pinned host memory is readable by the device directly, so a copy out of
    // one is a host-to-device copy rather than a staged round trip.
    if (gk_backend_buft_is_host(src_buft)) {
        GK_CUDA_CHECK(gkSetDevice(dctx->device));
        GK_CUDA_CHECK(gkMemcpy(dst->data, src->data, gk_nbytes(src), gkMemcpyHostToDevice));
        return true;
    }

    // Is the source one of ours?
    int src_device = -1;
    for (int i = 0; i < g_cuda_n_devices; ++i) {
        if (&g_cuda_devices[i].buft == src_buft) {
            src_device = g_cuda_devices[i].index;
            break;
        }
    }
    if (src_device < 0) {
        return false;
    }

    if (src_device == dctx->device) {
        GK_CUDA_CHECK(gkSetDevice(dctx->device));
        GK_CUDA_CHECK(gkMemcpy(dst->data, src->data, gk_nbytes(src), gkMemcpyDeviceToDevice));
        return true;
    }

    int can_peer = 0;
    GK_CUDA_CHECK(gkDeviceCanAccessPeer(&can_peer, dctx->device, src_device));
    if (!can_peer) {
        return false;
    }

    GK_CUDA_CHECK(gkSetDevice(dctx->device));
    GK_CUDA_CHECK(gkMemcpyPeerAsync(dst->data, dctx->device, src->data, src_device,
                                    gk_nbytes(src), 0));
    GK_CUDA_CHECK(gkStreamSynchronize(0));
    return true;
}

static const struct gk_backend_buffer_i g_cuda_buffer_iface = {
    /* .free_buffer   = */ gk_cuda_buffer_free,
    /* .get_base      = */ gk_cuda_buffer_get_base,
    /* .init_tensor   = */ NULL, // nothing per tensor; the pointer is enough
    /* .set_tensor    = */ gk_cuda_buffer_set_tensor,
    /* .get_tensor    = */ gk_cuda_buffer_get_tensor,
    /* .clear         = */ gk_cuda_buffer_clear,
    /* .memset_tensor = */ gk_cuda_buffer_memset_tensor,
    /* .cpy_tensor    = */ gk_cuda_buffer_cpy_tensor,
};

// --------------------------------------------------------------------------
// the device buffer type
// --------------------------------------------------------------------------

static const char * gk_cuda_buft_name(gk_backend_buffer_type_t buft) {
    return ((struct gk_cuda_device_ctx *) buft->context)->name;
}

static gk_backend_buffer_t gk_cuda_buft_alloc(gk_backend_buffer_type_t buft, size_t size) {
    struct gk_cuda_device_ctx * dev = (struct gk_cuda_device_ctx *) buft->context;

    struct gk_cuda_buffer_ctx * ctx =
        (struct gk_cuda_buffer_ctx *) malloc(sizeof(struct gk_cuda_buffer_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->device = dev->index;
    ctx->owned  = true;

    GK_CUDA_CHECK(gkSetDevice(dev->index));

    const gkError_t err = gkMalloc(&ctx->base, size);
    if (err != gkSuccess) {
        gk_logf("gk %s: failed to allocate %zu bytes on %s: %s\n",
                GK_CUDA_BACKEND_NAME, size, dev->name, gkGetErrorString(err));
        free(ctx);
        return NULL;
    }

    gk_backend_buffer_t buffer = gk_backend_buffer_init(buft, &g_cuda_buffer_iface, ctx, size);
    if (buffer == NULL) {
        GK_CUDA_CHECK(gkFree(ctx->base));
        free(ctx);
        return NULL;
    }

    return buffer;
}

static size_t gk_cuda_buft_alignment(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return GK_CUDA_ALIGN;
}

static const struct gk_backend_buffer_type_i g_cuda_buft_iface = {
    /* .get_name       = */ gk_cuda_buft_name,
    /* .alloc_buffer   = */ gk_cuda_buft_alloc,
    /* .get_alignment  = */ gk_cuda_buft_alignment,
    /* .get_alloc_size = */ NULL, // a tensor costs its own bytes here
    /* .is_host        = */ NULL, // device memory: the default false is right
};

// --------------------------------------------------------------------------
// pinned host memory
//
// Allocated by the device, addressable by the host, and copied from without
// the driver having to stage it - which is what makes it worth a separate
// buffer type rather than plain malloc. The engine puts the KV cache overflow
// and the input tensors here when it wants both sides to reach them.
// --------------------------------------------------------------------------

struct gk_cuda_host_buffer_ctx {
    void * base;
};

static void gk_cuda_host_buffer_free(gk_backend_buffer_t buffer) {
    struct gk_cuda_host_buffer_ctx * ctx = (struct gk_cuda_host_buffer_ctx *) buffer->context;
    if (ctx == NULL) {
        return;
    }
    GK_CUDA_CHECK(gkFreeHost(ctx->base));
    free(ctx);
}

static void * gk_cuda_host_buffer_get_base(gk_backend_buffer_t buffer) {
    return ((struct gk_cuda_host_buffer_ctx *) buffer->context)->base;
}

static void gk_cuda_host_buffer_set(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                    const void * data, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void gk_cuda_host_buffer_get(gk_backend_buffer_t buffer, const struct gk_tensor * tensor,
                                    void * data, size_t offset, size_t size) {
    GK_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void gk_cuda_host_buffer_clear(gk_backend_buffer_t buffer, uint8_t value) {
    struct gk_cuda_host_buffer_ctx * ctx = (struct gk_cuda_host_buffer_ctx *) buffer->context;
    memset(ctx->base, value, buffer->size);
}

static const struct gk_backend_buffer_i g_cuda_host_buffer_iface = {
    /* .free_buffer   = */ gk_cuda_host_buffer_free,
    /* .get_base      = */ gk_cuda_host_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .set_tensor    = */ gk_cuda_host_buffer_set,
    /* .get_tensor    = */ gk_cuda_host_buffer_get,
    /* .clear         = */ gk_cuda_host_buffer_clear,
    /* .memset_tensor = */ NULL,
    /* .cpy_tensor    = */ NULL,
};

static const char * gk_cuda_host_buft_name(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return GK_CUDA_BACKEND_NAME "_Host";
}

static gk_backend_buffer_t gk_cuda_host_buft_alloc(gk_backend_buffer_type_t buft, size_t size) {
    struct gk_cuda_device_ctx * dev = (struct gk_cuda_device_ctx *) buft->context;

    struct gk_cuda_host_buffer_ctx * ctx =
        (struct gk_cuda_host_buffer_ctx *) malloc(sizeof(struct gk_cuda_host_buffer_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    GK_CUDA_CHECK(gkSetDevice(dev->index));

    const gkError_t err = gkHostAlloc(&ctx->base, size);
    if (err != gkSuccess) {
        // Pinned memory is a finite resource and running out of it is not
        // fatal - ordinary host memory works, just slower - so this is a
        // warning and a NULL rather than an abort.
        gk_logf("gk %s: could not pin %zu bytes of host memory: %s\n",
                GK_CUDA_BACKEND_NAME, size, gkGetErrorString(err));
        free(ctx);
        return NULL;
    }

    return gk_backend_buffer_init(buft, &g_cuda_host_buffer_iface, ctx, size);
}

static size_t gk_cuda_host_buft_alignment(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return GK_MEM_ALIGN;
}

static bool gk_cuda_host_buft_is_host(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return true;
}

static const struct gk_backend_buffer_type_i g_cuda_host_buft_iface = {
    /* .get_name       = */ gk_cuda_host_buft_name,
    /* .alloc_buffer   = */ gk_cuda_host_buft_alloc,
    /* .get_alignment  = */ gk_cuda_host_buft_alignment,
    /* .get_alloc_size = */ NULL,
    /* .is_host        = */ gk_cuda_host_buft_is_host,
};

// --------------------------------------------------------------------------
// the backend
// --------------------------------------------------------------------------

struct gk_cuda_backend_ctx {
    struct gk_cuda_device_ctx * dev;
    gkStream_t                  stream;
    struct gk_cuda_scratch      scratch;
};

static const char * gk_cuda_backend_name(gk_backend_t backend) {
    return ((struct gk_cuda_backend_ctx *) backend->context)->dev->name;
}

static void gk_cuda_backend_free(gk_backend_t backend) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;
    if (ctx != NULL) {
        GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));
        GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
        // After the wait, so nothing is still reading it.
        if (ctx->scratch.ptr != NULL) {
            GK_CUDA_CHECK(gkFree(ctx->scratch.ptr));
        }
        GK_CUDA_CHECK(gkStreamDestroy(ctx->stream));
        free(ctx);
    }
    free(backend);
}

static gk_backend_buffer_type_t gk_cuda_backend_buft(gk_backend_t backend) {
    return &((struct gk_cuda_backend_ctx *) backend->context)->dev->buft;
}

// --------------------------------------------------------------------------
// per-op profile
//
// What the scheduler report is for placement, this is for time. GK_SCHED_REPORT
// answers "which backend ran this op"; this answers "where did the graph's
// milliseconds go", which is the only question worth asking once every op is
// already on the device and the whole thing is still slower than it should be.
//
// Set GK_OP_PROFILE in the environment. Every node is timed by synchronizing
// the stream around its launch - that serializes what would otherwise overlap,
// so the total reads high, but a graph this deep is nearly serial anyway and
// the shares are what the number is for. Rows are keyed by op and by the shape
// that op saw, because "mul_mat is expensive" is not actionable and "mul_mat,
// f16 weights, 2880x4096" is.
// --------------------------------------------------------------------------

#define GK_CU_PROF_MAX 512

struct gk_cu_prof_row {
    char   key[96];
    double ms;
    int64_t calls;
    double flops;   // multiply-accumulates, for the shapes where it means something
};

static struct gk_cu_prof_row g_prof[GK_CU_PROF_MAX];
static int                   g_prof_n       = 0;
static int                   g_prof_enabled = -1;
static double                g_prof_total   = 0.0;

static int gk_cu_prof_cmp(const void * a, const void * b) {
    const double x = ((const struct gk_cu_prof_row *) a)->ms;
    const double y = ((const struct gk_cu_prof_row *) b)->ms;
    return x < y ? 1 : (x > y ? -1 : 0);
}

struct gk_cu_scratch_stats g_gk_scratch_stats;

static void gk_cu_prof_dump(void) {
    if (g_prof_n == 0) {
        return;
    }

    extern double g_gk_mm_quant_ms;
    extern double g_gk_mm_tile_ms;
    gk_logf("\ngk cuda nvfp4 mma: %.1f ms quantizing activations, %.1f ms in the tile\n",
            g_gk_mm_quant_ms, g_gk_mm_tile_ms);

    {
        // GK_MM_FP4_STATS, if it was on. Zero counters mean it was not.
        double sq_err = 0.0, sq_ref = 0.0;
        unsigned long long zero = 0, groups = 0;

        gk_cuda_fp4_stats(&sq_err, &sq_ref, &zero, &groups);

        if (groups != 0) {
            gk_logf("\ngk cuda nvfp4 activations: rms error %.4f%% of signal, "
                    "%llu of %llu groups of 16 zeroed by scale underflow (%.3f%%)\n",
                    100.0 * (sq_ref > 0.0 ? sqrt(sq_err / sq_ref) : 0.0),
                    (unsigned long long) zero, (unsigned long long) groups,
                    100.0 * (double) zero / (double) groups);
        }
    }

    gk_logf("\ngk cuda scratch: %lld calls, %lld grows (%lld failed), %.1f ms in the grow path\n",
            (long long) g_gk_scratch_stats.calls, (long long) g_gk_scratch_stats.grows,
            (long long) g_gk_scratch_stats.fails, g_gk_scratch_stats.grow_ms);

    qsort(g_prof, (size_t) g_prof_n, sizeof(g_prof[0]), gk_cu_prof_cmp);

    gk_logf("\ngk cuda profile: %.1f ms over %d distinct shapes\n", g_prof_total, g_prof_n);
    gk_logf("  %-62s %10s %7s %8s %10s\n", "op / shape", "ms", "%", "calls", "GFLOP/s");

    for (int i = 0; i < g_prof_n; ++i) {
        char rate[16] = "-";
        if (g_prof[i].flops > 0.0 && g_prof[i].ms > 0.0) {
            snprintf(rate, sizeof(rate), "%.0f", g_prof[i].flops * 2.0 / (g_prof[i].ms * 1e6));
        }
        gk_logf("  %-62s %10.2f %6.1f%% %8lld %10s\n",
                g_prof[i].key, g_prof[i].ms,
                100.0 * g_prof[i].ms / g_prof_total,
                (long long) g_prof[i].calls, rate);
    }
}

static void gk_cu_prof_add(const char * key, double ms, double flops) {
    for (int i = 0; i < g_prof_n; ++i) {
        if (strcmp(g_prof[i].key, key) == 0) {
            g_prof[i].ms    += ms;
            g_prof[i].flops += flops;
            g_prof[i].calls++;
            g_prof_total    += ms;
            return;
        }
    }

    if (g_prof_n >= GK_CU_PROF_MAX) {
        return;
    }

    struct gk_cu_prof_row * row = &g_prof[g_prof_n++];
    snprintf(row->key, sizeof(row->key), "%s", key);
    row->ms    = ms;
    row->flops = flops;
    row->calls = 1;
    g_prof_total += ms;
}

// The name a row gets, and the work it did. Only the ops whose cost depends on
// more than the output size say anything beyond their shape.
static void gk_cu_prof_key(const struct gk_tensor * node, char * out, size_t out_size,
                           double * flops) {
    *flops = 0.0;

    switch ((int) node->op) {
        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID: {
            const struct gk_tensor * a = node->src[0];
            const struct gk_tensor * b = node->src[1];
            const int64_t k = a->ne[0];
            const int64_t m = node->ne[0];
            const int64_t n = node->ne[1] * node->ne[2] * node->ne[3];
            snprintf(out, out_size, "%s %-6s %lldx%lldx%lld [%s]",
                     gk_op_name(node->op), gk_type_name(a->type),
                     (long long) m, (long long) n, (long long) k,
                     node->op == GK_OP_MUL_MAT ? gk_cuda_mm_last_path() : "-");
            *flops = (double) m * (double) n * (double) k;
            GK_UNUSED(b);
            break;
        }
        case GK_OP_FLASH_ATTN_EXT: {
            const struct gk_tensor * q = node->src[0];
            const struct gk_tensor * k = node->src[1];
            snprintf(out, out_size, "flash_attn d=%lld q=%lld kv=%lld h=%lld",
                     (long long) q->ne[0], (long long) q->ne[1],
                     (long long) k->ne[1], (long long) q->ne[2]);
            *flops = 2.0 * (double) q->ne[0] * (double) q->ne[1] *
                     (double) k->ne[1] * (double) q->ne[2];
            break;
        }
        case GK_OP_IM2COL: {
            snprintf(out, out_size, "im2col %lldx%lldx%lld",
                     (long long) node->ne[0], (long long) node->ne[1],
                     (long long) node->ne[2]);
            break;
        }
        default:
            snprintf(out, out_size, "%s %lldx%lldx%lldx%lld",
                     gk_op_name(node->op),
                     (long long) node->ne[0], (long long) node->ne[1],
                     (long long) node->ne[2], (long long) node->ne[3]);
            break;
    }
}

static bool gk_cu_prof_on(void) {
    if (g_prof_enabled < 0) {
        const char * e = getenv("GK_OP_PROFILE");
        g_prof_enabled = e != NULL && e[0] != '0';
        if (g_prof_enabled) {
            atexit(gk_cu_prof_dump);
        }
    }
    return g_prof_enabled != 0;
}

static enum gk_status gk_cuda_backend_compute(gk_backend_t backend, struct gk_cgraph * graph) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));

    // The per-node check below reads the thread's error flag, and that flag
    // carries whatever the last unchecked runtime call left in it - from
    // anywhere, including code that has nothing to do with this graph. Clear it
    // once here so that what the loop reports is this graph's, not a stale
    // error attributed to node 0 because node 0 happened to look next.
    (void) gkGetLastError();

    const int  n    = gk_graph_n_nodes(graph);
    const bool prof = gk_cu_prof_on();

    for (int i = 0; i < n; ++i) {
        struct gk_tensor * node = gk_graph_node(graph, i);

        std::chrono::steady_clock::time_point t0;
        if (prof) {
            GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
            t0 = std::chrono::steady_clock::now();
        }

        if (!gk_cuda_compute_op(ctx->stream, &ctx->scratch, node)) {
            gk_logf("gk %s: no kernel for op %s (node %s)\n",
                    GK_CUDA_BACKEND_NAME, gk_op_name(node->op), node->name);
            return GK_STATUS_NO_STORAGE;
        }

        // A launch is rejected synchronously when its geometry is wrong, so the
        // check belongs next to the launch that caused it: checking once at the
        // end of the graph would name whichever node happened to be queued last
        // and leave the real one unnamed. It is a host-side flag read, not a
        // synchronization - the queue keeps running behind it.
        const gkError_t err = gkGetLastError();
        if (err != gkSuccess) {
            gk_logf("gk %s: %s (node %s, op %s, ne = [%lld %lld %lld %lld])\n",
                    GK_CUDA_BACKEND_NAME, gkGetErrorString(err),
                    node->name, gk_op_name(node->op),
                    (long long) node->ne[0], (long long) node->ne[1],
                    (long long) node->ne[2], (long long) node->ne[3]);
            return GK_STATUS_NO_STORAGE;
        }

        if (prof) {
            GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
            const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

            char   key[96];
            double flops = 0.0;
            gk_cu_prof_key(node, key, sizeof(key), &flops);
            gk_cu_prof_add(key,
                           std::chrono::duration<double, std::milli>(t1 - t0).count(),
                           flops);
        }
    }

    return GK_STATUS_SUCCESS;
}

static bool gk_cuda_backend_supports_op(gk_backend_t backend, const struct gk_tensor * op) {
    GK_UNUSED(backend);
    return gk_cuda_supports_op(op);
}

static bool gk_cuda_backend_supports_buft(gk_backend_t backend, gk_backend_buffer_type_t buft) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    // Its own memory, and any pinned host memory - which the device can read
    // directly, so a tensor there needs no staging.
    return buft == &ctx->dev->buft || buft == &ctx->dev->host_buft;
}

// Whether it is worth pulling an op here that would otherwise run elsewhere.
// The trade is one activation's worth of transfer against the op's work, so
// the answer is yes only for the ops whose work grows with the batch: a matmul
// over many tokens pays the copy back, an elementwise add never does.
static bool gk_cuda_backend_offload_op(gk_backend_t backend, const struct gk_tensor * op) {
    GK_UNUSED(backend);

    const int64_t min_batch = 32;

    switch ((int) op->op) {
        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID:
            return op->src[1]->ne[1] >= min_batch;
        case GK_OP_FLASH_ATTN_EXT:
            return op->src[0]->ne[1] >= min_batch;
        default:
            return false;
    }
}

static void gk_cuda_backend_synchronize(gk_backend_t backend) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));
    GK_CUDA_CHECK(gkStreamSynchronize(ctx->stream));
}

static void gk_cuda_backend_set_async(gk_backend_t backend, struct gk_tensor * tensor,
                                      const void * data, size_t offset, size_t size) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));
    GK_CUDA_CHECK(gkMemcpyAsync((char *) tensor->data + offset, data, size,
                                gkMemcpyHostToDevice, ctx->stream));
}

static void gk_cuda_backend_get_async(gk_backend_t backend, const struct gk_tensor * tensor,
                                      void * data, size_t offset, size_t size) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));
    GK_CUDA_CHECK(gkMemcpyAsync(data, (const char *) tensor->data + offset, size,
                                gkMemcpyDeviceToHost, ctx->stream));
}

static const struct gk_backend_i g_cuda_backend_iface = {
    /* .get_name                = */ gk_cuda_backend_name,
    /* .free                    = */ gk_cuda_backend_free,
    /* .get_default_buffer_type = */ gk_cuda_backend_buft,
    /* .graph_compute           = */ gk_cuda_backend_compute,
    /* .supports_op             = */ gk_cuda_backend_supports_op,
    /* .supports_buft           = */ gk_cuda_backend_supports_buft,
    /* .offload_op              = */ gk_cuda_backend_offload_op,
    /* .synchronize             = */ gk_cuda_backend_synchronize,
    /* .set_tensor_async        = */ gk_cuda_backend_set_async,
    /* .get_tensor_async        = */ gk_cuda_backend_get_async,
};

// --------------------------------------------------------------------------
// the device vtable
// --------------------------------------------------------------------------

static const char * gk_cuda_device_name(gk_device_t dev) {
    return ((struct gk_cuda_device_ctx *) dev->context)->name;
}

static const char * gk_cuda_device_description(gk_device_t dev) {
    return ((struct gk_cuda_device_ctx *) dev->context)->description;
}

static void gk_cuda_device_memory(gk_device_t dev, size_t * free_out, size_t * total_out) {
    struct gk_cuda_device_ctx * ctx = (struct gk_cuda_device_ctx *) dev->context;

    size_t free_mem = 0, total_mem = ctx->total_memory;

    GK_CUDA_CHECK(gkSetDevice(ctx->index));
    GK_CUDA_CHECK(gkMemGetInfo(&free_mem, &total_mem));

    if (free_out  != NULL) { *free_out  = free_mem; }
    if (total_out != NULL) { *total_out = total_mem; }
}

static enum gk_device_type gk_cuda_device_type(gk_device_t dev) {
    struct gk_cuda_device_ctx * ctx = (struct gk_cuda_device_ctx *) dev->context;
    // An integrated part shares the host's memory, and the scheduler treats
    // that differently: there is no bus to avoid crossing.
    return ctx->integrated ? GK_DEVICE_TYPE_IGPU : GK_DEVICE_TYPE_GPU;
}

static gk_backend_t gk_cuda_device_init_backend(gk_device_t dev) {
    struct gk_cuda_device_ctx * d = (struct gk_cuda_device_ctx *) dev->context;

    struct gk_cuda_backend_ctx * ctx =
        (struct gk_cuda_backend_ctx *) malloc(sizeof(struct gk_cuda_backend_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->dev          = d;
    ctx->scratch.ptr  = NULL;
    ctx->scratch.size = 0;
    ctx->scratch.n_sm = d->n_sm;
    ctx->scratch.cc   = d->cc;
    ctx->scratch.smem_max = d->smem_max;

    GK_CUDA_CHECK(gkSetDevice(d->index));

    const gkError_t err = gkStreamCreate(&ctx->stream);
    if (err != gkSuccess) {
        gk_logf("gk %s: could not create a stream on %s: %s\n",
                GK_CUDA_BACKEND_NAME, d->name, gkGetErrorString(err));
        free(ctx);
        return NULL;
    }

    gk_backend_t backend = (gk_backend_t) malloc(sizeof(struct gk_backend));
    if (backend == NULL) {
        GK_CUDA_CHECK(gkStreamDestroy(ctx->stream));
        free(ctx);
        return NULL;
    }

    backend->iface   = g_cuda_backend_iface;
    backend->context = ctx;
    backend->device  = &d->device;

    return backend;
}

static gk_backend_buffer_type_t gk_cuda_device_buft(gk_device_t dev) {
    return &((struct gk_cuda_device_ctx *) dev->context)->buft;
}

static gk_backend_buffer_type_t gk_cuda_device_host_buft(gk_device_t dev) {
    return &((struct gk_cuda_device_ctx *) dev->context)->host_buft;
}

static bool gk_cuda_device_supports_op(gk_device_t dev, const struct gk_tensor * op) {
    GK_UNUSED(dev);
    return gk_cuda_supports_op(op);
}

static bool gk_cuda_device_supports_buft(gk_device_t dev, gk_backend_buffer_type_t buft) {
    struct gk_cuda_device_ctx * ctx = (struct gk_cuda_device_ctx *) dev->context;
    return buft == &ctx->buft || buft == &ctx->host_buft;
}

static bool gk_cuda_device_offload_op(gk_device_t dev, const struct gk_tensor * op) {
    GK_UNUSED(dev);
    return gk_cuda_backend_offload_op(NULL, op);
}

// What this binary was compiled for, filled in by the build.
#ifndef GK_CUDA_ARCH_LIST
#define GK_CUDA_ARCH_LIST "unknown"
#endif

static const struct gk_feature * gk_cuda_device_features(gk_device_t dev) {
    return ((struct gk_cuda_device_ctx *) dev->context)->features;
}

// Filled at registration, once the properties are known. "built for" is the
// build's list rather than this device's own capability on purpose: a device
// running on a nearer arch than it was compiled for is the usual reason a GPU
// is slower than it should be, and the two numbers side by side say so.
static void gk_cuda_device_fill_features(struct gk_cuda_device_ctx * d) {
    snprintf(d->cc_str,   sizeof(d->cc_str),   "%d.%d", d->cc / 10, d->cc % 10);
    snprintf(d->n_sm_str, sizeof(d->n_sm_str), "%d",    d->n_sm);
    snprintf(d->smem_str, sizeof(d->smem_str), "%d KiB", d->smem_max / 1024);

    int n = 0;
    #define GK_CUDA_FEATURE(nm, val)                                             \
        do {                                                                     \
            GK_ASSERT(n < (int) (sizeof(d->features) / sizeof(d->features[0]))); \
            d->features[n].name  = (nm);                                         \
            d->features[n].value = (val);                                        \
            n++;                                                                 \
        } while (0)

    GK_CUDA_FEATURE("compute capability", d->cc_str);
    GK_CUDA_FEATURE("SMs",                d->n_sm_str);
    GK_CUDA_FEATURE("shared memory",      d->smem_str);
    GK_CUDA_FEATURE("built for",          GK_CUDA_ARCH_LIST);
    if (d->integrated) {
        GK_CUDA_FEATURE("integrated", "1");
    }
    GK_CUDA_FEATURE(NULL, NULL);

    #undef GK_CUDA_FEATURE
}

static const struct gk_device_i g_cuda_device_iface = {
    /* .get_name             = */ gk_cuda_device_name,
    /* .get_description      = */ gk_cuda_device_description,
    /* .get_memory           = */ gk_cuda_device_memory,
    /* .get_type             = */ gk_cuda_device_type,
    /* .init_backend         = */ gk_cuda_device_init_backend,
    /* .buffer_type          = */ gk_cuda_device_buft,
    /* .host_buffer_type     = */ gk_cuda_device_host_buft,
    /* .buffer_from_host_ptr = */ NULL, // mapping arbitrary host pages is not
                                        // something this backend does; the
                                        // loader copies weights in instead
    /* .supports_op          = */ gk_cuda_device_supports_op,
    /* .supports_buft        = */ gk_cuda_device_supports_buft,
    /* .offload_op           = */ gk_cuda_device_offload_op,
    /* .get_features         = */ gk_cuda_device_features,
};

// --------------------------------------------------------------------------
// discovery
// --------------------------------------------------------------------------

// Launched at discovery to find out whether the binary actually contains code
// this device can run. It does nothing; the launch itself is the question.
static __global__ void gk_cuda_probe_kernel(void) {}

// Whether a device can run any of the code in this binary.
//
// A build carries SASS for the architectures it was told about and PTX for at
// most one of them. Put a card in the machine that is newer than both and every
// launch on it fails with "no kernel image is available for execution on the
// device" - the first launch and every one after, because nothing about it is
// transient. It costs one empty launch to find that out here instead, before
// the device has been registered and before the scheduler has put a graph on
// it, and the difference in what the user sees is the whole point: a named
// device and a rebuild flag at startup, rather than a failed generation with a
// node number in it.
//
// The launch is on the default stream and is checked synchronously, because a
// rejected launch reports through gkGetLastError immediately.
static bool gk_cuda_device_has_kernel_image(int index) {
    if (gkSetDevice(index) != gkSuccess) {
        return false;
    }

    // Whatever came before is not this launch's fault; the flag is per-thread
    // and sticky until read.
    (void) gkGetLastError();

    gk_cuda_probe_kernel<<<1, 1>>>();

    const gkError_t err = gkGetLastError();
    if (err == gkSuccess) {
        return true;
    }

    if (err != gkErrorNoKernelImage) {
        // Something else is wrong with the device - out of memory at context
        // creation, a driver that has fallen over. Not this function's
        // question, and not a reason to claim the binary is at fault, so let
        // the device register and fail with its own error where it happens.
        return true;
    }

    return false;
}

// Whether this binary carries SASS a device of capability `cc` can run.
//
// A cubin runs on any part of the same major version with a minor at least as
// high as its own: sm_80 code runs on an sm_89 card, sm_89 code does not run on
// sm_86 and nothing crosses a major boundary. Anything else the device has to
// JIT from PTX.
static bool gk_cuda_arch_list_has_sass_for(int cc) {
    const char * p = GK_CUDA_ARCH_LIST;

    while (*p) {
        if (*p < '0' || *p > '9') {
            p++;
            continue;
        }

        int arch = 0;
        while (*p >= '0' && *p <= '9') {
            arch = arch * 10 + (*p - '0');
            p++;
        }

        if (arch / 10 == cc / 10 && arch % 10 <= cc % 10) {
            return true;
        }
    }

    return false;
}

extern "C" void gk_cuda_register_devices(void) {
    if (g_cuda_discovered) {
        return;
    }
    g_cuda_discovered = true;

    int count = 0;
    const gkError_t err = gkGetDeviceCount(&count);
    if (err != gkSuccess) {
        // No driver, no devices, or a driver too old for this build: all of
        // them mean "this machine has none", which is not an error - the CPU
        // backend is always there.
        return;
    }

    if (count > GK_CUDA_MAX_DEVICES) {
        count = GK_CUDA_MAX_DEVICES;
    }

    for (int i = 0; i < count; ++i) {
        gkDeviceProp_t prop;
        if (gkGetDeviceProperties(&prop, i) != gkSuccess) {
            continue;
        }

        if (!gk_cuda_device_has_kernel_image(i)) {
            // Registering it would mean the scheduler eventually places work
            // on it, and every one of those launches fails. Leaving it out
            // costs the device and keeps the run: the remaining GPUs, or the
            // CPU, take the graph instead.
            gk_logf("gk %s: skipping device %d (%s, compute capability %d.%d) - this build has "
                    "no code for it (built for: %s). Rebuild with "
                    "-DGK_CUDA_ARCHITECTURES=%d to use it.\n",
                    GK_CUDA_BACKEND_NAME, i, prop.name, prop.major, prop.minor,
                    GK_CUDA_ARCH_LIST, prop.major * 10 + prop.minor);
            continue;
        }

        if (!gk_cuda_arch_list_has_sass_for(prop.major * 10 + prop.minor)) {
            // The probe launch succeeded, so the driver can JIT this build's
            // PTX for the device - but it has to compile every kernel the run
            // touches before running it, and for a graph the size of a
            // diffusion model that is minutes of apparently frozen process
            // with no output between "loading tensors completed" and the first
            // step. It is cached afterwards (%APPDATA%\NVIDIA\ComputeCache,
            // ~/.nv/ComputeCache), so the run after this one looks fine and
            // nothing about it looks like a build problem. Say so once, here,
            // where the cost is about to be paid.
            gk_logf("gk %s: device %d (%s, compute capability %d.%d) has no native code in this "
                    "build (built for: %s); it will run on JIT-compiled PTX. The first launch "
                    "on it can take minutes before anything appears to happen - the driver "
                    "caches the result, so later runs start normally. Rebuild with "
                    "-DGK_CUDA_ARCHITECTURES=%d to avoid it.\n",
                    GK_CUDA_BACKEND_NAME, i, prop.name, prop.major, prop.minor,
                    GK_CUDA_ARCH_LIST, prop.major * 10 + prop.minor);
        }

        struct gk_cuda_device_ctx * d = &g_cuda_devices[g_cuda_n_devices];
        memset(d, 0, sizeof(*d));

        d->index        = i;
        d->total_memory = prop.totalGlobalMem;
        d->integrated   = prop.integrated != 0;
        d->n_sm         = prop.multiProcessorCount;
        d->cc           = prop.major * 10 + prop.minor;
        // Every part gives a block 48 KB without asking; Ampere and later will
        // give most of the multiprocessor's store to one block on request.
        d->smem_max     = (int) prop.sharedMemPerBlockOptin;
        if (d->smem_max < (int) prop.sharedMemPerBlock) {
            d->smem_max = (int) prop.sharedMemPerBlock;
        }

        snprintf(d->name, sizeof(d->name), "%s%d", GK_CUDA_BACKEND_NAME, i);
        snprintf(d->description, sizeof(d->description), "%s", prop.name);

        gk_cuda_device_fill_features(d);

        d->buft.iface   = g_cuda_buft_iface;
        d->buft.context = d;
        d->buft.device  = &d->device;

        d->host_buft.iface   = g_cuda_host_buft_iface;
        d->host_buft.context = d;
        d->host_buft.device  = &d->device;

        d->device.iface   = g_cuda_device_iface;
        d->device.backend = GK_CUDA_BACKEND_NAME;
        d->device.index   = i;
        d->device.context = d;
        snprintf(d->device.name, sizeof(d->device.name), "%s", d->name);

        // Peer access, where the pair supports it, is what lets the scheduler
        // move a tensor from one device to another without a host bounce. It
        // is enabled once here rather than per copy, because enabling it is
        // not free and the set of devices does not change.
        //
        // Paired against the devices already registered rather than every
        // index below this one: a device that failed the kernel-image probe is
        // not going to be given work, so there is no copy to make faster.
        for (int k = 0; k < g_cuda_n_devices; ++k) {
            const int peer = g_cuda_devices[k].index;
            int can = 0;
            if (gkDeviceCanAccessPeer(&can, i, peer) == gkSuccess && can) {
                GK_CUDA_CHECK(gkSetDevice(i));
                gkDeviceEnablePeerAccess(peer, 0);
                GK_CUDA_CHECK(gkSetDevice(peer));
                gkDeviceEnablePeerAccess(i, 0);
            }
        }

        g_cuda_n_devices++;

        gk_device_register(&d->device);
    }

    // Peer access that was already enabled comes back as an error rather than
    // as success, and an unread error is not discarded - it is handed to
    // whoever calls gkGetLastError next. That is the graph loop, which would
    // blame it on its first node. Read it here, where it means nothing.
    (void) gkGetLastError();
}

// --------------------------------------------------------------------------
// the direct entry point
// --------------------------------------------------------------------------

extern "C" gk_backend_t gk_backend_cuda_init(int device) {
    gk_cuda_register_devices();

    // `device` is a CUDA device index - the number in "CUDA1" and the one
    // CUDA_VISIBLE_DEVICES speaks - not a position in the registered array.
    // The two stop agreeing the moment a device is skipped for having no
    // kernel image, and the caller has no way of knowing that happened.
    for (int i = 0; i < g_cuda_n_devices; ++i) {
        if (g_cuda_devices[i].index == device) {
            return gk_cuda_device_init_backend(&g_cuda_devices[i].device);
        }
    }

    return NULL;
}
