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

#define GK_CUDA_MAX_DEVICES 16

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
    GK_CUDA_CHECK(gkMemcpy((char *) tensor->data + offset, data, size, gkMemcpyHostToDevice));
}

static void gk_cuda_buffer_get_tensor(gk_backend_buffer_t buffer, const struct gk_tensor * tensor,
                                      void * data, size_t offset, size_t size) {
    struct gk_cuda_buffer_ctx * ctx = (struct gk_cuda_buffer_ctx *) buffer->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->device));
    GK_CUDA_CHECK(gkMemcpy(data, (const char *) tensor->data + offset, size, gkMemcpyDeviceToHost));
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

static enum gk_status gk_cuda_backend_compute(gk_backend_t backend, struct gk_cgraph * graph) {
    struct gk_cuda_backend_ctx * ctx = (struct gk_cuda_backend_ctx *) backend->context;

    GK_CUDA_CHECK(gkSetDevice(ctx->dev->index));

    const int n = gk_graph_n_nodes(graph);

    for (int i = 0; i < n; ++i) {
        struct gk_tensor * node = gk_graph_node(graph, i);

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
};

// --------------------------------------------------------------------------
// discovery
// --------------------------------------------------------------------------

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

        struct gk_cuda_device_ctx * d = &g_cuda_devices[g_cuda_n_devices];
        memset(d, 0, sizeof(*d));

        d->index        = i;
        d->total_memory = prop.totalGlobalMem;
        d->integrated   = prop.integrated != 0;
        d->n_sm         = prop.multiProcessorCount;

        snprintf(d->name, sizeof(d->name), "%s%d", GK_CUDA_BACKEND_NAME, i);
        snprintf(d->description, sizeof(d->description), "%s", prop.name);

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
        for (int j = 0; j < i; ++j) {
            int can = 0;
            if (gkDeviceCanAccessPeer(&can, i, j) == gkSuccess && can) {
                GK_CUDA_CHECK(gkSetDevice(i));
                gkDeviceEnablePeerAccess(j, 0);
                GK_CUDA_CHECK(gkSetDevice(j));
                gkDeviceEnablePeerAccess(i, 0);
            }
        }

        g_cuda_n_devices++;

        gk_device_register(&d->device);
    }
}

// --------------------------------------------------------------------------
// the direct entry point
// --------------------------------------------------------------------------

extern "C" gk_backend_t gk_backend_cuda_init(int device) {
    gk_cuda_register_devices();

    if (device < 0 || device >= g_cuda_n_devices) {
        return NULL;
    }

    return gk_cuda_device_init_backend(&g_cuda_devices[device].device);
}
