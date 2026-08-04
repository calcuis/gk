#pragma once

// Device-side decoding of the GGUF block formats.
//
// Every function here answers one question: given a pointer to the start of a
// packed row and an element index within it, what is that element's value?
// Random access rather than whole-block decoding, because that is the shape
// the kernels want - a matmul lane walks a row at its own stride and never has
// a whole block to itself, and materialising one would cost more shared memory
// than the arithmetic saves.
//
// The layouts are the ones in ../../quantizer/src/kernels/qz_format.h and the
// arithmetic is qz_decode.c's, transcribed. That duplication is deliberate and
// is the only kind in this tree: the codec's C is not callable from device
// code, so the choice is between transcribing it and having no quantized
// weights on the GPU at all. What keeps the two honest is that the CPU pass is
// the reference and the differential tests compare against it - a transcription
// error here shows up as a wrong answer there, not as silence.
//
// The formats with codebooks - IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S - are deliberately absent. They need their grids resident on the
// device, and their share of published weights does not yet justify carrying
// them; a matmul against one is reported unsupported and runs on the CPU.

#include "gk_cuda_vendor.h"

#include <stdint.h>

// --------------------------------------------------------------------------
// the type enum, mirrored
//
// gk.h's enum is C and this is device code, so the values are repeated rather
// than included. They are the GGUF file constants and cannot change.
// --------------------------------------------------------------------------

#define GKT_F32     0
#define GKT_F16     1
#define GKT_Q4_0    2
#define GKT_Q4_1    3
#define GKT_Q5_0    6
#define GKT_Q5_1    7
#define GKT_Q8_0    8
#define GKT_Q8_1    9
#define GKT_Q2_K   10
#define GKT_Q3_K   11
#define GKT_Q4_K   12
#define GKT_Q5_K   13
#define GKT_Q6_K   14
#define GKT_Q8_K   15
#define GKT_IQ4_NL 20
#define GKT_IQ4_XS 23
#define GKT_I8     24
#define GKT_I16    25
#define GKT_I32    26
#define GKT_I64    27
#define GKT_F64    28
#define GKT_BF16   30
#define GKT_TQ1_0  34
#define GKT_TQ2_0  35
#define GKT_MXFP4  39
#define GKT_NVFP4  40
#define GKT_Q1_0   41
#define GKT_Q2_0   42

#define GK_QK 256 // the super-block size shared by the K and ternary formats

// --------------------------------------------------------------------------
// narrow floats
// --------------------------------------------------------------------------

static __device__ __forceinline__ float gk_cu_h2f(const uint8_t * p) {
    __half h;
    memcpy(&h, p, sizeof(h));
    return __half2float(h);
}

static __device__ __forceinline__ float gk_cu_bf2f(uint16_t bits) {
    const uint32_t u = (uint32_t) bits << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// E8M0 shared exponent, halved: the MXFP4 codebook stores doubled values.
static __device__ __forceinline__ float gk_cu_e8m0_half(uint8_t e) {
    uint32_t u;
    if (e >= 2) {
        u = (uint32_t) (e - 1) << 23;
    } else {
        u = 0x00200000u << e; // 2^-128 and 2^-127 are subnormal in f32
    }
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// UE4M3: NVFP4's per-group scale, also halved. 0 and the NaN slot decode to 0.
static __device__ __forceinline__ float gk_cu_ue4m3(uint8_t v) {
    if (v == 0 || v == 0x7f) {
        return 0.0f;
    }
    const uint32_t exp = (v >> 3) & 0xfu;
    const uint32_t man = v & 0x7u;

    if (exp == 0) {
        return (float) man * (1.0f / 512.0f) * 0.5f;
    }

    const uint32_t u = ((exp + 120u) << 23) | (man << 20);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f * 0.5f;
}

// One ternary digit out of a base-3 byte, the same multiply-and-shift the
// codec uses.
static __device__ __forceinline__ int gk_cu_trit(uint8_t byte, int pow3) {
    const uint8_t v = (uint8_t) (byte * pow3);
    return (int) (((uint16_t) v * 3) >> 8) - 1;
}

// The two 16-entry codebooks, as device constants.
static __device__ __constant__ int8_t gk_cu_iq4_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static __device__ __constant__ int8_t gk_cu_e2m1_values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static __device__ __constant__ uint8_t gk_cu_pow3[5] = { 1, 3, 9, 27, 81 };

// --------------------------------------------------------------------------
// block geometry
//
// Elements per block and bytes per block, for both host and device: the launch
// code needs them to size work and the kernels need them to find a block.
// --------------------------------------------------------------------------

static __device__ __host__ __forceinline__ int gk_cu_blck_size(int type) {
    switch (type) {
        case GKT_Q4_0: case GKT_Q4_1: case GKT_Q5_0: case GKT_Q5_1:
        case GKT_Q8_0: case GKT_Q8_1: case GKT_IQ4_NL: case GKT_MXFP4:
            return 32;
        case GKT_Q1_0:
            return 128;
        case GKT_Q2_0: case GKT_NVFP4:
            return 64;
        case GKT_Q2_K: case GKT_Q3_K: case GKT_Q4_K: case GKT_Q5_K:
        case GKT_Q6_K: case GKT_Q8_K: case GKT_IQ4_XS:
        case GKT_TQ1_0: case GKT_TQ2_0:
            return GK_QK;
        default:
            return 1;
    }
}

static __device__ __host__ __forceinline__ int gk_cu_type_size(int type) {
    switch (type) {
        case GKT_F32:    return 4;
        case GKT_F16:    return 2;
        case GKT_BF16:   return 2;
        case GKT_F64:    return 8;
        case GKT_I8:     return 1;
        case GKT_I16:    return 2;
        case GKT_I32:    return 4;
        case GKT_I64:    return 8;
        case GKT_Q4_0:   return 2 + 16;
        case GKT_Q4_1:   return 4 + 16;
        case GKT_Q5_0:   return 6 + 16;
        case GKT_Q5_1:   return 8 + 16;
        case GKT_Q8_0:   return 2 + 32;
        case GKT_Q8_1:   return 4 + 32;
        case GKT_Q1_0:   return 2 + 16;
        case GKT_Q2_0:   return 2 + 16;
        case GKT_MXFP4:  return 1 + 16;
        case GKT_NVFP4:  return 4 + 32;
        case GKT_Q2_K:   return 4 + GK_QK / 16 + GK_QK / 4;
        case GKT_Q3_K:   return 2 + GK_QK / 4 + GK_QK / 8 + 12;
        case GKT_Q4_K:   return 4 + 12 + GK_QK / 2;
        case GKT_Q5_K:   return 4 + 12 + GK_QK / 2 + GK_QK / 8;
        case GKT_Q6_K:   return 2 + GK_QK / 16 + 3 * GK_QK / 4;
        case GKT_Q8_K:   return 4 + GK_QK + GK_QK / 16 * 2;
        case GKT_IQ4_NL: return 2 + 16;
        case GKT_IQ4_XS: return 2 + 2 + GK_QK / 64 + GK_QK / 2;
        case GKT_TQ1_0:  return 2 + GK_QK / 64 + (GK_QK - 4 * GK_QK / 64) / 5;
        case GKT_TQ2_0:  return 2 + GK_QK / 4;
        default:         return 0;
    }
}

// Whether a matmul weight of this type can be read by the kernels below.
static __host__ __forceinline__ bool gk_cu_type_supported(int type) {
    switch (type) {
        case GKT_F32: case GKT_F16: case GKT_BF16:
        case GKT_Q4_0: case GKT_Q4_1: case GKT_Q5_0: case GKT_Q5_1: case GKT_Q8_0:
        case GKT_Q1_0: case GKT_Q2_0: case GKT_MXFP4: case GKT_NVFP4:
        case GKT_Q2_K: case GKT_Q3_K: case GKT_Q4_K: case GKT_Q5_K: case GKT_Q6_K:
        case GKT_IQ4_NL: case GKT_IQ4_XS: case GKT_TQ1_0: case GKT_TQ2_0:
            return true;
        default:
            return false;
    }
}

// --------------------------------------------------------------------------
// the decoders
//
// Each takes the address of the block containing the element and the index
// within that block.
// --------------------------------------------------------------------------

static __device__ __forceinline__ float gk_cu_dq_q4_0(const uint8_t * b, int j) {
    const float d = gk_cu_h2f(b);
    const uint8_t * qs = b + 2;
    const int q = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
    return d * (float) (q - 8);
}

static __device__ __forceinline__ float gk_cu_dq_q4_1(const uint8_t * b, int j) {
    const float d = gk_cu_h2f(b);
    const float m = gk_cu_h2f(b + 2);
    const uint8_t * qs = b + 4;
    const int q = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
    return d * (float) q + m;
}

static __device__ __forceinline__ float gk_cu_dq_q5_0(const uint8_t * b, int j) {
    const float d = gk_cu_h2f(b);
    uint32_t qh;
    memcpy(&qh, b + 2, sizeof(qh));
    const uint8_t * qs = b + 6;
    // The fifth bit of element j is always bit j of qh, whichever half the
    // nibble came from - which is why the two halves share one expression.
    const int nib = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
    const int q   = nib | (int) (((qh >> j) & 1u) << 4);
    return d * (float) (q - 16);
}

static __device__ __forceinline__ float gk_cu_dq_q5_1(const uint8_t * b, int j) {
    const float d = gk_cu_h2f(b);
    const float m = gk_cu_h2f(b + 2);
    uint32_t qh;
    memcpy(&qh, b + 4, sizeof(qh));
    const uint8_t * qs = b + 8;
    const int nib = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
    const int q   = nib | (int) (((qh >> j) & 1u) << 4);
    return d * (float) q + m;
}

static __device__ __forceinline__ float gk_cu_dq_q8_0(const uint8_t * b, int j) {
    const float d = gk_cu_h2f(b);
    const int8_t * qs = (const int8_t *) (b + 2);
    return d * (float) qs[j];
}

static __device__ __forceinline__ float gk_cu_dq_q1_0(const uint8_t * b, int j) {
    const float d = gk_cu_h2f(b);
    const uint8_t * qs = b + 2;
    return ((qs[j >> 3] >> (j & 7)) & 1) ? d : -d;
}

static __device__ __forceinline__ float gk_cu_dq_q2_0(const uint8_t * b, int j) {
    const float d = gk_cu_h2f(b);
    const uint8_t * qs = b + 2;
    const int q = (qs[j / 4] >> ((j % 4) * 2)) & 3;
    return d * (float) (q - 1);
}

static __device__ __forceinline__ float gk_cu_dq_mxfp4(const uint8_t * b, int j) {
    const float d = gk_cu_e8m0_half(b[0]);
    const uint8_t * qs = b + 1;
    const int code = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
    return d * (float) gk_cu_e2m1_values[code];
}

static __device__ __forceinline__ float gk_cu_dq_nvfp4(const uint8_t * b, int j) {
    const int sub = j / 16;
    const int jj  = j % 16;

    const float d = gk_cu_ue4m3(b[sub]);
    const uint8_t * qs = b + 4 + sub * 8;
    const int code = jj < 8 ? (qs[jj] & 0xf) : (qs[jj - 8] >> 4);
    return d * (float) gk_cu_e2m1_values[code];
}

static __device__ __forceinline__ float gk_cu_dq_q2_k(const uint8_t * b, int j) {
    const uint8_t * scales = b;
    const uint8_t * qs     = b + GK_QK / 16;
    const float d    = gk_cu_h2f(b + GK_QK / 16 + GK_QK / 4);
    const float dmin = gk_cu_h2f(b + GK_QK / 16 + GK_QK / 4 + 2);

    const int half  = j / 128;
    const int r     = j % 128;
    const int shift = 2 * (r / 32);
    const int part  = (r % 32) / 16;
    const int l     = r % 16;

    const int g = half * 8 + (shift / 2) * 2 + part;
    const uint8_t sc = scales[g];

    const int q = (qs[half * 32 + part * 16 + l] >> shift) & 3;

    return d * (float) (sc & 0xf) * (float) q - dmin * (float) (sc >> 4);
}

static __device__ __forceinline__ float gk_cu_dq_q3_k(const uint8_t * b, int j) {
    const uint8_t * hmask  = b;
    const uint8_t * qs     = b + GK_QK / 8;
    const uint8_t * scales = b + GK_QK / 8 + GK_QK / 4;
    const float d = gk_cu_h2f(b + GK_QK / 8 + GK_QK / 4 + 12);

    const int half  = j / 128;
    const int r     = j % 128;
    const int shift = 2 * (r / 32);
    const int bit   = half * 4 + (r / 32);
    const int part  = (r % 32) / 16;
    const int l     = r % 16;
    const int idx   = part * 16 + l;

    const int g = half * 8 + (shift / 2) * 2 + part;

    // the sixteen 6-bit scales: low nibbles in the first eight bytes, high two
    // bits packed two at a time into the last four
    const int low  = g < 8 ? (scales[g] & 0xf) : (scales[g - 8] >> 4);
    const int high = (scales[8 + (g % 4)] >> (2 * (g / 4))) & 3;
    const int sc   = (low | (high << 4)) - 32;

    const int lo = (qs[half * 32 + idx] >> shift) & 3;
    // the high bit is stored inverted: a set mask bit means "do not subtract 4"
    const int v = lo - ((hmask[idx] & (1u << bit)) ? 0 : 4);

    return d * (float) sc * (float) v;
}

// The 6-bit scale and min of group g, out of the twelve packed bytes.
static __device__ __forceinline__ void gk_cu_scale_min_6(const uint8_t * src, int g,
                                                         int * scale, int * min) {
    if (g < 4) {
        *scale = src[g] & 63;
        *min   = src[g + 4] & 63;
    } else {
        *scale = (src[g + 4] & 0xf) | ((src[g - 4] >> 6) << 4);
        *min   = (src[g + 4] >> 4)  | ((src[g]     >> 6) << 4);
    }
}

static __device__ __forceinline__ float gk_cu_dq_q4_k(const uint8_t * b, int j) {
    const float d    = gk_cu_h2f(b);
    const float dmin = gk_cu_h2f(b + 2);
    const uint8_t * scales = b + 4;
    const uint8_t * qs     = b + 4 + 12;

    const int g = j / 32;
    const int l = j % 32;

    int sc, mn;
    gk_cu_scale_min_6(scales, g, &sc, &mn);

    const int q = (qs[(g / 2) * 32 + l] >> ((g % 2) * 4)) & 0xf;

    return d * (float) sc * (float) q - dmin * (float) mn;
}

static __device__ __forceinline__ float gk_cu_dq_q5_k(const uint8_t * b, int j) {
    const float d    = gk_cu_h2f(b);
    const float dmin = gk_cu_h2f(b + 2);
    const uint8_t * scales = b + 4;
    const uint8_t * qh     = b + 4 + 12;
    const uint8_t * qs     = b + 4 + 12 + GK_QK / 8;

    const int g = j / 32;
    const int l = j % 32;

    int sc, mn;
    gk_cu_scale_min_6(scales, g, &sc, &mn);

    const int lo = (qs[(g / 2) * 32 + l] >> ((g % 2) * 4)) & 0xf;
    const int q  = lo | ((qh[l] & (1u << g)) ? 16 : 0);

    return d * (float) sc * (float) q - dmin * (float) mn;
}

static __device__ __forceinline__ float gk_cu_dq_q6_k(const uint8_t * b, int j) {
    const uint8_t * ql = b;
    const uint8_t * qh = b + GK_QK / 2;
    const int8_t  * sc = (const int8_t *) (b + GK_QK / 2 + GK_QK / 4);
    const float d = gk_cu_h2f(b + GK_QK / 2 + GK_QK / 4 + GK_QK / 16);

    const int half  = j / 128;
    const int r     = j % 128;
    const int which = r / 32;
    const int i     = r % 32;
    const int is    = i / 16;

    const uint8_t * l = ql + half * 64;
    const uint8_t * h = qh + half * 32;
    const int8_t  * s = sc + half * 8;

    int q;
    int scale;
    switch (which) {
        case 0: q = (l[i]      & 0xf) | (((h[i] >> 0) & 3) << 4); scale = s[is];     break;
        case 1: q = (l[i + 32] & 0xf) | (((h[i] >> 2) & 3) << 4); scale = s[is + 2]; break;
        case 2: q = (l[i]      >> 4)  | (((h[i] >> 4) & 3) << 4); scale = s[is + 4]; break;
        default:q = (l[i + 32] >> 4)  | (((h[i] >> 6) & 3) << 4); scale = s[is + 6]; break;
    }

    return d * (float) scale * (float) (q - 32);
}

static __device__ __forceinline__ float gk_cu_dq_iq4_nl(const uint8_t * b, int j) {
    const float d = gk_cu_h2f(b);
    const uint8_t * qs = b + 2;
    const int code = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
    return d * (float) gk_cu_iq4_values[code];
}

static __device__ __forceinline__ float gk_cu_dq_iq4_xs(const uint8_t * b, int j) {
    const float d = gk_cu_h2f(b);
    uint16_t scales_h;
    memcpy(&scales_h, b + 2, sizeof(scales_h));
    const uint8_t * scales_l = b + 4;
    const uint8_t * qs       = b + 4 + GK_QK / 64;

    const int g = j / 32;
    const int r = j % 32;

    const int ls = ((scales_l[g / 2] >> (4 * (g % 2))) & 0xf) |
                   (int) (((scales_h >> (2 * g)) & 3) << 4);
    const float dl = d * (float) (ls - 32);

    const uint8_t * q = qs + g * 16;
    const int code = r < 16 ? (q[r] & 0xf) : (q[r - 16] >> 4);

    return dl * (float) gk_cu_iq4_values[code];
}

static __device__ __forceinline__ float gk_cu_dq_tq1_0(const uint8_t * b, int j) {
    const uint8_t * qs = b;
    const uint8_t * qh = b + (GK_QK - 4 * GK_QK / 64) / 5;
    const float d = gk_cu_h2f(b + (GK_QK - 4 * GK_QK / 64) / 5 + GK_QK / 64);

    // Three regions, in the order the codec writes them: 160 elements five
    // digits deep over 32 bytes, then 80 over 16, then 16 four digits deep
    // out of qh.
    uint8_t byte;
    int n;
    if (j < 160) {
        n    = j / 32;
        byte = qs[j % 32];
    } else if (j < 240) {
        const int r = j - 160;
        n    = r / 16;
        byte = qs[32 + (r % 16)];
    } else {
        const int r = j - 240;
        n    = r / 4;
        byte = qh[r % 4];
    }

    return d * (float) gk_cu_trit(byte, gk_cu_pow3[n]);
}

static __device__ __forceinline__ float gk_cu_dq_tq2_0(const uint8_t * b, int j) {
    const uint8_t * qs = b;
    const float d = gk_cu_h2f(b + GK_QK / 4);

    const int half = j / 128;
    const int r    = j % 128;
    const int n    = r / 32;
    const int m    = r % 32;

    const int q = ((qs[half * 32 + m] >> (2 * n)) & 3) - 1;
    return d * (float) q;
}

// --------------------------------------------------------------------------
// the dispatch
//
// One element of a packed row. `row` points at the row's first block and `i`
// counts elements, so the block and the offset within it are worked out here
// rather than by every caller.
// --------------------------------------------------------------------------

static __device__ __forceinline__ float gk_cu_row_elem(const void * row, int type, int64_t i) {
    const uint8_t * p = (const uint8_t *) row;

    switch (type) {
        case GKT_F32:  return ((const float *)    p)[i];
        case GKT_F16:  return __half2float(((const __half *) p)[i]);
        case GKT_BF16: return gk_cu_bf2f(((const uint16_t *) p)[i]);
        case GKT_I32:  return (float) ((const int32_t *) p)[i];
        case GKT_I64:  return (float) ((const int64_t *) p)[i];
        case GKT_I16:  return (float) ((const int16_t *) p)[i];
        case GKT_I8:   return (float) ((const int8_t  *) p)[i];
        default: break;
    }

    const int blck = gk_cu_blck_size(type);
    const int tsz  = gk_cu_type_size(type);

    const uint8_t * b = p + (i / blck) * tsz;
    const int       j = (int) (i % blck);

    switch (type) {
        case GKT_Q4_0:   return gk_cu_dq_q4_0  (b, j);
        case GKT_Q4_1:   return gk_cu_dq_q4_1  (b, j);
        case GKT_Q5_0:   return gk_cu_dq_q5_0  (b, j);
        case GKT_Q5_1:   return gk_cu_dq_q5_1  (b, j);
        case GKT_Q8_0:   return gk_cu_dq_q8_0  (b, j);
        case GKT_Q1_0:   return gk_cu_dq_q1_0  (b, j);
        case GKT_Q2_0:   return gk_cu_dq_q2_0  (b, j);
        case GKT_MXFP4:  return gk_cu_dq_mxfp4 (b, j);
        case GKT_NVFP4:  return gk_cu_dq_nvfp4 (b, j);
        case GKT_Q2_K:   return gk_cu_dq_q2_k  (b, j);
        case GKT_Q3_K:   return gk_cu_dq_q3_k  (b, j);
        case GKT_Q4_K:   return gk_cu_dq_q4_k  (b, j);
        case GKT_Q5_K:   return gk_cu_dq_q5_k  (b, j);
        case GKT_Q6_K:   return gk_cu_dq_q6_k  (b, j);
        case GKT_IQ4_NL: return gk_cu_dq_iq4_nl(b, j);
        case GKT_IQ4_XS: return gk_cu_dq_iq4_xs(b, j);
        case GKT_TQ1_0:  return gk_cu_dq_tq1_0 (b, j);
        case GKT_TQ2_0:  return gk_cu_dq_tq2_0 (b, j);
        default:         return 0.0f; // unreachable: supports_op filtered it
    }
}
