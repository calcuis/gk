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

// The inverses of the two decoders above, for quantizing activations *into*
// nvfp4. Only the FP4 tensor-core path needs them - every other path quantizes
// activations to int8 - so they live here beside what they undo rather than in
// the codec, which never encodes on the device.
//
// Both work in *true* units, unlike `gk_cu_ue4m3` and `gk_cu_e2m1_value`,
// which are each off by a factor of two in opposite directions. Encoding is
// where that convention has to be paid off, because the consumer is hardware
// rather than the matching gk decoder.

// A float to the nearest ue4m3 byte, as a true value: 2^(e-7) * (1 + m/8),
// with e == 0 subnormal at 2^-6 * m/8. Round to nearest, ties away, saturating
// at 0x7e - 0x7f is the NaN slot and gk's decoder reads it as zero.
static __device__ __forceinline__ uint8_t gk_cu_f2ue4m3(float v) {
    if (!(v > 0.0f)) {
        return 0;   // negatives cannot happen (v is an amax) and NaN maps to 0
    }

    if (v >= 448.0f) {
        return 0x7e;
    }

    if (v < (1.0f / 64.0f)) {
        // Subnormal: the step is 2^-9, and eight steps reach the first normal.
        const int m = (int) rintf(v * 512.0f);
        return (uint8_t) (m > 7 ? 8 : m);   // 8 lands on e=1,m=0, which is 2^-6
    }

    int e;
    const float frac = frexpf(v, &e);   // v = frac * 2^e, frac in [0.5, 1)
    // frexpf's [0.5,1) against ue4m3's [1,2) is one exponent apart.
    int exp = e - 1 + 7;
    int man = (int) rintf((frac * 2.0f - 1.0f) * 8.0f);

    if (man == 8) {     // rounded up out of the mantissa
        man = 0;
        ++exp;
    }

    if (exp >= 15 && man >= 7) {
        return 0x7e;
    }

    return (uint8_t) ((exp << 3) | man);
}

// A float to the nearest e2m1 code. The eight magnitudes are 0, 0.5, 1, 1.5,
// 2, 3, 4 and 6, and the code is the index into that list with the sign in
// bit 3. Written as comparisons against midpoints rather than as arithmetic on
// the exponent: eight cases is short enough that a branchless ladder beats
// taking a logarithm, and the midpoints are what "nearest" means here.
static __device__ __forceinline__ int gk_cu_f2e2m1(float v) {
    const int   sign = v < 0.0f ? 8 : 0;
    const float a    = fabsf(v);

    int mag;
    if      (a < 0.25f) { mag = 0; }
    else if (a < 0.75f) { mag = 1; }
    else if (a < 1.25f) { mag = 2; }
    else if (a < 1.75f) { mag = 3; }
    else if (a < 2.50f) { mag = 4; }
    else if (a < 3.50f) { mag = 5; }
    else if (a < 5.00f) { mag = 6; }
    else                { mag = 7; }

    return sign | mag;
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

// The e2m1 codebook, as an immediate rather than a lookup.
//
// It was `__constant__`, and constant memory answers one address per cycle for
// a warp: lanes asking for the same entry are broadcast together, lanes asking
// for different ones are replayed one after another. A codebook index here is
// the weight itself, so a warp asks for as many entries as it has distinct
// codes - up to sixteen replays per lookup, in kernels that do tens of lookups
// per thread per step. Measured on the nvfp4 tensor-core tile, removing it was
// worth 1.6x.
//
// The eight magnitudes - 0, 1, 2, 3, 4, 6, 8, 12 - each fit in a nibble, so
// the table is one 32-bit immediate and an entry is a shift and a mask. The
// high bit of the code is the sign.
#define GK_CU_E2M1_NIBBLES 0xC8643210u

static __device__ __forceinline__ int gk_cu_e2m1_value(int code) {
    const int mag = (int) ((GK_CU_E2M1_NIBBLES >> (4 * (code & 7))) & 0xf);
    return (code & 8) ? -mag : mag;
}

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
//
// This list and GK_CU_MM_DISPATCH's must agree: supports_op promises the
// scheduler that a weight of this type will be handled, and the dispatch is
// what handles it. They are kept adjacent for that reason, and the dispatch
// says loudly rather than quietly if it is ever asked for a type it has no
// instantiation of.
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

// Turns a runtime weight type into a compile-time one, once per launch, so
// that everything inside the kernel can be specialized. `LAUNCH` is a macro
// taking the type constant; it is invoked exactly once, on the matching arm.
//
// The cost is instantiation count - one kernel per type per launcher - which
// is the trade this whole file is making: a larger binary and a longer build
// against a division and two branches per weight element at run time.
#define GK_CU_MM_DISPATCH(type, LAUNCH)                                       \
    switch (type) {                                                           \
        case GKT_F32:    LAUNCH(GKT_F32);    break;                           \
        case GKT_F16:    LAUNCH(GKT_F16);    break;                           \
        case GKT_BF16:   LAUNCH(GKT_BF16);   break;                           \
        case GKT_Q4_0:   LAUNCH(GKT_Q4_0);   break;                           \
        case GKT_Q4_1:   LAUNCH(GKT_Q4_1);   break;                           \
        case GKT_Q5_0:   LAUNCH(GKT_Q5_0);   break;                           \
        case GKT_Q5_1:   LAUNCH(GKT_Q5_1);   break;                           \
        case GKT_Q8_0:   LAUNCH(GKT_Q8_0);   break;                           \
        case GKT_Q1_0:   LAUNCH(GKT_Q1_0);   break;                           \
        case GKT_Q2_0:   LAUNCH(GKT_Q2_0);   break;                           \
        case GKT_MXFP4:  LAUNCH(GKT_MXFP4);  break;                           \
        case GKT_NVFP4:  LAUNCH(GKT_NVFP4);  break;                           \
        case GKT_Q2_K:   LAUNCH(GKT_Q2_K);   break;                           \
        case GKT_Q3_K:   LAUNCH(GKT_Q3_K);   break;                           \
        case GKT_Q4_K:   LAUNCH(GKT_Q4_K);   break;                           \
        case GKT_Q5_K:   LAUNCH(GKT_Q5_K);   break;                           \
        case GKT_Q6_K:   LAUNCH(GKT_Q6_K);   break;                           \
        case GKT_IQ4_NL: LAUNCH(GKT_IQ4_NL); break;                           \
        case GKT_IQ4_XS: LAUNCH(GKT_IQ4_XS); break;                           \
        case GKT_TQ1_0:  LAUNCH(GKT_TQ1_0);  break;                           \
        case GKT_TQ2_0:  LAUNCH(GKT_TQ2_0);  break;                           \
        default:                                                              \
            /* supports_op said yes to a type with no instantiation: the two  \
               lists above have drifted. Nothing is launched, so this would   \
               otherwise be an untouched output buffer rather than an error. */ \
            gk_logf("gk cuda: matmul has no kernel for weight type %d "       \
                    "(gk_cu_type_supported and GK_CU_MM_DISPATCH disagree)\n", \
                    (int) (type));                                            \
            break;                                                            \
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
    return d * (float) gk_cu_e2m1_value(code);
}

static __device__ __forceinline__ float gk_cu_dq_nvfp4(const uint8_t * b, int j) {
    const int sub = j / 16;
    const int jj  = j % 16;

    const float d = gk_cu_ue4m3(b[sub]);
    const uint8_t * qs = b + 4 + sub * 8;
    const int code = jj < 8 ? (qs[jj] & 0xf) : (qs[jj - 8] >> 4);
    return d * (float) gk_cu_e2m1_value(code);
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

// --------------------------------------------------------------------------
// the integer dot path
//
// Everything above turns packed weights into floats and multiplies. That costs
// a decode per element, and measurement says the decode is what a quantized
// matvec spends its time on: q4_K reads a third of the bytes f16 does and
// takes more time.
//
// This is the other way round. The activations are quantized to 8 bits once
// per matmul, and then a weight's own 4- or 8-bit codes are dotted against
// them as integers - four at a time through one instruction - with the scales
// applied once per 32-element group instead of once per element. Nothing is
// converted to float until the group is done.
//
// The accuracy question is already settled by the CPU: gk's own vec_dot
// converts the activation side to Q8_0/Q8_1/Q8_K before dotting, so this makes
// the device agree with the CPU more closely than the float path did, not less.
// --------------------------------------------------------------------------

// 32 activations, quantized. `s` is the sum of the codes, which is what the
// asymmetric formats need: their weights carry a minimum that multiplies the
// plain sum of the activations rather than the dot product.
struct gk_cu_q8blk {
    float   d;    // value ~= d * code
    float   s;    // sum of the 32 codes
    int32_t q[8]; // the codes, four to a word
};

// 64 activations, quantized to nvfp4 rather than to int8.
//
// This exists for one instruction. Blackwell's block-scaled FP4 mma takes both
// operands as e2m1 and applies a ue4m3 scale per sixteen elements *in
// hardware*, so the activation side has to arrive in exactly that shape - and
// when it does, nothing on the path decodes anything.
//
// `sc` is the four scales as the instruction wants them: one packed uint32,
// byte `i` being the scale for elements `16i..16i+15`. It is handed to the PTX
// verbatim. `q` is the codes in k order, eight nibbles to a word, which is
// also verbatim - word `w` is the fragment register for elements `8w..8w+7`.
//
// The scales are stored as *true* ue4m3, not as gk's halved decode. That is
// not a special case: gk's e2m1 table is doubled by exactly the same factor
// (see `gk_cu_e2m1_value`), so the halved scale and the doubled code have
// always multiplied out to the right answer, and the bytes on disk have always
// been true ue4m3. The hardware reads them as such.
struct gk_cu_fp4blk {
    uint32_t sc;   // four ue4m3 scales, one per sixteen elements
    uint32_t q[8]; // the codes, eight to a word, in k order
};

// Four 8-bit products accumulated into an int, in one instruction where the
// part has it. Every NVIDIA part since Pascal does; the fallback is here so
// that a build for an older one, or for HIP, is slow rather than broken.
static __device__ __forceinline__ int gk_cu_dp4a(int a, int b, int c) {
#if !defined(GK_USE_HIP) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610)
    return __dp4a(a, b, c);
#else
    const int8_t * va = (const int8_t *) &a;
    const int8_t * vb = (const int8_t *) &b;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        c += (int) va[i] * (int) vb[i];
    }
    return c;
#endif
}

// Four packed bytes as an int, read as two 16-bit halves.
//
// A direct 32-bit load would be wrong: a q4_0 block is 18 bytes, so every
// other block's payload starts two bytes off a 4-byte boundary. Every format
// here is at least 2-byte aligned, which this only needs.
static __device__ __forceinline__ int gk_cu_int_b2(const uint8_t * p, int i32) {
    const uint16_t * p16 = (const uint16_t *) p;
    return (int) ((uint32_t) p16[2 * i32] | ((uint32_t) p16[2 * i32 + 1] << 16));
}

// Whether this format has an integer dot below. The rest keep the float path.
template <int TYPE>
static __device__ __host__ __forceinline__ bool gk_cu_has_dp4a() {
    return TYPE == GKT_Q4_0 || TYPE == GKT_Q4_1 ||
           TYPE == GKT_Q8_0 || TYPE == GKT_Q4_K;
}

// One 32-element group of a weight row, as integer codes plus the two floats
// that turn a code-dot back into a value.
//
// Every format here has the same shape once unpacked: a value is
// `scale * code + offset`, so a group's contribution to a dot product against
// activations `a.d * c_j` is
//
//     a.d * ( scale * sum_j code_j*c_j  +  offset * sum_j c_j )
//
// - one integer dot and one multiple of the activation block's code sum. The
// bias in a symmetric format (q4_0's -8) and the minimum in an asymmetric one
// (q4_1's m, q4_K's dmin*mn) are the same term seen twice.
//
// `codes[i]` holds elements 4i..4i+3 in natural order, matching how the
// activation block packs its own, which is what lets the two be dotted
// directly.
template <int TYPE>
static __device__ __forceinline__ void gk_cu_wblk32(const uint8_t * row, int64_t g,
                                                    int (&codes)[8],
                                                    float & scale, float & offset) {
    if (TYPE == GKT_Q4_0 || TYPE == GKT_Q4_1) {
        const int       stride = TYPE == GKT_Q4_0 ? 18 : 20;
        const int       hdr    = TYPE == GKT_Q4_0 ?  2 :  4;
        const uint8_t * blk    = row + g * stride;
        const uint8_t * qs     = blk + hdr;

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            // one word holds four low nibbles and four high ones, which are
            // elements 4i.. and 16+4i.. - the order the decoder above reads
            const int w = gk_cu_int_b2(qs, i);
            codes[i]     = (w >> 0) & 0x0F0F0F0F;
            codes[i + 4] = (w >> 4) & 0x0F0F0F0F;
        }

        const float d = gk_cu_h2f(blk);
        scale  = d;
        offset = TYPE == GKT_Q4_0 ? -8.0f * d : gk_cu_h2f(blk + 2);
        return;
    }

    if (TYPE == GKT_Q8_0) {
        const uint8_t * blk = row + g * 34;
        const uint8_t * qs  = blk + 2;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            codes[i] = gk_cu_int_b2(qs, i);
        }

        scale  = gk_cu_h2f(blk);
        offset = 0.0f;
        return;
    }

    if (TYPE == GKT_Q4_K) {
        // eight groups to a super-block, each with its own 6-bit scale and
        // minimum out of the packed twelve-byte field
        const uint8_t * blk = row + (g / 8) * 144;
        const int       sub = (int) (g % 8);

        int sc, mn;
        gk_cu_scale_min_6(blk + 4, sub, &sc, &mn);

        const uint8_t * qs    = blk + 16 + (sub / 2) * 32;
        const int       shift = (sub % 2) * 4;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            codes[i] = (gk_cu_int_b2(qs, i) >> shift) & 0x0F0F0F0F;
        }

        scale  =  gk_cu_h2f(blk)     * (float) sc;
        offset = -gk_cu_h2f(blk + 2) * (float) mn;
        return;
    }

    // unreachable: gk_cu_has_dp4a filtered it
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        codes[i] = 0;
    }
    scale  = 0.0f;
    offset = 0.0f;
}

// The dot of one 32-element group of a weight row against one quantized
// activation block. `g` counts groups from the start of the row, so a format
// whose block holds more than 32 elements finds its block from it.
template <int TYPE>
static __device__ __forceinline__ float gk_cu_vecdot32(const uint8_t * row, int64_t g,
                                                       const gk_cu_q8blk & ab) {
    int   codes[8];
    float scale, offset;
    gk_cu_wblk32<TYPE>(row, g, codes, scale, offset);

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sumi = gk_cu_dp4a(codes[i], ab.q[i], sumi);
    }

    return ab.d * (scale * (float) sumi + offset * ab.s);
}

// One element of an already-located block.
//
// Split out of gk_cu_row_elem_t below so that a caller walking several
// elements of the same block can find the block once. That is the whole point:
// the block's header - a scale, sometimes a minimum, sometimes a packed set of
// sub-scales - costs the same whether one element is wanted from the block or
// thirty-two, and the per-element entry point pays it every time.
template <int TYPE>
static __device__ __forceinline__ float gk_cu_blk_elem_t(const uint8_t * b, int j) {
    switch (TYPE) {
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

// Whether gk_cu_blk_run_t below has anything better to offer this format than
// calling the per-element decoder in a loop. Where it does not, the caller is
// better off not gathering a run at all: the array costs registers, and both
// q6_K and mxfp4 measured slower going through one for no gain.
template <int TYPE>
static __device__ __host__ __forceinline__ bool gk_cu_has_run_path() {
    return TYPE == GKT_Q4_K || TYPE == GKT_Q5_K;
}

// A run of `n` consecutive elements of one block, into `out`.
//
// The generic body just calls the per-element decoder, which is already a big
// improvement over finding the block each time. The K formats get an explicit
// path because their header is the expensive one: two half-precision scales
// and a 6-bit sub-scale unpacked out of a packed twelve-byte field. A run of
// four stays inside one 32-element sub-group, so the sub-scale is the same for
// all of them and the whole header reduces to two multiplies hoisted out of
// the loop - which the compiler does not manage on its own through the
// per-element entry point.
//
// `j0` is a multiple of the maximum run, which is what guarantees a run never
// straddles a sub-group boundary.
template <int TYPE, int RUN>
static __device__ __forceinline__ void gk_cu_blk_run_t(const uint8_t * b, int j0, int n,
                                                       float (&out)[RUN]) {
    if (TYPE == GKT_Q4_K || TYPE == GKT_Q5_K) {
        const float d    = gk_cu_h2f(b);
        const float dmin = gk_cu_h2f(b + 2);
        const uint8_t * scales = b + 4;

        const int g = j0 / 32;

        int sc, mn;
        gk_cu_scale_min_6(scales, g, &sc, &mn);

        // the whole header, collapsed to the two constants it contributes
        const float dsc = d * (float) sc;
        const float dm  = dmin * (float) mn;

        const int shift = (g % 2) * 4;

        if (TYPE == GKT_Q4_K) {
            const uint8_t * qs = b + 4 + 12 + (g / 2) * 32;
#pragma unroll
            for (int e = 0; e < RUN; ++e) {
                if (e >= n) { break; }
                const int q = (qs[(j0 % 32) + e] >> shift) & 0xf;
                out[e] = dsc * (float) q - dm;
            }
        } else {
            const uint8_t * qh = b + 4 + 12;
            const uint8_t * qs = b + 4 + 12 + GK_QK / 8 + (g / 2) * 32;
#pragma unroll
            for (int e = 0; e < RUN; ++e) {
                if (e >= n) { break; }
                const int l  = (j0 % 32) + e;
                const int lo = (qs[l] >> shift) & 0xf;
                const int q  = lo | ((qh[l] & (1u << g)) ? 16 : 0);
                out[e] = dsc * (float) q - dm;
            }
        }
        return;
    }

#pragma unroll
    for (int e = 0; e < RUN; ++e) {
        if (e >= n) { break; }
        out[e] = gk_cu_blk_elem_t<TYPE>(b, j0 + e);
    }
}

// Whether a type is packed into blocks at all, which decides whether the
// caller above has a block to find.
template <int TYPE>
static __device__ __forceinline__ constexpr bool gk_cu_is_blocked() {
    return TYPE != GKT_F32 && TYPE != GKT_F16 && TYPE != GKT_BF16 &&
           TYPE != GKT_I8  && TYPE != GKT_I16 && TYPE != GKT_I32 && TYPE != GKT_I64;
}

// The same decode, with the type known at compile time.
//
// This exists because the runtime version below is expensive in a way that is
// invisible when you read it. `gk_cu_blck_size(type)` and `gk_cu_type_size(type)`
// return runtime values, so `i / blck` is a 64-bit integer division by a
// quantity the compiler cannot see - and no NVIDIA part has an integer divide
// instruction, so that is tens of instructions, per element, before any
// decoding happens. The two switches are branches for the same reason.
//
// With TYPE a template parameter all of it folds: every block size here is a
// power of two, so the division becomes a shift and the modulo a mask, and
// both switches become the one arm that survives dead-code elimination.
//
// The measurement that prompted this: a q4_0 matvec reads 3.6x less memory
// than the same matmul in f32 and took longer, and every quantized format
// landed within 25% of every other regardless of how many bytes it moved.
// That is the signature of being bound by decode instructions rather than by
// the memory system.
template <int TYPE>
static __device__ __forceinline__ float gk_cu_row_elem_t(const void * row, int64_t i) {
    const uint8_t * p = (const uint8_t *) row;

    // Not a switch: TYPE is a constant here, so each of these is either the
    // whole function or nothing at all.
    if (TYPE == GKT_F32)  { return ((const float *)    p)[i]; }
    if (TYPE == GKT_F16)  { return __half2float(((const __half *) p)[i]); }
    if (TYPE == GKT_BF16) { return gk_cu_bf2f(((const uint16_t *) p)[i]); }
    if (TYPE == GKT_I32)  { return (float) ((const int32_t *) p)[i]; }
    if (TYPE == GKT_I64)  { return (float) ((const int64_t *) p)[i]; }
    if (TYPE == GKT_I16)  { return (float) ((const int16_t *) p)[i]; }
    if (TYPE == GKT_I8)   { return (float) ((const int8_t  *) p)[i]; }

    const int blck = gk_cu_blck_size(TYPE);
    const int tsz  = gk_cu_type_size(TYPE);

    const uint8_t * b = p + (i / blck) * tsz;
    const int       j = (int) (i % blck);

    switch (TYPE) {
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
