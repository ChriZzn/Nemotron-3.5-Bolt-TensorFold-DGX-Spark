// Tensor-core GEMMs for packed weights. Y[T][N] = X[T][K] * W^T with T up to 16 rows of fp32 activations.
//
// A warp owns tiles of 16 weight rows and walks K in steps of 64. The weights are repacked at load time so
// that lane (g, t) = (lane / 4, lane % 4) fetches the A fragments of mma.m16n8k16 with one 16-byte load:
//   a0 = (row g,   k 2t, 2t+1)   a1 = (row g+8, k 2t, 2t+1)
//   a2 = (row g,   k 2t+8, 2t+9) a3 = (row g+8, k 2t+8, 2t+9)
// Activations are the B operand (K x 8 tokens), converted from fp32 as they are loaded. Quantized values are
// exact in fp16/bf16, and the per-group scales are applied to the fp32 accumulators, so the only rounding
// beyond the weights themselves is the activation conversion.
//
// Byte layouts (rt = row tile, lane = g * 4 + t):
//   NVFP4  q  [rt][k/64][lane][4 x u32]; word j is k16 step j. Nibble r (bits 4r..4r+3) is the first element
//          of a_r and nibble 4+r (bits 16+4r..) its second. sc [rt][k/64][g][8 bytes]; byte 2j+h is the e4m3
//          scale of row g+8h, k group 4s+j.
//   Q8_0   q  [rt][k/32][lane][16 bytes]; bytes 8u+2r, 8u+2r+1 are the elements of a_r in k16 step u of the
//          block. d [rt][k/32][g][u32]; low half row g, high half row g+8.
//   bf16   w  [rt][k/16][lane][4 x u32] = a0..a3.
//   IQ2    q  [rt][k/64][row 16][16 bytes] = the row's two 8-byte groups of the step; d [rt][k/256][row 16] half.

#include "bolt_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>
#include <cstdlib>

#define GGML_TABLE_BEGIN(type, name, size) static __device__ type name[size] = {
#define GGML_TABLE_END() };
#include "iq2_tables.inc"

namespace {

__device__ __forceinline__ void mma_f16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// cp.async pieces for the staged weight rings (the tile kernel below has its own copies of these)
__device__ __forceinline__ void w_cp16(void* smem, const void* gmem) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void w_cp8(void* smem, const void* gmem) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void w_cp4(void* smem, const void* gmem) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void w_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void w_cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <bool BF>
__device__ __forceinline__ uint32_t pack2(float a, float b) {
    if constexpr (BF) {
        __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<uint32_t*>(&h);
    } else {
        a = fminf(fmaxf(a, -65504.f), 65504.f);
        b = fminf(fmaxf(b, -65504.f), 65504.f);
        __half2 h = __floats2half2_rn(a, b);
        return *reinterpret_cast<uint32_t*>(&h);
    }
}

__device__ __forceinline__ uint32_t hmul2u(uint32_t a, __half2 s) {
    __half2 v = __hmul2(*reinterpret_cast<__half2*>(&a), s);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t bmul2u(uint32_t a, __nv_bfloat162 s) {
    __nv_bfloat162 v = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), s);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float e4m3f(unsigned b) {
    __nv_fp8_e4m3 f;
    f.__x = (unsigned char)b;
    return (float)f;
}

__device__ __forceinline__ float half_lo(uint32_t v) { return __half2float(__ushort_as_half((unsigned short)(v & 0xffff))); }
__device__ __forceinline__ float half_hi(uint32_t v) { return __half2float(__ushort_as_half((unsigned short)(v >> 16))); }

// ------------------------------------------------------------------------------------------------------
// Policies: how one lane loads and uses a 64-wide k step of one 16-row tile.
// ------------------------------------------------------------------------------------------------------

struct PNV4 {
    using W = TNV4;
    struct Smem { int unused; };
    struct Raw { uint4 q; uint2 sc; };
    static constexpr bool kPermK = false;   // B fragments in natural k order
    static constexpr bool kBF = false;
    static __device__ __forceinline__ int steps(const W& w) { return w.k >> 6; }
    static __device__ __forceinline__ int tiles(const W& w) { return (w.n + 15) >> 4; }
    static __device__ __forceinline__ void init(Smem&) {}
    static __device__ __forceinline__ float scale(const W& w) { return w.s2; }
    static __device__ __forceinline__ Raw load(const W& w, int tile, int s, int lane, int g) {
        size_t S = w.k >> 6;
        Raw r;
        r.q = __ldg(w.q + ((size_t)tile * S + s) * 32 + lane);
        r.sc = __ldg(w.sc + ((size_t)tile * S + s) * 8 + g);
        return r;
    }
    static constexpr int kStage = 512 + 64;   // 32 x 16 B of nibbles, 8 x 8 B of scales
    static __device__ __forceinline__ void stage(const W& w, int tile, int s, int lane, int g, unsigned char* dst) {
        const size_t S = w.k >> 6;
        w_cp16(dst + lane * 16, w.q + ((size_t)tile * S + s) * 32 + lane);
        if (lane < 8) w_cp8(dst + 512 + lane * 8, w.sc + ((size_t)tile * S + s) * 8 + lane);
    }
    static __device__ __forceinline__ Raw unstage(const unsigned char* src, int lane, int g) {
        Raw r;
        r.q = *reinterpret_cast<const uint4*>(src + lane * 16);
        r.sc = *reinterpret_cast<const uint2*>(src + 512 + g * 8);
        return r;
    }
    static constexpr bool kMmaBF = false;
    // the step's four A fragments (k 16j + 2t.. for rows g and g+8), scales folded in, as step() builds them
    static __device__ __forceinline__ void frag(const Raw& r, uint32_t (&f)[4][4], Smem&, int, int) {
        const uint32_t words[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
        const uint32_t scb[2] = {r.sc.x, r.sc.y};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t wd = words[j];
            const uint32_t lo = __byte_perm(0x3E3C3800u, 0x46444240u, wd & 0x7777u);
            const uint32_t hi = __byte_perm(0x3E3C3800u, 0x46444240u, (wd >> 16) & 0x7777u);
            uint32_t a[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint32_t v = __byte_perm(lo, hi, q | (q << 4) | ((4 + q) << 8) | ((4 + q) << 12));
                a[q] = (v & 0xFF00FF00u) | ((wd << (12 - 4 * q)) & 0x80008000u);
            }
            const float sg = e4m3f((scb[(2 * j) >> 2] >> (8 * ((2 * j) & 3))) & 0xff);
            const float sg8 = e4m3f((scb[(2 * j + 1) >> 2] >> (8 * ((2 * j + 1) & 3))) & 0xff);
            const __half2 h0 = __float2half2_rn(sg), h1 = __float2half2_rn(sg8);
            f[j][0] = hmul2u(a[0], h0); f[j][1] = hmul2u(a[1], h1); f[j][2] = hmul2u(a[2], h0); f[j][3] = hmul2u(a[3], h1);
        }
    }
    template <int NT>
    static __device__ __forceinline__ void step(const Raw& r, const uint32_t (&b)[4][NT][2], float (&acc)[NT][4], Smem&, int, int) {
        const uint32_t words[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
        const uint32_t scb[2] = {r.sc.x, r.sc.y};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t wd = words[j];
            const uint32_t lo = __byte_perm(0x3E3C3800u, 0x46444240u, wd & 0x7777u);
            const uint32_t hi = __byte_perm(0x3E3C3800u, 0x46444240u, (wd >> 16) & 0x7777u);
            uint32_t a[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint32_t v = __byte_perm(lo, hi, q | (q << 4) | ((4 + q) << 8) | ((4 + q) << 12));
                a[q] = (v & 0xFF00FF00u) | ((wd << (12 - 4 * q)) & 0x80008000u);
            }
            // The block scales are folded into the A fragment: an e2m1 value (one mantissa bit) times an e4m3 scale (three)
            // is exact in fp16, so the mma accumulates the scaled products and nothing is left to do per mma. Rows gi
            // (a0, a2) and gi + 8 (a1, a3) have their own scale.
            const float sg = e4m3f((scb[(2 * j) >> 2] >> (8 * ((2 * j) & 3))) & 0xff);
            const float sg8 = e4m3f((scb[(2 * j + 1) >> 2] >> (8 * ((2 * j + 1) & 3))) & 0xff);
            const __half2 h0 = __float2half2_rn(sg), h1 = __float2half2_rn(sg8);
            const uint32_t m0 = hmul2u(a[0], h0), m1 = hmul2u(a[1], h1), m2 = hmul2u(a[2], h0), m3 = hmul2u(a[3], h1);
#pragma unroll
            for (int nt = 0; nt < NT; nt++) mma_f16(acc[nt], m0, m1, m2, m3, b[j][nt][0], b[j][nt][1]);
        }
    }
};

struct PQ8 {
    using W = TQ8;
    struct Smem { int unused; };
    struct Raw { uint4 q0, q1; uint32_t d0, d1; };
    static constexpr bool kPermK = false;   // B fragments in natural k order
    static constexpr bool kBF = false;
    static __device__ __forceinline__ int steps(const W& w) { return w.k >> 6; }
    static __device__ __forceinline__ int tiles(const W& w) { return (w.n + 15) >> 4; }
    static __device__ __forceinline__ void init(Smem&) {}
    static __device__ __forceinline__ float scale(const W&) { return 1.f; }
    static __device__ __forceinline__ Raw load(const W& w, int tile, int s, int lane, int g) {
        size_t KB = w.k >> 5;
        Raw r;
        size_t b0 = (size_t)tile * KB + 2 * s;
        r.q0 = __ldg(w.q + b0 * 32 + lane);
        r.q1 = __ldg(w.q + (b0 + 1) * 32 + lane);
        r.d0 = __ldg(w.d + b0 * 8 + g);
        r.d1 = __ldg(w.d + (b0 + 1) * 8 + g);
        return r;
    }
    static constexpr int kStage = 1024 + 64;   // two blocks of 32 x 16 B, two of 8 x 4 B scales
    static __device__ __forceinline__ void stage(const W& w, int tile, int s, int lane, int g, unsigned char* dst) {
        const size_t KB = w.k >> 5, b0 = (size_t)tile * KB + 2 * s;
        w_cp16(dst + lane * 16, w.q + b0 * 32 + lane);
        w_cp16(dst + 512 + lane * 16, w.q + (b0 + 1) * 32 + lane);
        if (lane < 16) w_cp4(dst + 1024 + lane * 4, w.d + b0 * 8 + lane);   // d0 (8 words) then d1 (8 words), adjacent
    }
    static __device__ __forceinline__ Raw unstage(const unsigned char* src, int lane, int g) {
        Raw r;
        r.q0 = *reinterpret_cast<const uint4*>(src + lane * 16);
        r.q1 = *reinterpret_cast<const uint4*>(src + 512 + lane * 16);
        r.d0 = *reinterpret_cast<const uint32_t*>(src + 1024 + g * 4);
        r.d1 = *reinterpret_cast<const uint32_t*>(src + 1024 + 32 + g * 4);
        return r;
    }
    static __device__ __forceinline__ uint32_t cvt(uint32_t x, uint32_t sel) {
        uint32_t v = __byte_perm(x, 0x64646464u, sel);
        __half2 h = __hsub2(*reinterpret_cast<__half2*>(&v), __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
        return *reinterpret_cast<uint32_t*>(&h);
    }
    static constexpr bool kMmaBF = false;
    static __device__ __forceinline__ void frag(const Raw& r, uint32_t (&f)[4][4], Smem&, int, int) {
#pragma unroll
        for (int kbi = 0; kbi < 2; kbi++) {
            const uint4 q = kbi ? r.q1 : r.q0;
            const uint32_t dd = kbi ? r.d1 : r.d0;
            const __half2 s_lo = __half2half2(__ushort_as_half((unsigned short)(dd & 0xffff)));
            const __half2 s_hi = __half2half2(__ushort_as_half((unsigned short)(dd >> 16)));
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const uint32_t x0 = (u ? q.z : q.x) ^ 0x80808080u;
                const uint32_t x1 = (u ? q.w : q.y) ^ 0x80808080u;
                f[kbi * 2 + u][0] = hmul2u(cvt(x0, 0x4140), s_lo); f[kbi * 2 + u][1] = hmul2u(cvt(x0, 0x4342), s_hi);
                f[kbi * 2 + u][2] = hmul2u(cvt(x1, 0x4140), s_lo); f[kbi * 2 + u][3] = hmul2u(cvt(x1, 0x4342), s_hi);
            }
        }
    }
    template <int NT>
    static __device__ __forceinline__ void step(const Raw& r, const uint32_t (&b)[4][NT][2], float (&acc)[NT][4], Smem&, int, int) {
#pragma unroll
        for (int kbi = 0; kbi < 2; kbi++) {
            const uint4 q = kbi ? r.q1 : r.q0;
            const uint32_t dd = kbi ? r.d1 : r.d0;
            // the block scale goes into the A fragment (an int8 times an fp16 scale: one rounding of 2^-11), rows gi
            // (a0, a2) and gi + 8 (a1, a3) each with their own, so the mma accumulates across blocks as it is
            const __half2 s_lo = __half2half2(__ushort_as_half((unsigned short)(dd & 0xffff)));
            const __half2 s_hi = __half2half2(__ushort_as_half((unsigned short)(dd >> 16)));
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const uint32_t x0 = (u ? q.z : q.x) ^ 0x80808080u;
                const uint32_t x1 = (u ? q.w : q.y) ^ 0x80808080u;
                const uint32_t a0 = hmul2u(cvt(x0, 0x4140), s_lo), a1 = hmul2u(cvt(x0, 0x4342), s_hi);
                const uint32_t a2 = hmul2u(cvt(x1, 0x4140), s_lo), a3 = hmul2u(cvt(x1, 0x4342), s_hi);
#pragma unroll
                for (int nt = 0; nt < NT; nt++) mma_f16(acc[nt], a0, a1, a2, a3, b[kbi * 2 + u][nt][0], b[kbi * 2 + u][nt][1]);
            }
        }
    }
};

struct PBF {
    using W = TBF;
    struct Smem { int unused; };
    struct Raw { uint4 a[4]; };
    static constexpr bool kPermK = false;   // B fragments in natural k order
    static constexpr bool kBF = true;
    static __device__ __forceinline__ int steps(const W& w) { return w.k >> 6; }
    static __device__ __forceinline__ int tiles(const W& w) { return (w.n + 15) >> 4; }
    static __device__ __forceinline__ void init(Smem&) {}
    static __device__ __forceinline__ float scale(const W&) { return 1.f; }
    static __device__ __forceinline__ Raw load(const W& w, int tile, int s, int lane, int) {
        size_t K16 = w.k >> 4;
        Raw r;
#pragma unroll
        for (int j = 0; j < 4; j++) r.a[j] = __ldg(w.w + ((size_t)tile * K16 + 4 * s + j) * 32 + lane);
        return r;
    }
    static constexpr int kStage = 2048;   // four blocks of 32 x 16 B
    static __device__ __forceinline__ void stage(const W& w, int tile, int s, int lane, int, unsigned char* dst) {
        const size_t K16 = w.k >> 4;
#pragma unroll
        for (int j = 0; j < 4; j++) w_cp16(dst + (j * 32 + lane) * 16, w.w + ((size_t)tile * K16 + 4 * s + j) * 32 + lane);
    }
    static __device__ __forceinline__ Raw unstage(const unsigned char* src, int lane, int) {
        Raw r;
#pragma unroll
        for (int j = 0; j < 4; j++) r.a[j] = *reinterpret_cast<const uint4*>(src + (j * 32 + lane) * 16);
        return r;
    }
    static constexpr bool kMmaBF = true;
    static __device__ __forceinline__ void frag(const Raw& r, uint32_t (&f)[4][4], Smem&, int, int) {
#pragma unroll
        for (int j = 0; j < 4; j++) { f[j][0] = r.a[j].x; f[j][1] = r.a[j].y; f[j][2] = r.a[j].z; f[j][3] = r.a[j].w; }
    }
    template <int NT>
    static __device__ __forceinline__ void step(const Raw& r, const uint32_t (&b)[4][NT][2], float (&acc)[NT][4], Smem&, int, int) {
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) mma_bf16(acc[nt], r.a[j].x, r.a[j].y, r.a[j].z, r.a[j].w, b[j][nt][0], b[j][nt][1]);
    }
};

struct PIQ2 {
    using W = TIQ2;
    struct Smem { uint32_t tbl[4][256]; uint32_t msk[4][128]; };
    struct Raw { uint4 qa, qb; float da, db; };
    static constexpr bool kPermK = false;   // B fragments in natural k order
    static constexpr bool kBF = true;
    static __device__ __forceinline__ int steps(const W& w) { return w.k >> 6; }
    static __device__ __forceinline__ int tiles(const W& w) { return (w.n + 15) >> 4; }
    static __device__ __forceinline__ float scale(const W&) { return 1.f; }
    // lane t of the warp reads elements 2t, 2t+1 of each 8-weight grid entry and of its sign byte
    static __device__ void init(Smem& sm) {
        for (int i = threadIdx.x; i < 4 * 256; i += blockDim.x) {
            int t = i >> 8, idx = i & 255;
            unsigned long long gr = iq2xxs_grid[idx];
            float lo = (float)((gr >> (16 * t)) & 0xff), hi = (float)((gr >> (16 * t + 8)) & 0xff);
            __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
            sm.tbl[t][idx] = *reinterpret_cast<uint32_t*>(&h);
        }
        for (int i = threadIdx.x; i < 4 * 128; i += blockDim.x) {
            int t = i >> 7, sidx = i & 127;
            unsigned sg = sidx | ((__popc(sidx) & 1) << 7);
            sm.msk[t][sidx] = (((sg >> (2 * t)) & 1) ? 0x00008000u : 0u) | (((sg >> (2 * t + 1)) & 1) ? 0x80000000u : 0u);
        }
        __syncthreads();
    }
    static __device__ __forceinline__ Raw load(const W& w, int tile, int s, int lane, int g) {
        size_t C2 = w.k >> 6, NB = w.k >> 8;
        Raw r;
        const uint4* base = w.q + ((size_t)tile * C2 + s) * 16;
        r.qa = __ldg(base + g);
        r.qb = __ldg(base + g + 8);
        const uint16_t* dp = w.d + ((size_t)tile * NB + (s >> 2)) * 16;
        r.da = __half2float(__ushort_as_half(__ldg(dp + g)));
        r.db = __half2float(__ushort_as_half(__ldg(dp + g + 8)));
        return r;
    }
    static constexpr int kStage = 256 + 32;   // 16 x 16 B of grid bytes, 16 halves of block scales
    static __device__ __forceinline__ void stage(const W& w, int tile, int s, int lane, int, unsigned char* dst) {
        const size_t C2 = w.k >> 6, NB = w.k >> 8;
        if (lane < 16) w_cp16(dst + lane * 16, w.q + ((size_t)tile * C2 + s) * 16 + lane);
        else if (lane < 24) w_cp4(dst + 256 + (lane - 16) * 4, w.d + ((size_t)tile * NB + (s >> 2)) * 16 + (lane - 16) * 2);
    }
    static __device__ __forceinline__ Raw unstage(const unsigned char* src, int lane, int g) {
        Raw r;
        r.qa = *reinterpret_cast<const uint4*>(src + g * 16);
        r.qb = *reinterpret_cast<const uint4*>(src + (g + 8) * 16);
        const uint16_t* dp = reinterpret_cast<const uint16_t*>(src + 256);
        r.da = __half2float(__ushort_as_half(dp[g]));
        r.db = __half2float(__ushort_as_half(dp[g + 8]));
        return r;
    }
    static constexpr bool kMmaBF = true;
    static __device__ __forceinline__ void frag(const Raw& r, uint32_t (&f)[4][4], Smem& sm, int, int t) {
        const uint32_t* tbl = sm.tbl[t];
        const uint32_t* msk = sm.msk[t];
#pragma unroll
        for (int ch = 0; ch < 2; ch++) {
            const uint32_t ax = ch ? r.qa.z : r.qa.x, ay = ch ? r.qa.w : r.qa.y;
            const uint32_t bx = ch ? r.qb.z : r.qb.x, by = ch ? r.qb.w : r.qb.y;
            const float dba = r.da * (0.5f + (float)(ay >> 28)) * 0.25f;
            const float dbb = r.db * (0.5f + (float)(by >> 28)) * 0.25f;
            const __nv_bfloat162 sa = __float2bfloat162_rn(dba), sb = __float2bfloat162_rn(dbb);
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int l0 = 2 * u, l1 = 2 * u + 1;
                f[ch * 2 + u][0] = bmul2u(tbl[(ax >> (8 * l0)) & 255] ^ msk[(ay >> (7 * l0)) & 127], sa);
                f[ch * 2 + u][2] = bmul2u(tbl[(ax >> (8 * l1)) & 255] ^ msk[(ay >> (7 * l1)) & 127], sa);
                f[ch * 2 + u][1] = bmul2u(tbl[(bx >> (8 * l0)) & 255] ^ msk[(by >> (7 * l0)) & 127], sb);
                f[ch * 2 + u][3] = bmul2u(tbl[(bx >> (8 * l1)) & 255] ^ msk[(by >> (7 * l1)) & 127], sb);
            }
        }
    }
    template <int NT>
    static __device__ __forceinline__ void step(const Raw& r, const uint32_t (&b)[4][NT][2], float (&acc)[NT][4], Smem& sm, int, int t) {
        const uint32_t* tbl = sm.tbl[t];
        const uint32_t* msk = sm.msk[t];
#pragma unroll
        for (int ch = 0; ch < 2; ch++) {
            const uint32_t ax = ch ? r.qa.z : r.qa.x, ay = ch ? r.qa.w : r.qa.y;
            const uint32_t bx = ch ? r.qb.z : r.qb.x, by = ch ? r.qb.w : r.qb.y;
            // the chunk's scale (block d times the 4-bit sub-scale) goes into the A fragment in bf16: the grid values
            // are exact, the product is rounded once (2^-9, far below the 2-bit quantization), and the mma accumulates
            // across chunks without a per-chunk epilogue
            const float dba = r.da * (0.5f + (float)(ay >> 28)) * 0.25f;
            const float dbb = r.db * (0.5f + (float)(by >> 28)) * 0.25f;
            const __nv_bfloat162 sa = __float2bfloat162_rn(dba), sb = __float2bfloat162_rn(dbb);
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int l0 = 2 * u, l1 = 2 * u + 1;
                const uint32_t a0 = bmul2u(tbl[(ax >> (8 * l0)) & 255] ^ msk[(ay >> (7 * l0)) & 127], sa);
                const uint32_t a2 = bmul2u(tbl[(ax >> (8 * l1)) & 255] ^ msk[(ay >> (7 * l1)) & 127], sa);
                const uint32_t a1 = bmul2u(tbl[(bx >> (8 * l0)) & 255] ^ msk[(by >> (7 * l0)) & 127], sb);
                const uint32_t a3 = bmul2u(tbl[(bx >> (8 * l1)) & 255] ^ msk[(by >> (7 * l1)) & 127], sb);
#pragma unroll
                for (int nt = 0; nt < NT; nt++) mma_bf16(acc[nt], a0, a1, a2, a3, b[ch * 2 + u][nt][0], b[ch * 2 + u][nt][1]);
            }
        }
    }
};

struct PIQ2W {
    using W = TIQ2;
    struct Smem { uint4 tbl8[256]; uint4 msk8[128]; };
    struct Raw { uint4 qa, qb; float da, db; };
    // Whole grid entries: inside each chunk of 32 k, lane t owns grid entry t (real k 8t..8t+7) for its fragment slots of
    // the chunk's two MMAs, and the B fragments are read at the same real k (4 consecutive halves, one 8-byte load).
    // One 16-byte table read for the eight values and one for their signs instead of eight 4-byte ones; the products
    // are the same, only the order in which the tensor core adds the 16 of an MMA changes.
    static constexpr bool kPermK = true;
    static constexpr bool kBF = true;
    static __device__ __forceinline__ int steps(const W& w) { return w.k >> 6; }
    static __device__ __forceinline__ int tiles(const W& w) { return (w.n + 15) >> 4; }
    static __device__ __forceinline__ float scale(const W&) { return 1.f; }
    // lane t of the warp reads elements 2t, 2t+1 of each 8-weight grid entry and of its sign byte
    static __device__ void init(Smem& sm) {
        for (int idx = threadIdx.x; idx < 256; idx += blockDim.x) {
            const unsigned long long gr = iq2xxs_grid[idx];
            uint32_t wv[4];
#pragma unroll
            for (int t = 0; t < 4; t++) {
                const float lo = (float)((gr >> (16 * t)) & 0xff), hi = (float)((gr >> (16 * t + 8)) & 0xff);
                __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
                wv[t] = *reinterpret_cast<uint32_t*>(&h);
            }
            sm.tbl8[idx] = make_uint4(wv[0], wv[1], wv[2], wv[3]);
        }
        for (int sidx = threadIdx.x; sidx < 128; sidx += blockDim.x) {
            const unsigned sg = sidx | ((__popc(sidx) & 1) << 7);
            uint32_t wv[4];
#pragma unroll
            for (int t = 0; t < 4; t++) wv[t] = (((sg >> (2 * t)) & 1) ? 0x00008000u : 0u) | (((sg >> (2 * t + 1)) & 1) ? 0x80000000u : 0u);
            sm.msk8[sidx] = make_uint4(wv[0], wv[1], wv[2], wv[3]);
        }
        __syncthreads();
    }
    static __device__ __forceinline__ Raw load(const W& w, int tile, int s, int lane, int g) {
        size_t C2 = w.k >> 6, NB = w.k >> 8;
        Raw r;
        const uint4* base = w.q + ((size_t)tile * C2 + s) * 16;
        r.qa = __ldg(base + g);
        r.qb = __ldg(base + g + 8);
        const uint16_t* dp = w.d + ((size_t)tile * NB + (s >> 2)) * 16;
        r.da = __half2float(__ushort_as_half(__ldg(dp + g)));
        r.db = __half2float(__ushort_as_half(__ldg(dp + g + 8)));
        return r;
    }
    static constexpr int kStage = 256 + 32;   // 16 x 16 B of grid bytes, 16 halves of block scales
    static __device__ __forceinline__ void stage(const W& w, int tile, int s, int lane, int, unsigned char* dst) {
        const size_t C2 = w.k >> 6, NB = w.k >> 8;
        if (lane < 16) w_cp16(dst + lane * 16, w.q + ((size_t)tile * C2 + s) * 16 + lane);
        else if (lane < 24) w_cp4(dst + 256 + (lane - 16) * 4, w.d + ((size_t)tile * NB + (s >> 2)) * 16 + (lane - 16) * 2);
    }
    static __device__ __forceinline__ Raw unstage(const unsigned char* src, int lane, int g) {
        Raw r;
        r.qa = *reinterpret_cast<const uint4*>(src + g * 16);
        r.qb = *reinterpret_cast<const uint4*>(src + (g + 8) * 16);
        const uint16_t* dp = reinterpret_cast<const uint16_t*>(src + 256);
        r.da = __half2float(__ushort_as_half(dp[g]));
        r.db = __half2float(__ushort_as_half(dp[g + 8]));
        return r;
    }
    static constexpr bool kMmaBF = true;
    static __device__ __forceinline__ void frag(const Raw& r, uint32_t (&f)[4][4], Smem& sm, int, int t) {
#pragma unroll
        for (int ch = 0; ch < 2; ch++) {
            const uint32_t ax = ch ? r.qa.z : r.qa.x, ay = ch ? r.qa.w : r.qa.y;
            const uint32_t bx = ch ? r.qb.z : r.qb.x, by = ch ? r.qb.w : r.qb.y;
            const __nv_bfloat162 sa = __float2bfloat162_rn(r.da * (0.5f + (float)(ay >> 28)) * 0.25f);
            const __nv_bfloat162 sb = __float2bfloat162_rn(r.db * (0.5f + (float)(by >> 28)) * 0.25f);
            const uint4 ga = sm.tbl8[(ax >> (8 * t)) & 255], ma = sm.msk8[(ay >> (7 * t)) & 127];
            const uint4 gb = sm.tbl8[(bx >> (8 * t)) & 255], mb = sm.msk8[(by >> (7 * t)) & 127];
            // MMA u of the chunk: a0 = values 4u, 4u+1 (row g), a2 = 4u+2, 4u+3; a1, a3 the same for row g+8
            f[ch * 2][0] = bmul2u(ga.x ^ ma.x, sa); f[ch * 2][2] = bmul2u(ga.y ^ ma.y, sa);
            f[ch * 2][1] = bmul2u(gb.x ^ mb.x, sb); f[ch * 2][3] = bmul2u(gb.y ^ mb.y, sb);
            f[ch * 2 + 1][0] = bmul2u(ga.z ^ ma.z, sa); f[ch * 2 + 1][2] = bmul2u(ga.w ^ ma.w, sa);
            f[ch * 2 + 1][1] = bmul2u(gb.z ^ mb.z, sb); f[ch * 2 + 1][3] = bmul2u(gb.w ^ mb.w, sb);
        }
    }
    template <int NT>
    static __device__ __forceinline__ void step(const Raw& r, const uint32_t (&b)[4][NT][2], float (&acc)[NT][4], Smem& sm, int g, int t) {
        uint32_t f[4][4];
        frag(r, f, sm, g, t);
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) mma_bf16(acc[nt], f[j][0], f[j][1], f[j][2], f[j][3], b[j][nt][0], b[j][nt][1]);
    }
};

// ------------------------------------------------------------------------------------------------------
// The GEMM. A block is four warps splitting K for RT row tiles; NT is how many 8-token tiles the rows need.
// ------------------------------------------------------------------------------------------------------

constexpr int RING_STAGES = 3;
template <class P, int RT>
__host__ __device__ constexpr size_t ring_bytes() { return (size_t)4 * RING_STAGES * RT * P::kStage; }   // a CTA's four warps

// RING: the warp's weight pieces come through a per-warp three-stage cp.async ring, two steps ahead (about 2 KB in
// flight a warp for the NVFP4 projections instead of one 16-byte piece a lane); the same bytes, the same sums.
template <class P, int NT, int RT, class X = float, bool RING = false>
__device__ __forceinline__ void mma_body(const typename P::W& w, const X* __restrict__ x, int ldx, float* __restrict__ y, int ldy,
                                         int Tn, int epi, int tile0, unsigned char* ring = nullptr) {
    __shared__ typename P::Smem sm;
    __shared__ float red[4][RT][NT][4][32];
    P::init(sm);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ntiles = P::tiles(w), steps = P::steps(w);
    float acc[RT][NT][4];
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int n = 0; n < NT; n++) { acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f; }

    typename P::Raw cur[RT], nxt[RT];
    unsigned char* wring = ring ? ring + (size_t)warp * RING_STAGES * RT * P::kStage : nullptr;
    const int nsw = steps > warp ? (steps - warp + 3) / 4 : 0;   // steps this warp walks: warp, warp + 4, ...
    auto issue = [&](int i) {
#pragma unroll
        for (int r = 0; r < RT; r++) P::stage(w, min(tile0 + r, ntiles - 1), warp + 4 * i, lane, g, wring + ((i % RING_STAGES) * RT + r) * P::kStage);
        w_cp_commit();
    };
    int s = warp;
    if constexpr (RING) {
        if (nsw > 0) issue(0);
        if (nsw > 1) issue(1);
    } else if (s < steps) {
#pragma unroll
        for (int r = 0; r < RT; r++) cur[r] = P::load(w, min(tile0 + r, ntiles - 1), s, lane, g);
    }
    int si = 0;
    for (; s < steps; s += 4, si++) {
        const int sn = s + 4;
        if constexpr (RING) {
            if (si + 1 < nsw) w_cp_wait<1>(); else w_cp_wait<0>();
            __syncwarp();
#pragma unroll
            for (int r = 0; r < RT; r++) cur[r] = P::unstage(wring + ((si % RING_STAGES) * RT + r) * P::kStage, lane, g);
            __syncwarp();   // every lane has its pieces before the stage can be refilled
            if (si + 2 < nsw) issue(si + 2);
        } else if (sn < steps) {
#pragma unroll
            for (int r = 0; r < RT; r++) nxt[r] = P::load(w, min(tile0 + r, ntiles - 1), sn, lane, g);
        }
        uint32_t b[4][NT][2];
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const int tok = nt * 8 + g;
                if (tok < Tn) {
                    if constexpr (P::kPermK) {
                        // the lane's four consecutive k of this MMA (chunk j >> 1, half j & 1, entry t)
                        const size_t off = (size_t)tok * ldx + s * 64 + (j >> 1) * 32 + 8 * t + 4 * (j & 1);
                        if constexpr (std::is_same<X, uint16_t>::value) {
                            const uint2 v = __ldg(reinterpret_cast<const uint2*>(x + off));
                            b[j][nt][0] = v.x; b[j][nt][1] = v.y;
                        } else {
                            const float4 v = __ldg(reinterpret_cast<const float4*>(x + off));
                            b[j][nt][0] = pack2<P::kBF>(v.x, v.y); b[j][nt][1] = pack2<P::kBF>(v.z, v.w);
                        }
                    } else if constexpr (std::is_same<X, uint16_t>::value) {
                        // the rows were packed once for the launch (pack_x_k, the same rounding), 2 bytes an element
                        const uint16_t* p = x + (size_t)tok * ldx + s * 64 + 16 * j + 2 * t;
                        b[j][nt][0] = __ldg(reinterpret_cast<const uint32_t*>(p));
                        b[j][nt][1] = __ldg(reinterpret_cast<const uint32_t*>(p + 8));
                    } else {
                        const float* p = x + (size_t)tok * ldx + s * 64 + 16 * j + 2 * t;
                        float2 v0 = __ldg(reinterpret_cast<const float2*>(p));
                        float2 v1 = __ldg(reinterpret_cast<const float2*>(p + 8));
                        b[j][nt][0] = pack2<P::kBF>(v0.x, v0.y);
                        b[j][nt][1] = pack2<P::kBF>(v1.x, v1.y);
                    }
                } else {
                    b[j][nt][0] = 0u;
                    b[j][nt][1] = 0u;
                }
            }
#pragma unroll
        for (int r = 0; r < RT; r++) P::template step<NT>(cur[r], b, acc[r], sm, g, t);
        if constexpr (!RING) {
            if (sn < steps) {
#pragma unroll
                for (int r = 0; r < RT; r++) cur[r] = nxt[r];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int q = 0; q < 4; q++) red[warp][r][n][q][lane] = acc[r][n][q];
    __syncthreads();
    const float s2 = P::scale(w);
    for (int e = threadIdx.x; e < RT * NT * 4 * 32; e += blockDim.x) {
        const int ln = e & 31, q = (e >> 5) & 3, n = (e >> 7) % NT, r = (e >> 7) / NT;
        float v = red[0][r][n][q][ln] + red[1][r][n][q][ln] + red[2][r][n][q][ln] + red[3][r][n][q][ln];
        const int lg = ln >> 2, lt = ln & 3;
        const int row = (tile0 + r) * 16 + lg + ((q >> 1) ? 8 : 0);
        const int tok = n * 8 + 2 * lt + (q & 1);
        if (row < w.n && tok < Tn) {
            v *= s2;
            if (epi == 1) v = v > 0.f ? v * v : 0.f;
            y[(size_t)tok * ldy + row] = v;
        }
    }
}

// Many tokens (prompt chunks): the CTA owns 4 warps x RT row tiles and a group of 8*NT tokens. Each warp walks all
// of K for its own rows; the group's activations are converted and staged in shared memory 64 k at a time, so a
// token's row is read from L2 once per CTA and every warp takes its B fragments from shared memory.
template <class P, int NT, int RT, class X = float>
__global__ void __launch_bounds__(128) mma_big_k(typename P::W w, const X* __restrict__ x, int ldx, float* __restrict__ y, int ldy,
                                                 int Tn, int epi) {
    pdl_wait();
    constexpr int TOK = NT * 8;
    constexpr int STR = 72;                                  // halves per staged token row: 64 + 8 keeps fragment loads conflict-free
    __shared__ typename P::Smem sm;
    __shared__ __align__(16) uint16_t xs[2][TOK * STR];
    P::init(sm);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int g0 = blockIdx.x * TOK;
    const int tile0 = (blockIdx.y * 4 + warp) * RT;
    const int ntiles = P::tiles(w), steps = P::steps(w);
    const int tn = min(Tn - g0, TOK);
    float acc[RT][NT][4];
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int n = 0; n < NT; n++) { acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f; }

    auto stage = [&](int buf, int s) {   // 64 k of TOK tokens into shared memory as packed halves
        for (int i = threadIdx.x; i < TOK * 16; i += 128) {
            const int tok = i >> 4, k4 = (i & 15) * 4;
            uint2 pk;
            if constexpr (std::is_same<X, uint16_t>::value) {
                // packed once for the launch (pack_x_k, the same rounding)
                pk = tok < tn ? __ldg(reinterpret_cast<const uint2*>(x + (size_t)(g0 + tok) * ldx + s * 64 + k4)) : make_uint2(0u, 0u);
            } else {
                float4 v = tok < tn ? __ldg(reinterpret_cast<const float4*>(x + (size_t)(g0 + tok) * ldx + s * 64 + k4)) : make_float4(0.f, 0.f, 0.f, 0.f);
                pk = make_uint2(pack2<P::kBF>(v.x, v.y), pack2<P::kBF>(v.z, v.w));
            }
            *reinterpret_cast<uint2*>(&xs[buf][tok * STR + k4]) = pk;
        }
    };
    typename P::Raw cur[RT], nxt[RT];
    if (steps > 0) {
#pragma unroll
        for (int r = 0; r < RT; r++) cur[r] = P::load(w, min(tile0 + r, ntiles - 1), 0, lane, g);
    }
    stage(0, 0);
    for (int s = 0; s < steps; s++) {
        __syncthreads();
        const int buf = s & 1;
        if (s + 1 < steps) {
#pragma unroll
            for (int r = 0; r < RT; r++) nxt[r] = P::load(w, min(tile0 + r, ntiles - 1), s + 1, lane, g);
            stage(buf ^ 1, s + 1);
        }
        uint32_t b[4][NT][2];
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                if constexpr (P::kPermK) {
                    const uint2 v = *reinterpret_cast<const uint2*>(&xs[buf][(nt * 8 + g) * STR + (j >> 1) * 32 + 8 * t + 4 * (j & 1)]);
                    b[j][nt][0] = v.x; b[j][nt][1] = v.y;
                } else {
                    const uint16_t* p = &xs[buf][(nt * 8 + g) * STR + 16 * j + 2 * t];
                    b[j][nt][0] = *reinterpret_cast<const uint32_t*>(p);
                    b[j][nt][1] = *reinterpret_cast<const uint32_t*>(p + 8);
                }
            }
#pragma unroll
        for (int r = 0; r < RT; r++) P::template step<NT>(cur[r], b, acc[r], sm, g, t);
        if (s + 1 < steps) {
#pragma unroll
            for (int r = 0; r < RT; r++) cur[r] = nxt[r];
        }
    }
    const float s2 = P::scale(w);
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int row = (tile0 + r) * 16 + g + ((q >> 1) ? 8 : 0);
                const int tok = n * 8 + 2 * t + (q & 1);
                if (row < w.n && tok < tn) {
                    float v = acc[r][n][q] * s2;
                    if (epi == 1) v = v > 0.f ? v * v : 0.f;
                    y[(size_t)(g0 + tok) * ldy + row] = v;
                }
            }
    pdl_trigger();
}

// ------------------------------------------------------------------------------------------------------
// Prompt chunks, tiled: a CTA of 8 warps owns 128 weight rows x 128 tokens. Each K step of 64, every warp
// dequantizes one 16-row tile into shared memory (the fragments step() would build, so the products are the
// same), the tokens' packed rows arrive by cp.async into a three-stage ring, and both operands reach the tensor
// cores through ldmatrix: a warp (4 x 2 grid: 32 rows x 64 tokens) does 64 MMAs a step from 8 + 16 shared-memory
// instructions. Sums run over K in order, as mma_big_k does, so the outputs are the same.
// ------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

constexpr int TILE_BM = 128, TILE_BN = 128, TILE_BK = 64, TILE_STR = TILE_BK + 8, TILE_STAGES = 3;
template <class P>
__host__ __device__ constexpr size_t tile_smem_head() { return (sizeof(typename P::Smem) + 15) & ~(size_t)15; }   // the tables, then 16-byte aligned buffers
template <class P>
__host__ __device__ constexpr size_t tile_smem_bytes() {
    return tile_smem_head<P>() + (size_t)2 * TILE_BM * TILE_STR * 2 + (size_t)TILE_STAGES * TILE_BN * TILE_STR * 2;
}

template <class P>
__global__ void __launch_bounds__(256, 1) mma_tile_k(typename P::W w, const uint16_t* __restrict__ x, int ldx, float* __restrict__ y, int ldy,
                                                     int Tn, int epi) {
    pdl_wait();
    extern __shared__ __align__(16) unsigned char tsm[];
    typename P::Smem& sm = *reinterpret_cast<typename P::Smem*>(tsm);
    uint16_t* wa = reinterpret_cast<uint16_t*>(tsm + tile_smem_head<P>());                // [2][BM][STR]
    uint16_t* xb = wa + 2 * TILE_BM * TILE_STR;                                            // [STAGES][BN][STR]
    P::init(sm);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wr = warp >> 1, wt = warp & 1;
    const int g0 = blockIdx.x * TILE_BN;
    const int ntiles = P::tiles(w), steps = P::steps(w);
    const int tn = min(Tn - g0, TILE_BN);
    const int mytile = min(blockIdx.y * 8 + warp, ntiles - 1);
    float acc[2][8][4];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int n = 0; n < 8; n++) { acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f; }

    auto issue = [&](int stage, int s) {   // the tokens' packed rows for step s: 128 rows x 128 bytes, 16 B a piece
        uint16_t* base = xb + (size_t)stage * TILE_BN * TILE_STR;
        for (int i = tid; i < TILE_BN * 8; i += 256) {
            const int tok = i >> 3, p8 = (i & 7) * 8;
            uint16_t* dst = base + tok * TILE_STR + p8;
            if (tok < tn) cp_async16(dst, x + (size_t)(g0 + tok) * ldx + s * TILE_BK + p8);
            else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        }
        cp_async_commit();
    };
    // ldmatrix lane addressing: matrix lane >> 3, row lane & 7
    const int lm = lane >> 3, lr = lane & 7;
    auto dequant = [&](const typename P::Raw& r, int buf) {   // this warp's tile of a step into wa[buf]
        uint32_t f[4][4];
        P::frag(r, f, sm, g, t);
        uint16_t* wrow = wa + (size_t)buf * TILE_BM * TILE_STR + (warp * 16 + g) * TILE_STR;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            *reinterpret_cast<uint32_t*>(wrow + 16 * j + 2 * t) = f[j][0];
            *reinterpret_cast<uint32_t*>(wrow + 8 * TILE_STR + 16 * j + 2 * t) = f[j][1];
            *reinterpret_cast<uint32_t*>(wrow + 16 * j + 8 + 2 * t) = f[j][2];
            *reinterpret_cast<uint32_t*>(wrow + 8 * TILE_STR + 16 * j + 8 + 2 * t) = f[j][3];
        }
    };
    // Software pipeline: the weights are two steps ahead in registers and the tokens two stages ahead in flight;
    // a step issues its MMAs first and dequantizes the next step's tile while they drain, then one barrier.
    typename P::Raw cur, nxt, nxt2;
    cur = P::load(w, mytile, 0, lane, g);
    if (steps > 1) nxt = P::load(w, mytile, 1, lane, g);
    issue(0, 0);
    if (steps > 1) issue(1, 1);
    dequant(cur, 0);
    if (steps > 1) cp_async_wait<1>(); else cp_async_wait<0>();
    __syncthreads();
    for (int s = 0; s < steps; s++) {
        if (s + 2 < steps) { issue((s + 2) % TILE_STAGES, s + 2); nxt2 = P::load(w, mytile, s + 2, lane, g); }
        const uint16_t* wbuf = wa + (size_t)(s & 1) * TILE_BM * TILE_STR;
        const uint16_t* xbuf = xb + (size_t)(s % TILE_STAGES) * TILE_BN * TILE_STR;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t a[2][4];
#pragma unroll
            for (int rt = 0; rt < 2; rt++) {
                const int row = wr * 32 + rt * 16 + lr + (lm & 1) * 8;
                ldsm_x4(a[rt][0], a[rt][1], a[rt][2], a[rt][3], wbuf + row * TILE_STR + 16 * j + (lm >> 1) * 8);
            }
#pragma unroll
            for (int tp = 0; tp < 4; tp++) {   // two token tiles an ldmatrix
                uint32_t b0, b1, b2, b3;
                const int tok = wt * 64 + tp * 16 + lr + (lm >> 1) * 8;
                ldsm_x4(b0, b1, b2, b3, xbuf + tok * TILE_STR + 16 * j + (lm & 1) * 8);
#pragma unroll
                for (int rt = 0; rt < 2; rt++) {
                    if constexpr (P::kMmaBF) {
                        mma_bf16(acc[rt][2 * tp], a[rt][0], a[rt][1], a[rt][2], a[rt][3], b0, b1);
                        mma_bf16(acc[rt][2 * tp + 1], a[rt][0], a[rt][1], a[rt][2], a[rt][3], b2, b3);
                    } else {
                        mma_f16(acc[rt][2 * tp], a[rt][0], a[rt][1], a[rt][2], a[rt][3], b0, b1);
                        mma_f16(acc[rt][2 * tp + 1], a[rt][0], a[rt][1], a[rt][2], a[rt][3], b2, b3);
                    }
                }
            }
        }
        if (s + 1 < steps) dequant(nxt, (s + 1) & 1);
        if (s + 2 < steps) cp_async_wait<1>(); else cp_async_wait<0>();
        __syncthreads();
        cur = nxt; nxt = nxt2;
    }
    const float s2 = P::scale(w);
    const int row0 = blockIdx.y * TILE_BM + wr * 32;
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int row = row0 + rt * 16 + g + ((q >> 1) ? 8 : 0);
                const int tok = wt * 64 + n * 8 + 2 * t + (q & 1);
                if (row < w.n && tok < tn) {
                    float v = acc[rt][n][q] * s2;
                    if (epi == 1) v = v > 0.f ? v * v : 0.f;
                    y[(size_t)(g0 + tok) * ldy + row] = v;
                }
            }
    pdl_trigger();
}

template <class P>
static void launch_tile(cudaStream_t s, const typename P::W& w, const uint16_t* xp, float* y, int ldy, int T, int epi) {
    static bool opted = false;
    if (!opted) { cudaFuncSetAttribute(mma_tile_k<P>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tile_smem_bytes<P>()); opted = true; }
    const int tiles = (w.n + 15) >> 4;
    launch_pdl(mma_tile_k<P>, dim3((T + TILE_BN - 1) / TILE_BN, (tiles + 7) / 8), 256, tile_smem_bytes<P>(), s, w, xp, w.k, y, ldy, T, epi);
}

// blockIdx.x walks the token groups (8 * NT rows each) so CTAs that share weights run side by side and
// the weights come from DRAM once.
template <class P, int NT, int RT, class X, bool RING = false>
__global__ void __launch_bounds__(128) mma_k(typename P::W w, const X* __restrict__ x, int ldx, float* __restrict__ y, int ldy, int Tn, int epi) {
    pdl_wait();
    extern __shared__ __align__(16) unsigned char ring_dyn[];
    const int g0 = blockIdx.x * NT * 8;
    mma_body<P, NT, RT, X, RING>(w, x + (size_t)g0 * ldx, ldx, y + (size_t)g0 * ldy, ldy, min(Tn - g0, NT * 8), epi, blockIdx.y * RT, RING ? ring_dyn : nullptr);
    pdl_trigger();
}
// the ring kernels need their dynamic shared memory allowed once an instantiation
template <class P, int NT, int RT, class X>
static void launch_ring(dim3 grid, cudaStream_t s, const typename P::W& w, const X* x, int ldx, float* y, int ldy, int T, int epi) {
    static bool opted = false;
    if (!opted) { cudaFuncSetAttribute(mma_k<P, NT, RT, X, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ring_bytes<P, RT>()); opted = true; }
    launch_pdl(mma_k<P, NT, RT, X, true>, grid, 128, ring_bytes<P, RT>(), s, w, x, ldx, y, ldy, T, epi);
}
// BOLT_DENSE_RING=1: measured level (0.92 to 1.02x, identical outputs): the NVFP4 and Q8_0 decode GEMMs already run at the
// DRAM ceiling (230 to 245 GB/s on the large matrices) and the small ones are launch-bound; off by default
static bool dense_ring() { static const char* e = getenv("BOLT_DENSE_RING"); return e && atoi(e) != 0; }

// Verify blocks of many rows: every CTA of the K-split kernel re-reads its tokens' activations for each pair of row
// tiles, so at 64 rows the fp32 rows cost more L2 traffic than the weights cost DRAM. These pack the rows once a
// launch into halves (the rounding mma_body would apply), in a scratch of the launch's stream; the GEMM's sums are
// the same in the same order.
template <bool BF>
__global__ void pack_x_k(const float* __restrict__ x, int ldx, uint16_t* __restrict__ xp, int T, int K) {
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int per = K >> 2;
    if (i >= (size_t)T * per) return;
    const int tok = (int)(i / per), k4 = (int)(i % per) * 4;
    const float4 v = __ldg(reinterpret_cast<const float4*>(x + (size_t)tok * ldx + k4));
    *reinterpret_cast<uint2*>(xp + (size_t)tok * K + k4) = make_uint2(pack2<BF>(v.x, v.y), pack2<BF>(v.z, v.w));
    pdl_trigger();
}
struct PackSlot { cudaStream_t s; uint16_t* buf; size_t elems; };
static PackSlot g_pack[4];
static int g_npack = 0;
static bool g_pack_on = true;
static uint16_t* pack_for(cudaStream_t s, size_t need) {
    if (!g_pack_on) return nullptr;
    for (int i = 0; i < g_npack; i++) if (g_pack[i].s == s && g_pack[i].elems >= need) return g_pack[i].buf;
    return nullptr;
}
template <bool BF>
static uint16_t* pack_rows(cudaStream_t s, const float* x, int ldx, int T, int K) {
    uint16_t* xp = pack_for(s, (size_t)T * K);
    if (!xp) return nullptr;
    const size_t n4 = (size_t)T * (K >> 2);
    launch_pdl(pack_x_k<BF>, (unsigned)((n4 + 255) / 256), 256, 0, s, x, ldx, xp, T, K);
    return xp;
}

// Decode and verify blocks sum K in the K-split kernel's order at any row count, so a stream's rounding does not change
// when other streams grow the block past 32 rows; the prompt pass keeps the faster large-block kernel.
static bool g_decode_rows = false;

template <class P>
void launch_mma(cudaStream_t s, const typename P::W& w, const float* x, int ldx, float* y, int ldy, int T, int epi) {
    const int tiles = (w.n + 15) >> 4;
    constexpr int RT = 2;
    const int rg = (tiles + RT - 1) / RT;
    // BOLT_RT1=1: one row tile a CTA for small N (the shared expert, fc1/fc2, out_proj), twice the warps in flight.
    // Measured slower at 16 streams (the activations are re-read per CTA), level alone; off by default.
    static const bool rt1 = getenv("BOLT_RT1") && atoi(getenv("BOLT_RT1")) != 0;
    if (rt1 && tiles < 640 && (T <= 32 || g_decode_rows)) {
        if (T <= 8) { launch_pdl(mma_k<P, 1, 1, float>, dim3(1, tiles), 128, 0, s, w, x, ldx, y, ldy, T, epi); return; }
        const uint16_t* xp = pack_rows<P::kBF>(s, x, ldx, T, w.k);
        if (xp) {
            if (T <= 16) launch_pdl(mma_k<P, 2, 1, uint16_t>, dim3(1, tiles), 128, 0, s, w, xp, w.k, y, ldy, T, epi);
            else if (T <= 32) launch_pdl(mma_k<P, 4, 1, uint16_t>, dim3(1, tiles), 128, 0, s, w, xp, w.k, y, ldy, T, epi);
            else launch_pdl(mma_k<P, 8, 1, uint16_t>, dim3((T + 63) / 64, tiles), 128, 0, s, w, xp, w.k, y, ldy, T, epi);
            return;
        }
    }
    const bool ring = dense_ring();
    if (T <= 8) { if (ring) launch_ring<P, 1, RT, float>(dim3(1, rg), s, w, x, ldx, y, ldy, T, epi); else launch_pdl(mma_k<P, 1, RT, float>, dim3(1, rg), 128, 0, s, w, x, ldx, y, ldy, T, epi); }
    else if (T <= 32 || g_decode_rows) {
        const uint16_t* xp = pack_rows<P::kBF>(s, x, ldx, T, w.k);
        if (xp && ring) {
            if (T <= 16) launch_ring<P, 2, RT, uint16_t>(dim3(1, rg), s, w, xp, w.k, y, ldy, T, epi);
            else if (T <= 32) launch_ring<P, 4, RT, uint16_t>(dim3(1, rg), s, w, xp, w.k, y, ldy, T, epi);
            else launch_ring<P, 8, RT, uint16_t>(dim3((T + 63) / 64, rg), s, w, xp, w.k, y, ldy, T, epi);
        } else if (xp) {
            if (T <= 16) launch_pdl(mma_k<P, 2, RT, uint16_t>, dim3(1, rg), 128, 0, s, w, xp, w.k, y, ldy, T, epi);
            else if (T <= 32) launch_pdl(mma_k<P, 4, RT, uint16_t>, dim3(1, rg), 128, 0, s, w, xp, w.k, y, ldy, T, epi);
            else launch_pdl(mma_k<P, 8, RT, uint16_t>, dim3((T + 63) / 64, rg), 128, 0, s, w, xp, w.k, y, ldy, T, epi);
        } else if (T <= 16) launch_pdl(mma_k<P, 2, RT, float>, dim3(1, rg), 128, 0, s, w, x, ldx, y, ldy, T, epi);
        else if (T <= 32) launch_pdl(mma_k<P, 4, RT, float>, dim3(1, rg), 128, 0, s, w, x, ldx, y, ldy, T, epi);
        else launch_pdl(mma_k<P, 8, RT, float>, dim3((T + 63) / 64, rg), 128, 0, s, w, x, ldx, y, ldy, T, epi);   // 64 tokens a CTA: the weights are decoded once per 64
    } else {
        constexpr int BR = 2;                       // row tiles a warp owns
        const int rg2 = (tiles + 4 * BR - 1) / (4 * BR);
        // the prompt pass: every CTA stages its 64 tokens' rows from L2 for each of its row tiles, so the rows are
        // packed to halves once a launch when the scratch holds them (half the traffic, no conversion in the kernel)
        // BOLT_TILE=1: the 128x128 tile kernel (mma_tile_k). Measured on the microbenchmark at 0.9 to 0.97x of mma_big_k for
        // the quantized shapes (both near 60 % of the 107 TFLOPS fp16 mma.sync peak of GB10) and 1.28x for bf16; off by default.
        static const bool tiled = getenv("BOLT_TILE") && atoi(getenv("BOLT_TILE")) != 0;
        if (const uint16_t* xp = pack_rows<P::kBF>(s, x, ldx, T, w.k)) {
            if (T < 64) launch_pdl(mma_big_k<P, 4, BR, uint16_t>, dim3((T + 31) / 32, rg2), 128, 0, s, w, xp, w.k, y, ldy, T, epi);
            else if (tiled && T >= 256 && !P::kPermK) launch_tile<P>(s, w, xp, y, ldy, T, epi);
            else launch_pdl(mma_big_k<P, 8, BR, uint16_t>, dim3((T + 63) / 64, rg2), 128, 0, s, w, xp, w.k, y, ldy, T, epi);
        } else if (T < 64) launch_pdl(mma_big_k<P, 4, BR>, dim3((T + 31) / 32, rg2), 128, 0, s, w, x, ldx, y, ldy, T, epi);
        else launch_pdl(mma_big_k<P, 8, BR>, dim3((T + 63) / 64, rg2), 128, 0, s, w, x, ldx, y, ldy, T, epi);
    }
}

template <int NT, int RT, class X>
__global__ void __launch_bounds__(128) mma_q8_multi_k(TQ8Multi m, const X* __restrict__ x, int ldx, int Tn) {
    pdl_wait();
    int t0 = blockIdx.y * RT;
    int which = 0;
    while (which < 2 && t0 >= ((m.w[which].n + 15) >> 4)) { t0 -= (m.w[which].n + 15) >> 4; which++; }
    const int g0 = blockIdx.x * NT * 8;
    mma_body<PQ8, NT, RT, X>(m.w[which], x + (size_t)g0 * ldx, ldx, m.y[which] + (size_t)g0 * m.ldy[which], m.ldy[which], min(Tn - g0, NT * 8), 0, t0);
    pdl_trigger();
}

}  // namespace

void mma_decode_rows(bool on) { g_decode_rows = on; }
void mma_pack_buffer(cudaStream_t s, uint16_t* buf, size_t elems) {
    static bool first = true;
    if (first) { first = false; const char* e = getenv("BOLT_PACK"); if (e && atoi(e) == 0) g_pack_on = false; }
    if (g_npack < 4) g_pack[g_npack++] = {s, buf, elems};
}
void mma_nv4(cudaStream_t s, const TNV4& w, const float* x, int ldx, float* y, int ldy, int T) { launch_mma<PNV4>(s, w, x, ldx, y, ldy, T, 0); }
void mma_q8(cudaStream_t s, const TQ8& w, const float* x, int ldx, float* y, int ldy, int T) { launch_mma<PQ8>(s, w, x, ldx, y, ldy, T, 0); }
void mma_bf16(cudaStream_t s, const TBF& w, const float* x, int ldx, float* y, int ldy, int T) { launch_mma<PBF>(s, w, x, ldx, y, ldy, T, 0); }
void mma_iq2(cudaStream_t s, const TIQ2& w, const float* x, int ldx, float* y, int ldy, int T, int epi) {
    // BOLT_IQ2_WHOLE=1: whole grid entries with K permuted inside the MMA (1.8 to 2x on the kernel alone, fp32-rounding-level
    // differences). Off by default: the shared expert runs on a side stream under the routed experts, so end to end it
    // measured within noise while changing replies; a numerics-tolerant profile can turn it on.
    static const bool whole = getenv("BOLT_IQ2_WHOLE") && atoi(getenv("BOLT_IQ2_WHOLE")) != 0;
    if (whole && (g_decode_rows || T <= 32)) launch_mma<PIQ2W>(s, w, x, ldx, y, ldy, T, epi);
    else launch_mma<PIQ2>(s, w, x, ldx, y, ldy, T, epi);
}
void mma_q8_multi(cudaStream_t s, const TQ8Multi& m, const float* x, int ldx, int T) {
    int tiles = 0;
    for (int i = 0; i < 3; i++) tiles += (m.w[i].n + 15) >> 4;
    int rg = (tiles + 1) / 2;
    if (T <= 8) { launch_pdl(mma_q8_multi_k<1, 2, float>, dim3(1, rg), 128, 0, s, m, x, ldx, T); return; }
    const int K = m.w[0].k;
    if (const uint16_t* xp = pack_rows<PQ8::kBF>(s, x, ldx, T, K)) {
        if (T <= 16) launch_pdl(mma_q8_multi_k<2, 2, uint16_t>, dim3(1, rg), 128, 0, s, m, xp, K, T);
        else launch_pdl(mma_q8_multi_k<4, 2, uint16_t>, dim3((T + 31) / 32, rg), 128, 0, s, m, xp, K, T);
    } else if (T <= 16) launch_pdl(mma_q8_multi_k<2, 2, float>, dim3(1, rg), 128, 0, s, m, x, ldx, T);
    else launch_pdl(mma_q8_multi_k<4, 2, float>, dim3((T + 31) / 32, rg), 128, 0, s, m, x, ldx, T);
}

// ------------------------------------------------------------------------------------------------------
// Repacking from the checkpoint layouts.
// ------------------------------------------------------------------------------------------------------

size_t tile_bytes_nv4(int n, int k) { return (size_t)((n + 15) / 16) * (k / 64) * 32 * 16; }
size_t tile_scale_bytes_nv4(int n, int k) { return (size_t)((n + 15) / 16) * (k / 64) * 64; }
size_t tile_bytes_q8(int n, int k) { return (size_t)((n + 15) / 16) * (k / 32) * 32 * 16; }
size_t tile_scale_bytes_q8(int n, int k) { return (size_t)((n + 15) / 16) * (k / 32) * 8 * 4; }
size_t tile_bytes_bf16(int n, int k) { return (size_t)((n + 15) / 16) * (k / 16) * 32 * 16; }
size_t tile_bytes_iq2(int n, int k) { return (size_t)((n + 15) / 16) * (k / 64) * 16 * 16; }
size_t tile_scale_bytes_iq2(int n, int k) { return (size_t)((n + 15) / 16) * (k / 256) * 16 * 2; }

namespace {

__global__ void tile_nv4_k(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sc, uint32_t* __restrict__ qt,
                           uint8_t* __restrict__ sct, int n, int k, int n16, int S) {
    pdl_wait();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n16 * S * 32) return;
    const int lane = i & 31;
    const size_t sidx = i >> 5;
    const int s = sidx % S, rt = sidx / S;
    const int g = lane >> 2, t = lane & 3;
    for (int j = 0; j < 4; j++) {
        uint32_t word = 0;
        for (int r = 0; r < 4; r++) {
            const int row = rt * 16 + g + 8 * (r & 1);
            const int kk = s * 64 + 16 * j + 8 * (r >> 1) + 2 * t;
            uint32_t lo = 0, hi = 0;
            if (row < n) {
                uint8_t byte = q[(size_t)row * (k / 2) + kk / 2];  // kk even: low nibble is kk, high nibble kk+1
                lo = byte & 15;
                hi = byte >> 4;
            }
            word |= (lo << (4 * r)) | (hi << (16 + 4 * r));
        }
        qt[(sidx * 32 + lane) * 4 + j] = word;
    }
    if (t == 0) {
        for (int j = 0; j < 4; j++)
            for (int h = 0; h < 2; h++) {
                const int row = rt * 16 + g + 8 * h;
                sct[(sidx * 8 + g) * 8 + 2 * j + h] = row < n ? sc[(size_t)row * (k / 16) + 4 * s + j] : 0;
            }
    }
    pdl_trigger();
}
void tile_nv4_launch(cudaStream_t s, const uint8_t* q, const uint8_t* sc, void* qt, void* sct, int n, int k) {
    int n16 = (n + 15) / 16, S = k / 64;
    size_t th = (size_t)n16 * S * 32;
    launch_pdl(tile_nv4_k, (unsigned)((th + 255) / 256), 256, 0, s, q, sc, (uint32_t*)qt, (uint8_t*)sct, n, k, n16, S);
}

__global__ void tile_q8_k(const uint8_t* __restrict__ raw, uint8_t* __restrict__ qt, uint32_t* __restrict__ dt, int n, int k, int n16, int KB) {
    pdl_wait();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n16 * KB * 32) return;
    const int lane = i & 31;
    const size_t bidx = i >> 5;
    const int kb = bidx % KB, rt = bidx / KB;
    const int g = lane >> 2, t = lane & 3;
    uint8_t* out = qt + (bidx * 32 + lane) * 16;
    for (int u = 0; u < 2; u++)
        for (int r = 0; r < 4; r++) {
            const int row = rt * 16 + g + 8 * (r & 1);
            const int kk = 16 * u + 8 * (r >> 1) + 2 * t;
            uint8_t b0 = 0, b1 = 0;
            if (row < n) {
                const uint8_t* blk = raw + ((size_t)row * KB + kb) * 34 + 2;
                b0 = blk[kk];
                b1 = blk[kk + 1];
            }
            out[8 * u + 2 * r] = b0;
            out[8 * u + 2 * r + 1] = b1;
        }
    if (t == 0) {
        uint32_t d = 0;
        for (int h = 0; h < 2; h++) {
            const int row = rt * 16 + g + 8 * h;
            uint32_t half = 0;
            if (row < n) {
                const uint8_t* blk = raw + ((size_t)row * KB + kb) * 34;
                half = (uint32_t)blk[0] | ((uint32_t)blk[1] << 8);
            }
            d |= half << (16 * h);
        }
        dt[bidx * 8 + g] = d;
    }
    pdl_trigger();
}
void tile_q8_launch(cudaStream_t s, const uint8_t* raw, void* qt, void* dt, int n, int k) {
    int n16 = (n + 15) / 16, KB = k / 32;
    size_t th = (size_t)n16 * KB * 32;
    launch_pdl(tile_q8_k, (unsigned)((th + 255) / 256), 256, 0, s, raw, (uint8_t*)qt, (uint32_t*)dt, n, k, n16, KB);
}

__global__ void tile_bf16_k(const uint16_t* __restrict__ w, uint32_t* __restrict__ wt, int n, int k, int n16, int K16) {
    pdl_wait();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n16 * K16 * 32) return;
    const int lane = i & 31;
    const size_t sidx = i >> 5;
    const int s = sidx % K16, rt = sidx / K16;
    const int g = lane >> 2, t = lane & 3;
    for (int r = 0; r < 4; r++) {
        const int row = rt * 16 + g + 8 * (r & 1);
        const int kk = 16 * s + 8 * (r >> 1) + 2 * t;
        uint32_t v = 0;
        if (row < n) v = (uint32_t)w[(size_t)row * k + kk] | ((uint32_t)w[(size_t)row * k + kk + 1] << 16);
        wt[(sidx * 32 + lane) * 4 + r] = v;
    }
    pdl_trigger();
}
void tile_bf16_launch(cudaStream_t s, const uint16_t* w, void* wt, int n, int k) {
    int n16 = (n + 15) / 16, K16 = k / 16;
    size_t th = (size_t)n16 * K16 * 32;
    launch_pdl(tile_bf16_k, (unsigned)((th + 255) / 256), 256, 0, s, w, (uint32_t*)wt, n, k, n16, K16);
}

__global__ void tile_iq2_k(const uint8_t* __restrict__ raw, uint8_t* __restrict__ qt, uint16_t* __restrict__ dt, int n, int k, int n16, int C2, int NB) {
    pdl_wait();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n16 * C2 * 16) return;
    const int r16 = i & 15;
    const size_t cidx = i >> 4;
    const int c2 = cidx % C2, rt = cidx / C2;
    const int row = rt * 16 + r16;
    const int b = c2 >> 2;
    uint8_t* out = qt + i * 16;
    if (row < n) {
        const uint8_t* blk = raw + ((size_t)row * NB + b) * 66;
        const uint8_t* src = blk + 2 + (c2 & 3) * 16;
        for (int j = 0; j < 16; j++) out[j] = src[j];
        if ((c2 & 3) == 0) dt[((size_t)rt * NB + b) * 16 + r16] = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    } else {
        for (int j = 0; j < 16; j++) out[j] = 0;
        if ((c2 & 3) == 0) dt[((size_t)rt * NB + b) * 16 + r16] = 0;
    }
    pdl_trigger();
}
void tile_iq2_launch(cudaStream_t s, const uint8_t* raw, void* qt, void* dt, int n, int k) {
    int n16 = (n + 15) / 16, C2 = k / 64, NB = k / 256;
    size_t th = (size_t)n16 * C2 * 16;
    launch_pdl(tile_iq2_k, (unsigned)((th + 255) / 256), 256, 0, s, raw, (uint8_t*)qt, (uint16_t*)dt, n, k, n16, C2, NB);
}

}  // namespace

void tile_nv4(cudaStream_t s, const uint8_t* q, const uint8_t* sc, void* qt, void* sct, int n, int k) { tile_nv4_launch(s, q, sc, qt, sct, n, k); }
void tile_q8(cudaStream_t s, const uint8_t* raw, void* qt, void* dt, int n, int k) { tile_q8_launch(s, raw, qt, dt, n, k); }
void tile_bf16(cudaStream_t s, const uint16_t* w, void* wt, int n, int k) { tile_bf16_launch(s, w, wt, n, k); }
void tile_iq2(cudaStream_t s, const uint8_t* raw, void* qt, void* dt, int n, int k) { tile_iq2_launch(s, raw, qt, dt, n, k); }

// ------------------------------------------------------------------------------------------------------
// Prompt attention on tensor cores. A CTA is eight warps; warp w takes query t0 + w with all 16 heads of one KV
// head as the 16 rows of an m16 tile. S = Q K^T and O += P V are m16n8k16 bf16 mmas, softmax is online in
// registers, and the S accumulators become the next mma's A fragments directly. K is staged as [key][dim] and V
// transposed as [dim][key], both bf16 as stored in the cache. Keys are split across CTAs; a combine kernel
// merges the partials.
// ------------------------------------------------------------------------------------------------------

namespace {

constexpr int FA_Q = 8;                      // queries (warps) per CTA
constexpr int FA_KS = 136;                   // K tile row stride in bf16: 272 bytes, conflict-free fragment loads
constexpr int FA_VS = 40;                    // V^T tile row stride in bf16: 80 bytes
constexpr int FA_PART = FA_Q * (32 + 16 * 128);  // per CTA partial: per warp m[16], l[16], o[16][128]

// MULTI: a CTA is one segment (a verify window of up to 8 rows of one sequence, warp w = its row w) instead of eight
// consecutive rows of the prompt chunk.
// Eight e4m3 bytes times a scale as eight bf16.
__device__ __forceinline__ uint4 fp8x8_to_bf16(uint2 raw, float scale) {
    uint32_t w[4];
    const uint32_t in[2] = {raw.x, raw.y};
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)((in[i] >> (16 * j)) & 0xffff), __NV_E4M3);
            float2 f = __half22float2(__half2(hr));
            __nv_bfloat162 b = __floats2bfloat162_rn(f.x * scale, f.y * scale);
            w[2 * i + j] = *reinterpret_cast<uint32_t*>(&b);
        }
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

template <bool MULTI>
__global__ void __launch_bounds__(256) attn_mma_part_k(const Ctl* __restrict__ ctl, const float* __restrict__ q,
                                                       KvView kvc, float* __restrict__ part, int splits, int T) {
    pdl_wait();
    const int sp = blockIdx.x, g = blockIdx.y, qb = blockIdx.z;
    if (MULTI && qb >= ctl->nseg) return;
    const int sgi = MULTI ? qb : 0;
    const size_t seq = ctl->seg_seq[sgi];
    const int ctx = kvc.ctx;
    const bool fp8 = kvc.fp8;
    const int base0 = ctl->seg_base[sgi];
    __shared__ __align__(16) uint16_t Ks[32 * FA_KS];
    __shared__ __align__(16) uint16_t Vt[128 * FA_VS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gi = lane >> 2, ti = lane & 3;
    const int row0 = MULTI ? ctl->seg_row0[sgi] : qb * FA_Q;                 // first query row of this CTA
    const int nrow = MULTI ? ctl->seg_row0[sgi + 1] - row0 : min(T - qb * FA_Q, FA_Q);
    const int t = row0 + warp;                                                // query row (global)
    const int local = MULTI ? warp : qb * FA_Q + warp;                        // its position within the segment
    const bool live = warp < nrow;
    const int len = live ? base0 + local + 1 : 0;
    const int maxlen = base0 + (MULTI ? nrow : min(T, qb * FA_Q + FA_Q));
    const int chunk = MULTI ? win_chunk(base0) : ((maxlen + splits - 1) / splits + 31) / 32 * 32;
    const int p0 = sp * chunk, p1 = min(maxlen, p0 + chunk);

    // Q fragments: rows are the 16 heads of this KV head, k is the head dimension
    uint32_t qa[8][4];
    const float scale = rsqrtf(128.f);
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
        const float* r0 = q + (size_t)(live ? t : 0) * 4096 + (g * 16 + gi) * 128 + ks * 16 + 2 * ti;
        const float* r1 = r0 + 8 * 128;
        float2 a = *reinterpret_cast<const float2*>(r0), b = *reinterpret_cast<const float2*>(r1);
        float2 c = *reinterpret_cast<const float2*>(r0 + 8), d = *reinterpret_cast<const float2*>(r1 + 8);
        qa[ks][0] = pack2<true>(a.x * scale, a.y * scale);
        qa[ks][1] = pack2<true>(b.x * scale, b.y * scale);
        qa[ks][2] = pack2<true>(c.x * scale, c.y * scale);
        qa[ks][3] = pack2<true>(d.x * scale, d.y * scale);
    }
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;

    for (int base = p0; base < p1; base += 32) {
        __syncthreads();
        // stage K [key][dim] and V transposed [dim][key]; keys past the end are zero
        for (int i = tid; i < 32 * 16; i += 256) {
            int key = i >> 4, c8 = (i & 15) * 8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (base + key < p1) {
                const size_t row = (seq * ctx + base + key) * 2 + g;
                if (!fp8) {
                    kv = __ldg(reinterpret_cast<const uint4*>((const uint16_t*)kvc.k + row * 128 + c8));
                    vv = __ldg(reinterpret_cast<const uint4*>((const uint16_t*)kvc.v + row * 128 + c8));
                } else {
                    kv = fp8x8_to_bf16(__ldg(reinterpret_cast<const uint2*>((const uint8_t*)kvc.k + row * 128 + c8)), kvc.ks[row]);
                    vv = fp8x8_to_bf16(__ldg(reinterpret_cast<const uint2*>((const uint8_t*)kvc.v + row * 128 + c8)), kvc.vs[row]);
                }
            }
            *reinterpret_cast<uint4*>(&Ks[key * FA_KS + c8]) = kv;
            const uint16_t* vp = reinterpret_cast<const uint16_t*>(&vv);
#pragma unroll
            for (int j = 0; j < 8; j++) Vt[(c8 + j) * FA_VS + key] = vp[j];
        }
        __syncthreads();
        if (!live) continue;
        // S = Q K^T for 32 keys: four n8 tiles
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f; }
#pragma unroll
        for (int ks = 0; ks < 8; ks++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const uint32_t* kp = reinterpret_cast<const uint32_t*>(&Ks[(nt * 8 + gi) * FA_KS + ks * 16 + 2 * ti]);
                mma_bf16(s[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], kp[0], kp[4]);
            }
        // causal and end-of-range mask, then the online softmax update for rows gi and gi + 8
        float tm0 = -1e30f, tm1 = -1e30f;
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                int key = base + nt * 8 + 2 * ti + (c & 1);
                if (key >= len || key >= p1) s[nt][c] = -1e30f;
                if (c < 2) tm0 = fmaxf(tm0, s[nt][c]); else tm1 = fmaxf(tm1, s[nt][c]);
            }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1)); tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1)); tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const float sc0 = __expf(m0 - n0), sc1 = __expf(m1 - n1);
        m0 = n0; m1 = n1;
        l0 *= sc0; l1 *= sc1;
#pragma unroll
        for (int i = 0; i < 16; i++) { o[i][0] *= sc0; o[i][1] *= sc0; o[i][2] *= sc1; o[i][3] *= sc1; }
        uint32_t pa[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            float p00 = s[nt][0] > -1e29f ? __expf(s[nt][0] - n0) : 0.f, p01 = s[nt][1] > -1e29f ? __expf(s[nt][1] - n0) : 0.f;
            float p10 = s[nt][2] > -1e29f ? __expf(s[nt][2] - n1) : 0.f, p11 = s[nt][3] > -1e29f ? __expf(s[nt][3] - n1) : 0.f;
            l0 += p00 + p01; l1 += p10 + p11;
            const int j = nt >> 1, hi = nt & 1;   // k16 step j; n-tile 2j feeds a0,a1 and n-tile 2j+1 feeds a2,a3
            pa[j][hi * 2 + 0] = pack2<true>(p00, p01);
            pa[j][hi * 2 + 1] = pack2<true>(p10, p11);
        }
        // O += P V: sixteen n8 tiles of the head dimension
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int d = 0; d < 16; d++) {
                const uint32_t* vp = reinterpret_cast<const uint32_t*>(&Vt[(d * 8 + gi) * FA_VS + j * 16 + 2 * ti]);
                mma_bf16(o[d], pa[j][0], pa[j][1], pa[j][2], pa[j][3], vp[0], vp[4]);
            }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    float* out = part + (size_t)((qb * 2 + g) * splits + sp) * FA_PART + warp * (32 + 16 * 128);
    if (ti == 0) { out[gi] = m0; out[gi + 8] = m1; out[16 + gi] = l0; out[16 + gi + 8] = l1; }
#pragma unroll
    for (int d = 0; d < 16; d++) {
        float* r0 = out + 32 + gi * 128 + d * 8 + 2 * ti;
        float* r1 = out + 32 + (gi + 8) * 128 + d * 8 + 2 * ti;
        r0[0] = o[d][0]; r0[1] = o[d][1]; r1[0] = o[d][2]; r1[1] = o[d][3];
    }
    pdl_trigger();
}

template <bool MULTI>
__global__ void attn_mma_comb_k(const Ctl* __restrict__ ctl, const float* __restrict__ part, float* __restrict__ out, int splits) {
    pdl_wait();
    int t = blockIdx.x, h = blockIdx.y;
    int g = h >> 4, hh = h & 15;
    int qb = t / FA_Q, w = t % FA_Q;
    int d = threadIdx.x;
    if (MULTI) {
        if (t >= ctl->nrows) { out[(size_t)t * 4096 + h * 128 + d] = 0.f; return; }   // padding row
        int sg = 0;
        while (sg + 1 < ctl->nseg && t >= ctl->seg_row0[sg + 1]) sg++;
        qb = sg; w = t - ctl->seg_row0[sg];
    }
    const float* base = part + (size_t)((qb * 2 + g) * splits) * FA_PART + w * (32 + 16 * 128);
    float M = -1e30f;
    for (int s = 0; s < splits; s++) M = fmaxf(M, base[(size_t)s * FA_PART + hh]);
    float L = 0.f, O = 0.f;
    for (int s = 0; s < splits; s++) {
        const float* p = base + (size_t)s * FA_PART;
        float wgt = __expf(p[hh] - M);
        L += wgt * p[16 + hh];
        O += wgt * p[32 + hh * 128 + d];
    }
    out[(size_t)t * 4096 + h * 128 + d] = O / L;
    pdl_trigger();
}

}  // namespace

void k_attn_mma(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out,
                int T, int splits) {
    launch_pdl(attn_mma_part_k<false>, dim3(splits, 2, (T + FA_Q - 1) / FA_Q), 256, 0, s, ctl, q, kv, part, splits, T);
    launch_pdl(attn_mma_comb_k<false>, dim3(T, 32), 128, 0, s, ctl, part, out, splits);
}
// Verify windows of up to nseg sequences (nseg a launch bound, ctl->nseg of them used), each of at most 8 rows.
void k_attn_mma_windows(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out,
                        int T, int nseg, int splits) {
    launch_pdl(attn_mma_part_k<true>, dim3(splits, 2, nseg), 256, 0, s, ctl, q, kv, part, splits, T);
    launch_pdl(attn_mma_comb_k<true>, dim3(T, 32), 128, 0, s, ctl, part, out, splits);
}
size_t attn_mma_part_floats(int T, int splits) { return (size_t)((T + FA_Q - 1) / FA_Q) * 2 * splits * FA_PART; }
