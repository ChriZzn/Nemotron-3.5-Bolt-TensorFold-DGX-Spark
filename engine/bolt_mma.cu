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
    static __device__ __forceinline__ uint32_t cvt(uint32_t x, uint32_t sel) {
        uint32_t v = __byte_perm(x, 0x64646464u, sel);
        __half2 h = __hsub2(*reinterpret_cast<__half2*>(&v), __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
        return *reinterpret_cast<uint32_t*>(&h);
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

// ------------------------------------------------------------------------------------------------------
// The GEMM. A block is four warps splitting K for RT row tiles; NT is how many 8-token tiles the rows need.
// ------------------------------------------------------------------------------------------------------

template <class P, int NT, int RT>
__device__ __forceinline__ void mma_body(const typename P::W& w, const float* __restrict__ x, int ldx, float* __restrict__ y, int ldy,
                                         int Tn, int epi, int tile0) {
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
    int s = warp;
    if (s < steps) {
#pragma unroll
        for (int r = 0; r < RT; r++) cur[r] = P::load(w, min(tile0 + r, ntiles - 1), s, lane, g);
    }
    for (; s < steps; s += 4) {
        const int sn = s + 4;
        if (sn < steps) {
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
                    const float* p = x + (size_t)tok * ldx + s * 64 + 16 * j + 2 * t;
                    float2 v0 = __ldg(reinterpret_cast<const float2*>(p));
                    float2 v1 = __ldg(reinterpret_cast<const float2*>(p + 8));
                    b[j][nt][0] = pack2<P::kBF>(v0.x, v0.y);
                    b[j][nt][1] = pack2<P::kBF>(v1.x, v1.y);
                } else {
                    b[j][nt][0] = 0u;
                    b[j][nt][1] = 0u;
                }
            }
#pragma unroll
        for (int r = 0; r < RT; r++) P::template step<NT>(cur[r], b, acc[r], sm, g, t);
        if (sn < steps) {
#pragma unroll
            for (int r = 0; r < RT; r++) cur[r] = nxt[r];
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
template <class P, int NT, int RT>
__global__ void __launch_bounds__(128) mma_big_k(typename P::W w, const float* __restrict__ x, int ldx, float* __restrict__ y, int ldy,
                                                 int Tn, int epi) {
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

    auto stage = [&](int buf, int s) {   // fp32 -> packed halves, 64 k of TOK tokens
        for (int i = threadIdx.x; i < TOK * 16; i += 128) {
            const int tok = i >> 4, k4 = (i & 15) * 4;
            float4 v = tok < tn ? __ldg(reinterpret_cast<const float4*>(x + (size_t)(g0 + tok) * ldx + s * 64 + k4)) : make_float4(0.f, 0.f, 0.f, 0.f);
            uint2 pk = make_uint2(pack2<P::kBF>(v.x, v.y), pack2<P::kBF>(v.z, v.w));
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
                const uint16_t* p = &xs[buf][(nt * 8 + g) * STR + 16 * j + 2 * t];
                b[j][nt][0] = *reinterpret_cast<const uint32_t*>(p);
                b[j][nt][1] = *reinterpret_cast<const uint32_t*>(p + 8);
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
}

// blockIdx.x walks the token groups (8 * NT rows each) so CTAs that share weights run side by side and
// the weights come from DRAM once.
template <class P, int NT, int RT>
__global__ void __launch_bounds__(128) mma_k(typename P::W w, const float* __restrict__ x, int ldx, float* __restrict__ y, int ldy, int Tn, int epi) {
    const int g0 = blockIdx.x * NT * 8;
    mma_body<P, NT, RT>(w, x + (size_t)g0 * ldx, ldx, y + (size_t)g0 * ldy, ldy, min(Tn - g0, NT * 8), epi, blockIdx.y * RT);
}

// Decode and verify blocks sum K in the K-split kernel's order at any row count, so a stream's rounding does not change
// when other streams grow the block past 32 rows; the prompt pass keeps the faster large-block kernel.
static bool g_decode_rows = false;

template <class P>
void launch_mma(cudaStream_t s, const typename P::W& w, const float* x, int ldx, float* y, int ldy, int T, int epi) {
    const int tiles = (w.n + 15) >> 4;
    constexpr int RT = 2;
    const int rg = (tiles + RT - 1) / RT;
    if (T <= 8) mma_k<P, 1, RT><<<dim3(1, rg), 128, 0, s>>>(w, x, ldx, y, ldy, T, epi);
    else if (T <= 16) mma_k<P, 2, RT><<<dim3(1, rg), 128, 0, s>>>(w, x, ldx, y, ldy, T, epi);
    else if (T <= 32) mma_k<P, 4, RT><<<dim3(1, rg), 128, 0, s>>>(w, x, ldx, y, ldy, T, epi);
    else if (g_decode_rows) mma_k<P, 8, RT><<<dim3((T + 63) / 64, rg), 128, 0, s>>>(w, x, ldx, y, ldy, T, epi);   // 64 tokens a CTA: the weights are decoded once per 64
    else {
        constexpr int BR = 2;                       // row tiles a warp owns
        const int rg2 = (tiles + 4 * BR - 1) / (4 * BR);
        if (T < 64) mma_big_k<P, 4, BR><<<dim3((T + 31) / 32, rg2), 128, 0, s>>>(w, x, ldx, y, ldy, T, epi);
        else mma_big_k<P, 8, BR><<<dim3((T + 63) / 64, rg2), 128, 0, s>>>(w, x, ldx, y, ldy, T, epi);
    }
}

template <int NT, int RT>
__global__ void __launch_bounds__(128) mma_q8_multi_k(TQ8Multi m, const float* __restrict__ x, int ldx, int Tn) {
    int t0 = blockIdx.y * RT;
    int which = 0;
    while (which < 2 && t0 >= ((m.w[which].n + 15) >> 4)) { t0 -= (m.w[which].n + 15) >> 4; which++; }
    const int g0 = blockIdx.x * NT * 8;
    mma_body<PQ8, NT, RT>(m.w[which], x + (size_t)g0 * ldx, ldx, m.y[which] + (size_t)g0 * m.ldy[which], m.ldy[which], min(Tn - g0, NT * 8), 0, t0);
}

}  // namespace

void mma_decode_rows(bool on) { g_decode_rows = on; }
void mma_nv4(cudaStream_t s, const TNV4& w, const float* x, int ldx, float* y, int ldy, int T) { launch_mma<PNV4>(s, w, x, ldx, y, ldy, T, 0); }
void mma_q8(cudaStream_t s, const TQ8& w, const float* x, int ldx, float* y, int ldy, int T) { launch_mma<PQ8>(s, w, x, ldx, y, ldy, T, 0); }
void mma_bf16(cudaStream_t s, const TBF& w, const float* x, int ldx, float* y, int ldy, int T) { launch_mma<PBF>(s, w, x, ldx, y, ldy, T, 0); }
void mma_iq2(cudaStream_t s, const TIQ2& w, const float* x, int ldx, float* y, int ldy, int T, int epi) { launch_mma<PIQ2>(s, w, x, ldx, y, ldy, T, epi); }
void mma_q8_multi(cudaStream_t s, const TQ8Multi& m, const float* x, int ldx, int T) {
    int tiles = 0;
    for (int i = 0; i < 3; i++) tiles += (m.w[i].n + 15) >> 4;
    int rg = (tiles + 1) / 2;
    if (T <= 8) mma_q8_multi_k<1, 2><<<dim3(1, rg), 128, 0, s>>>(m, x, ldx, T);
    else if (T <= 16) mma_q8_multi_k<2, 2><<<dim3(1, rg), 128, 0, s>>>(m, x, ldx, T);
    else mma_q8_multi_k<4, 2><<<dim3((T + 31) / 32, rg), 128, 0, s>>>(m, x, ldx, T);
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
}
void tile_nv4_launch(cudaStream_t s, const uint8_t* q, const uint8_t* sc, void* qt, void* sct, int n, int k) {
    int n16 = (n + 15) / 16, S = k / 64;
    size_t th = (size_t)n16 * S * 32;
    tile_nv4_k<<<(unsigned)((th + 255) / 256), 256, 0, s>>>(q, sc, (uint32_t*)qt, (uint8_t*)sct, n, k, n16, S);
}

__global__ void tile_q8_k(const uint8_t* __restrict__ raw, uint8_t* __restrict__ qt, uint32_t* __restrict__ dt, int n, int k, int n16, int KB) {
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
}
void tile_q8_launch(cudaStream_t s, const uint8_t* raw, void* qt, void* dt, int n, int k) {
    int n16 = (n + 15) / 16, KB = k / 32;
    size_t th = (size_t)n16 * KB * 32;
    tile_q8_k<<<(unsigned)((th + 255) / 256), 256, 0, s>>>(raw, (uint8_t*)qt, (uint32_t*)dt, n, k, n16, KB);
}

__global__ void tile_bf16_k(const uint16_t* __restrict__ w, uint32_t* __restrict__ wt, int n, int k, int n16, int K16) {
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
}
void tile_bf16_launch(cudaStream_t s, const uint16_t* w, void* wt, int n, int k) {
    int n16 = (n + 15) / 16, K16 = k / 16;
    size_t th = (size_t)n16 * K16 * 32;
    tile_bf16_k<<<(unsigned)((th + 255) / 256), 256, 0, s>>>(w, (uint32_t*)wt, n, k, n16, K16);
}

__global__ void tile_iq2_k(const uint8_t* __restrict__ raw, uint8_t* __restrict__ qt, uint16_t* __restrict__ dt, int n, int k, int n16, int C2, int NB) {
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
}
void tile_iq2_launch(cudaStream_t s, const uint8_t* raw, void* qt, void* dt, int n, int k) {
    int n16 = (n + 15) / 16, C2 = k / 64, NB = k / 256;
    size_t th = (size_t)n16 * C2 * 16;
    tile_iq2_k<<<(unsigned)((th + 255) / 256), 256, 0, s>>>(raw, (uint8_t*)qt, (uint16_t*)dt, n, k, n16, C2, NB);
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
}

template <bool MULTI>
__global__ void attn_mma_comb_k(const Ctl* __restrict__ ctl, const float* __restrict__ part, float* __restrict__ out, int splits) {
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
}

}  // namespace

void k_attn_mma(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out,
                int T, int splits) {
    attn_mma_part_k<false><<<dim3(splits, 2, (T + FA_Q - 1) / FA_Q), 256, 0, s>>>(ctl, q, kv, part, splits, T);
    attn_mma_comb_k<false><<<dim3(T, 32), 128, 0, s>>>(ctl, part, out, splits);
}
// Verify windows of up to nseg sequences (nseg a launch bound, ctl->nseg of them used), each of at most 8 rows.
void k_attn_mma_windows(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out,
                        int T, int nseg, int splits) {
    attn_mma_part_k<true><<<dim3(splits, 2, nseg), 256, 0, s>>>(ctl, q, kv, part, splits, T);
    attn_mma_comb_k<true><<<dim3(T, 32), 128, 0, s>>>(ctl, part, out, splits);
}
size_t attn_mma_part_floats(int T, int splits) { return (size_t)((T + FA_Q - 1) / FA_Q) * 2 * splits * FA_PART; }
