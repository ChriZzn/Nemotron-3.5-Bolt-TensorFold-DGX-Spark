// Bolt kernels. Weights stay packed (NVFP4, Q8_0, IQ2_XXS) and are dequantized in registers while a warp
// streams its rows once; activations are fp32. Tables for IQ2_XXS are ggml's (MIT).

#include "bolt_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstring>

#define GGML_TABLE_BEGIN(type, name, size) static __device__ type name[size] = {
#define GGML_TABLE_END() };
#include "iq2_tables.inc"

static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

static __device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }

// ---------------------------------------------------------------------------------------------------------
// Format policies. A chunk is 32 consecutive weights of one row; a lane owns every 32nd chunk.
// load() fetches the raw bytes of a chunk, deq() turns 8 of its weights (sub 0..3) into scaled floats.
// ---------------------------------------------------------------------------------------------------------

struct FmtQ8 {
    using W = Q8W;
    struct Chunk { uint4 a, b; float d; };
    static constexpr bool kGrid = false;
    static __device__ __forceinline__ Chunk load(const W& w, int row, int c) {
        Chunk ch;
        const uint4* p = reinterpret_cast<const uint4*>(w.q + (size_t)row * w.k + (size_t)c * 32);
        ch.a = __ldg(p);
        ch.b = __ldg(p + 1);
        ch.d = __half2float(__ldg(reinterpret_cast<const __half*>(w.d) + (size_t)row * (w.k >> 5) + c));
        return ch;
    }
    static __device__ __forceinline__ void deq(const Chunk& ch, int sub, float* out, const void*) {
        uint2 v;
        if (sub == 0) v = make_uint2(ch.a.x, ch.a.y);
        else if (sub == 1) v = make_uint2(ch.a.z, ch.a.w);
        else if (sub == 2) v = make_uint2(ch.b.x, ch.b.y);
        else v = make_uint2(ch.b.z, ch.b.w);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            out[j] = (float)(int8_t)((v.x >> (8 * j)) & 0xff) * ch.d;
            out[4 + j] = (float)(int8_t)((v.y >> (8 * j)) & 0xff) * ch.d;
        }
    }
};

struct FmtNV4 {
    using W = NV4W;
    struct Chunk { uint4 q; unsigned sc; float s2; };
    static constexpr bool kGrid = false;
    static __device__ __forceinline__ Chunk load(const W& w, int row, int c) {
        Chunk ch;
        ch.q = __ldg(reinterpret_cast<const uint4*>(w.q + (size_t)row * (w.k >> 1) + (size_t)c * 16));
        ch.sc = __ldg(reinterpret_cast<const unsigned short*>(w.sc + (size_t)row * (w.k >> 4) + (size_t)c * 2));
        ch.s2 = w.s2;
        return ch;
    }
    static __device__ __forceinline__ float e4m3(unsigned b) {
        __nv_fp8_e4m3 f;
        f.__x = (unsigned char)b;
        return (float)f;
    }
    static __device__ __forceinline__ void deq(const Chunk& ch, int sub, float* out, const void*) {
        unsigned word = sub == 0 ? ch.q.x : sub == 1 ? ch.q.y : sub == 2 ? ch.q.z : ch.q.w;
        float s = e4m3((ch.sc >> (8 * (sub >> 1))) & 0xff) * ch.s2;
        // e2m1 -> float: the three magnitude bits land in the exponent's low bits and the mantissa MSB,
        // and 2^126 puts the exponent back (denormal e=0 gives the 0.5 step).
        const float fix = 8.507059173023462e37f * 1.f;  // 2^126
#pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned nib = (word >> (4 * j)) & 0xf;
            unsigned bits = ((nib & 7u) << 22) | ((nib & 8u) << 28);
            out[j] = __uint_as_float(bits) * fix * s;
        }
    }
};

struct FmtBF16 {
    using W = BF16W;
    struct Chunk { uint4 a, b, c, d; };
    static constexpr bool kGrid = false;
    static __device__ __forceinline__ Chunk load(const W& w, int row, int c) {
        Chunk ch;
        const uint4* p = reinterpret_cast<const uint4*>(w.w + (size_t)row * w.k + (size_t)c * 32);
        ch.a = __ldg(p); ch.b = __ldg(p + 1); ch.c = __ldg(p + 2); ch.d = __ldg(p + 3);
        return ch;
    }
    static __device__ __forceinline__ void deq(const Chunk& ch, int sub, float* out, const void*) {
        uint4 v = sub == 0 ? ch.a : sub == 1 ? ch.b : sub == 2 ? ch.c : ch.d;
        out[0] = __uint_as_float(v.x << 16); out[1] = __uint_as_float(v.x & 0xffff0000u);
        out[2] = __uint_as_float(v.y << 16); out[3] = __uint_as_float(v.y & 0xffff0000u);
        out[4] = __uint_as_float(v.z << 16); out[5] = __uint_as_float(v.z & 0xffff0000u);
        out[6] = __uint_as_float(v.w << 16); out[7] = __uint_as_float(v.w & 0xffff0000u);
    }
};

// IQ2_XXS: a chunk is one 8-byte group (four grid indices, four 7-bit sign indices, a 4-bit scale).
struct FmtIQ2 {
    using W = IQ2W;
    struct Chunk { uint2 q; float db; };
    static constexpr bool kGrid = true;
    static __device__ __forceinline__ Chunk load(const W& w, int row, int c) {
        Chunk ch;
        int nb = w.k >> 8;
        const uint2* p = reinterpret_cast<const uint2*>(w.qs + ((size_t)row * nb) * 64) + c;
        ch.q = __ldg(p);
        float d = __half2float(__ldg(reinterpret_cast<const __half*>(w.d) + (size_t)row * nb + (c >> 3)));
        ch.db = d * (0.5f + (float)(ch.q.y >> 28)) * 0.25f;
        return ch;
    }
    static __device__ __forceinline__ void deq(const Chunk& ch, int sub, float* out, const void* grid_smem) {
        const uint2* grid = reinterpret_cast<const uint2*>(grid_smem);
        unsigned gi = (ch.q.x >> (8 * sub)) & 0xff;
        uint2 g = grid[gi];
        unsigned si = (ch.q.y >> (7 * sub)) & 127u;
        unsigned sg = si | ((__popc(si) & 1u) << 7);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned byte = ((j < 4 ? g.x : g.y) >> (8 * (j & 3))) & 0xffu;
            float v = __uint_as_float(0x4B000000u | byte) - 8388608.f;
            unsigned flip = (sg << (31 - j)) & 0x80000000u;
            out[j] = __uint_as_float(__float_as_uint(v) ^ flip) * ch.db;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------
// The shared GEMV body: T activation rows, R weight rows per warp, four warps per block.
// ---------------------------------------------------------------------------------------------------------

template <class F, int T, int R, int EPI>
static __device__ __forceinline__ void gemv_body(const typename F::W& w, const float* __restrict__ x, int ldx,
                                                 float* __restrict__ y, int ldy, int N, int rows_base, int Tn) {
    __shared__ uint2 grid_s[F::kGrid ? 256 : 1];
    if constexpr (F::kGrid) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            unsigned long long g = iq2xxs_grid[i];
            grid_s[i] = make_uint2((unsigned)g, (unsigned)(g >> 32));
        }
        __syncthreads();
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = rows_base + warp * R;
    if (row0 >= N) return;
    const int nchunk = w.k >> 5;
    float acc[R][T];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < T; t++) acc[r][t] = 0.f;

    typename F::Chunk cur[R], nxt[R];
    int c = lane;
    if (c < nchunk) {
#pragma unroll
        for (int r = 0; r < R; r++) cur[r] = F::load(w, min(row0 + r, N - 1), c);
    }
    for (; c < nchunk; c += 32) {
        const int cn = c + 32;
        if (cn < nchunk) {
#pragma unroll
            for (int r = 0; r < R; r++) nxt[r] = F::load(w, min(row0 + r, N - 1), cn);
        }
#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            float wv[R][8];
#pragma unroll
            for (int r = 0; r < R; r++) F::deq(cur[r], sub, wv[r], grid_s);
#pragma unroll
            for (int t = 0; t < T; t++) {
                const float4* xp = reinterpret_cast<const float4*>(x + (size_t)t * ldx + (size_t)c * 32 + sub * 8);
                float4 xa = __ldg(xp), xb = __ldg(xp + 1);
#pragma unroll
                for (int r = 0; r < R; r++) {
                    acc[r][t] += wv[r][0] * xa.x + wv[r][1] * xa.y + wv[r][2] * xa.z + wv[r][3] * xa.w +
                                 wv[r][4] * xb.x + wv[r][5] * xb.y + wv[r][6] * xb.z + wv[r][7] * xb.w;
                }
            }
        }
        if (cn < nchunk) {
#pragma unroll
            for (int r = 0; r < R; r++) cur[r] = nxt[r];
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int t = 0; t < T; t++) {
            float v = warp_sum(acc[r][t]);
            if (lane == ((r * T + t) & 31) && row0 + r < N && t < Tn) {
                if (EPI == 1) v = v > 0.f ? v * v : 0.f;
                y[(size_t)t * ldy + row0 + r] = v;
            }
        }
    }
}

template <class F, int T, int R, int EPI>
__global__ void __launch_bounds__(128) gemv_k(typename F::W w, const float* __restrict__ x, int ldx, float* __restrict__ y, int ldy, int Tn) {
    gemv_body<F, T, R, EPI>(w, x, ldx, y, ldy, w.n, blockIdx.x * 4 * R, Tn);
}

template <class F, int T, int R, int EPI>
static void launch_gemv(cudaStream_t s, const typename F::W& w, const float* x, int ldx, float* y, int ldy, int Tn) {
    int rows_per_cta = 4 * R;
    gemv_k<F, T, R, EPI><<<(w.n + rows_per_cta - 1) / rows_per_cta, 128, 0, s>>>(w, x, ldx, y, ldy, Tn);
}

template <class F, int EPI>
static void dispatch_gemv(cudaStream_t s, const typename F::W& w, const float* x, int ldx, float* y, int ldy, int T) {
    if (T <= 1) launch_gemv<F, 1, 4, EPI>(s, w, x, ldx, y, ldy, T);
    else if (T == 2) launch_gemv<F, 2, 4, EPI>(s, w, x, ldx, y, ldy, T);
    else if (T <= 4) launch_gemv<F, 4, 4, EPI>(s, w, x, ldx, y, ldy, T);
    else if (T <= 8) launch_gemv<F, 8, 2, EPI>(s, w, x, ldx, y, ldy, T);
    else launch_gemv<F, 16, 2, EPI>(s, w, x, ldx, y, ldy, T);
}

void gemv_q8(cudaStream_t s, const Q8W& w, const float* x, int ldx, float* y, int ldy, int T, int epi) {
    if (epi) dispatch_gemv<FmtQ8, 1>(s, w, x, ldx, y, ldy, T);
    else dispatch_gemv<FmtQ8, 0>(s, w, x, ldx, y, ldy, T);
}
void gemv_nv4(cudaStream_t s, const NV4W& w, const float* x, int ldx, float* y, int ldy, int T) {
    dispatch_gemv<FmtNV4, 0>(s, w, x, ldx, y, ldy, T);
}
void gemv_iq2(cudaStream_t s, const IQ2W& w, const float* x, int ldx, float* y, int ldy, int T, int epi) {
    if (epi) dispatch_gemv<FmtIQ2, 1>(s, w, x, ldx, y, ldy, T);
    else dispatch_gemv<FmtIQ2, 0>(s, w, x, ldx, y, ldy, T);
}
void gemv_bf16(cudaStream_t s, const BF16W& w, const float* x, int ldx, float* y, int ldy, int T) {
    dispatch_gemv<FmtBF16, 0>(s, w, x, ldx, y, ldy, T);
}

// q, k and v in one launch: a block belongs to one matrix.
template <int T, int R>
__global__ void __launch_bounds__(128) q8_multi_k(Q8Multi m, const float* __restrict__ x, int ldx, int Tn) {
    int rows_per_cta = 4 * R;
    int b = blockIdx.x * rows_per_cta;
    int which = 0;
    while (which < 2 && b >= m.rows[which]) { b -= m.rows[which]; which++; }
    gemv_body<FmtQ8, T, R, 0>(m.w[which], x, ldx, m.y[which], m.ldy[which], m.w[which].n, b, Tn);
}

void gemv_q8_multi(cudaStream_t s, const Q8Multi& m, const float* x, int ldx, int T) {
    int total = m.rows[0] + m.rows[1] + m.rows[2];
    auto grid = [&](int rpc) { return (total + rpc - 1) / rpc; };
    if (T <= 1) q8_multi_k<1, 4><<<grid(16), 128, 0, s>>>(m, x, ldx, T);
    else if (T == 2) q8_multi_k<2, 4><<<grid(16), 128, 0, s>>>(m, x, ldx, T);
    else if (T <= 4) q8_multi_k<4, 4><<<grid(16), 128, 0, s>>>(m, x, ldx, T);
    else if (T <= 8) q8_multi_k<8, 2><<<grid(8), 128, 0, s>>>(m, x, ldx, T);
    else q8_multi_k<16, 2><<<grid(8), 128, 0, s>>>(m, x, ldx, T);
}

// One launch for every (token, expert) pair. Pair order[blockIdx.y] belongs to token pair / TOPN and uses expert
// ridx[pair]. The activation vector is quantized to int8 per 32 (llama.cpp's q8_1 scheme), so a group of
// eight weights is two dp4a; the grid magnitudes get their signs from ggml's ksigns64 byte masks.
// Routed experts on int8 tensor cores. The (token, expert) pairs are grouped by expert and cut into work items of at
// most MOE_B pairs. Activations are quantized to int8 per 32 (a row's scale is amax / 127); an item's tokens sit in
// shared memory as the B operand while each warp walks a 16-row tile of the expert's rows chunk by chunk (32
// weights): the IQ2_XXS grid and sign tables turn the chunk into an int8 A fragment of m16n8k32, one mma gives the
// tile's 16 rows x up to 16 tokens for the chunk, and the chunk's scales are applied to the int32 result.
constexpr int MOE_B = 32;   // most pairs of a work item; the grouping cuts at the launch's item size (8, 16 or 32)

__global__ void moe_group_k(const int* __restrict__ ridx, int pairs, int item_b, int* __restrict__ order, int* __restrict__ work_first,
                            int* __restrict__ work_n, int* __restrict__ work_exp, int* __restrict__ n_work) {
    __shared__ int cnt[512], start[512], fill[512], act[512], sstart[512], slen[512], sexp[512], nbat[512];
    const int tid = threadIdx.x;  // 512 threads
    cnt[tid] = 0; fill[tid] = 0;
    __syncthreads();
    for (int p = tid; p < pairs; p += 512) atomicAdd(&cnt[ridx[p]], 1);
    __syncthreads();
    start[tid] = cnt[tid];
    act[tid] = cnt[tid] > 0;
    __syncthreads();
    for (int off = 1; off < 512; off <<= 1) {
        int a = tid >= off ? start[tid - off] : 0, b = tid >= off ? act[tid - off] : 0;
        __syncthreads();
        start[tid] += a; act[tid] += b;
        __syncthreads();
    }
    const int pos0 = start[tid] - cnt[tid];
    const int slot = act[tid] - (cnt[tid] > 0);
    const int nact = act[511];
    sstart[tid] = 0; slen[tid] = 0; sexp[tid] = 0;
    __syncthreads();
    if (cnt[tid] > 0) { sstart[slot] = pos0; slen[slot] = cnt[tid]; sexp[slot] = tid; }
    __syncthreads();
    nbat[tid] = tid < nact ? (slen[tid] + item_b - 1) / item_b : 0;
    __syncthreads();
    const int mine = nbat[tid];
    for (int off = 1; off < 512; off <<= 1) {
        int a = tid >= off ? nbat[tid - off] : 0;
        __syncthreads();
        nbat[tid] += a;
        __syncthreads();
    }
    if (tid < nact) {
        int w0 = nbat[tid] - mine;
        for (int i = 0; i < mine; i++) {
            work_first[w0 + i] = sstart[tid] + i * item_b;
            work_n[w0 + i] = min(item_b, slen[tid] - i * item_b);
            work_exp[w0 + i] = sexp[tid];
        }
    }
    if (tid == 511) n_work[0] = nbat[511];
    start[tid] = pos0;
    __syncthreads();
    for (int p = tid; p < pairs; p += 512) {
        int e = ridx[p];
        order[start[e] + atomicAdd(&fill[e], 1)] = p;
    }
}
// Pairs a work item holds for a launch of `pairs` pairs.
int moe_item_pairs(int pairs) {
    static const char* e = getenv("BOLT_MOE_B");
    if (e && pairs > 4096) return atoi(e);
    return pairs <= 8 * 9 ? 8 : 16;
}
void k_moe_group(cudaStream_t s, const int* ridx, int pairs, int* order, int* work_first, int* work_n, int* work_exp, int* n_work) {
    moe_group_k<<<1, 512, 0, s>>>(ridx, pairs, moe_item_pairs(pairs), order, work_first, work_n, work_exp, n_work);
}

// int8 rows with one scale per 32: q[r][k], sc[r][k/32] = amax / 127
__global__ void quant_rows_k(const float* __restrict__ x, int ldx, int8_t* __restrict__ q, int ldq, float* __restrict__ sc, int ldsc, int k) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int b = warp; b < (k >> 5); b += blockDim.x >> 5) {
        float v = x[(size_t)r * ldx + b * 32 + lane];
        float am = fabsf(v);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
        float inv = am > 0.f ? 127.f / am : 0.f;
        q[(size_t)r * ldq + b * 32 + lane] = (int8_t)__float2int_rn(v * inv);
        if (lane == 0) sc[(size_t)r * ldsc + b] = am / 127.f;
    }
}
void k_quant_rows(cudaStream_t s, const float* x, int ldx, int8_t* q, int ldq, float* sc, int ldsc, int rows, int k) {
    quant_rows_k<<<rows, 256, 0, s>>>(x, ldx, q, ldq, sc, ldsc, k);
}

static __device__ __forceinline__ void mma_s8(int* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

constexpr int MOE_SMEM_K = 2560;

template <int NT, int EPI>
__global__ void __launch_bounds__(256) moe_mma_k(IQ2Slab slab, const int* __restrict__ order, const int* __restrict__ work_first,
                                                 const int* __restrict__ work_n, const int* __restrict__ work_exp,
                                                 const int* __restrict__ n_work, const int8_t* __restrict__ xq,
                                                 const float* __restrict__ xs, int x_per_pair, float* __restrict__ y, int ldy,
                                                 int n, int k) {
    if ((int)blockIdx.y >= n_work[0]) return;
    extern __shared__ __align__(16) unsigned char smem[];
    const int stride = k + 16;                       // bytes per staged token row (bank-conflict free fragment loads)
    int8_t* xsm = reinterpret_cast<int8_t*>(smem);   // [NT*8][stride]
    float* ssm = reinterpret_cast<float*>(smem + (size_t)NT * 8 * stride);  // [NT*8][k/32]
    __shared__ unsigned gx[256], gy[256], sx[128], sy[128];
    __shared__ int pid[MOE_B];
    const int nb = k >> 8, nchunk = k >> 5;
    const int e = work_exp[blockIdx.y], first = work_first[blockIdx.y], cnt = work_n[blockIdx.y];
    const int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 256) {
        unsigned long long g = iq2xxs_grid[i];
        gx[i] = (unsigned)g; gy[i] = (unsigned)(g >> 32);
    }
    if (tid < 128) { unsigned long long m = ksigns64[tid]; sx[tid] = (unsigned)m; sy[tid] = (unsigned)(m >> 32); }
    if (tid < MOE_B) pid[tid] = tid < cnt ? order[first + tid] : -1;
    __syncthreads();
    // stage the item's tokens, all (row, 16-byte piece) pairs at once so the loads overlap; the columns past cnt hold
    // whatever the shared memory had and are never stored
    {
        const int per = k >> 4;   // 16-byte pieces a row
#pragma unroll 4
        for (int idx = tid; idx < cnt * per; idx += 256) {
            const int r = idx / per, i = idx - r * per;
            const int p = pid[r];
            const int row = x_per_pair ? p : p / TOPN;
            *reinterpret_cast<uint4*>(xsm + (size_t)r * stride + i * 16) = __ldg(reinterpret_cast<const uint4*>(xq + (size_t)row * k) + i);
        }
        for (int idx = tid; idx < cnt * nchunk; idx += 256) {
            const int r = idx / nchunk, i = idx - r * nchunk;
            const int p = pid[r];
            const int row = x_per_pair ? p : p / TOPN;
            ssm[r * nchunk + i] = __ldg(xs + (size_t)row * nchunk + i);
        }
    }
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31, gi = lane >> 2, ti = lane & 3;
    for (int tile = blockIdx.x * 8 + warp; tile * 16 < n; tile += gridDim.x * 8) {
        const int rowA = min(tile * 16 + gi, n - 1), rowB = min(tile * 16 + gi + 8, n - 1);
        const uint8_t* qbase = slab.qs + (size_t)e * slab.qs_stride;
        const __half* dbase = reinterpret_cast<const __half*>(slab.d) + (size_t)e * slab.d_stride;
        const uint4* qa_p = reinterpret_cast<const uint4*>(qbase + (size_t)rowA * nb * 64);   // 16 bytes = two chunks
        const uint4* qb_p = reinterpret_cast<const uint4*>(qbase + (size_t)rowB * nb * 64);
        const __half* da_p = dbase + (size_t)rowA * nb;
        const __half* db_p = dbase + (size_t)rowB * nb;
        const int l0 = ti >> 1, l2 = l0 + 2;
        const unsigned* gt = (ti & 1) ? gy : gx;
        const unsigned* mt = (ti & 1) ? sy : sx;
        float acc[NT][4];
    #pragma unroll
        for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

        // four chunks (one 32-byte sector of each row) per step, the next step's loads in flight
        uint4 a0 = __ldg(qa_p), a1 = __ldg(qa_p + 1), b0 = __ldg(qb_p), b1 = __ldg(qb_p + 1);
        for (int c4 = 0; c4 < nchunk; c4 += 4) {
            const uint4 A0 = a0, A1 = a1, B0 = b0, B1 = b1;
            if (c4 + 4 < nchunk) {
                const int u = (c4 + 4) >> 1;
                a0 = __ldg(qa_p + u); a1 = __ldg(qa_p + u + 1); b0 = __ldg(qb_p + u); b1 = __ldg(qb_p + u + 1);
            }
            const float dhA = __half2float(__ldg(da_p + (c4 >> 3))), dhB = __half2float(__ldg(db_p + (c4 >> 3)));
    #pragma unroll
            for (int cc = 0; cc < 4; cc++) {
                const int c = c4 + cc;
                const uint2 ca = cc == 0 ? make_uint2(A0.x, A0.y) : cc == 1 ? make_uint2(A0.z, A0.w) : cc == 2 ? make_uint2(A1.x, A1.y) : make_uint2(A1.z, A1.w);
                const uint2 cb = cc == 0 ? make_uint2(B0.x, B0.y) : cc == 1 ? make_uint2(B0.z, B0.w) : cc == 2 ? make_uint2(B1.x, B1.y) : make_uint2(B1.z, B1.w);
                const float dA = dhA * (0.5f + (float)(ca.y >> 28)) * 0.25f;
                const float dB = dhB * (0.5f + (float)(cb.y >> 28)) * 0.25f;
                unsigned ga0 = gt[(ca.x >> (8 * l0)) & 255], ma0 = mt[(ca.y >> (7 * l0)) & 127];
                unsigned ga2 = gt[(ca.x >> (8 * l2)) & 255], ma2 = mt[(ca.y >> (7 * l2)) & 127];
                unsigned gb0 = gt[(cb.x >> (8 * l0)) & 255], mb0 = mt[(cb.y >> (7 * l0)) & 127];
                unsigned gb2 = gt[(cb.x >> (8 * l2)) & 255], mb2 = mt[(cb.y >> (7 * l2)) & 127];
                const unsigned q0 = __vsub4(ga0 ^ ma0, ma0), q1 = __vsub4(gb0 ^ mb0, mb0);
                const unsigned q2 = __vsub4(ga2 ^ ma2, ma2), q3 = __vsub4(gb2 ^ mb2, mb2);
    #pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const int8_t* xr = xsm + (size_t)(nt * 8 + gi) * stride + c * 32 + 4 * ti;
                    int dd[4] = {0, 0, 0, 0};
                    mma_s8(dd, q0, q1, q2, q3, *reinterpret_cast<const unsigned*>(xr), *reinterpret_cast<const unsigned*>(xr + 16));
                    const float s0 = ssm[(nt * 8 + 2 * ti) * nchunk + c], s1 = ssm[(nt * 8 + 2 * ti + 1) * nchunk + c];
                    acc[nt][0] += dA * s0 * (float)dd[0];
                    acc[nt][1] += dA * s1 * (float)dd[1];
                    acc[nt][2] += dB * s0 * (float)dd[2];
                    acc[nt][3] += dB * s1 * (float)dd[3];
                }
            }
        }
    #pragma unroll
        for (int nt = 0; nt < NT; nt++)
    #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int tok = nt * 8 + 2 * ti + (j & 1);
                const int row = tile * 16 + gi + ((j >> 1) ? 8 : 0);
                if (tok < cnt && row < n) {
                    float v = acc[nt][j];
                    if (EPI == 1) v = v > 0.f ? v * v : 0.f;
                    y[(size_t)pid[tok] * ldy + row] = v;
                }
            }
    }
}

static size_t moe_smem(int nt, int k) { return (size_t)nt * 8 * (k + 16) + (size_t)nt * 8 * (k >> 5) * 4; }

template <int NT, int EPI>
static void launch_moe_mma(cudaStream_t s, const IQ2Slab& slab, const MoeWork& w, const int8_t* xq, const float* xs, int x_per_pair,
                           float* y, int ldy, int n, int k, int items) {
    static bool opted = false;
    size_t sm = moe_smem(NT, MOE_SMEM_K);
    if (!opted) {
        cudaFuncSetAttribute(moe_mma_k<NT, EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        opted = true;
    }
    // each CTA stages its item's tokens once and walks several weight tiles per warp; items past n_work exit at once, so
    // the launch only has to cover the most items `items` pairs can make
    static const char* ec = getenv("BOLT_MOE_CTAS");
    const int ctas = ec ? atoi(ec) : 4;
    items = std::min(items, items / (NT * 8) + 512);
    dim3 grid(std::min((n / 16 + 7) / 8, ctas), items);
    moe_mma_k<NT, EPI><<<grid, 256, moe_smem(NT, k), s>>>(slab, w.order, w.first, w.count, w.expert, w.n_work, xq, xs, x_per_pair, y, ldy, n, k);
}

void moe_up(cudaStream_t s, const IQ2Slab& up, const MoeWork& w, const int8_t* xq, const float* xs, float* act, int width, int pairs) {
    const int b = moe_item_pairs(pairs);
    if (b == 8) launch_moe_mma<1, 1>(s, up, w, xq, xs, 0, act, width, width, 1024, pairs);
    else if (b == 16) launch_moe_mma<2, 1>(s, up, w, xq, xs, 0, act, width, width, 1024, pairs);
    else launch_moe_mma<4, 1>(s, up, w, xq, xs, 0, act, width, width, 1024, pairs);
}
void moe_down(cudaStream_t s, const IQ2Slab& down, const MoeWork& w, const int8_t* xq, const float* xs, float* out, int width, int pairs) {
    const int b = moe_item_pairs(pairs);
    if (b == 8) launch_moe_mma<1, 0>(s, down, w, xq, xs, 1, out, 1024, 1024, width, pairs);
    else if (b == 16) launch_moe_mma<2, 0>(s, down, w, xq, xs, 1, out, 1024, 1024, width, pairs);
    else launch_moe_mma<4, 0>(s, down, w, xq, xs, 1, out, 1024, 1024, width, pairs);
}

// ---------------------------------------------------------------------------------------------------------
// Load-time repacking of routed experts: ggml's interleaved blocks into planes so every load is aligned.
// ---------------------------------------------------------------------------------------------------------

__global__ void repack_iq2_k(const uint8_t* __restrict__ src, uint8_t* __restrict__ qs, __half* __restrict__ d, size_t nblocks) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint8_t* b = src + i * 66;
    unsigned short h = (unsigned short)b[0] | ((unsigned short)b[1] << 8);
    d[i] = __ushort_as_half(h);
    unsigned short* o = reinterpret_cast<unsigned short*>(qs + i * 64);
    for (int j = 0; j < 32; j++) o[j] = (unsigned short)b[2 + 2 * j] | ((unsigned short)b[3 + 2 * j] << 8);
}
void repack_iq2(cudaStream_t s, const uint8_t* src, uint8_t* qs, void* d, size_t rows, int k) {
    size_t nb = rows * (size_t)(k >> 8);
    repack_iq2_k<<<(unsigned)((nb + 255) / 256), 256, 0, s>>>(src, qs, (__half*)d, nb);
}
void repack_q8(cudaStream_t, const uint8_t*, int8_t*, void*, size_t, int) {}

// ---------------------------------------------------------------------------------------------------------
// Block building blocks
// ---------------------------------------------------------------------------------------------------------

__global__ void embed_k(const uint16_t* __restrict__ table, const uint16_t* __restrict__ extra, const Ctl* __restrict__ ctl, float* __restrict__ x) {
    int t = blockIdx.x;
    int id = ctl->tokens[t];
    // a negative token takes row t of `extra` (an image or video embedding copied there for this block)
    const uint16_t* row = id < 0 ? extra + (size_t)t * HID : table + (size_t)((unsigned)id >= (unsigned)VOCAB ? 0 : id) * HID;
    const uint2* src = reinterpret_cast<const uint2*>(row);
    float4* dst = reinterpret_cast<float4*>(x + (size_t)t * HID);
    for (int i = threadIdx.x; i < HID / 4; i += blockDim.x) {
        uint2 v = __ldg(src + i);
        dst[i] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                             __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
    }
}
void k_embed(cudaStream_t s, const uint16_t* table, const uint16_t* extra, const Ctl* ctl, float* x, int T) { embed_k<<<T, 256, 0, s>>>(table, extra, ctl, x); }

// x += r1 + r2 (when given), xn = rmsnorm(x) * w. cap (optional) receives the updated residual.
__global__ void add_rms_k(float* __restrict__ x, const float* __restrict__ r1, const float* __restrict__ r2,
                          const float* __restrict__ w, float* __restrict__ xn, float* __restrict__ cap, int cap_stride, float eps) {
    int t = blockIdx.x;
    float4* xp = reinterpret_cast<float4*>(x + (size_t)t * HID);
    const float4* a = r1 ? reinterpret_cast<const float4*>(r1 + (size_t)t * HID) : nullptr;
    const float4* b = r2 ? reinterpret_cast<const float4*>(r2 + (size_t)t * HID) : nullptr;
    float4 v[4];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int i = threadIdx.x + j * 256;
        float4 u = xp[i];
        if (a) { float4 q = a[i]; u.x += q.x; u.y += q.y; u.z += q.z; u.w += q.w; }
        if (b) { float4 q = b[i]; u.x += q.x; u.y += q.y; u.z += q.z; u.w += q.w; }
        v[j] = u;
        ss += u.x * u.x + u.y * u.y + u.z * u.z + u.w * u.w;
    }
    ss = warp_sum(ss);
    __shared__ float sm[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) tot += sm[i];
    float inv = rsqrtf(tot / (float)HID + eps);
    float4* np = reinterpret_cast<float4*>(xn + (size_t)t * HID);
    const float4* wp = reinterpret_cast<const float4*>(w);
    float4* cp = cap ? reinterpret_cast<float4*>(cap + (size_t)t * cap_stride) : nullptr;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int i = threadIdx.x + j * 256;
        xp[i] = v[j];
        float4 g = __ldg(wp + i);
        np[i] = make_float4(v[j].x * inv * g.x, v[j].y * inv * g.y, v[j].z * inv * g.z, v[j].w * inv * g.w);
        if (cp) cp[i] = v[j];
    }
}
void k_add_rms(cudaStream_t s, float* x, const float* r1, const float* r2, const float* w, float* xn, float* cap,
               int cap_stride, float eps, int T) {
    add_rms_k<<<T, 256, 0, s>>>(x, r1, r2, w, xn, cap, cap_stride, eps);
}

__device__ __forceinline__ int seg_of(const Ctl* ctl, int row) {
    int i = 0;
    while (i + 1 < ctl->nseg && row >= ctl->seg_row0[i + 1]) i++;
    return i;
}

// Mamba conv1d (kernel 4, causal, SiLU) over the block's rows, one thread per channel and segment. State is the
// last three inputs. The segment's slot is read; with record every row's state goes to the following slots of its
// ring (so a window can be cut after any row), otherwise only the final state goes to the next slot.
__global__ void conv_k(const Ctl* __restrict__ ctl, float* __restrict__ pool,
                       const float* __restrict__ w, const float* __restrict__ bias, const float* __restrict__ proj, int ldp,
                       float* __restrict__ xbc, int record, float* __restrict__ hist_x, float* __restrict__ hist_dt) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    const int sg = blockIdx.y;
    if (sg >= ctl->nseg) return;
    const int r0 = ctl->seg_row0[sg], r1 = ctl->seg_row0[sg + 1];
    float* ring = pool + (size_t)ctl->seg_seq[sg] * CONV_SLOTS * CONV_ELEMS;
    const int cur = ctl->seg_conv[sg];
    const float* in = ring + (size_t)cur * CONV_ELEMS + (size_t)c * 3;
    float s0 = in[0], s1 = in[1], s2 = in[2];
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3], b = bias[c];
    for (int t = r0; t < r1; t++) {
        float xin = proj[(size_t)t * ldp + 8192 + c];
        float a = b + w0 * s0 + w1 * s1 + w2 * s2 + w3 * xin;
        float act = silu(a);
        xbc[(size_t)t * 10240 + c] = act;
        if (hist_x) {
            hist_x[(size_t)t * 10240 + c] = act;
            if (c < 128) hist_dt[t * 128 + c] = proj[(size_t)t * ldp + 18432 + c];
        }
        s0 = s1; s1 = s2; s2 = xin;
        if (record) {
            float* o = ring + (size_t)((cur + 1 + (t - r0)) % CONV_SLOTS) * CONV_ELEMS + (size_t)c * 3;
            o[0] = s0; o[1] = s1; o[2] = s2;
        }
    }
    if (!record) {
        float* o = ring + (size_t)((cur + 1) % CONV_SLOTS) * CONV_ELEMS + (size_t)c * 3;
        o[0] = s0; o[1] = s1; o[2] = s2;
    }
}
// Long single-segment blocks: the conv is a 4-tap filter with no recurrence, so rows are split over blocks.
#define CONV_ROWS 16
__global__ void conv_long_k(const Ctl* __restrict__ ctl, float* __restrict__ pool,
                            const float* __restrict__ w, const float* __restrict__ bias, const float* __restrict__ proj, int ldp,
                            float* __restrict__ xbc, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    float* ring = pool + (size_t)ctl->seg_seq[0] * CONV_SLOTS * CONV_ELEMS;
    const int cur = ctl->seg_conv[0];
    const float* in = ring + (size_t)cur * CONV_ELEMS + (size_t)c * 3;
    const int t0 = blockIdx.y * CONV_ROWS, t1 = min(T, t0 + CONV_ROWS);
    // inputs t0-3 .. t0-1 (state when negative)
    float s0, s1, s2;
    {
        float v[3];
#pragma unroll
        for (int j = 0; j < 3; j++) {
            int t = t0 - 3 + j;
            v[j] = t >= 0 ? proj[(size_t)t * ldp + 8192 + c] : in[3 + t];
        }
        s0 = v[0]; s1 = v[1]; s2 = v[2];
    }
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3], b = bias[c];
    float xin[CONV_ROWS];
#pragma unroll
    for (int j = 0; j < CONV_ROWS; j++) xin[j] = t0 + j < t1 ? proj[(size_t)(t0 + j) * ldp + 8192 + c] : 0.f;
#pragma unroll
    for (int j = 0; j < CONV_ROWS; j++) {
        if (t0 + j < t1) {
            float a = b + w0 * s0 + w1 * s1 + w2 * s2 + w3 * xin[j];
            xbc[(size_t)(t0 + j) * 10240 + c] = silu(a);
        }
        s0 = s1; s1 = s2; s2 = xin[j];
    }
    if (t1 == T) {   // the block holding the last row writes the new state: inputs T-3 .. T-1
        float* o = ring + (size_t)((cur + 1) % CONV_SLOTS) * CONV_ELEMS + (size_t)c * 3;
#pragma unroll
        for (int j = 0; j < 3; j++) {
            int t = T - 3 + j;
            o[j] = t >= 0 ? proj[(size_t)t * ldp + 8192 + c] : in[3 + t];
        }
    }
}

void k_conv(cudaStream_t s, const Ctl* ctl, float* pool, const float* w, const float* b,
            const float* proj, int ldp, float* xbc, int T, int record, float* hist_x, float* hist_dt, int nseg) {
    if (nseg == 1 && T >= 64 && !record && !hist_x) {
        conv_long_k<<<dim3(10240 / 128, (T + CONV_ROWS - 1) / CONV_ROWS), 128, 0, s>>>(ctl, pool, w, b, proj, ldp, xbc, T);
        return;
    }
    conv_k<<<dim3((10240 + 255) / 256, nseg), 256, 0, s>>>(ctl, pool, w, b, proj, ldp, xbc, record, hist_x, hist_dt);
}

// Mamba-2 selective scan, one warp per (head, channel, segment): the warp's 128 state entries live 4 to a lane.
// The segment reads its SSM slot and writes the other one.
__global__ void __launch_bounds__(256) scan_k(const Ctl* __restrict__ ctl, float* __restrict__ pool,
                                               const float* __restrict__ xbc, const float* __restrict__ proj, int ldp,
                                               const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                               const float* __restrict__ Dp, float* __restrict__ y) {
    const int sg = blockIdx.y;
    if (sg >= ctl->nseg) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.x >> 3;
    const int d = (blockIdx.x & 7) * 8 + warp;
    const int g = h >> 4;  // 16 heads per group
    const int r0 = ctl->seg_row0[sg], r1 = ctl->seg_row0[sg + 1];
    float* st_pool = pool + (size_t)ctl->seg_seq[sg] * 2 * SSM_ELEMS;
    const int cur = ctl->seg_ssm[sg];
    const size_t off = ((size_t)h * 64 + d) * 128 + 4 * lane;
    float4 st = *reinterpret_cast<const float4*>(st_pool + (size_t)cur * SSM_ELEMS + off);
    const float A = -__expf(A_log[h]);
    const float bias = dt_bias[h];
    const float Dh = Dp[h];
    for (int t = r0; t < r1; t++) {
        float v = proj[(size_t)t * ldp + 18432 + h] + bias;
        float sp = v > 20.f ? v : log1pf(__expf(v));
        float dA = __expf(sp * A);
        const float* row = xbc + (size_t)t * 10240;
        float xv = row[h * 64 + d];
        float4 B = *reinterpret_cast<const float4*>(row + 8192 + g * 128 + 4 * lane);
        float4 C = *reinterpret_cast<const float4*>(row + 9216 + g * 128 + 4 * lane);
        float sx = sp * xv;
        st.x = st.x * dA + sx * B.x;
        st.y = st.y * dA + sx * B.y;
        st.z = st.z * dA + sx * B.z;
        st.w = st.w * dA + sx * B.w;
        float part = st.x * C.x + st.y * C.y + st.z * C.z + st.w * C.w;
        part = warp_sum(part);
        if (lane == 0) y[(size_t)t * 8192 + h * 64 + d] = part + Dh * xv;
    }
    *reinterpret_cast<float4*>(st_pool + (size_t)(cur ^ 1) * SSM_ELEMS + off) = st;
}

// Long single-segment blocks (prompt chunks): a thread pair owns one channel, 64 of the 128 state entries each, so
// the dot with C needs one shuffle instead of five. B, C, x and the decay of 16 steps are staged in shared memory;
// one block per head.
#define SCAN_TC 16
__global__ void __launch_bounds__(128) scan_t_k(const Ctl* __restrict__ ctl, float* __restrict__ pool,
                                                 const float* __restrict__ xbc, const float* __restrict__ proj, int ldp,
                                                 const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                                 const float* __restrict__ Dp, float* __restrict__ y, int T) {
    __shared__ __align__(16) float Bs[SCAN_TC][136], Cs[SCAN_TC][136], xs[SCAN_TC][64];
    __shared__ float dAs[SCAN_TC], sps[SCAN_TC];
    const int tid = threadIdx.x, half = tid & 1, d = tid >> 1;
    const int h = blockIdx.x, g = h >> 4;
    float* st_pool = pool + (size_t)ctl->seg_seq[0] * 2 * SSM_ELEMS;
    const int cur = ctl->seg_ssm[0];
    float4 st[16];
    {
        const float4* src = reinterpret_cast<const float4*>(st_pool + (size_t)cur * SSM_ELEMS + ((size_t)h * 64 + d) * 128 + half * 64);
#pragma unroll
        for (int i = 0; i < 16; i++) st[i] = src[i];
    }
    const float A = -__expf(A_log[h]);
    const float bias = dt_bias[h];
    const float Dh = Dp[h];
    for (int c0 = 0; c0 < T; c0 += SCAN_TC) {
        const int tc = min(SCAN_TC, T - c0);
        for (int idx = tid; idx < tc * 32; idx += 128) {
            int row = idx >> 5, q = idx & 31, n = 4 * q;
            const float* r = xbc + (size_t)(c0 + row) * 10240;
            *reinterpret_cast<float4*>(&Bs[row][n + (n >> 6) * 4]) = *reinterpret_cast<const float4*>(r + 8192 + g * 128 + n);
            *reinterpret_cast<float4*>(&Cs[row][n + (n >> 6) * 4]) = *reinterpret_cast<const float4*>(r + 9216 + g * 128 + n);
        }
        for (int idx = tid; idx < tc * 16; idx += 128) {
            int row = idx >> 4, q = idx & 15;
            *reinterpret_cast<float4*>(&xs[row][4 * q]) = *reinterpret_cast<const float4*>(xbc + (size_t)(c0 + row) * 10240 + h * 64 + 4 * q);
        }
        if (tid < tc) {
            float v = proj[(size_t)(c0 + tid) * ldp + 18432 + h] + bias;
            float sp = v > 20.f ? v : log1pf(__expf(v));
            sps[tid] = sp;
            dAs[tid] = __expf(sp * A);
        }
        __syncthreads();
        for (int tl = 0; tl < tc; tl++) {
            const float sp = sps[tl], dA = dAs[tl], xv = xs[tl][d], sx = sp * xv;
            const float4* Bp = reinterpret_cast<const float4*>(&Bs[tl][half * 68]);
            const float4* Cp = reinterpret_cast<const float4*>(&Cs[tl][half * 68]);
            float part = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                float4 B = Bp[i], C = Cp[i];
                st[i].x = st[i].x * dA + sx * B.x;
                st[i].y = st[i].y * dA + sx * B.y;
                st[i].z = st[i].z * dA + sx * B.z;
                st[i].w = st[i].w * dA + sx * B.w;
                part += st[i].x * C.x + st[i].y * C.y + st[i].z * C.z + st[i].w * C.w;
            }
            part += __shfl_xor_sync(0xffffffffu, part, 1);
            if (half == 0) y[(size_t)(c0 + tl) * 8192 + h * 64 + d] = part + Dh * xv;
        }
        __syncthreads();
    }
    float4* dst = reinterpret_cast<float4*>(st_pool + (size_t)(cur ^ 1) * SSM_ELEMS + ((size_t)h * 64 + d) * 128 + half * 64);
#pragma unroll
    for (int i = 0; i < 16; i++) dst[i] = st[i];
}
void k_scan(cudaStream_t s, const Ctl* ctl, float* pool, const float* xbc, const float* proj,
            int ldp, const float* A_log, const float* dt_bias, const float* D, float* y, int T, int record, int nseg) {
    if (nseg == 1 && T >= 64 && !record) {
        scan_t_k<<<128, 128, 0, s>>>(ctl, pool, xbc, proj, ldp, A_log, dt_bias, D, y, T);
        return;
    }
    scan_k<<<dim3(128 * 8, nseg), 256, 0, s>>>(ctl, pool, xbc, proj, ldp, A_log, dt_bias, D, y);
}

// out = RMSNorm_group(y * silu(z)) * w, groups of 1024.
__global__ void gate_rms_k(const float* __restrict__ y, const float* __restrict__ proj, int ldp, const float* __restrict__ w,
                           float* __restrict__ out, float eps) {
    int t = blockIdx.x >> 3, g = blockIdx.x & 7;
    const float4* yp = reinterpret_cast<const float4*>(y + (size_t)t * 8192 + g * 1024);
    const float4* zp = reinterpret_cast<const float4*>(proj + (size_t)t * ldp + g * 1024);
    float4 v = {0, 0, 0, 0};
    float ss = 0.f;
    {
        float4 a = yp[threadIdx.x], z = zp[threadIdx.x];
        v = make_float4(a.x * silu(z.x), a.y * silu(z.y), a.z * silu(z.z), a.w * silu(z.w));
        ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = warp_sum(ss);
    __shared__ float sm[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) tot += sm[i];
    float inv = rsqrtf(tot / 1024.f + eps);
    float4 g4 = __ldg(reinterpret_cast<const float4*>(w + g * 1024) + threadIdx.x);
    reinterpret_cast<float4*>(out + (size_t)t * 8192 + g * 1024)[threadIdx.x] =
        make_float4(v.x * inv * g4.x, v.y * inv * g4.y, v.z * inv * g4.z, v.w * inv * g4.w);
}
void k_gate_rms(cudaStream_t s, const float* y, const float* proj, int ldp, const float* w, float* out, float eps, int T) {
    gate_rms_k<<<T * 8, 256, 0, s>>>(y, proj, ldp, w, out, eps);
}

// Sigmoid router: the top TOPN of sigmoid(logit) + bias (ties to the lower index), weights are the plain
// sigmoids normalised to one and scaled. One warp per token; E <= 512.
__global__ void route_k(const float* __restrict__ logits, int ldl, const float* __restrict__ bias, int* __restrict__ ridx,
                        float* __restrict__ rw, int E, float scale) {
    int t = blockIdx.x, lane = threadIdx.x;
    float sig[16], sel[16];
#pragma unroll
    for (int j = 0; j < 16; j++) {
        int i = lane + 32 * j;
        if (i < E) {
            float z = logits[(size_t)t * ldl + i];
            sig[j] = 1.f / (1.f + __expf(-z));
            sel[j] = sig[j] + bias[i];
        } else {
            sig[j] = 0.f;
            sel[j] = -1e30f;
        }
    }
    float picked[TOPN];
    int pid[TOPN];
#pragma unroll
    for (int k = 0; k < TOPN; k++) {
        float bv = -3e38f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int i = lane + 32 * j;
            if (sel[j] > bv) { bv = sel[j]; bi = i; }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        pid[k] = bi;
        float ps = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            if (lane + 32 * j == bi) { ps = sig[j]; sel[j] = -1e30f; }
        }
        picked[k] = warp_sum(ps);
    }
    float sum = 1e-20f;
#pragma unroll
    for (int k = 0; k < TOPN; k++) sum += picked[k];
    if (lane < TOPN) {
        int idx = 0;
        float val = 0.f;
#pragma unroll
        for (int k = 0; k < TOPN; k++) if (k == lane) { idx = pid[k]; val = picked[k]; }
        ridx[t * TOPN + lane] = idx;
        rw[t * TOPN + lane] = val / sum * scale;
    }
}
void k_route(cudaStream_t s, const float* logits, int ldl, const float* bias, int* ridx, float* rw, int E, float scale, int T) {
    route_k<<<T, 32, 0, s>>>(logits, ldl, bias, ridx, rw, E, scale);
}

// out[t] = sum_k rw[t][k] * o[t*TOPN + k]  (1024 wide)
__global__ void combine_k(const float* __restrict__ o, const float* __restrict__ rw, float* __restrict__ out) {
    int t = blockIdx.x;
    float4 acc = {0, 0, 0, 0};
    for (int k = 0; k < TOPN; k++) {
        float wk = rw[t * TOPN + k];
        float4 v = reinterpret_cast<const float4*>(o + (size_t)(t * TOPN + k) * 1024)[threadIdx.x];
        acc.x += wk * v.x; acc.y += wk * v.y; acc.z += wk * v.z; acc.w += wk * v.w;
    }
    reinterpret_cast<float4*>(out + (size_t)t * 1024)[threadIdx.x] = acc;
}
void k_combine(cudaStream_t s, const float* o, const float* rw, float* out, int T) { combine_k<<<T, 256, 0, s>>>(o, rw, out); }

// ---------------------------------------------------------------------------------------------------------
// Attention: 32 query heads over 2 KV heads, bf16 cache [pos][2][128]. Split-KV so long contexts use the GPU.
// ---------------------------------------------------------------------------------------------------------

// fp8 e4m3 cache: one fp32 scale per (token, head), amax / 448, so every row uses the format's whole range.
// A block takes one token: warps 0-1 are K heads 0-1, warps 2-3 V heads 0-1.
__global__ void __launch_bounds__(128) kv_store_k(const Ctl* __restrict__ ctl, const float* __restrict__ k, const float* __restrict__ v, KvView kv) {
    int t = blockIdx.x;
    if (t >= ctl->nrows) return;   // padding rows of a block rounded up to a stable shape
    int sg = seg_of(ctl, t);
    size_t pos = (size_t)ctl->seg_seq[sg] * kv.ctx + (size_t)(ctl->seg_base[sg] + t - ctl->seg_row0[sg]);
    if (!kv.fp8) {
        uint16_t* kc = (uint16_t*)kv.k;
        uint16_t* vc = (uint16_t*)kv.v;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            kc[pos * 256 + i] = __bfloat16_as_ushort(__float2bfloat16(k[t * 256 + i]));
            vc[pos * 256 + i] = __bfloat16_as_ushort(__float2bfloat16(v[t * 256 + i]));
        }
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int isv = warp >> 1, h = warp & 1;
    const float* src = (isv ? v : k) + (size_t)t * 256 + h * 128;
    float4 x = reinterpret_cast<const float4*>(src)[lane];
    float am = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
    const float sc = am > 0.f ? am * (1.f / 448.f) : 1.f, inv = 1.f / sc;
    __nv_fp8x4_e4m3 q(make_float4(x.x * inv, x.y * inv, x.z * inv, x.w * inv));
    uint8_t* dst = (uint8_t*)(isv ? kv.v : kv.k) + (pos * 2 + h) * 128;
    reinterpret_cast<uint32_t*>(dst)[lane] = *reinterpret_cast<uint32_t*>(&q);
    if (lane == 0) (isv ? kv.vs : kv.ks)[pos * 2 + h] = sc;
}
void k_kv_store(cudaStream_t s, const Ctl* ctl, const float* k, const float* v, const KvView& kv, int T) {
    kv_store_k<<<T, 128, 0, s>>>(ctl, k, v, kv);
}

// One K or V element of (sequence, position, head, dim) for the CUDA-core kernels.
__device__ __forceinline__ float kv_elem(const KvView& kv, bool isv, size_t seq, size_t pos, int g, int d) {
    const size_t row = (seq * kv.ctx + pos) * 2 + g;
    if (!kv.fp8) return __uint_as_float((unsigned)((const uint16_t*)(isv ? kv.v : kv.k))[row * 128 + d] << 16);
    const uint8_t b = ((const uint8_t*)(isv ? kv.v : kv.k))[row * 128 + d];
    __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(hr)) * (isv ? kv.vs : kv.ks)[row];
}

constexpr int ATT_PART = 16 + 16 + 16 * 128;  // m[16], l[16], o[16][128]

__global__ void __launch_bounds__(256) attn_part_k(const Ctl* __restrict__ ctl, const float* __restrict__ q, KvView kvc,
                                                   float* __restrict__ part, int splits) {
    const int sp = blockIdx.x, g = blockIdx.y, t = blockIdx.z;
    const bool pad = t >= ctl->nrows;   // padding rows of a block rounded up to a stable shape read one key of sequence 0
    const int sg = pad ? 0 : seg_of(ctl, t);
    const int len = pad ? 1 : ctl->seg_base[sg] + (t - ctl->seg_row0[sg]) + 1;
    const size_t seq = pad ? 0 : ctl->seg_seq[sg];
    int chunk = ((len + splits - 1) / splits + 31) / 32 * 32;
    const int p0 = sp * chunk, p1 = min(len, p0 + chunk);
    float* out = part + (size_t)((t * 2 + g) * splits + sp) * ATT_PART;

    __shared__ float qs[16][128];
    __shared__ float ks[128][33];
    __shared__ float vs[32][128];
    __shared__ float ps[16][32];
    __shared__ float mrun[16], lrun[16], scl[16];
    const int tid = threadIdx.x;
    const float qscale = rsqrtf(128.f);
    for (int i = tid; i < 16 * 128; i += 256) qs[i >> 7][i & 127] = q[(size_t)t * 4096 + g * 2048 + i] * qscale;
    if (tid < 16) { mrun[tid] = -1e30f; lrun[tid] = 0.f; }
    const int ph = tid >> 4, pl = tid & 15;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.f;
    __syncthreads();
    for (int base = p0; base < p1; base += 32) {
        int n = min(32, p1 - base);
        // stage the K tile transposed and the V tile
        for (int i = tid; i < 32 * 128; i += 256) {
            int p = i >> 7, d = i & 127;
            float kv = 0.f, vv = 0.f;
            if (p < n) {
                kv = kv_elem(kvc, false, seq, base + p, g, d);
                vv = kv_elem(kvc, true, seq, base + p, g, d);
            }
            ks[d][p] = kv;
            vs[p][d] = vv;
        }
        __syncthreads();
        for (int pair = tid; pair < 16 * 32; pair += 256) {
            int h = pair >> 5, p = pair & 31;
            float dot = 0.f;
#pragma unroll 8
            for (int d = 0; d < 128; d++) dot += qs[h][d] * ks[d][p];
            ps[h][p] = p < n ? dot : -1e30f;
        }
        __syncthreads();
        if (tid < 16) {
            float mx = mrun[tid];
            for (int p = 0; p < 32; p++) mx = fmaxf(mx, ps[tid][p]);
            float sc = __expf(mrun[tid] - mx);
            float l = lrun[tid] * sc;
            for (int p = 0; p < 32; p++) {
                float e = ps[tid][p] > -1e29f ? __expf(ps[tid][p] - mx) : 0.f;
                ps[tid][p] = e;
                l += e;
            }
            mrun[tid] = mx; lrun[tid] = l; scl[tid] = sc;
        }
        __syncthreads();
        {
            float sc = scl[ph];
#pragma unroll
            for (int i = 0; i < 8; i++) acc[i] *= sc;
            for (int p = 0; p < 32; p++) {
                float pr = ps[ph][p];
#pragma unroll
                for (int i = 0; i < 8; i++) acc[i] += pr * vs[p][pl + 16 * i];
            }
        }
        __syncthreads();
    }
    if (tid < 16) { out[tid] = mrun[tid]; out[16 + tid] = lrun[tid]; }
#pragma unroll
    for (int i = 0; i < 8; i++) out[32 + ph * 128 + pl + 16 * i] = acc[i];
}

__global__ void attn_comb_k(const float* __restrict__ part, float* __restrict__ out, int splits) {
    int t = blockIdx.x, h = blockIdx.y;
    int g = h >> 4, hh = h & 15;
    int d = threadIdx.x;
    const float* base = part + (size_t)((t * 2 + g) * splits) * ATT_PART;
    float M = -1e30f;
    for (int s = 0; s < splits; s++) M = fmaxf(M, base[(size_t)s * ATT_PART + hh]);
    float L = 0.f, O = 0.f;
    for (int s = 0; s < splits; s++) {
        const float* p = base + (size_t)s * ATT_PART;
        float w = __expf(p[hh] - M);
        L += w * p[16 + hh];
        O += w * p[32 + hh * 128 + d];
    }
    out[(size_t)t * 4096 + h * 128 + d] = O / L;
}

// Prompt chunks: a CTA takes QB queries x 16 heads of one KV head against a split of the keys, so each K/V tile is
// staged once for QB queries instead of once per query.
constexpr int QB = 4;
constexpr int BROWS = QB * 16;
constexpr int BPART = BROWS + BROWS + BROWS * 128;  // m, l, o[BROWS][128]
constexpr int BLK_SMEM = (BROWS * 128 + 128 * 33 + 32 * 128 + BROWS * 32) * 4;

__global__ void __launch_bounds__(256) attn_blk_part_k(const Ctl* __restrict__ ctl, const float* __restrict__ q,
                                                       const uint16_t* __restrict__ kc_all, const uint16_t* __restrict__ vc_all,
                                                       float* __restrict__ part, int splits, int T, int ctx) {   // bf16 cache only
    const uint16_t* kc = kc_all + (size_t)ctl->seg_seq[0] * ctx * 256;
    const uint16_t* vc = vc_all + (size_t)ctl->seg_seq[0] * ctx * 256;
    const int base0 = ctl->seg_base[0];
    extern __shared__ float sm[];
    float* qs = sm;                       // [BROWS][128]
    float* ks = qs + BROWS * 128;         // [128][33]
    float* vs = ks + 128 * 33;            // [32][128]
    float* ps = vs + 32 * 128;            // [BROWS][32]
    __shared__ float mrun[BROWS], lrun[BROWS], scl[BROWS];
    __shared__ int rlen[BROWS];
    const int sp = blockIdx.x, g = blockIdx.y, qt = blockIdx.z;
    const int tid = threadIdx.x;
    const int maxlen = base0 + min(T, qt * QB + QB);
    int chunk = ((maxlen + splits - 1) / splits + 31) / 32 * 32;
    const int p0 = sp * chunk, p1 = min(maxlen, p0 + chunk);
    float* out = part + (size_t)((qt * 2 + g) * splits + sp) * BPART;
    const float qscale = rsqrtf(128.f);
    for (int i = tid; i < BROWS * 128; i += 256) {
        int r = i >> 7, d = i & 127;
        int t = qt * QB + (r >> 4), hh = r & 15;
        qs[i] = t < T ? q[(size_t)t * 4096 + (g * 16 + hh) * 128 + d] * qscale : 0.f;
    }
    if (tid < BROWS) {
        int t = qt * QB + (tid >> 4);
        rlen[tid] = t < T ? base0 + t + 1 : 0;
        mrun[tid] = -1e30f;
        lrun[tid] = 0.f;
    }
    const int pr = tid >> 2, pc = (tid & 3) * 32;  // PV: row, first dim
    float acc[32];
#pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
    __syncthreads();
    for (int base = p0; base < p1; base += 32) {
        const int n = min(32, p1 - base);
        for (int i = tid; i < 32 * 128; i += 256) {
            int p = i >> 7, d = i & 127;
            float kv = 0.f, vv = 0.f;
            if (p < n) {
                size_t o = ((size_t)(base + p) * 2 + g) * 128 + d;
                kv = __uint_as_float((unsigned)kc[o] << 16);
                vv = __uint_as_float((unsigned)vc[o] << 16);
            }
            ks[d * 33 + p] = kv;
            vs[p * 128 + d] = vv;
        }
        __syncthreads();
        for (int pair = tid; pair < BROWS * 32; pair += 256) {
            int r = pair >> 5, p = pair & 31;
            float dot = 0.f;
#pragma unroll 8
            for (int d = 0; d < 128; d++) dot += qs[r * 128 + d] * ks[d * 33 + p];
            ps[r * 32 + p] = (p < n && base + p < rlen[r]) ? dot : -1e30f;
        }
        __syncthreads();
        if (tid < BROWS) {
            float mx = mrun[tid];
            for (int p = 0; p < 32; p++) mx = fmaxf(mx, ps[tid * 32 + p]);
            float sc = __expf(mrun[tid] - mx);
            float l = lrun[tid] * sc;
            for (int p = 0; p < 32; p++) {
                float e = ps[tid * 32 + p] > -1e29f ? __expf(ps[tid * 32 + p] - mx) : 0.f;
                ps[tid * 32 + p] = e;
                l += e;
            }
            mrun[tid] = mx; lrun[tid] = l; scl[tid] = sc;
        }
        __syncthreads();
        {
            float sc = scl[pr];
#pragma unroll
            for (int i = 0; i < 32; i++) acc[i] *= sc;
            for (int p = 0; p < 32; p++) {
                float pe = ps[pr * 32 + p];
#pragma unroll
                for (int i = 0; i < 32; i++) acc[i] += pe * vs[p * 128 + pc + i];
            }
        }
        __syncthreads();
    }
    if (tid < BROWS) { out[tid] = mrun[tid]; out[BROWS + tid] = lrun[tid]; }
#pragma unroll
    for (int i = 0; i < 32; i++) out[2 * BROWS + pr * 128 + pc + i] = acc[i];
}

__global__ void attn_blk_comb_k(const float* __restrict__ part, float* __restrict__ out, int splits) {
    int t = blockIdx.x, h = blockIdx.y;
    int g = h >> 4, hh = h & 15;
    int qt = t / QB, r = (t % QB) * 16 + hh;
    int d = threadIdx.x;
    const float* base = part + (size_t)((qt * 2 + g) * splits) * BPART;
    float M = -1e30f;
    for (int s = 0; s < splits; s++) M = fmaxf(M, base[(size_t)s * BPART + r]);
    float L = 0.f, O = 0.f;
    for (int s = 0; s < splits; s++) {
        const float* p = base + (size_t)s * BPART;
        float w = __expf(p[r] - M);
        L += w * p[BROWS + r];
        O += w * p[2 * BROWS + r * 128 + d];
    }
    out[(size_t)t * 4096 + h * 128 + d] = O / L;
}

void k_attn(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out,
            int T, int splits, int multi, int) {
    if (T <= 8 || multi) {
        attn_part_k<<<dim3(splits, 2, T), 256, 0, s>>>(ctl, q, kv, part, splits);
        attn_comb_k<<<dim3(T, 32), 128, 0, s>>>(part, out, splits);
        return;
    }
    // the query-blocked prompt kernel (bf16 cache only; prompts use k_attn_mma)
    static bool opted = false;
    if (!opted) {
        cudaFuncSetAttribute(attn_blk_part_k, cudaFuncAttributeMaxDynamicSharedMemorySize, BLK_SMEM);
        opted = true;
    }
    attn_blk_part_k<<<dim3(splits, 2, (T + QB - 1) / QB), 256, BLK_SMEM, s>>>(ctl, q, (const uint16_t*)kv.k, (const uint16_t*)kv.v, part, splits, T, kv.ctx);
    attn_blk_comb_k<<<dim3(T, 32), 128, 0, s>>>(part, out, splits);
}

// ---------------------------------------------------------------------------------------------------------
// Row-wise top candidates of the logits. Stage one: 128 blocks per row, each selects its slice's best
// `topk`; stage two merges them. Ties go to the lower id, so the order is total.
// ---------------------------------------------------------------------------------------------------------

constexpr int CAND_BLOCKS = 128;

static __device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

__global__ void __launch_bounds__(256) cand1_k(const float* __restrict__ logits, int topk, float* __restrict__ sv, int* __restrict__ si) {
    const int t = blockIdx.y, b = blockIdx.x;
    const float* row = logits + (size_t)t * VOCAB + (size_t)b * 1024;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = row[threadIdx.x + 256 * j];
    __shared__ float wv[2][8];
    __shared__ int wi[2][8];
    for (int r = 0; r < topk; r++) {
        float bv = -3.4e38f;
        int bj = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) if (v[j] > bv) { bv = v[j]; bj = j; }
        int bi = threadIdx.x + 256 * bj;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        int slot = r & 1;
        if ((threadIdx.x & 31) == 0) { wv[slot][threadIdx.x >> 5] = bv; wi[slot][threadIdx.x >> 5] = bi; }
        __syncthreads();
        float fv = wv[slot][0];
        int fi = wi[slot][0];
#pragma unroll
        for (int w = 1; w < 8; w++) if (better(wv[slot][w], wi[slot][w], fv, fi)) { fv = wv[slot][w]; fi = wi[slot][w]; }
        if (threadIdx.x == 0) {
            size_t o = ((size_t)t * CAND_BLOCKS + b) * CAND + r;
            sv[o] = fv;
            si[o] = b * 1024 + fi;
        }
        if ((fi & 255) == threadIdx.x) v[fi >> 8] = -3.4e38f;
    }
}

__global__ void __launch_bounds__(256) cand2_k(int topk, const float* __restrict__ sv, const int* __restrict__ si,
                                                float* __restrict__ vals, int* __restrict__ ids) {
    const int t = blockIdx.x;
    // CAND_BLOCKS * topk candidates, 32 per thread at most
    const int total = CAND_BLOCKS * topk;
    float v[32];
    int id[32];
#pragma unroll
    for (int j = 0; j < 32; j++) {
        int k = threadIdx.x + 256 * j;
        if (k < total) {
            int blk = k / topk, r = k % topk;
            size_t o = ((size_t)t * CAND_BLOCKS + blk) * CAND + r;
            v[j] = sv[o];
            id[j] = si[o];
        } else {
            v[j] = -3.4e38f;
            id[j] = 0x7fffffff;
        }
    }
    __shared__ float wv[2][8];
    __shared__ int wi[2][8];
    for (int r = 0; r < topk; r++) {
        float bv = -3.4e38f;
        int bi = 0x7fffffff, bj = 0;
#pragma unroll
        for (int j = 0; j < 32; j++) if (better(v[j], id[j], bv, bi)) { bv = v[j]; bi = id[j]; bj = j; }
        float lv = bv; int li = bi;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        int slot = r & 1;
        if ((threadIdx.x & 31) == 0) { wv[slot][threadIdx.x >> 5] = bv; wi[slot][threadIdx.x >> 5] = bi; }
        __syncthreads();
        float fv = wv[slot][0];
        int fi = wi[slot][0];
#pragma unroll
        for (int w = 1; w < 8; w++) if (better(wv[slot][w], wi[slot][w], fv, fi)) { fv = wv[slot][w]; fi = wi[slot][w]; }
        if (threadIdx.x == 0) {
            vals[(size_t)t * CAND + r] = fv;
            ids[(size_t)t * CAND + r] = fi;
        }
        // retire the winner from the thread that holds it
        if (li == fi && lv == fv) {
#pragma unroll
            for (int j = 0; j < 32; j++) if (j == bj) { v[j] = -3.4e38f; id[j] = 0x7fffffff; }
        }
    }
}

// The same for chosen rows among only the tokens a structured-output mask allows (bit t of a row's `words` u32s): stage one
// of row sel[i] reads its mask from masks + i * words, a disallowed token counting as -infinity.
__global__ void __launch_bounds__(256) cand1m_k(const float* __restrict__ logits, const int* __restrict__ sel, const uint32_t* __restrict__ masks,
                                                 int words, int topk, float* __restrict__ sv, int* __restrict__ si) {
    const int i = blockIdx.y, b = blockIdx.x;
    const float* row = logits + (size_t)sel[i] * VOCAB + (size_t)b * 1024;
    const uint32_t* mk = masks + (size_t)i * words;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int tok = b * 1024 + threadIdx.x + 256 * j;
        const bool ok = (tok >> 5) < words && ((mk[tok >> 5] >> (tok & 31)) & 1u);
        v[j] = ok ? row[threadIdx.x + 256 * j] : -3.4e38f;
    }
    __shared__ float wv[2][8];
    __shared__ int wi[2][8];
    for (int r = 0; r < topk; r++) {
        float bv = -3.4e38f;
        int bj = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) if (v[j] > bv) { bv = v[j]; bj = j; }
        int bi = threadIdx.x + 256 * bj;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        int slot = r & 1;
        if ((threadIdx.x & 31) == 0) { wv[slot][threadIdx.x >> 5] = bv; wi[slot][threadIdx.x >> 5] = bi; }
        __syncthreads();
        float fv = wv[slot][0];
        int fi = wi[slot][0];
#pragma unroll
        for (int w = 1; w < 8; w++) if (better(wv[slot][w], wi[slot][w], fv, fi)) { fv = wv[slot][w]; fi = wi[slot][w]; }
        if (threadIdx.x == 0) {
            size_t o = ((size_t)i * CAND_BLOCKS + b) * CAND + r;
            sv[o] = fv;
            si[o] = b * 1024 + fi;
        }
        if ((fi & 255) == threadIdx.x) v[fi >> 8] = -3.4e38f;
    }
}

void k_candidates_masked(cudaStream_t s, const float* logits, const int* sel, const uint32_t* masks, int words, int n, int topk,
                         float* vals, int* ids, float* sv, int* si) {
    if (topk < 1) topk = 1;
    if (topk > CAND) topk = CAND;
    cand1m_k<<<dim3(CAND_BLOCKS, n), 256, 0, s>>>(logits, sel, masks, words, topk, sv, si);
    cand2_k<<<n, 256, 0, s>>>(topk, sv, si, vals, ids);
}

void k_candidates(cudaStream_t s, const float* logits, int T, int topk, float* vals, int* ids, float* sv, int* si) {
    if (topk < 1) topk = 1;
    if (topk > CAND) topk = CAND;
    cand1_k<<<dim3(CAND_BLOCKS, T), 256, 0, s>>>(logits, topk, sv, si);
    cand2_k<<<T, 256, 0, s>>>(topk, sv, si, vals, ids);
}

// Replay the first n rows of a verify window from the pre-window state: grid.y selects the layer.
__global__ void __launch_bounds__(256) scan_replay_k(const ReplayLayer* __restrict__ layers, int seq, int slot, int row0, int n) {
    const ReplayLayer L = layers[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.x >> 3;
    const int d = (blockIdx.x & 7) * 8 + warp;
    const int g = h >> 4;
    const size_t off = ((size_t)h * 64 + d) * 128 + 4 * lane;
    float* pool = L.ssm + (size_t)seq * 2 * SSM_ELEMS;
    float4 st = *reinterpret_cast<const float4*>(pool + (size_t)slot * SSM_ELEMS + off);
    const float A = -__expf(L.A_log[h]);
    const float bias = L.dt_bias[h];
    for (int t = row0; t < row0 + n; t++) {
        float v = L.hist_dt[t * 128 + h] + bias;
        float sp = v > 20.f ? v : log1pf(__expf(v));
        float dA = __expf(sp * A);
        const float* row = L.hist_x + (size_t)t * 10240;
        float sx = sp * row[h * 64 + d];
        float4 B = *reinterpret_cast<const float4*>(row + 8192 + g * 128 + 4 * lane);
        st.x = st.x * dA + sx * B.x;
        st.y = st.y * dA + sx * B.y;
        st.z = st.z * dA + sx * B.z;
        st.w = st.w * dA + sx * B.w;
    }
    *reinterpret_cast<float4*>(pool + (size_t)(slot ^ 1) * SSM_ELEMS + off) = st;
}
void k_scan_replay(cudaStream_t s, const ReplayLayer* layers, int nlayers, int seq, int slot, int row0, int n) {
    scan_replay_k<<<dim3(128 * 8, nlayers), 256, 0, s>>>(layers, seq, slot, row0, n);
}
