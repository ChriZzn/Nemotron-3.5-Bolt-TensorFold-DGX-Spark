// C-RADIO v4-H and the vision projector on CUDA. fp32 activations, bf16 weights through the tile-major tensor-core GEMMs
// of bolt_mma.cu (which round the activations to bf16, as the checkpoint's own bf16 model does).
#include "bolt_vision.h"

#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { fprintf(stderr, "bolt vision: %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(_e)); return -1; } } while (0)

namespace {

// ---------------------------------------------------------------------------------------------------------
// The image processor's sizing rules (image_processing_nemotron_h_omni.py)
// ---------------------------------------------------------------------------------------------------------

constexpr int DS = 2;  // pixel-shuffle factor

void rule_image(int width, int height, int avail, int* cols, int* rows) {
    // Python's round() is banker's: x.5 goes to the even neighbour, and x + 0.5 is x.5 whenever the size is a multiple of 16
    auto pyround = [](double v) { double r = std::floor(v + 0.5); if (std::fabs(v - std::floor(v) - 0.5) < 1e-12 && std::fmod(r, 2.0) != 0.0) r -= 1.0; return (int)r; };
    const int ch = pyround(height / (double)VIT_PATCH + 0.5), cw = pyround(width / (double)VIT_PATCH + 0.5);
    const double patches = (double)ch * cw;
    const double f = std::min(std::sqrt(avail / patches), 1.0);
    int th = (int)std::floor(f * ch), tw = (int)std::floor(f * cw);
    if (avail > VIT_MIN_PATCHES && th * tw < VIT_MIN_PATCHES) {
        const double up = std::sqrt((double)VIT_MIN_PATCHES / (th * tw));
        th = (int)std::ceil(up * th);
        tw = (int)std::ceil(up * tw);
    }
    int rem = th % DS;
    if (rem) {
        const int inc = DS - rem;
        if ((th + inc) * tw <= avail) th += inc; else th = std::max(DS, th - rem);
    }
    rem = tw % DS;
    if (rem) {
        const int inc = DS - rem;
        if (th * (tw + inc) <= avail) tw += inc; else tw = std::max(DS, tw - rem);
    }
    *cols = tw;
    *rows = th;
}

}  // namespace

void vision_grid(int width, int height, int* cols, int* rows) {
    int budget = (VIT_MAX_MODEL_LEN - 4) * DS * DS;
    budget = std::max(budget, VIT_MIN_PATCHES);
    const int per = std::max(std::min(budget, VIT_MAX_PATCHES), VIT_MIN_PATCHES);
    rule_image(width, height, per, cols, rows);
}

void vision_grid_video(int width, int height, int* cols, int* rows) {
    const double target = VIT_VIDEO_PATCHES, aspect = width / (double)std::max(height, 1);
    auto pyr = [](double v) { double r = std::floor(v + 0.5); if (std::fabs(v - std::floor(v) - 0.5) < 1e-12 && std::fmod(r, 2.0) != 0.0) r -= 1.0; return (int)r; };
    int ph = std::max(pyr(std::sqrt(target / aspect)), 1), pw = std::max(pyr(std::sqrt(target * aspect)), 1);
    const int remh = ph % DS, remw = pw % DS;
    const int ph_up = ph + (remh ? DS - remh : 0), ph_down = ph - remh, pw_up = pw + (remw ? DS - remw : 0), pw_down = pw - remw;
    if (ph_up * pw_up <= VIT_VIDEO_PATCHES) { ph = ph_up; pw = pw_up; }
    else { ph = std::max(DS, ph_down); pw = std::max(DS, pw_down); }
    *cols = pw;
    *rows = ph;
}

namespace {

// ---------------------------------------------------------------------------------------------------------
// Resizing: torch's antialiased bicubic (upsample_bicubic2d_aa, a = -0.5), as separable weight tables
// ---------------------------------------------------------------------------------------------------------

double cubic_filter(double x) {
    const double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

struct Axis {
    int out = 0, taps = 0;
    std::vector<int> start, count;
    std::vector<float> w;   // [out][taps]
};

Axis make_axis(int in, int out) {
    Axis ax;
    ax.out = out;
    const double scale = (double)in / out;
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    ax.taps = (int)std::ceil(support) * 2 + 2;
    ax.start.resize(out); ax.count.resize(out);
    ax.w.assign((size_t)out * ax.taps, 0.f);
    for (int i = 0; i < out; i++) {
        const double center = scale * (i + 0.5);
        const int xmin = std::max((int)(center - support + 0.5), 0);
        const int xmax = std::min((int)(center + support + 0.5), in);
        const int n = std::max(xmax - xmin, 1);
        double total = 0;
        std::vector<double> ww(n);
        for (int j = 0; j < n; j++) { ww[j] = cubic_filter((j + xmin - center + 0.5) * invscale); total += ww[j]; }
        ax.start[i] = xmin; ax.count[i] = n;
        for (int j = 0; j < n; j++) ax.w[(size_t)i * ax.taps + j] = (float)(total != 0 ? ww[j] / total : ww[j]);
    }
    return ax;
}

// pass 1: tmp[y][xo][c] = sum_j wx[xo][j] * src[y][start + j][c]
__global__ void resize_h_k(const uint8_t* __restrict__ src, int sw, int sh, float* __restrict__ tmp, int ow,
                           const int* __restrict__ st, const int* __restrict__ cnt, const float* __restrict__ w, int taps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sh * ow * 3) return;
    const int c = i % 3, xo = (i / 3) % ow, y = i / (3 * ow);
    float acc = 0.f;
    const uint8_t* row = src + (size_t)y * sw * 3 + c;
    const int s0 = st[xo], n = cnt[xo];
    for (int j = 0; j < n; j++) acc += w[(size_t)xo * taps + j] * (float)row[(size_t)(s0 + j) * 3];
    tmp[i] = acc;
}

// pass 2: vertical resize, then (v / 255 - mean) / std straight into the patch layout [patch][chan][py][px]
__global__ void resize_v_patch_k(const float* __restrict__ tmp, int ow, float* __restrict__ pat, int pd, int cols, int oh, int cbase,
                                 const int* __restrict__ st, const int* __restrict__ cnt, const float* __restrict__ w, int taps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= oh * ow * 3) return;
    const int c = i % 3, xo = (i / 3) % ow, yo = i / (3 * ow);
    float acc = 0.f;
    const int s0 = st[yo], n = cnt[yo];
    for (int j = 0; j < n; j++) acc += w[(size_t)yo * taps + j] * tmp[((size_t)(s0 + j) * ow + xo) * 3 + c];
    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f}, stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    const float v = (acc / 255.f - mean[c]) / stdv[c];
    const int patch = (yo / VIT_PATCH) * cols + xo / VIT_PATCH;
    pat[(size_t)patch * pd + (cbase + c) * 256 + (yo % VIT_PATCH) * VIT_PATCH + xo % VIT_PATCH] = v;
}

// ---------------------------------------------------------------------------------------------------------
// Tower kernels
// ---------------------------------------------------------------------------------------------------------

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }

// prefix tokens, and the cropped position embedding: the 128 x 128 table resampled bilinearly (no antialiasing, corners
// aligned by half pixels) to max(rows, cols) square, then cropped to rows x cols
__global__ void pos_add_k(float* __restrict__ x, const float* __restrict__ cls, const uint16_t* __restrict__ pos, int rows, int cols) {
    const int tok = blockIdx.x;   // 0 .. 9 prefix, then patches
    if (tok < VIT_PREFIX) {
        for (int c = threadIdx.x; c < VIT_HID; c += blockDim.x) x[(size_t)tok * VIT_HID + c] = cls[tok * VIT_HID + c];
        return;
    }
    const int p = tok - VIT_PREFIX, r = p / cols, c = p % cols;
    const int md = max(rows, cols);
    const float scale = (float)VIT_POS_SIDE / md;
    auto src = [&](int d) { float s = (d + 0.5f) * scale - 0.5f; return s < 0.f ? 0.f : s; };
    const float sy = src(r), sx = src(c);
    const int y0 = min((int)sy, VIT_POS_SIDE - 1), x0 = min((int)sx, VIT_POS_SIDE - 1);
    const int y1 = min(y0 + 1, VIT_POS_SIDE - 1), x1 = min(x0 + 1, VIT_POS_SIDE - 1);
    const float ly = sy - y0, lx = sx - x0;
    for (int ch = threadIdx.x; ch < VIT_HID; ch += blockDim.x) {
        const float a = bf2f(pos[((size_t)y0 * VIT_POS_SIDE + x0) * VIT_HID + ch]), b = bf2f(pos[((size_t)y0 * VIT_POS_SIDE + x1) * VIT_HID + ch]);
        const float cc = bf2f(pos[((size_t)y1 * VIT_POS_SIDE + x0) * VIT_HID + ch]), d = bf2f(pos[((size_t)y1 * VIT_POS_SIDE + x1) * VIT_HID + ch]);
        const float v = (1.f - ly) * ((1.f - lx) * a + lx * b) + ly * ((1.f - lx) * cc + lx * d);
        x[(size_t)tok * VIT_HID + ch] += v;
    }
}

__device__ __forceinline__ float block_sum256(float v, float* sm) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) t += sm[i];
    return t;
}

// LayerNorm over 1280 with weight and bias (eps 1e-6), one block per row
__global__ void ln_k(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y) {
    __shared__ float sm[8];
    const float* xr = x + (size_t)blockIdx.x * VIT_HID;
    float v[5];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 5; i++) { v[i] = xr[threadIdx.x + 256 * i]; s += v[i]; }
    const float mean = block_sum256(s, sm) / VIT_HID;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < 5; i++) { v[i] -= mean; q += v[i] * v[i]; }
    const float inv = rsqrtf(block_sum256(q, sm) / VIT_HID + 1e-6f);
#pragma unroll
    for (int i = 0; i < 5; i++) {
        const int c = threadIdx.x + 256 * i;
        y[(size_t)blockIdx.x * VIT_HID + c] = v[i] * inv * w[c] + b[c];
    }
}

__global__ void bias_k(float* __restrict__ y, const float* __restrict__ b, int n, size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] += b[i % n];
}
__global__ void bias_residual_k(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ b, int n, size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += y[i] + b[i % n];
}
__global__ void bias_gelu_k(float* __restrict__ y, const float* __restrict__ b, int n, size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float v = y[i] + b[i % n];
    y[i] = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
}

// ---- bidirectional attention, head dim 80: bf16 mma, online softmax ----

__device__ __forceinline__ void mma_bf16_k(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

constexpr int AK = 88;   // K tile row stride (bf16): 176 bytes, conflict-free fragment loads
constexpr int AV = 40;   // V^T tile row stride (bf16)

// grid (ceil(n / 64), 16 heads), 4 warps; warp w takes queries 16 * w .. of the CTA's 64
__global__ void __launch_bounds__(128) vit_attn_k(const float* __restrict__ qkv, float* __restrict__ out, int n) {
    __shared__ __align__(16) uint16_t Ks[32 * AK];
    __shared__ __align__(16) uint16_t Vt[VIT_HD * AV];
    const int head = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gi = lane >> 2, ti = lane & 3;
    const int q0 = blockIdx.x * 64 + warp * 16;
    const float scale = rsqrtf((float)VIT_HD);
    const int ldq = 3 * VIT_HID;
    uint32_t qa[5][4];
    {
        const int ra = min(q0 + gi, n - 1), rb = min(q0 + gi + 8, n - 1);
        const float* qa_row = qkv + (size_t)ra * ldq + head * VIT_HD;
        const float* qb_row = qkv + (size_t)rb * ldq + head * VIT_HD;
#pragma unroll
        for (int ks = 0; ks < 5; ks++) {
            const int c = ks * 16 + 2 * ti;
            const float2 a = *reinterpret_cast<const float2*>(qa_row + c), b = *reinterpret_cast<const float2*>(qb_row + c);
            const float2 cc = *reinterpret_cast<const float2*>(qa_row + c + 8), d = *reinterpret_cast<const float2*>(qb_row + c + 8);
            qa[ks][0] = pack_bf(a.x * scale, a.y * scale);
            qa[ks][1] = pack_bf(b.x * scale, b.y * scale);
            qa[ks][2] = pack_bf(cc.x * scale, cc.y * scale);
            qa[ks][3] = pack_bf(d.x * scale, d.y * scale);
        }
    }
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    float o[10][4];
#pragma unroll
    for (int i = 0; i < 10; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;

    for (int base = 0; base < n; base += 32) {
        __syncthreads();
        // stage 32 keys: K [key][dim] and V transposed [dim][key], bf16; keys past n are zero
        for (int i = tid; i < 32 * 10; i += 128) {
            const int key = i / 10, c8 = (i % 10) * 8;
            float kf[8] = {0, 0, 0, 0, 0, 0, 0, 0}, vf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (base + key < n) {
                const float* kp = qkv + (size_t)(base + key) * ldq + VIT_HID + head * VIT_HD + c8;
                const float* vp = qkv + (size_t)(base + key) * ldq + 2 * VIT_HID + head * VIT_HD + c8;
                const float4 k0 = *reinterpret_cast<const float4*>(kp), k1 = *reinterpret_cast<const float4*>(kp + 4);
                const float4 v0 = *reinterpret_cast<const float4*>(vp), v1 = *reinterpret_cast<const float4*>(vp + 4);
                kf[0] = k0.x; kf[1] = k0.y; kf[2] = k0.z; kf[3] = k0.w; kf[4] = k1.x; kf[5] = k1.y; kf[6] = k1.z; kf[7] = k1.w;
                vf[0] = v0.x; vf[1] = v0.y; vf[2] = v0.z; vf[3] = v0.w; vf[4] = v1.x; vf[5] = v1.y; vf[6] = v1.z; vf[7] = v1.w;
            }
            uint32_t kw[4];
#pragma unroll
            for (int j = 0; j < 4; j++) kw[j] = pack_bf(kf[2 * j], kf[2 * j + 1]);
            *reinterpret_cast<uint4*>(&Ks[key * AK + c8]) = make_uint4(kw[0], kw[1], kw[2], kw[3]);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat16 b = __float2bfloat16(vf[j]);
                Vt[(c8 + j) * AV + key] = *reinterpret_cast<uint16_t*>(&b);
            }
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 5; ks++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const uint32_t* kp = reinterpret_cast<const uint32_t*>(&Ks[(nt * 8 + gi) * AK + ks * 16 + 2 * ti]);
                mma_bf16_k(s[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], kp[0], kp[4]);
            }
        float tm0 = -1e30f, tm1 = -1e30f;
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int key = base + nt * 8 + 2 * ti + (c & 1);
                if (key >= n) s[nt][c] = -1e30f;
                if (c < 2) tm0 = fmaxf(tm0, s[nt][c]); else tm1 = fmaxf(tm1, s[nt][c]);
            }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1)); tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1)); tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const float sc0 = __expf(m0 - n0), sc1 = __expf(m1 - n1);
        m0 = n0; m1 = n1;
        l0 *= sc0; l1 *= sc1;
#pragma unroll
        for (int i = 0; i < 10; i++) { o[i][0] *= sc0; o[i][1] *= sc0; o[i][2] *= sc1; o[i][3] *= sc1; }
        uint32_t pa[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const float p00 = s[nt][0] > -1e29f ? __expf(s[nt][0] - n0) : 0.f, p01 = s[nt][1] > -1e29f ? __expf(s[nt][1] - n0) : 0.f;
            const float p10 = s[nt][2] > -1e29f ? __expf(s[nt][2] - n1) : 0.f, p11 = s[nt][3] > -1e29f ? __expf(s[nt][3] - n1) : 0.f;
            l0 += p00 + p01; l1 += p10 + p11;
            const int j = nt >> 1, hi = nt & 1;
            pa[j][hi * 2 + 0] = pack_bf(p00, p01);
            pa[j][hi * 2 + 1] = pack_bf(p10, p11);
        }
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int d = 0; d < 10; d++) {
                const uint32_t* vp = reinterpret_cast<const uint32_t*>(&Vt[(d * 8 + gi) * AV + j * 16 + 2 * ti]);
                mma_bf16_k(o[d], pa[j][0], pa[j][1], pa[j][2], pa[j][3], vp[0], vp[4]);
            }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    const int ra = q0 + gi, rb = q0 + gi + 8;
#pragma unroll
    for (int d = 0; d < 10; d++) {
        const int c = head * VIT_HD + d * 8 + 2 * ti;
        if (ra < n) *reinterpret_cast<float2*>(out + (size_t)ra * VIT_HID + c) = make_float2(o[d][0] * i0, o[d][1] * i0);
        if (rb < n) *reinterpret_cast<float2*>(out + (size_t)rb * VIT_HID + c) = make_float2(o[d][2] * i1, o[d][3] * i1);
    }
}

// ---- projector ----

// features of token (r2, c2) = the four patches of its 2x2 block, channel order (dr, dc, k); then RMSNorm (eps 1e-5)
__global__ void shuffle_rms_k(const float* __restrict__ x, int rows, int cols, const float* __restrict__ w, float* __restrict__ y) {
    __shared__ float sm[8];
    const int tok = blockIdx.x, c2n = cols / 2;
    const int r2 = tok / c2n, c2 = tok % c2n;
    float v[20];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 20; i++) {
        const int ch = threadIdx.x + 256 * i;            // 0 .. 5119
        const int blk = ch / VIT_HID, k = ch % VIT_HID, dr = blk >> 1, dc = blk & 1;
        const int patch = (2 * r2 + dr) * cols + 2 * c2 + dc;
        v[i] = x[(size_t)(VIT_PREFIX + patch) * VIT_HID + k];
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(block_sum256(ss, sm) / 5120.f + 1e-5f);
#pragma unroll
    for (int i = 0; i < 20; i++) {
        const int ch = threadIdx.x + 256 * i;
        y[(size_t)tok * 5120 + ch] = v[i] * inv * w[ch];
    }
}
__global__ void relu2_k(float* __restrict__ y, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float v = y[i]; y[i] = v > 0.f ? v * v : 0.f; }
}
__global__ void to_bf16_k(const float* __restrict__ y, uint16_t* __restrict__ o, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { __nv_bfloat16 b = __float2bfloat16(y[i]); o[i] = *reinterpret_cast<uint16_t*>(&b); }
}

}  // namespace

struct Vision {
    VisionW w;
    // scratch, grown to the largest block seen
    size_t cap_tokens = 0;
    float *x = nullptr, *xn = nullptr, *qkv = nullptr, *att = nullptr, *hid = nullptr, *pat = nullptr;
    size_t cap_src = 0, cap_tmp = 0;
    uint8_t* src = nullptr;
    float* tmp = nullptr;
};

Vision* vision_new(const VisionW& w) {
    Vision* v = new Vision();
    v->w = w;
    return v;
}

namespace {

void gemm(cudaStream_t s, const TBF& w, const float* x, int ldx, float* y, int ldy, int T) {
    for (int base = 0; base < T; base += TMAX) {
        const int nt = std::min(TMAX, T - base);
        mma_bf16(s, w, x + (size_t)base * ldx, ldx, y + (size_t)base * ldy, ldy, nt);
    }
}

#define BLK(n) (unsigned)(((n) + 255) / 256)

int ensure(Vision* v, size_t tokens, size_t pd) {
    if (tokens <= v->cap_tokens) return 0;
    if (v->x) { cudaFree(v->x); cudaFree(v->xn); cudaFree(v->qkv); cudaFree(v->att); cudaFree(v->hid); cudaFree(v->pat); }
    const size_t t = tokens;
    CK(cudaMalloc(&v->x, t * VIT_HID * 4));
    CK(cudaMalloc(&v->xn, t * VIT_HID * 4));
    CK(cudaMalloc(&v->qkv, t * 3 * VIT_HID * 4));
    CK(cudaMalloc(&v->att, t * VIT_HID * 4));
    CK(cudaMalloc(&v->hid, t * VIT_FF * 4));
    CK(cudaMalloc(&v->pat, t * 1536 * 4));
    (void)pd;
    v->cap_tokens = tokens;
    return 0;
}

// Run the tower and the projector on one block of nch-channel pictures already laid out as patches in v->pat.
// Returns 0; stage_cap > 0 copies an intermediate to host (debug).
int run_tower(Vision* v, int cols, int rows, bool video, cudaStream_t s, int stage, float* host, size_t cap, float** emb_out) {
    const VisionW& W = v->w;
    const int np = cols * rows, nt = VIT_PREFIX + np;
    gemm(s, video ? W.vpatch : W.patch, v->pat, video ? 1536 : 768, v->x + (size_t)VIT_PREFIX * VIT_HID, VIT_HID, np);
    pos_add_k<<<nt, 256, 0, s>>>(v->x, W.cls, W.pos, rows, cols);
    if (stage == 0) { CK(cudaStreamSynchronize(s)); CK(cudaMemcpy(host, v->x, std::min(cap, (size_t)nt * VIT_HID) * 4, cudaMemcpyDeviceToHost)); return 0; }
    const size_t nh = (size_t)nt * VIT_HID;
    for (int l = 0; l < VIT_LAYERS; l++) {
        const VLayerW& L = W.L[l];
        ln_k<<<nt, 256, 0, s>>>(v->x, L.n1w, L.n1b, v->xn);
        gemm(s, L.qkv, v->xn, VIT_HID, v->qkv, 3 * VIT_HID, nt);
        bias_k<<<BLK((size_t)nt * 3 * VIT_HID), 256, 0, s>>>(v->qkv, L.qkv_b, 3 * VIT_HID, (size_t)nt * 3 * VIT_HID);
        vit_attn_k<<<dim3((nt + 63) / 64, VIT_HEADS), 128, 0, s>>>(v->qkv, v->att, nt);
        gemm(s, L.proj, v->att, VIT_HID, v->xn, VIT_HID, nt);
        bias_residual_k<<<BLK(nh), 256, 0, s>>>(v->x, v->xn, L.proj_b, VIT_HID, nh);
        ln_k<<<nt, 256, 0, s>>>(v->x, L.n2w, L.n2b, v->xn);
        gemm(s, L.fc1, v->xn, VIT_HID, v->hid, VIT_FF, nt);
        bias_gelu_k<<<BLK((size_t)nt * VIT_FF), 256, 0, s>>>(v->hid, L.fc1_b, VIT_FF, (size_t)nt * VIT_FF);
        gemm(s, L.fc2, v->hid, VIT_FF, v->xn, VIT_HID, nt);
        bias_residual_k<<<BLK(nh), 256, 0, s>>>(v->x, v->xn, L.fc2_b, VIT_HID, nh);
        if (l == 0 && stage == 1) { CK(cudaStreamSynchronize(s)); CK(cudaMemcpy(host, v->x, std::min(cap, nh) * 4, cudaMemcpyDeviceToHost)); return 0; }
    }
    if (stage == 2) { CK(cudaStreamSynchronize(s)); CK(cudaMemcpy(host, v->x, std::min(cap, nh) * 4, cudaMemcpyDeviceToHost)); return 0; }
    // projector: shuffled/normalized tokens in xn, the 20480-wide hidden in hid, embeddings (fp32) in att
    const int ntok = np / 4;
    shuffle_rms_k<<<ntok, 256, 0, s>>>(v->x, rows, cols, W.mlp_norm, v->xn);
    gemm(s, W.mlp1, v->xn, 5120, v->hid, 20480, ntok);
    relu2_k<<<BLK((size_t)ntok * 20480), 256, 0, s>>>(v->hid, (size_t)ntok * 20480);
    gemm(s, W.mlp2, v->hid, 20480, v->att, 4096, ntok);
    *emb_out = v->att;
    return 0;
}

// uploads the frames, resizes and lays out the patches. g selects the frame pair (video) or the image.
int prepare_patches(Vision* v, const uint8_t* const* rgb, const int* w, const int* h, int first, int nch_frames, int cols, int rows, cudaStream_t s) {
    const int ow = cols * VIT_PATCH, oh = rows * VIT_PATCH, pd = nch_frames * 768;
    for (int f = 0; f < nch_frames; f++) {
        const int sw = w[first + f], sh = h[first + f];
        const size_t sbytes = (size_t)sw * sh * 3, tbytes = (size_t)sh * ow * 3 * 4;
        if (sbytes > v->cap_src) { if (v->src) cudaFree(v->src); CK(cudaMalloc(&v->src, sbytes)); v->cap_src = sbytes; }
        if (tbytes > v->cap_tmp) { if (v->tmp) cudaFree(v->tmp); CK(cudaMalloc(&v->tmp, tbytes)); v->cap_tmp = tbytes; }
        CK(cudaMemcpyAsync(v->src, rgb[first + f], sbytes, cudaMemcpyHostToDevice, s));
        Axis ax = make_axis(sw, ow), ay = make_axis(sh, oh);
        int *dxs, *dxc, *dys, *dyc;
        float *dxw, *dyw;
        CK(cudaMalloc(&dxs, ow * 4)); CK(cudaMalloc(&dxc, ow * 4)); CK(cudaMalloc(&dxw, ax.w.size() * 4));
        CK(cudaMalloc(&dys, oh * 4)); CK(cudaMalloc(&dyc, oh * 4)); CK(cudaMalloc(&dyw, ay.w.size() * 4));
        CK(cudaMemcpyAsync(dxs, ax.start.data(), ow * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(dxc, ax.count.data(), ow * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(dxw, ax.w.data(), ax.w.size() * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(dys, ay.start.data(), oh * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(dyc, ay.count.data(), oh * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(dyw, ay.w.data(), ay.w.size() * 4, cudaMemcpyHostToDevice, s));
        resize_h_k<<<BLK((size_t)sh * ow * 3), 256, 0, s>>>(v->src, sw, sh, v->tmp, ow, dxs, dxc, dxw, ax.taps);
        resize_v_patch_k<<<BLK((size_t)oh * ow * 3), 256, 0, s>>>(v->tmp, ow, v->pat, pd, cols, oh, f * 3, dys, dyc, dyw, ay.taps);
        CK(cudaStreamSynchronize(s));   // the tables are freed below and src/tmp are reused by the next frame
        cudaFree(dxs); cudaFree(dxc); cudaFree(dxw); cudaFree(dys); cudaFree(dyc); cudaFree(dyw);
    }
    return 0;
}

}  // namespace

int vision_encode(Vision* v, const uint8_t* const* rgb, const int* w, const int* h, int nframes, int cols, int rows,
                  uint16_t** out, int* ntok, cudaStream_t s) {
    const bool video = nframes > 1;
    if (video && (nframes % 2)) return -2;
    if (cols < 2 || rows < 2 || (cols | rows) & 1) return -3;
    const int groups = video ? nframes / 2 : 1, np = cols * rows, per = np / 4;
    if (ensure(v, VIT_PREFIX + np, 1536)) return -1;
    uint16_t* dst = nullptr;
    CK(cudaMalloc(&dst, (size_t)groups * per * 4096 * 2));
    for (int g = 0; g < groups; g++) {
        if (prepare_patches(v, rgb, w, h, video ? 2 * g : 0, video ? 2 : 1, cols, rows, s)) return -1;
        float* emb = nullptr;
        if (run_tower(v, cols, rows, video, s, -1, nullptr, 0, &emb)) return -1;
        to_bf16_k<<<BLK((size_t)per * 4096), 256, 0, s>>>(emb, dst + (size_t)g * per * 4096, (size_t)per * 4096);
        CK(cudaStreamSynchronize(s));
    }
    *out = dst;
    *ntok = groups * per;
    return 0;
}

int vision_debug(Vision* v, const uint8_t* rgb, int w, int h, int stage, float* host, size_t cap, cudaStream_t s) {
    int cols, rows;
    vision_grid(w, h, &cols, &rows);
    if (ensure(v, VIT_PREFIX + (size_t)cols * rows, 1536)) return -1;
    const uint8_t* fr[1] = {rgb};
    if (prepare_patches(v, fr, &w, &h, 0, 1, cols, rows, s)) return -1;
    float* emb = nullptr;
    if (stage == 3) {
        if (run_tower(v, cols, rows, false, s, -1, nullptr, 0, &emb)) return -1;
        CK(cudaMemcpy(host, emb, std::min(cap, (size_t)cols * rows / 4 * 4096) * 4, cudaMemcpyDeviceToHost));
        return 0;
    }
    return run_tower(v, cols, rows, false, s, stage, host, cap, &emb);
}

// ---------------------------------------------------------------------------------------------------------
// Decoding (stb_image: JPEG, PNG, BMP, GIF; the alpha of RGBA is dropped, as PIL's convert("RGB") does)
// ---------------------------------------------------------------------------------------------------------

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "stb_image.h"

// 0 on success; rgb is malloc'd (free with bolt_free_host). Refuses pictures over 100 megapixels.
extern "C" int bolt_image_decode(const uint8_t* data, size_t n, uint8_t** rgb, int* w, int* h) {
    int x = 0, y = 0, c = 0;
    if (n > 0x7fffffff || !stbi_info_from_memory(data, (int)n, &x, &y, &c)) return -1;
    if ((double)x * y > 100e6 || x < 1 || y < 1) return -2;
    uint8_t* px = stbi_load_from_memory(data, (int)n, &x, &y, &c, 3);
    if (!px) return -3;
    *rgb = px; *w = x; *h = y;
    return 0;
}
extern "C" void bolt_free_host(void* p) { free(p); }
extern "C" int bolt_image_grid(int w, int h, int* cols, int* rows) { vision_grid(w, h, cols, rows); return (*cols) * (*rows) / 4; }
extern "C" int bolt_video_grid(int w, int h, int* cols, int* rows) { vision_grid_video(w, h, cols, rows); return (*cols) * (*rows) / 4; }

// ---------------------------------------------------------------------------------------------------------
// Video: ffprobe / ffmpeg do the container and the codec; frames come back already resized to the patch grid
// ---------------------------------------------------------------------------------------------------------

#include <unistd.h>
#include <sys/stat.h>

namespace {

struct TempFile {
    char path[64] = "/tmp/bolt-video-XXXXXX";
    int fd = -1;
    bool ok() const { return fd >= 0; }
    TempFile() { fd = mkstemp(path); }
    ~TempFile() { if (fd >= 0) { close(fd); unlink(path); } }
};

// stdout of a shell command (our own, over a temp path that holds no special characters), bounded to `cap` bytes
bool run_capture(const std::string& cmd, std::vector<uint8_t>& out, size_t cap) {
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return false;
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, p)) > 0) {
        if (out.size() + n > cap) { pclose(p); return false; }
        out.insert(out.end(), buf, buf + n);
    }
    return pclose(p) == 0;
}

}  // namespace

// Sample a video at `fps` (default 2; 1 above 720 rows), at most `max_frames` frames (default 256; 128 above 720 rows),
// spread over the whole clip if it is longer, resized to the video patch grid (aspect kept, about 1024 patches), padded
// with a copy of its last frame to an even count. rgb is one malloc'd block of nframes * h * w * 3 bytes (free with
// bolt_free_host), times one float a real frame in seconds. Returns 0 on success.
extern "C" int bolt_video_decode(const uint8_t* data, size_t n, double fps, int max_frames, uint8_t** rgb, int* nframes, int* w, int* h,
                                 float** times, int* ntimes, int* cols, int* rows) {
    TempFile tf;
    if (!tf.ok()) return -1;
    for (size_t off = 0; off < n;) {
        ssize_t k = write(tf.fd, data + off, n - off);
        if (k <= 0) return -1;
        off += (size_t)k;
    }
    std::vector<uint8_t> info;
    const std::string probe = std::string("ffprobe -v error -select_streams v:0 -show_entries stream=width,height:stream_side_data=rotation:format=duration "
                                          "-of default=noprint_wrappers=1 ") + tf.path + " 2>/dev/null";
    if (!run_capture(probe, info, 1 << 16)) return -2;
    std::string text(info.begin(), info.end());
    int sw = 0, sh = 0, rot = 0;
    double dur = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t e = text.find('\n', pos);
        if (e == std::string::npos) e = text.size();
        std::string line = text.substr(pos, e - pos);
        pos = e + 1;
        if (line.rfind("width=", 0) == 0) sw = atoi(line.c_str() + 6);
        else if (line.rfind("height=", 0) == 0) sh = atoi(line.c_str() + 7);
        else if (line.rfind("rotation=", 0) == 0) rot = atoi(line.c_str() + 9);
        else if (line.rfind("duration=", 0) == 0 && line[9] != 'N') dur = atof(line.c_str() + 9);
    }
    if (sw < 16 || sh < 16 || sw > 16384 || sh > 16384) return -3;
    if (rot == 90 || rot == -90 || rot == 270 || rot == -270) std::swap(sw, sh);
    if (!(dur > 0)) dur = 1.0;
    const bool large = sh > 720;
    if (!(fps > 0) && getenv("BOLT_VIDEO_FPS")) fps = atof(getenv("BOLT_VIDEO_FPS"));
    if (max_frames <= 0 && getenv("BOLT_VIDEO_FRAMES")) max_frames = atoi(getenv("BOLT_VIDEO_FRAMES"));
    if (!(fps > 0)) fps = large ? 1.0 : 2.0;
    if (max_frames <= 0) max_frames = large ? 128 : 256;
    int cc, rr;
    vision_grid_video(sw, sh, &cc, &rr);
    // frames to take: fps over the clip, spread thinner if that is more than max_frames
    double eff = fps;
    int want = (int)std::floor(dur * eff);
    if (want > max_frames) { eff = max_frames / dur; want = max_frames; }
    if (want < 1) want = 1;
    char vf[256];
    snprintf(vf, sizeof vf, "fps=%.6f,scale=%d:%d:flags=bicubic", eff, cc * VIT_PATCH, rr * VIT_PATCH);
    char cmd[512];
    snprintf(cmd, sizeof cmd, "timeout 180 ffmpeg -v error -nostdin -i %s -vf '%s' -frames:v %d -f rawvideo -pix_fmt rgb24 - 2>/dev/null", tf.path, vf, want);
    const size_t frame_bytes = (size_t)cc * VIT_PATCH * rr * VIT_PATCH * 3;
    std::vector<uint8_t> raw;
    if (!run_capture(cmd, raw, frame_bytes * (size_t)(max_frames + 2))) return -4;
    int got = (int)(raw.size() / frame_bytes);
    if (got < 1) {
        // a clip shorter than one sampling step (the fps filter then emits nothing): take its first frame
        snprintf(cmd, sizeof cmd, "timeout 60 ffmpeg -v error -nostdin -i %s -vf 'scale=%d:%d:flags=bicubic' -frames:v 1 -f rawvideo -pix_fmt rgb24 - 2>/dev/null",
                 tf.path, cc * VIT_PATCH, rr * VIT_PATCH);
        raw.clear();
        if (!run_capture(cmd, raw, frame_bytes * 2)) return -4;
        got = (int)(raw.size() / frame_bytes);
    }
    if (got < 1) return -5;
    const int padded = got + (got & 1);
    uint8_t* buf = (uint8_t*)malloc(frame_bytes * padded);
    float* tt = (float*)malloc(sizeof(float) * got);
    if (!buf || !tt) { free(buf); free(tt); return -6; }
    memcpy(buf, raw.data(), frame_bytes * got);
    if (padded > got) memcpy(buf + frame_bytes * got, buf + frame_bytes * (got - 1), frame_bytes);
    for (int i = 0; i < got; i++) tt[i] = (float)(i / eff);
    *rgb = buf; *nframes = padded; *w = cc * VIT_PATCH; *h = rr * VIT_PATCH;
    *times = tt; *ntimes = got; *cols = cc; *rows = rr;
    return 0;
}
