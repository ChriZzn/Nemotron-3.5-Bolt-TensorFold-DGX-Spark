// Shared declarations for the Bolt engine: packed weight views, the per-launch control block and the
// kernel launchers in bolt_kernels.cu. Everything here is C++ and private to libbolt.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

constexpr int TMAX = 1024;     // most rows one block forward carries (prompt chunks)
constexpr int BATCH_MAX = 128; // most rows of a block that carries several sequences' verify windows
constexpr int LOGIT_ROWS = BATCH_MAX; // rows whose logits are kept (verify windows and the last prompt row)
constexpr int SEG_MAX = 16;    // most sequences in one block
constexpr int VERIFY_MAX = 8;  // most rows one sequence's verify window carries
constexpr int CONV_SLOTS = VERIFY_MAX + 1;  // conv state slots a sequence keeps (one per window row, plus the start)

// Keys one CTA of a verify window's attention covers: 128, doubled until 32 CTAs reach the end of the sequence and its
// window. It depends on the sequence's own length alone, so a stream's rounding is the same alone, among other streams,
// or in a wider window; the split of the keys is the only part of the forward whose order depended on the batch.
__host__ __device__ inline int win_chunk(int base0) {
    int k = 128;
    while (k * 32 < base0 + VERIFY_MAX) k *= 2;
    return k;
}
constexpr int HID = 4096;
constexpr int VOCAB = 131072;
constexpr int TOPN = 9;        // routed experts a token uses
constexpr int CAND = 64;       // sampling candidates kept per row

// Q8_0, planar: q [N][K] int8, d [N][K/32] half.
struct Q8W { const int8_t* q; const void* d; int n, k; };
// ModelOpt NVFP4: q [N][K/2] (low nibble first), sc [N][K/16] e4m3, one fp32 tensor scale.
struct NV4W { const uint8_t* q; const uint8_t* sc; float s2; int n, k; };
// ggml IQ2_XXS, planar: qs [N][K/256][64] bytes, d [N][K/256] half.
struct IQ2W { const uint8_t* qs; const void* d; int n, k; };
// Row-major bf16, used by the router and the drafter.
struct BF16W { const uint16_t* w; int n, k; };

// All experts of one layer share a slab: expert e starts at qs + e * qs_stride, d + e * d_stride (elements).
struct IQ2Slab { const uint8_t* qs; const void* d; size_t qs_stride, d_stride; };

// Tile-major weights for the tensor-core GEMMs: rows come in tiles of 16, laid out so that each lane of a warp
// reads the A fragments of m16n8k16 with one 16-byte load. n and k are the logical sizes; the tile count is
// ceil(n / 16) and missing rows are zero. See bolt_mma.cu for the exact byte layouts.
struct TNV4 { const uint4* q; const uint2* sc; float s2; int n, k; };   // NVFP4
struct TQ8  { const uint4* q; const uint32_t* d; int n, k; };           // Q8_0
struct TBF  { const uint4* w; int n, k; };                              // bf16
struct TIQ2 { const uint4* q; const uint16_t* d; int n, k; };           // IQ2_XXS

// Device-visible values that change from launch to launch, so a captured graph can be replayed.
// A block is a list of segments: consecutive rows of one sequence (a prompt chunk is one segment, a shared verify
// round has one window per sequence). Row r belongs to the segment with seg_row0[i] <= r < seg_row0[i + 1].
struct Ctl {
    int tokens[TMAX];
    int nrows;                  // rows in this block
    int nseg;
    int seg_row0[SEG_MAX + 1];  // first row of each segment; [nseg] == nrows
    int seg_base[SEG_MAX];      // the sequence's cache length before the segment
    int seg_seq[SEG_MAX];       // which sequence's state the segment uses
    int seg_ssm[SEG_MAX];       // its SSM slot to read (0 or 1; the other one is written)
    int seg_conv[SEG_MAX];      // its conv ring slot to read
};

struct Q8Multi {
    Q8W w[3];
    float* y[3];
    int ldy[3];
    int rows[3];  // rows of each matrix
};

// GEMV launchers. x is [T][ldx] fp32, y is [T][ldy] fp32. epi: 0 store, 1 store relu(v)^2.
void gemv_q8(cudaStream_t s, const Q8W& w, const float* x, int ldx, float* y, int ldy, int T, int epi = 0);
void gemv_q8_multi(cudaStream_t s, const Q8Multi& m, const float* x, int ldx, int T);
void gemv_nv4(cudaStream_t s, const NV4W& w, const float* x, int ldx, float* y, int ldy, int T);
void gemv_iq2(cudaStream_t s, const IQ2W& w, const float* x, int ldx, float* y, int ldy, int T, int epi = 0);
void gemv_bf16(cudaStream_t s, const BF16W& w, const float* x, int ldx, float* y, int ldy, int T);

// Tensor-core GEMMs for T rows (1..TMAX): x is [T][ldx] fp32, y is [T][ldy] fp32. epi 1: relu(v)^2.
void mma_decode_rows(bool on);   // the GEMMs of the blocks enqueued next are decode or verify rows (summation order independent of the row count)
void mma_nv4(cudaStream_t s, const TNV4& w, const float* x, int ldx, float* y, int ldy, int T);
void mma_q8(cudaStream_t s, const TQ8& w, const float* x, int ldx, float* y, int ldy, int T);
void mma_bf16(cudaStream_t s, const TBF& w, const float* x, int ldx, float* y, int ldy, int T);
void mma_iq2(cudaStream_t s, const TIQ2& w, const float* x, int ldx, float* y, int ldy, int T, int epi = 0);
struct TQ8Multi { TQ8 w[3]; float* y[3]; int ldy[3]; };
void mma_q8_multi(cudaStream_t s, const TQ8Multi& m, const float* x, int ldx, int T);

// Load-time repacking into the tile layouts. Sources are the checkpoint's own layouts; the destination
// buffers must hold ceil(n/16) tiles. Sizes in bytes: see tile_bytes_*.
size_t tile_bytes_nv4(int n, int k);
size_t tile_scale_bytes_nv4(int n, int k);
size_t tile_bytes_q8(int n, int k);
size_t tile_scale_bytes_q8(int n, int k);
size_t tile_bytes_bf16(int n, int k);
size_t tile_bytes_iq2(int n, int k);
size_t tile_scale_bytes_iq2(int n, int k);
void tile_nv4(cudaStream_t s, const uint8_t* q, const uint8_t* sc, void* qt, void* sct, int n, int k);
void tile_q8(cudaStream_t s, const uint8_t* raw, void* qt, void* dt, int n, int k);
void tile_bf16(cudaStream_t s, const uint16_t* w, void* wt, int n, int k);
void tile_iq2(cudaStream_t s, const uint8_t* raw, void* qt, void* dt, int n, int k);

// Expert kernels: one launch covers every (token, expert) pair of the block.
// Work items of the routed experts: pairs grouped by expert, at most 16 pairs per item (see bolt_kernels.cu).
struct MoeWork { int *order, *first, *count, *expert, *n_work; };
void k_moe_group(cudaStream_t s, const int* ridx, int pairs, int* order, int* work_first, int* work_n, int* work_exp, int* n_work);
void k_quant_rows(cudaStream_t s, const float* x, int ldx, int8_t* q, int ldq, float* sc, int ldsc, int rows, int k);
// One launch covers every item. xq/xs are the quantized inputs: up reads one row per token, down one per pair.
void moe_up(cudaStream_t s, const IQ2Slab& up, const MoeWork& w, const int8_t* xq, const float* xs, float* act, int width, int pairs);
void moe_down(cudaStream_t s, const IQ2Slab& down, const MoeWork& w, const int8_t* xq, const float* xs, float* out, int width, int pairs);

// Repack and format helpers used at load time.
void repack_q8(cudaStream_t s, const uint8_t* src, int8_t* q, void* d, size_t rows, int k);
void repack_iq2(cudaStream_t s, const uint8_t* src, uint8_t* qs, void* d, size_t rows, int k);

// Block building blocks.
// Rows whose token is negative come from row t of `extra` (bf16 [T][HID]): media embeddings.
void k_embed(cudaStream_t s, const uint16_t* table, const uint16_t* extra, const Ctl* ctl, float* x, int T);
void k_add_rms(cudaStream_t s, float* x, const float* r1, const float* r2, const float* w, float* xn, float* cap,
               int cap_stride, float eps, int T);
// Conv state: pool + seq * CONV_SLOTS * CONV_ELEMS + slot * CONV_ELEMS. A record block writes the state after every row
// into the next slots of the ring (so a window can be cut anywhere); otherwise only the state after the segment, one on.
void k_conv(cudaStream_t s, const Ctl* ctl, float* conv_pool, const float* w, const float* b,
            const float* proj, int ldp, float* xbc, int T, int record, float* hist_x, float* hist_dt, int nseg);
// SSM state: pool + seq * 2 * SSM_ELEMS + slot * SSM_ELEMS; the segment reads its slot and writes the other one.
void k_scan(cudaStream_t s, const Ctl* ctl, float* ssm_pool, const float* xbc,
            const float* proj, int ldp, const float* A_log, const float* dt_bias, const float* D, float* y, int T, int record, int nseg);
void k_gate_rms(cudaStream_t s, const float* y, const float* proj, int ldp, const float* w, float* out, float eps, int T);
void k_route(cudaStream_t s, const float* logits, int ldl, const float* bias, int* ridx, float* rw, int E, float scale, int T);
void k_combine(cudaStream_t s, const float* o, const float* rw, float* out, int T);
// The attention cache of one layer: K and V rows [seq][ctx][2 heads][128]. bf16, or (fp8) e4m3 bytes with one fp32 scale
// per (token, head) in ks/vs [seq][ctx][2]; a value is the byte times its scale. ctx is the per-sequence capacity.
struct KvView { void* k; void* v; float* ks; float* vs; int ctx; int fp8; };
void k_kv_store(cudaStream_t s, const Ctl* ctl, const float* k, const float* v, const KvView& kv, int T);
// multi: rows of several sequences (or a verify window): every row runs on its own. Otherwise one prompt segment.
void k_attn(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out,
            int T, int splits, int multi, int nseg_unused = 0);
// Row-wise top candidates of the logits: vals/ids are [T][CAND] sorted descending (ties to the lower id).
void k_candidates(cudaStream_t s, const float* logits, int T, int topk, float* vals, int* ids, float* scratch_v, int* scratch_i);

// Draft-side kernels reuse the same primitives; see bolt_draft.cu.

// Verify windows keep only the final SSM state; the scan inputs of each row are saved so a shorter accepted
// prefix can be replayed from the pre-window state (one launch over all Mamba layers).
struct ReplayLayer { float *ssm, *A_log, *dt_bias, *hist_x, *hist_dt; };
// Rebuild the SSM state after the first n rows of the segment whose rows start at row0 of the last block: read slot
// `slot` of sequence `seq`, write the other slot.
void k_scan_replay(cudaStream_t s, const ReplayLayer* layers, int nlayers, int seq, int slot, int row0, int n);
// Prompt attention on tensor cores (bolt_mma.cu); part must hold attn_mma_part_floats(T, splits) floats.
void k_attn_mma(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out, int T, int splits);
size_t attn_mma_part_floats(int T, int splits);
// The same for verify windows: one CTA per sequence's window (up to 8 rows), nseg a launch bound, part holds nseg * 2 * splits * 16640 floats.
void k_attn_mma_windows(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out,
                        int T, int nseg, int splits);

// Sizes of one sequence's recurrent state (floats per Mamba layer).
constexpr int SSM_ELEMS = 128 * 64 * 128;
constexpr int CONV_ELEMS = 10240 * 3;
