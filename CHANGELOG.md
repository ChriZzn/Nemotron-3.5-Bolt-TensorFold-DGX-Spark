# Changelog

Newest first. This recipe targets one DGX Spark. The image is `tensorfold-nemotron-bolt:zig-46a08c1`, built by
`scripts/prepare.sh` from TensorFold `46a08c18` (1.0.12) plus `patches/`.

## Unreleased

The Bolt backend serves the checkpoint. Measured on one GB10: decode 4.3 tok/s to 36 (no drafts) and 70 to 110 (with
LiLiCorr, greedy chat or code), 55 to 60 sampled; prefill 4.5 tok/s to 1,300 to 1,470 (from 1k to 50k tokens);
sixteen streams at once, 167 tok/s aggregate sampled and 172 greedy.

### Concurrent replies equal solo ones

A stream's replies used to depend on the batch it ran in: `bench_concurrent.py --alone` found every prose ("chat") reply
at four streams different from its solo run (0 of 12 equal), and code replies equal. The cause was not the GEMM kernels
chosen by row count, as first written up; it was three batch-dependent summation orders, all fixed now:

- **Attention key chunks.** A verify window merged partial softmaxes over key chunks sized by the number of streams in the
  block (about 32 splits for one stream, 8 for four) and by the window's width. The chunk size is now `win_chunk(base)`
  (128 keys, doubled until 32 chunks reach the sequence's end and window), a function of the sequence's own length.
- **GEMM order.** Blocks of more than 32 rows used `mma_big_k`, which sums K in one sequence; smaller blocks used the
  K-split kernel's four interleaved partials. Decode and verify blocks now always use the K-split kernel
  (`mma_decode_rows`, 64 tokens a CTA); the prompt pass keeps `mma_big_k`.
- **The plain step.** The non-speculative one-token step used the CUDA-core attention kernel; it now runs as a one-row
  window (then kept), so drafted and plain decoding share one arithmetic.

Tests: perplexity of passages scored alone and in blocks of 2, 3, 4, 8 and 16 streams with 8-row windows (up to 128 rows
a block; also 1- and 5-row windows at four streams) is identical to five decimals; sixteen sequences match their solo
runs for 768 of 768 tokens, plain steps and 4-row windows alike; `tools/bench_concurrent.py --alone --serial` on the production image
(`--parallel 8`, greedy, 256 tokens, 3 repetitions):

| prompt | streams | aggregate | each stream | concurrent equals solo | drafted equals `"draft": false` |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 1 | 115.1 tok/s | 115.1 tok/s | 3 of 3 | 1 of 1 |
| code | 4 | 207.9 tok/s | 54.8 tok/s | 12 of 12 | 1 of 1 |
| code | 8 | 252.6 tok/s | 35.2 tok/s | 24 of 24 | 1 of 1 |
| chat | 1 | 60.7 tok/s | 60.7 tok/s | 3 of 3 | 1 of 1 |
| chat | 4 | 139.1 tok/s | 35.6 tok/s | 12 of 12 | 1 of 1 |
| chat | 8 | 174.7 tok/s | 23.6 tok/s | 24 of 24 | 1 of 1 |

Perplexity on the 1,499-token passage is unchanged (5.286 one token at a time). Cost: the K-split kernel decodes the
weights again for each 64-token group, so the widest shared rounds are slower, 16 streams of 8-row windows (128 rows)
220.3 ms against 176.9 ms (about 25 %); a round of 64 rows is 130.4 ms against 135.5 ms for 8 streams of 8-row windows
and 141.7 ms against 133.2 ms for 16 streams of 4-row windows. A 16-sequence round of one row each is unchanged
(90.3 ms).

### TensorFold 1.0.12, images and video

- **Rebased onto TensorFold 1.0.12.** Upstream now ships a RADIO tower and image input, so the recipe uses them: the
  tower, the image and video server code and the Anthropic and Responses media routes are upstream's, the engine's own
  ViT, stb_image decoder and ffmpeg subprocess are gone. Bolt is the family `nemotron_h_puzzle`, built with
  `zig build bolt`; libbolt takes image rows by device pointer (`bolt_prefill_seq_rows`) and checks an ABI version.
- **Video** through the same tower: tubelets of two frames through `video_embedder`, the processor's frame labels,
  `<video>` turned into one token before encoding, `--vision-max-videos` (patch 0003). Decoding is upstream's libtfvideo
  over FFmpeg 9.0.2, built in the image.
- **A request holds up to 64 images and 4 videos** (`VISION_MAX_IMAGES`, `VISION_MAX_VIDEOS`); each prompt piece encodes
  only the images it crosses. Tested: 50 images and 4 videos in one chat, and 50 images of 2560×1920 (165,600 tokens).
- Formats follow upstream's: PNG and JPEG (libjpeg-turbo 3.1.4.1, in the image), HTTPS URLs behind `VISION_URLS=1`;
  GIF, BMP and CMYK JPEG are no longer read. Text speed and outputs are unchanged (perplexity 5.286; greedy replies
  identical to the previous base).
- `--parallel` reaches libbolt through the backend (no `BOLT_SEQS` in `start.sh`).

### Sixteen streams, a faster prompt pass

- **Streams.** libbolt keeps `BOLT_SEQS` (= `--parallel`, default 16) independent sequences. A block is a list of
  segments; a shared decode round carries every stream's verify window in one forward, so the weights are read once
  for all of them. Recurrent state is two SSM copies and a conv ring per sequence; a window reads one copy and writes
  the other, and a partial accept replays from the untouched one. The drafter, its ingest and its proposals are
  batched across streams too (25 ms for sixteen streams against 9 for one). `PARALLEL=16` is the default.
- **Memory follows use.** The attention rows of every stream have the full 262,144-token window in the address
  space, but are mapped 4,096 tokens at a time through CUDA virtual memory and unmapped when the stream ends; 16
  streams cost 0.4 GiB each from load, not 2 GiB each. `libbolt.so` now links `-lcuda`.
- **Prompt pass 380 -> 1,450 tok/s.** Tensor-core flash attention (prompt and verify windows), routed experts on int8
  tensor cores (tokens staged with overlapping loads, several weight tiles a warp), a shared-memory-staged GEMM for
  blocks over 32 rows, a row-parallel conv and a thread-pair scan, 1,024 rows a chunk, and the block scales of the
  NVFP4, Q8_0 and IQ2 GEMMs folded into the decoded weights (NVFP4's are exact in fp16) so the mma runs through K
  with no per-block epilogue. Prefill used to fall to 190 tok/s at 50k tokens and now holds 1,290.
- **Backend.** `prefill_step` (pieces of 2048 rows, 512 while other streams decode), `LaneMemory` when every sequence is
  busy, snapshots per stream, `draft_streams` and shared-round costs measured at load.
- **Checks.** The unit tests cover attention including several windows in one launch; sixteen sequences decoded in one
  block match their solo runs, and a leak test compared a stream's logits across neighbours (it missed the batch-dependent
  summation orders fixed under "Concurrent replies equal solo ones"); the rollback check now compares
  against a window of exactly the kept rows.

- **fp8 KV cache.** `KV_DTYPE=fp8` stores K and V as e4m3 with a scale per token and head: 4,224 bytes a token instead
  of 8,192, in the cache, the lazily mapped pool and prompt-cache snapshots. No speed change (prefill 1,420 tok/s at
  16k, a 30k-context round 68 ms either way); perplexity 2.760 against 2.761 over 4,000 tokens, per-token log-prob
  drift 0.026 against a 0.018 noise floor, the 195k needle and tool calls pass. bf16 stays the default.

- **Pinned KV.** `KV_PIN=1` (`BOLT_KV_PIN`) maps every stream's whole window at load and never unmaps it, for a guaranteed
  `PARALLEL` x `CONTEXT`; a start that cannot fit it fails with a message instead of failing a request later.

### Tried and did not help

- A grouped dp4a expert kernel (pairs grouped by expert, CUDA cores): no faster than one pair a thread.
- Work items of 32 pairs for the tensor-core expert kernel (fewer, wider items for big chunks): 10 % slower than 16.
- Prompt chunks of 2048 and 4096 rows: 1,000 and 750 tok/s against 1,200 at 1,024 (512 and 768 are within 3 %).
- Activations quantized per 256 (llama.cpp's Q8_K) so the expert kernel could scale once a block: no faster, and
  perplexity moved 0.2 %.
- 4 CTAs per SM for the expert kernel (64 registers): the spills cost more than the occupancy bought.
- Eight-token groups in the old large-batch GEMM: slower; the real cost was re-reading fp32 activations through L2
  (94 % L2 throughput), which staging them in shared memory fixed.
- Splitting the scan over more warps per channel (2 or 4 channels a warp): 2 % at most; the thread-pair scan with
  shared-memory staging took conv+scan from 3.6 to 1.2 ms a layer.
- fp16 or bf16 SSM state would cut the sixteen-stream round by about 12 ms, but the recurrence accumulates the
  rounding and NVIDIA recommends fp32 for this model: not done.

### Earlier, in this release

- `engine/bolt_native.zig`: the lane backend (prefill, verify windows, rollback, held drafts, keyed draws) in place of
  the one-row stub. `--drafts` is the default; `DRAFTS=0` serves one row at a time.
- `libbolt.so` rewritten: packed weights read once by tensor-core GEMMs, one CUDA graph a block, grouped expert
  kernel, per-request reset, KV for the whole window (it stopped at 4,096 before).
- Fixes behind the 0/7 acceptance: the shared expert took the norm of the updated residual, Mamba's `dt` floor was
  0.001 (the model's is 0), and the drafter's attention convolution ran before `o_proj` instead of after.
- Prompt cache: libbolt copies and restores its state (`bolt_snap_*`), the backend resumes prompts from kept prefixes and
  cuts the prompt pass at the cache's marks. `PROMPT_CACHE_GIB=8` by default.
- `scripts/prepare.sh` downloads the LiLiCorr drafter; `start.sh` serves by default (`SERVE=0` stops after the check).

### Before that

The checkpoint contract and the load gate. No Spark boot, and no decode or prefill numbers.

- Pin `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` at `a326822af111290f2af5ac59ef4b5040347dabc4`.
  `contract/bolt.json` records the 88-layer schedule, the three quant groups and the C-RADIO tower.
  `scripts/ckptcheck.py` checks a snapshot against it.
- `patches/0001-refuse-bolt-checkpoint.patch` on TensorFold's Nemotron `parse`: a Bolt omni or puzzle config
  returns `UnsupportedCheckpoint` and names the checkpoint. Lightning's `nemotron_h` path is unchanged.
- `./start.sh` prepares and checks. `SERVE=1` launches `tensorfold-native`. On this pin the process exits at load.
