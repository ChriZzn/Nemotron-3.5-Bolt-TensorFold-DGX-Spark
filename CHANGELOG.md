# Changelog

Newest first. This recipe targets one DGX Spark. The image is `tensorfold-nemotron-bolt:zig-c7137bb`, built by
`scripts/prepare.sh` from TensorFold `c7137bb3` plus `patches/`.

## Unreleased

The Bolt backend serves the checkpoint. Measured on one GB10: decode 4.3 tok/s to 36 (no drafts) and 70 to 110 (with
LiLiCorr, greedy chat or code), 55 to 60 sampled; prefill 4.5 tok/s to 1,300 to 1,470 (from 1k to 50k tokens);
sixteen streams at once, 167 tok/s aggregate sampled and 172 greedy.

### Images and video

- **C-RADIO v4-H in libbolt** (`engine/bolt_vision.cu`): GPU resize (antialiased bicubic, as the checkpoint's
  processor), 32-layer ViT on the tensor-core GEMMs with a bidirectional flash-attention kernel for head dim 80,
  cropped position embedding with 10 prefix tokens, 2×2 pixel shuffle, RMSNorm and the ReLU² MLP projector. Against the
  PyTorch reference the final embeddings have relative L2 error 4.5e-3 (cos 0.99999). Video runs the same tower on
  two-frame tubelets; ffmpeg decodes and resizes. JPEG/PNG/BMP/GIF decode with the public-domain `stb_image.h`.
- **`patches/0003-media.patch`** and the backend: `image_url` and `video_url` parts as base64 data URLs on
  `/v1/chat/completions`, prompt-cache keys salted by pixel hash, embeddings spliced in during the prompt pass in
  pieces. `VISION=1` is now the default (`VISION=0` for text only); the image installs ffmpeg.
- Up to 64 pictures a request (50 images and 4 videos tested in one chat); embeddings are freed as the prompt pass moves past them.
- Text-only speed and outputs are unchanged; the tower loads only when `BOLT_VISION=1`.

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
  block match their solo runs, and a stream's logits do not depend on its neighbours; the rollback check now compares
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
