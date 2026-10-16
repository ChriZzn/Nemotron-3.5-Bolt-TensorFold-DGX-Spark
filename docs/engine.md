# The Bolt backend and engine

`./start.sh` builds TensorFold at `c7137bb3`, applies `patches/`, copies `engine/bolt_native.zig` into
`zig/src/families/nemotron/` and compiles `engine/*.cu` into `/opt/tensorfold/lib/libbolt.so`. `tensorfold-native`
opens the checkpoint through the `nemotron_h_omni` entry that `patches/0002` registers, `dlopen`s libbolt and serves.

## Why a separate library

TensorFold's native Nemotron kernels serve Lightning: hidden 2688, 128 experts, top-6, MLX 4-bit or ModelOpt NVFP4
weights. Bolt is a different shape and packing.

| | Lightning | Bolt 24GB |
| --- | --- | --- |
| Hidden / layers | 2688 | 4096, 88 (40 Mamba, 40 MoE, 8 attention) |
| Routed experts | 128, one width, top-6 | 240 to 496 a layer, widths 1536 to 2560, top-9 |
| MoE | shared expert | latent size 1024 around the routed experts, shared width 5376, sigmoid router, routed scale 5.0, ReLU² |
| Quantization | MLX affine 4-bit / NVFP4 | IQ2_XXS experts (ggml), NVFP4 Mamba projections, Q8_0 attention, latent projections and `lm_head` |
| Draft head | MTP in the checkpoint | none; the separate LiLiCorr checkpoint |

## The lane backend (`engine/bolt_native.zig`)

TensorFold's round loop (`core/lanes`) asks a backend for a handful of things. Bolt answers them over the C API of
libbolt. Every live stream owns one *sequence* of libbolt (a `Slot`, found from the stream's pointer); the lane host
asks for a stream only after a sequence is free, otherwise the backend returns `LaneMemory` and the job waits at the
head of the queue.

| Backend entry | What it does |
| --- | --- |
| `prefill_step` | takes a sequence, resumes it from a kept prompt state when the cache has one, then runs the prompt in pieces (2048 rows alone, 512 while other streams decode, cut at the cache's marks) so a cancel lands quickly and the other streams' rounds run between pieces |
| `first` / `queue` | the keyed draw of one row at the given position |
| `verify` | **one forward over every stream's window** (up to 16 windows, 128 rows): `pending`, the held drafts and any host drafts, each row drawn with its stream's `Sampling` at its own position |
| `keep` | roll each stream back to its accepted prefix (`bolt_keep_seq`; a repeat is a no-op) |
| `draft` | cuts each stream back to its kept rows first (a shared round drafts before it keeps), then LiLiCorr proposes seven tokens for every stream in **one pass over the drafter's weights**; the first `depth` are held for the next window |
| `release` | gives the sequence back and unmaps the attention memory it grew into |

Draws use `lanes.sampling.choose` (Gumbel over the top candidates, `top_k` / `top_p` / `min_p`, keyed by seed and
position), so a speculative run draws what a plain run would. libbolt returns the top 20 or 64 candidates a row
(1 when greedy); `top_k` above 64 is capped. Facts reported to the round loop: window 8, drafts 7, window costs and
shared-round costs measured at load (rows 2 to 128 over all sequences), the drafter's cost, `hidden_rows` and
`draft_streams` on when there is more than one sequence, acceptance prior 0.80 falling to 0.68. Logprobs, grammars and
trees are not served by this backend; the round loop is told so.

`--parallel N` sets how many streams run together; `start.sh` also hands N to libbolt as `BOLT_SEQS`, which sizes the
recurrent state. Extra jobs beyond N queue.

## libbolt (`engine/`)

- `bolt_host.cu`: loading, the block forward, the C API, sequences, snapshots, calibration. A *block* is a list of
  segments, each a run of rows of one sequence (a prompt chunk is one segment; a shared decode round has one verify
  window per stream). `Ctl` carries what changes between replays: tokens, and per segment its first row, cache length,
  sequence and state slots. A block of up to 1024 rows (128 for a shared round) is one launch sequence, captured into
  a CUDA graph the second time a shape repeats; shared rounds are padded to a few row counts (multiples of 8, then 16)
  so a handful of graphs serve them. The shared expert and the latent projection run on side streams.
- `bolt_mma.cu`: tensor-core GEMMs for NVFP4, Q8_0, bf16 and IQ2_XXS from 16-row tiles (layouts in the file header);
  `mma_big_k` stages activations in shared memory for blocks over 32 rows. Prompt attention and verify-window
  attention on bf16 mma with online softmax (`attn_mma_part_k`; for windows one CTA is one stream's window).
- `bolt_kernels.cu`: norms, the Mamba conv and scan (a row-parallel conv and a thread-pair scan for long blocks, a
  warp-per-channel scan for windows), the router, the routed experts on int8 tensor cores, candidate selection.
- `bolt_vision.cu`: the image and video tower and its decoders (below).
- `bolt_draft.cu`: the LiLiCorr drafter: five DFlash layers (grouped conv, sliding-window sink attention, NVFP4 MLP)
  over the six captured residuals, then the candidate lattice and its greedy walk. A call drafts for up to 16 streams;
  the kept rows of every stream are ingested in one pass too.

### Images and video (`bolt_vision.cu`)

With `BOLT_VISION=1` the loader also reads `vision_model.*` and `mlp1.*` (1.68 GB bf16) and the C API gains
`bolt_image_decode` (stb_image), `bolt_video_decode` (ffprobe and ffmpeg through a temp file; frames come back already
resized), `bolt_embed_new/free` and `bolt_prefill_seq_media`. The pieces, checked against the checkpoint's
`modeling_radio.py` and `modeling_nemotron_h_omni.py` (the final embeddings match a PyTorch reference to 4.5e-3 relative L2):

- **Resize**: the processor's grid rule (1,024 to 13,312 patches of 16 px, aspect kept, Python's banker's rounding),
  then antialiased bicubic with torch's weights built on the host, two passes on the GPU, normalize and patchify fused.
- **Tower**: patch embedding (a GEMM), the cropped position embedding (the 128×128 table resampled to the larger side
  and cropped), ten prefix tokens, 32 pre-norm blocks of LayerNorm, QKV, bidirectional attention
  (`vit_attn_k`, mma flash attention at head dim 80) and a GELU MLP, on the same `mma_bf16` GEMMs as the language model.
- **Projector**: drop the prefix, 2×2 pixel shuffle, RMSNorm, `mlp1.1` (5120 to 20480), ReLU², `mlp1.3` (20480 to 4096).
  A video group of two frames uses `video_embedder` (1280×1536) and the same tower.
- **Splice**: the prompt holds `<image>` (id 18) placeholders, salted to ids 100..999 by the server so the prompt cache
  keys differ per picture. The backend hands `bolt_prefill_seq_media` runs of `{pos, count, embed, row}`; `k_embed`
  reads a negative token as a row of that embedding buffer. The buffers are freed after the last prompt piece.
  The language model treats the rows as ordinary positions (causal), as in the reference.

### State of a sequence

- **SSM**: two copies of the 4 MiB-a-layer fp32 state. A block reads one and writes the other, so a verify window only
  writes its final state; `keep(n)` for a partial accept replays the first `n` rows from the pre-window copy
  (`k_scan_replay`, using the window's saved scan inputs). Rolling back to zero rows is a pointer flip.
- **Conv**: a ring of nine slots; a window writes the state after each row, `keep(n)` picks slot `n`.
- **Attention K/V**: bf16, 8 KB a token; with `BOLT_KV=fp8` (`KV_DTYPE=fp8`) e4m3 bytes plus an fp32 scale per token and
  head (amax / 448, set when the row is stored), 4.1 KB a token. The kernels dequantize while staging tiles, so the
  cache costs no speed. Upstream TensorFold has no Nemotron cache quantization (its int8/int4 `--kv-dtype` is Flash
  Next's) and the checkpoint ships no KV scales, hence the dynamic ones. Reserved at the full window for every sequence but mapped 4096 tokens at a
  time (CUDA virtual memory, 2 MiB granules) as the sequence grows, and unmapped down to one granule when it is
  released, so 16 streams can each have the 262,144-token window while memory follows use. If the driver cannot do
  that, or `BOLT_KV_EAGER=1`, everything is allocated up front. `BOLT_KV_RESERVE_GIB` (default 3) is the free memory
  a growing stream leaves alone; past it the forward fails instead of freezing the machine.
- **Drafter**: a 2048-entry K/V ring per layer and the anchor row; prompt rows older than 1,100 never reach it.

Memory a sequence holds from load: 0.4 GiB (two SSM copies over 40 layers, the conv ring) plus the drafter's 21 MB.

### Why a round costs what it does

A shared round is bound by DRAM, not arithmetic. At 16 streams with one row each it reads, per round: every stream's
SSM state in and out (40 layers × 4 MiB × 2 × 16 = 5.4 GB), about 130 distinct experts a layer (6.9 GB), the dense
weights (2.8 GB). That is the 85 ms the profile shows (`BOLT_PROF=1`), and about 80 % of
the GB10's practical bandwidth. More rows per stream (draft depth) cost the extra experts they touch but no more state,
which is why TensorFold's cost rule asks for longer windows when streams are few. Measured shared rounds, ms by total
rows over 16 sequences: 16 → 61, 32 → 96, 64 → 141, 128 → 197.

## Checks

The engine is checked against references that live in the development harness, not in this repository: kernel,
tensor-core GEMM and attention unit tests (attention includes several windows in one launch), and engine-level gates:
perplexity 5.284 on the first 1,499 tokens of this README (5.288 through 8-row windows), a window cut to `n` rows equal
to a window of `n` rows bit for bit, a resumed snapshot equal to a whole prefill, the drafter accepting 4.3 of 7 tokens
on the model's own text, sixteen sequences decoded in one block each equal to its solo run, and a sequence's logits
bit-identical whatever its neighbours hold. `BOLT_PROF=1` times each stage of a round.

Kernel choice follows the row count, so logits differ in the second decimal between a stream decoded alone and the
same stream in a block of 64 rows (|logit| around 50); neighbours never matter.

## Checking a snapshot

`scripts/ckptcheck.py` checks a snapshot against `contract/bolt.json`. A change that serves different shapes fails that
check on purpose.
