# Credits

- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors). This recipe pins
  `46a08c18608153bfc566703a88acc883bc6140f4` (1.0.12): its Zig server, RADIO image tower, video decoder (`libtfvideo`)
  and [Nemotron recipes](https://github.com/ashhart/TensorFold/blob/46a08c18608153bfc566703a88acc883bc6140f4/docs/recipes/nemotron-3.5-super-vl.md).
- [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) by
  NVIDIA, revision `a326822af111290f2af5ac59ef4b5040347dabc4`. The weights are under the OpenMDW License Agreement,
  version 1.1, and are downloaded at setup. They are not part of this repository.
- The companion drafter
  [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr)
  by NVIDIA, revision `4c6a5613b6d887194a3a5493e3a18a5b3930bb86`. It is downloaded at setup and runs in `engine/bolt_draft.cu`.
- The vision tower in the checkpoint is NVIDIA's C-RADIO v4-H; TensorFold's `families/radio` runs it, and the video
  sampling and labels follow the checkpoint's published processor (`processing_nemotron_h_omni.py`).
- [ggml](https://github.com/ggml-org/ggml) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT): the IQ2_XXS and
  Q8_0 formats the checkpoint is packed in, and the IQ2_XXS tables in `engine/iq2_tables.inc`.
- [FFmpeg](https://ffmpeg.org) 9.0.2 (decoders only, built from source) inside `libtfvideo`, and
  [libjpeg-turbo](https://libjpeg-turbo.org) 3.1.4.1, which the image builds for TensorFold's JPEG decoder.
- [Zig](https://ziglang.org) 0.17.0 by the Zig Software Foundation and the Zig contributors.
- NVIDIA's [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) `26.07-py3`, the image
  base. Its license is in [`NOTICE`](NOTICE).
- The script layout (start, stop, prepare, config, banner, checkpoint check, image labels) follows MiaAI-Lab's
  [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold).
