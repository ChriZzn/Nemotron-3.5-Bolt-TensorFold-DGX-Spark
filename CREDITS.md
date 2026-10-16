# Credits

- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors). This recipe pins
  `c7137bb31a7dad924fb541a7068422127b9a94b2` and its
  [Nemotron 3.5 Lightning recipe](https://github.com/ashhart/TensorFold/blob/c7137bb31a7dad924fb541a7068422127b9a94b2/docs/recipes/nemotron-3.5.md).
- [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) by
  NVIDIA, revision `a326822af111290f2af5ac59ef4b5040347dabc4`. The weights are under the OpenMDW License Agreement,
  version 1.1, and are downloaded at setup. They are not part of this repository.
- The companion drafter
  [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr)
  by NVIDIA, revision `4c6a5613b6d887194a3a5493e3a18a5b3930bb86`. It is downloaded at setup and runs in `engine/bolt_draft.cu`.
- The vision tower in the checkpoint is NVIDIA's C-RADIO v4-H; `engine/bolt_vision.cu` reimplements it from the
  checkpoint's published modeling and image-processing code (`modeling_radio.py`, `image_processing_nemotron_h_omni.py`).
- [ggml](https://github.com/ggml-org/ggml) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT): the IQ2_XXS and
  Q8_0 formats the checkpoint is packed in, and the IQ2_XXS tables in `engine/iq2_tables.inc`.
- [stb_image](https://github.com/nothings/stb) by Sean Barrett and contributors, public domain, vendored as
  `engine/stb_image.h` to decode JPEG, PNG, BMP and GIF pictures. [FFmpeg](https://ffmpeg.org) decodes video inside the image.
- [Zig](https://ziglang.org) 0.17.0 by the Zig Software Foundation and the Zig contributors.
- NVIDIA's [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) `26.07-py3`, the image
  base. Its license is in [`NOTICE`](NOTICE).
- The script layout (start, stop, prepare, config, banner, checkpoint check, image labels) follows MiaAI-Lab's
  [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold).
