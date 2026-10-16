#pragma once
#include "bolt_common.h"

// The vision path of Nemotron 3.5 Bolt: C-RADIO v4-H (32 layers, hidden 1280, 16 heads, patch 16, dynamic resolution
// with a cropped position embedding, 3 cls + 7 register tokens) -> 2x2 pixel shuffle -> RMSNorm -> MLP (ReLU²) into
// the language model's 4096-wide embedding space. Reference: the checkpoint's modeling_radio.py,
// modeling_nemotron_h_omni.py and image_processing_nemotron_h_omni.py.

constexpr int VIT_HID = 1280, VIT_HEADS = 16, VIT_HD = 80, VIT_LAYERS = 32, VIT_PREFIX = 10, VIT_FF = 5120;
constexpr int VIT_PATCH = 16, VIT_POS_SIDE = 128;
constexpr int VIT_MIN_PATCHES = 1024, VIT_MAX_PATCHES = 13312, VIT_MAX_MODEL_LEN = 16384, VIT_VIDEO_PATCHES = 1024;

struct VLayerW {
    TBF qkv, proj, fc1, fc2;                       // [3840][1280], [1280][1280], [5120][1280], [1280][5120]
    float *qkv_b, *proj_b, *fc1_b, *fc2_b;
    float *n1w, *n1b, *n2w, *n2b;
};

struct VisionW {
    TBF patch, vpatch;                             // [1280][768] images, [1280][1536] two-frame video groups
    const uint16_t* pos;                           // [128 * 128][1280] bf16
    float* cls;                                    // [10][1280]
    VLayerW L[VIT_LAYERS];
    float* mlp_norm;                               // [5120]
    TBF mlp1, mlp2;                                // [20480][5120], [4096][20480]
};

struct Vision;
Vision* vision_new(const VisionW& w);

// The patch grid (columns, rows) the image rule picks for a width x height picture, and the video rule's.
void vision_grid(int width, int height, int* cols, int* rows);
void vision_grid_video(int width, int height, int* cols, int* rows);

// Encode one image (nframes 1, 3 channels) or one video (nframes a multiple of 2: groups of two frames). rgb[i] is frame
// i, uint8 [h][w][3]. The frames are resized to cols*16 x rows*16 (bicubic with antialiasing, as the reference), normalized
// and run through the tower and the projector. out receives a new device buffer of bf16 [ntok][4096] (cudaFree it);
// a video gives (nframes / 2) groups of cols*rows/4 tokens. Returns 0 on success.
int vision_encode(Vision* v, const uint8_t* const* rgb, const int* w, const int* h, int nframes, int cols, int rows,
                  uint16_t** out, int* ntok, cudaStream_t s);

// The pieces, exposed for tests: dump (fp32, host) of the embedding output after the patch embedding and position
// add (stage 0), the first block (1), the tower's features (2) and the projector's output (3) of an image.
int vision_debug(Vision* v, const uint8_t* rgb, int w, int h, int stage, float* host_out, size_t cap, cudaStream_t s);
