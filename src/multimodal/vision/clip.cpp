// NOTE: This is modified from clip.cpp only for LLaVA,
// so there might be still unnecessary artifacts hanging around
// I'll gradually clean and extend it
// Note: Even when using identical normalized image inputs (see normalize_image_u8_to_f32()) we have a significant difference in resulting embeddings compared to pytorch
#include "clip.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

//#ifdef GGML_USE_CUDA
//#include "ggml-cuda.h"
//#endif
//
//#ifdef GGML_USE_SYCL
//#include "ggml-sycl.h"
//#endif
//
//#ifdef GGML_USE_METAL
//#include "ggml-metal.h"
//#endif
//
//#ifdef GGML_USE_CANN
//#include "ggml-cann.h"
//#endif
//
//#ifdef GGML_USE_VULKAN
//#include "ggml-vulkan.h"
//#endif
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <cinttypes>
#include <limits>

#if defined(LLAVA_LOG_OFF)
#   define LOG_INF(...)
#   define LOG_WRN(...)
#   define LOG_ERR(...)
#   define LOG_DBG(...)
#else // defined(LLAVA_LOG_OFF)
#   define LOG_INF(...) do { fprintf(stdout, __VA_ARGS__); } while (0)
#   define LOG_WRN(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#   define LOG_ERR(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#   define LOG_DBG(...) do { fprintf(stdout, __VA_ARGS__); } while (0)
#endif // defined(LLAVA_LOG_OFF)

//#define CLIP_DEBUG_FUNCTIONS

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

//
// key constants
//

#define KEY_FTYPE               "general.file_type"
#define KEY_NAME                "general.name"
#define KEY_DESCRIPTION         "general.description"
#define KEY_HAS_TEXT_ENC        "clip.has_text_encoder"
#define KEY_HAS_VIS_ENC         "clip.has_vision_encoder"
#define KEY_HAS_RWKV_ADAPTER    "clip.has_rwkv_adapter"
#define KEY_USE_GELU            "clip.use_gelu"
#define KEY_USE_SILU            "clip.use_silu"
#define KEY_N_EMBD              "clip.%s.embedding_length"
#define KEY_N_FF                "clip.%s.feed_forward_length"
#define KEY_N_BLOCK             "clip.%s.block_count"
#define KEY_N_HEAD              "clip.%s.attention.head_count"
#define KEY_LAYER_NORM_EPS      "clip.%s.attention.layer_norm_epsilon"
#define KEY_PROJ_DIM            "clip.%s.projection_dim"
#define KEY_TOKENS              "tokenizer.ggml.tokens"
#define KEY_N_POSITIONS         "clip.text.context_length"
#define KEY_IMAGE_SIZE          "clip.vision.image_size"
#define KEY_PATCH_SIZE          "clip.vision.patch_size"
#define KEY_IMAGE_MEAN          "clip.vision.image_mean"
#define KEY_IMAGE_STD           "clip.vision.image_std"
#define KEY_PROJ_TYPE           "clip.projector_type"

#define KEY_MM_PATCH_MERGE_TYPE   "clip.vision.mm_patch_merge_type"
#define KEY_IMAGE_GRID_PINPOINTS  "clip.vision.image_grid_pinpoints"
#define KEY_IMAGE_CROP_RESOLUTION "clip.vision.image_crop_resolution"


//
// tensor name constants
//

#define TN_TOKEN_EMBD      "%s.token_embd.weight"
#define TN_POS_EMBD        "%s.position_embd.weight"
#define TN_CLASS_EMBD      "v.class_embd"
#define TN_PATCH_EMBD      "v.patch_embd.weight"  // not rename tensor with ".0" postfix for backwrad compat
#define TN_PATCH_EMBD_1    "v.patch_embd.weight.1"
#define TN_PATCH_BIAS      "v.patch_embd.bias"
#define TN_ATTN_K          "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q          "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V          "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT     "%s.blk.%d.attn_out.%s"
#define TN_FFN_DOWN        "%s.blk.%d.ffn_down.%s"
#define TN_FFN_UP          "%s.blk.%d.ffn_up.%s"
#define TN_LN_1            "%s.blk.%d.ln1.%s"
#define TN_LN_2            "%s.blk.%d.ln2.%s"
#define TN_LN_PRE          "%s.pre_ln.%s"
#define TN_LN_POST         "%s.post_ln.%s"
#define TN_TEXT_PROJ       "text_projection.weight"
#define TN_VIS_PROJ        "visual_projection.weight"
#define TN_LLAVA_PROJ      "mm.%d.%s"
#define TN_MVLM_PROJ_MLP   "mm.model.mlp.%d.%s"
#define TN_MVLM_PROJ_BLOCK "mm.model.mb_block.%d.block.%d.%s"
#define TN_MVLM_PROJ_PEG   "mm.model.peg.%d.%s"
#define TN_IMAGE_NEWLINE   "model.image_newline"

#define TN_MINICPMV_POS_EMBD_K "resampler.pos_embed_k"
#define TN_MINICPMV_QUERY "resampler.query"
#define TN_MINICPMV_PROJ "resampler.proj.weight"
#define TN_MINICPMV_KV_PROJ "resampler.kv.weight"
#define TN_MINICPMV_ATTN "resampler.attn.%s.%s"
#define TN_MINICPMV_LN "resampler.ln_%s.%s"

#define TN_GLM_ADAPER_CONV "adapter.conv.%s"
#define TN_GLM_ADAPTER_LINEAR "adapter.linear.linear.%s"
#define TN_GLM_ADAPTER_NORM_1 "adapter.linear.norm1.%s"
#define TN_GLM_ADAPTER_D_H_2_4H "adapter.linear.dense_h_to_4h.%s"
#define TN_GLM_ADAPTER_GATE "adapter.linear.gate.%s"
#define TN_GLM_ADAPTER_D_4H_2_H "adapter.linear.dense_4h_to_h.%s"
#define TN_GLM_BOI_W "adapter.boi"
#define TN_GLM_EOI_W "adapter.eoi"

#define TN_RWKV_ADAPTER_PROJ "rwkv_adapter.proj.%d.%s"
#define TN_RWKV_ADAPTER_NORM "rwkv_adapter.pre_norm.%s"

enum projector_type {
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_MLP_NORM,
    PROJECTOR_TYPE_LDP,
    PROJECTOR_TYPE_LDPV2,
    PROJECTOR_TYPE_RESAMPLER,
    PROJECTOR_TYPE_GLM_EDGE,
    PROJECTOR_TYPE_MERGER,
    PROJECTOR_TYPE_RWKV,
    PROJECTOR_TYPE_UNKNOWN,
};

static std::map<projector_type, std::string> PROJECTOR_TYPE_NAMES = {
    { PROJECTOR_TYPE_MLP, "mlp" },
    { PROJECTOR_TYPE_LDP, "ldp" },
    { PROJECTOR_TYPE_LDPV2, "ldpv2"},
    { PROJECTOR_TYPE_RESAMPLER, "resampler"},
    { PROJECTOR_TYPE_GLM_EDGE, "adapter"},
    { PROJECTOR_TYPE_MERGER, "qwen2vl_merger"},
    { PROJECTOR_TYPE_RWKV, "rwkv"},
};


//
// utilities to get data from a gguf file
//

static int get_key_idx(const gguf_context * ctx, const char * key) {
    int i = gguf_find_key(ctx, key);
    if (i == -1) {
        LOG_ERR("key %s not found in file\n", key);
        throw std::runtime_error(format("Missing required key: %s", key));
    }

    return i;
}

static uint32_t get_u32(const gguf_context * ctx, const std::string & key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_u32(ctx, i);
}

static float get_f32(const gguf_context * ctx, const std::string & key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_f32(ctx, i);
}

static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        throw std::runtime_error(format("%s: unable to find tensor %s\n", __func__, name.c_str()));
    }

    return cur;
}

static std::string get_ftype(int ftype) {
    return ggml_type_name(static_cast<ggml_type>(ftype));
}

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const bool *)data)[i] ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}

static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

static std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replace_all(val, "\\", "\\\\");
                        replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

static void print_tensor_info(const ggml_tensor * tensor, const char * prefix = "") {
    size_t tensor_size = ggml_nbytes(tensor);
    LOG_INF("%s: n_dims = %d, name = %s, tensor_size=%zu, shape:[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], type = %s\n",
            prefix, ggml_n_dims(tensor), tensor->name, tensor_size,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], ggml_type_name(tensor->type));
}

static projector_type clip_projector_type_from_string(const std::string & name) {
    for (const auto & kv : PROJECTOR_TYPE_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }
    return PROJECTOR_TYPE_UNKNOWN;
}

#ifdef CLIP_DEBUG_FUNCTIONS
static void clip_image_write_image_to_ppm(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    // PPM header: P6 format, width, height, and max color value
    file << "P6\n" << img.nx << " " << img.ny << "\n255\n";

    // Write pixel data
    for (size_t i = 0; i < img.buf.size(); i += 3) {
        // PPM expects binary data in RGB format, which matches our image buffer
        file.write(reinterpret_cast<const char*>(&img.buf[i]), 3);
    }

    file.close();
}

static void clip_image_save_to_bmp(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    int fileSize = 54 + 3 * img.nx * img.ny; // File header + info header + pixel data
    int bytesPerPixel = 3;
    int widthInBytes = img.nx * bytesPerPixel;
    int paddingAmount = (4 - (widthInBytes % 4)) % 4;
    int stride = widthInBytes + paddingAmount;

    // Bitmap file header
    unsigned char fileHeader[14] = {
        'B','M',     // Signature
        0,0,0,0,    // Image file size in bytes
        0,0,0,0,    // Reserved
        54,0,0,0    // Start of pixel array
    };

    // Total file size
    fileSize = 54 + (stride * img.ny);
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);

    // Bitmap information header (BITMAPINFOHEADER)
    unsigned char infoHeader[40] = {
        40,0,0,0,   // Size of this header (40 bytes)
        0,0,0,0,    // Image width
        0,0,0,0,    // Image height
        1,0,        // Number of color planes
        24,0,       // Bits per pixel
        0,0,0,0,    // No compression
        0,0,0,0,    // Image size (can be 0 for no compression)
        0,0,0,0,    // X pixels per meter (not specified)
        0,0,0,0,    // Y pixels per meter (not specified)
        0,0,0,0,    // Total colors (color table not used)
        0,0,0,0     // Important colors (all are important)
    };

    // Width and height in the information header
    infoHeader[4] = (unsigned char)(img.nx);
    infoHeader[5] = (unsigned char)(img.nx >> 8);
    infoHeader[6] = (unsigned char)(img.nx >> 16);
    infoHeader[7] = (unsigned char)(img.nx >> 24);
    infoHeader[8] = (unsigned char)(img.ny);
    infoHeader[9] = (unsigned char)(img.ny >> 8);
    infoHeader[10] = (unsigned char)(img.ny >> 16);
    infoHeader[11] = (unsigned char)(img.ny >> 24);

    // Write file headers
    file.write(reinterpret_cast<char*>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char*>(infoHeader), sizeof(infoHeader));

    // Pixel data
    std::vector<unsigned char> padding(3, 0); // Max padding size to be added to each row
    for (int y = img.ny - 1; y >= 0; --y) { // BMP files are stored bottom-to-top
        for (int x = 0; x < img.nx; ++x) {
            // Each pixel
            size_t pixelIndex = (y * img.nx + x) * 3;
            unsigned char pixel[3] = {
                img.buf[pixelIndex + 2], // BMP stores pixels in BGR format
                img.buf[pixelIndex + 1],
                img.buf[pixelIndex]
            };
            file.write(reinterpret_cast<char*>(pixel), 3);
        }
        // Write padding for the row
        file.write(reinterpret_cast<char*>(padding.data()), paddingAmount);
    }

    file.close();
}

// debug function to convert f32 to u8
static void clip_image_convert_f32_to_u8(const clip_image_f32& src, clip_image_u8& dst) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(3 * src.nx * src.ny);
    for (size_t i = 0; i < src.buf.size(); ++i) {
        dst.buf[i] = static_cast<uint8_t>(std::min(std::max(int(src.buf[i] * 255.0f), 0), 255));
    }
}
#endif


//
// clip layers
//

struct clip_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;

    float eps;

    char mm_patch_merge_type[32] = "flat"; // spatial_unpad or flat (default)

    int32_t image_grid_pinpoints[32];
    int32_t image_crop_resolution;
};

struct clip_layer {
    // attention
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    struct ggml_tensor * o_w;
    struct ggml_tensor * o_b;

    // layernorm 1
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_1_b;

    // ff
    struct ggml_tensor * ff_i_w;
    struct ggml_tensor * ff_i_b;

    struct ggml_tensor * ff_o_w;
    struct ggml_tensor * ff_o_b;

    // layernorm 2
    struct ggml_tensor * ln_2_w;
    struct ggml_tensor * ln_2_b;
};

struct clip_vision_model {
    struct clip_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding;
    struct ggml_tensor * patch_embeddings_0;
    struct ggml_tensor * patch_embeddings_1;  // second Conv2D kernel when we decouple Conv3D along temproal dimension (Qwen2VL)
    struct ggml_tensor * patch_bias;
    struct ggml_tensor * position_embeddings;

    struct ggml_tensor * pre_ln_w;
    struct ggml_tensor * pre_ln_b;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;

    // LLaVA projection
    struct ggml_tensor * mm_0_w = NULL;
    struct ggml_tensor * mm_0_b = NULL;
    struct ggml_tensor * mm_2_w = NULL;
    struct ggml_tensor * mm_2_b = NULL;

    struct ggml_tensor * image_newline = NULL;

    struct ggml_tensor * mm_1_w = NULL; // Yi type models have 0, 1, 3, 4
    struct ggml_tensor * mm_1_b = NULL;
};

struct clip_ctx {
    bool has_text_encoder    = false;
    bool has_vision_encoder  = false;
    bool has_rwkv_adapter = false;

    struct clip_vision_model vision_model;
    projector_type proj_type = PROJECTOR_TYPE_RWKV;

    float image_mean[3];
    float image_std[3];
    bool use_gelu = false;
    bool use_silu = false;
    int32_t ftype = 1;

    bool has_class_embedding = true;
    bool has_pre_norm = true;
    bool has_post_norm = false;
    bool has_patch_bias = false;

    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    // memory buffers to evaluate the model
    ggml_backend_buffer_t params_buffer  = NULL;

    ggml_backend_t backend       = NULL;
    ggml_gallocr_t compute_alloc = NULL;

    struct clip_image_size * load_image_size;
};

static ggml_cgraph * clip_image_build_graph(clip_ctx * ctx, const clip_image_f32_batch * imgs, struct clip_image_size * load_image_size, bool is_inf = false) {
    if (!ctx->has_vision_encoder) {
        LOG_ERR("This gguf file seems to have no vision encoder\n");
        return nullptr;
    }

    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size = hparams.image_size;
    int image_size_width  = image_size;
    int image_size_height = image_size;

    const int patch_size           = hparams.patch_size;
    const int num_patches          = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int patches_w            = image_size_width / patch_size;
    const int patches_h            = image_size_height / patch_size;
    const int num_positions        = num_patches + (ctx->has_class_embedding ? 1 : 0);
    const int num_position_ids     = num_positions;
    const int hidden_size          = hparams.hidden_size;
    const int n_head               = hparams.n_head;
    const int d_head               = hidden_size / n_head;
    int n_layer                    = hparams.n_layer;
    const float eps                = hparams.eps;
    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    const int batch_size = imgs->size;

    if (ctx->has_rwkv_adapter) {
        GGML_ASSERT(batch_size == 1);
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size_width, image_size_height, 3, batch_size);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

    if (ctx->has_patch_bias) {
        // inp = ggml_add(ctx0, inp, ggml_repeat(ctx0, model.patch_bias, inp));
        inp = ggml_add(ctx0, inp, model.patch_bias);
    }
    struct ggml_tensor * embeddings = inp;
    struct ggml_tensor * pos_embed = nullptr;

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    embeddings = ggml_add(ctx0, embeddings, ggml_get_rows(ctx0, model.position_embeddings, positions));

    // pre-layernorm
    if (ctx->has_pre_norm) {
        embeddings = ggml_norm(ctx0, embeddings, eps);
        ggml_set_name(embeddings, "pre_ln");

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.pre_ln_w), model.pre_ln_b);
    }

    // loop over layers
    if (ctx->has_rwkv_adapter) {
        n_layer += 1;
    }
    for (int il = 0; il < n_layer - 1; il++) {
        struct ggml_tensor * cur = embeddings; // embeddings = residual, cur = hidden_states

        //const size_t nb_q_w = model.layers[il].q_w->nb[0];

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_1_w),
                           model.layers[il].ln_1_b);
        }

        // self-attention
        {

            struct ggml_tensor * Q =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].q_w, cur), model.layers[il].q_b);

            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_positions, batch_size);
            Q = ggml_scale_inplace(ctx0, Q, 1.0f / sqrt((float)d_head));
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * K =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].k_w, cur), model.layers[il].k_b);

            K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * V =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].v_w, cur), model.layers[il].v_b);

            V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, batch_size);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head * batch_size);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_soft_max_inplace(ctx0, KQ);
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_positions, n_head, batch_size);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cont_3d(ctx0, KQV, hidden_size, num_positions, batch_size);
        }

        // attention output
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].o_w, cur), model.layers[il].o_b);

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_2_w), model.layers[il].ln_2_b);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ff_i_b);

        if (ctx->use_gelu) {
            cur = ggml_gelu_inplace(ctx0, cur);
        } else if (ctx->use_silu) {
            cur = ggml_silu_inplace(ctx0, cur);
        } else {
            cur = ggml_gelu_quick_inplace(ctx0, cur);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ff_o_b);

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);

        embeddings = cur;

    }

    // post-layernorm
    if (ctx->has_post_norm) {
        embeddings = ggml_norm(ctx0, embeddings, eps);
        ggml_set_name(embeddings, "post_ln");

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.post_ln_w), model.post_ln_b);
    }

    if (ctx->has_rwkv_adapter) {
        embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);

        embeddings = ggml_relu(ctx0, embeddings);
        embeddings = ggml_mul_mat(ctx0, model.mm_2_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_2_b);

        struct ggml_tensor * emb_norm = ggml_norm(ctx0, embeddings, 1e-5);
        emb_norm = ggml_add(ctx0, ggml_mul(ctx0, emb_norm, model.mm_1_w), model.mm_1_b);
        embeddings = ggml_add(ctx0, embeddings, emb_norm);
    }

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    ggml_free(ctx0);

    return gf;
}

// read and create ggml_context containing the tensors and their data
struct clip_ctx * clip_model_load(const char * fname, const char * fname_adapter, const int verbosity = 1) {
    struct ggml_context * meta = NULL;
    struct ggml_context * meta_adapter = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname, params);
    if (!ctx) {
        throw std::runtime_error(format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname));
    }


    struct gguf_init_params params_adapter = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta_adapter,
    };

    struct gguf_context * ctx_adapter = NULL;
    if (fname_adapter) {
        ctx_adapter = gguf_init_from_file(fname_adapter, params_adapter);
        if (!ctx_adapter) {
            throw std::runtime_error(format("%s: failed to load rwkv adapter file from %s. Does this file exist?\n", __func__, fname_adapter));
        }
    }

    if (verbosity >= 1) {
        const int n_tensors = gguf_get_n_tensors(ctx);
        const int n_kv = gguf_get_n_kv(ctx);
        const int ftype = get_u32(ctx, KEY_FTYPE);
        const std::string ftype_str = get_ftype(ftype);
        const int idx_desc = get_key_idx(ctx, KEY_DESCRIPTION);
        const std::string description = gguf_get_val_str(ctx, idx_desc);
        const int idx_name = gguf_find_key(ctx, KEY_NAME);
        if (idx_name != -1) { // make name optional temporarily as some of the uploaded models missing it due to a bug
            const std::string name = gguf_get_val_str(ctx, idx_name);
            LOG_INF("%s: model name:   %s\n", __func__, name.c_str());
        }
        LOG_INF("%s: description:  %s\n", __func__, description.c_str());
        LOG_INF("%s: GGUF version: %d\n", __func__, gguf_get_version(ctx));
        LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx));
        LOG_INF("%s: n_tensors:    %d\n", __func__, n_tensors);
        LOG_INF("%s: n_kv:         %d\n", __func__, n_kv);
        LOG_INF("%s: ftype:        %s\n", __func__, ftype_str.c_str());
        LOG_INF("\n");
    }
    const int n_tensors = gguf_get_n_tensors(ctx);
    const int n_tensors_adapter = ctx_adapter ? gguf_get_n_tensors(ctx_adapter) : 0;

    // kv
    const int n_kv = gguf_get_n_kv(ctx);
    LOG_INF("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n",
        __func__, n_kv, n_tensors, fname);
    if (ctx_adapter) {
        LOG_INF("%s: loaded rwkv adapter with %d tensors from %s\n",
            __func__, n_tensors_adapter, fname_adapter);
    }

    {
        std::map<enum ggml_type, uint32_t> n_type;

        for (int i = 0; i < n_tensors; i++) {
            enum ggml_type type = gguf_get_tensor_type(ctx, i);

            n_type[type]++;
        }

        LOG_INF("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);
        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(ctx, i);
            const enum gguf_type type   = gguf_get_kv_type(ctx, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%d]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(ctx, i)), gguf_get_arr_n(ctx, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(ctx, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LOG_INF("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LOG_INF("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    // data
    size_t model_size = 0;
    {
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);
            enum ggml_type type = gguf_get_tensor_type(ctx, i);
            struct ggml_tensor * cur = ggml_get_tensor(meta, name);
            size_t tensor_size = ggml_nbytes(cur);
            model_size += tensor_size;
            if (verbosity >= 3) {
                LOG_INF("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                       __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    if (ctx_adapter) {
        for (int i = 0; i < n_tensors_adapter; ++i) {
            const char * name = gguf_get_tensor_name(ctx_adapter, i);
            const size_t offset = gguf_get_tensor_offset(ctx_adapter, i);
            enum ggml_type type = gguf_get_tensor_type(ctx_adapter, i);
            struct ggml_tensor * cur = ggml_get_tensor(meta_adapter, name);
            size_t tensor_size = ggml_nbytes(cur);
            model_size += tensor_size;
            if (verbosity >= 3) {
                LOG_INF("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                       __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    clip_ctx * new_clip = new clip_ctx{};

    // update projector type
    {
        int idx = gguf_find_key(ctx, KEY_PROJ_TYPE);
        if (idx != -1) {
            const std::string proj_type = gguf_get_val_str(ctx, idx);
            new_clip->proj_type = clip_projector_type_from_string(proj_type);
        } else {
            new_clip->proj_type = PROJECTOR_TYPE_MLP;
        }

        if (new_clip->proj_type == PROJECTOR_TYPE_MLP) {
            if (gguf_find_tensor(ctx, format(TN_LLAVA_PROJ, 3, "weight").c_str()) != -1) {
                new_clip->proj_type = PROJECTOR_TYPE_MLP_NORM;
            }
        }
    }

//#ifdef GGML_USE_CUDA
//    new_clip->backend = ggml_backend_cuda_init(0);
//    LOG_INF("%s: CLIP using CUDA backend\n", __func__);
//#endif
//
//#ifdef GGML_USE_METAL
//    new_clip->backend = ggml_backend_metal_init();
//    LOG_INF("%s: CLIP using Metal backend\n", __func__);
//#endif
//
//#ifdef GGML_USE_CANN
//    new_clip->backend = ggml_backend_cann_init(0);
//    LOG_INF("%s: CLIP using CANN backend\n", __func__);
//#endif
//
//#ifdef GGML_USE_VULKAN
//    new_clip->backend = ggml_backend_vk_init(0);
//    LOG_INF("%s: CLIP using Vulkan backend\n", __func__);
//#endif
//
//#ifdef GGML_USE_SYCL
//    new_clip->backend = ggml_backend_sycl_init(0);
//    LOG_INF("%s: CLIP using SYCL backend\n", __func__);
//#endif

    if (!new_clip->backend) {
        new_clip->backend = ggml_backend_cpu_init();
        LOG_INF("%s: CLIP using CPU backend\n", __func__);
    }

    // model size and capabilities
    {
        int idx = get_key_idx(ctx, KEY_HAS_TEXT_ENC);
        new_clip->has_text_encoder = gguf_get_val_bool(ctx, idx);

        idx = get_key_idx(ctx, KEY_HAS_VIS_ENC);
        new_clip->has_vision_encoder = gguf_get_val_bool(ctx, idx);

        idx = gguf_find_key(ctx, KEY_HAS_RWKV_ADAPTER);
        if (idx != -1) {
            new_clip->has_rwkv_adapter = gguf_get_val_bool(ctx, idx);
        }

        // GGML_ASSERT(new_clip->has_llava_projector); // see monatis/clip.cpp for image and/or text encoding for semantic search

        GGML_ASSERT(new_clip->has_vision_encoder);
        GGML_ASSERT(!new_clip->has_text_encoder);

        idx = get_key_idx(ctx, KEY_USE_GELU);
        new_clip->use_gelu = gguf_get_val_bool(ctx, idx);

        try {
            idx = get_key_idx(ctx, KEY_USE_SILU);
            new_clip->use_silu = gguf_get_val_bool(ctx, idx);
        } catch (std::runtime_error & /*e*/) {
            new_clip->use_silu = false;
        }

        if (verbosity >= 1) {
            LOG_INF("%s: text_encoder:   %d\n", __func__, new_clip->has_text_encoder);
            LOG_INF("%s: vision_encoder: %d\n", __func__, new_clip->has_vision_encoder);
            LOG_INF("%s: rwkv_adapter:   %d\n", __func__, new_clip->has_rwkv_adapter);
            LOG_INF("%s: model size:     %.2f MB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:  %.2f MB\n", __func__, ggml_get_mem_size(meta) / 1024.0 / 1024.0);
        }
    }

    LOG_INF("%s: params backend buffer size = % 6.2f MB (%i tensors)\n", __func__, model_size / (1024.0 * 1024.0), n_tensors);

    // load tensors
    {
        std::vector<uint8_t> read_buf;
        struct ggml_init_params params = {
            /*.mem_size =*/ (n_tensors + n_tensors_adapter + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };

        new_clip->ctx_data = ggml_init(params);
        if (!new_clip->ctx_data) {
            LOG_ERR("%s: ggml_init() failed\n", __func__);
            clip_free(new_clip);
            gguf_free(ctx);
            gguf_free(ctx_adapter);
            return nullptr;
        }

        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            LOG_ERR("cannot open model file for loading tensors\n");
            clip_free(new_clip);
            gguf_free(ctx);
            gguf_free(ctx_adapter);
            return nullptr;
        }

        // add tensors to context
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor * t = ggml_get_tensor(meta, name);
            struct ggml_tensor * cur = ggml_dup_tensor(new_clip->ctx_data, t);
            ggml_set_name(cur, name);
        }

        if (ctx_adapter) {
            for (int i = 0; i < n_tensors_adapter; ++i) {
                const char * name = gguf_get_tensor_name(ctx_adapter, i);
                struct ggml_tensor * t = ggml_get_tensor(meta_adapter, name);
                struct ggml_tensor * cur = ggml_dup_tensor(new_clip->ctx_data, t);
                ggml_set_name(cur, name);
            }
        }

        // alloc memory and offload data
        new_clip->params_buffer = ggml_backend_alloc_ctx_tensors(new_clip->ctx_data, new_clip->backend);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor * cur = ggml_get_tensor(new_clip->ctx_data, name);
            const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                LOG_ERR("%s: failed to seek for tensor %s\n", __func__, name);
                clip_free(new_clip);
                gguf_free(ctx);
                return nullptr;
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(new_clip->params_buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        fin.close();

        if (ctx_adapter) {
            fin = std::ifstream(fname_adapter, std::ios::binary);
            if (!fin) {
                LOG_ERR("cannot open rwkv adapter file for loading tensors\n");
                clip_free(new_clip);
                gguf_free(ctx);
                gguf_free(ctx_adapter);
                return nullptr;
            }

            for (int i = 0; i < n_tensors_adapter; ++i) {
                const char * name = gguf_get_tensor_name(ctx_adapter, i);
                struct ggml_tensor * cur = ggml_get_tensor(new_clip->ctx_data, name);
                const size_t offset = gguf_get_data_offset(ctx_adapter) + gguf_get_tensor_offset(ctx_adapter, i);
                fin.seekg(offset, std::ios::beg);
                if (!fin) {
                    LOG_ERR("%s: failed to seek for tensor %s\n", __func__, name);
                    clip_free(new_clip);
                    gguf_free(ctx);
                    gguf_free(ctx_adapter);
                    return nullptr;
                }
                int num_bytes = ggml_nbytes(cur);
                if (ggml_backend_buffer_is_host(new_clip->params_buffer)) {
                    // for the CPU and Metal backend, we can read directly into the tensor
                    fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
                } else {
                    // read into a temporary buffer first, then copy to device memory
                    read_buf.resize(num_bytes);
                    fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
                }
            }
            fin.close();
            new_clip->has_rwkv_adapter = true;
        }
    }

    // vision model
    if (new_clip->has_vision_encoder) {
        // load vision model
        auto & vision_model = new_clip->vision_model;
        auto & hparams = vision_model.hparams;
        hparams.hidden_size    = get_u32(ctx, format(KEY_N_EMBD, "vision"));
        hparams.n_head         = get_u32(ctx, format(KEY_N_HEAD, "vision"));
        hparams.n_intermediate = get_u32(ctx, format(KEY_N_FF, "vision"));
        hparams.n_layer        = get_u32(ctx, format(KEY_N_BLOCK, "vision"));
        hparams.image_size     = get_u32(ctx, KEY_IMAGE_SIZE);
        hparams.patch_size     = get_u32(ctx, KEY_PATCH_SIZE);
        hparams.projection_dim = get_u32(ctx, format(KEY_PROJ_DIM, "vision"));
        hparams.eps            = get_f32(ctx, format(KEY_LAYER_NORM_EPS, "vision"));

        try {
            int idx = get_key_idx(ctx, KEY_IMAGE_GRID_PINPOINTS);
            int n = gguf_get_arr_n(ctx, idx);
            const int32_t * pinpoints = (const int32_t *)gguf_get_arr_data(ctx, idx);
            for (int i = 0; i < 32 && i < n && pinpoints[i] != 0; ++i) {
                hparams.image_grid_pinpoints[i] = pinpoints[i];
            }
            if (n < 32)
                hparams.image_grid_pinpoints[n] = 0;
        } catch (std::runtime_error & /*e*/) {
            hparams.image_grid_pinpoints[0]=0;
        }

        try {
            int idx = get_key_idx(ctx, KEY_MM_PATCH_MERGE_TYPE);
            strcpy(hparams.mm_patch_merge_type, gguf_get_val_str(ctx, idx));
        } catch (std::runtime_error & /*e*/) {
            strcpy(hparams.mm_patch_merge_type, "flat");
        }

        try {
            hparams.image_crop_resolution = get_u32(ctx, KEY_IMAGE_CROP_RESOLUTION); // llava-1.6
        } catch(const std::exception& /*e*/) {
            hparams.image_crop_resolution = hparams.image_size;
        }

        int idx_mean = get_key_idx(ctx, KEY_IMAGE_MEAN);
        int idx_std  = get_key_idx(ctx, KEY_IMAGE_STD);

        const float * mean_data = (const float *)gguf_get_arr_data(ctx, idx_mean);
        const float * std_data  = (const float *)gguf_get_arr_data(ctx, idx_std);

        for (int i = 0; i < 3; ++i) {
            new_clip->image_mean[i] = mean_data[i];
            new_clip->image_std[i]  = std_data[i];
        }

        if (verbosity >= 2) {
            LOG_INF("\n%s: vision model hparams\n", __func__);
            LOG_INF("image_size         %d\n", hparams.image_size);
            LOG_INF("patch_size         %d\n", hparams.patch_size);
            LOG_INF("v_hidden_size      %d\n", hparams.hidden_size);
            LOG_INF("v_n_intermediate   %d\n", hparams.n_intermediate);
            LOG_INF("v_projection_dim   %d\n", hparams.projection_dim);
            LOG_INF("v_n_head           %d\n", hparams.n_head);
            LOG_INF("v_n_layer          %d\n", hparams.n_layer);
            LOG_INF("v_eps              %f\n", hparams.eps);
            LOG_INF("v_image_mean       %f %f %f\n", new_clip->image_mean[0], new_clip->image_mean[1], new_clip->image_mean[2]);
            LOG_INF("v_image_std        %f %f %f\n", new_clip->image_std[0], new_clip->image_std[1], new_clip->image_std[2]);
            LOG_INF("v_image_grid_pinpoints: ");
            for (int i = 0; i < 32 && (hparams.image_grid_pinpoints[i] != 0); ++i) {
                LOG_INF("%d ", hparams.image_grid_pinpoints[i]);
            }
            LOG_INF("\n");
            LOG_INF("v_mm_patch_merge_type: %s\n", hparams.mm_patch_merge_type);

        }

        try {
            vision_model.class_embedding  = get_tensor(new_clip->ctx_data, TN_CLASS_EMBD);
            new_clip->has_class_embedding = true;
        } catch (const std::exception& /*e*/) {
            new_clip->has_class_embedding = false;
        }

        try {
            vision_model.pre_ln_w  = get_tensor(new_clip->ctx_data, format(TN_LN_PRE, "v", "weight"));
            vision_model.pre_ln_b  = get_tensor(new_clip->ctx_data, format(TN_LN_PRE, "v", "bias"));
            new_clip->has_pre_norm = true;
        } catch (std::exception & /*e*/) {
            new_clip->has_pre_norm = false;
        }

        try {
            vision_model.post_ln_w  = get_tensor(new_clip->ctx_data, format(TN_LN_POST, "v", "weight"));
            vision_model.post_ln_b  = get_tensor(new_clip->ctx_data, format(TN_LN_POST, "v", "bias"));
            new_clip->has_post_norm = true;
        } catch (std::exception & /*e*/) {
            new_clip->has_post_norm = false;
        }

        try {
            vision_model.patch_bias = get_tensor(new_clip->ctx_data, TN_PATCH_BIAS);
            new_clip->has_patch_bias = true;
        } catch (std::exception & /*e*/) {
            new_clip->has_patch_bias = false;
        }

        try {
            vision_model.patch_embeddings_0    = get_tensor(new_clip->ctx_data, TN_PATCH_EMBD);
            vision_model.position_embeddings = get_tensor(new_clip->ctx_data, format(TN_POS_EMBD, "v"));
        } catch(const std::exception& /*e*/) {
            LOG_ERR("%s: failed to load vision model tensors\n", __func__);
        }

        try {
            vision_model.mm_0_w              = get_tensor(new_clip->ctx_data, format(TN_RWKV_ADAPTER_PROJ, 0, "weight"));
            vision_model.mm_0_b              = get_tensor(new_clip->ctx_data, format(TN_RWKV_ADAPTER_PROJ, 0, "bias"));

            vision_model.mm_1_w              = get_tensor(new_clip->ctx_data, format(TN_RWKV_ADAPTER_NORM, "weight"));
            vision_model.mm_1_b              = get_tensor(new_clip->ctx_data, format(TN_RWKV_ADAPTER_NORM, "bias"));

            vision_model.mm_2_w              = get_tensor(new_clip->ctx_data, format(TN_RWKV_ADAPTER_PROJ, 2, "weight"));
            vision_model.mm_2_b              = get_tensor(new_clip->ctx_data, format(TN_RWKV_ADAPTER_PROJ, 2, "bias"));
        } catch(const std::exception& /*e*/) {
            LOG_ERR("%s: failed to load vision model tensors\n", __func__);
        }

        vision_model.layers.resize(hparams.n_layer);

        for (int il = 0; il < hparams.n_layer; ++il) {
            auto & layer = vision_model.layers[il];
            layer.k_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_K,      "v", il, "weight"));
            layer.q_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_Q,      "v", il, "weight"));
            layer.v_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_V,      "v", il, "weight"));
            layer.o_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_OUTPUT, "v", il, "weight"));
            layer.ln_1_w = get_tensor(new_clip->ctx_data, format(TN_LN_1,        "v", il, "weight"));
            layer.ln_2_w = get_tensor(new_clip->ctx_data, format(TN_LN_2,        "v", il, "weight"));
            layer.ff_i_w = get_tensor(new_clip->ctx_data, format(TN_FFN_DOWN,    "v", il, "weight"));
            layer.ff_o_w = get_tensor(new_clip->ctx_data, format(TN_FFN_UP,      "v", il, "weight"));
            layer.k_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_K,      "v", il, "bias"));
            layer.q_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_Q,      "v", il, "bias"));
            layer.v_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_V,      "v", il, "bias"));
            layer.o_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_OUTPUT, "v", il, "bias"));
            layer.ln_1_b = get_tensor(new_clip->ctx_data, format(TN_LN_1,        "v", il, "bias"));
            layer.ln_2_b = get_tensor(new_clip->ctx_data, format(TN_LN_2,        "v", il, "bias"));
            layer.ff_i_b = get_tensor(new_clip->ctx_data, format(TN_FFN_DOWN,    "v", il, "bias"));
            layer.ff_o_b = get_tensor(new_clip->ctx_data, format(TN_FFN_UP,      "v", il, "bias"));
        }
    }

    ggml_free(meta);

    new_clip->ctx_gguf = ctx;

    // measure mem requirement and allocate
    {
        new_clip->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
        new_clip->compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(new_clip->backend));
        clip_image_f32_batch batch;
        batch.size = 1;
        batch.data = nullptr;
        ggml_cgraph * gf = clip_image_build_graph(new_clip, &batch, nullptr, false);
        ggml_gallocr_reserve(new_clip->compute_alloc, gf);
        size_t compute_memory_buffer_size = ggml_gallocr_get_buffer_size(new_clip->compute_alloc, 0);
        LOG_INF("%s: compute allocated memory: %.2f MB\n", __func__, compute_memory_buffer_size /1024.0/1024.0);
    }

    return new_clip;
}

void clip_add_load_image_size(struct clip_ctx * ctx_clip, struct clip_image_size * load_image_size) {
    ctx_clip->load_image_size = load_image_size;
}

struct clip_image_size * clip_get_load_image_size(struct clip_ctx * ctx_clip) {
    return ctx_clip->load_image_size;
}

struct clip_image_size * clip_image_size_init() {
    struct clip_image_size * load_image_size = new struct clip_image_size();
    load_image_size->width = 448;
    load_image_size->height = 448;
    return load_image_size;
}

struct clip_image_u8 * clip_image_u8_init() {
    return new clip_image_u8();
}

struct clip_image_f32 * clip_image_f32_init() {
    return new clip_image_f32();
}

void clip_image_u8_free(struct clip_image_u8  * img) { delete img; }
void clip_image_f32_free(struct clip_image_f32 * img) { delete img; }
void clip_image_u8_batch_free(struct clip_image_u8_batch  * batch) {
    if (batch->size > 0) {
        delete[] batch->data;
        batch->size = 0;
    }
}
void clip_image_f32_batch_free(struct clip_image_f32_batch  * batch) {
    if (batch->size > 0) {
        delete[] batch->data;
        batch->size = 0;
    }
}

static void build_clip_img_from_data(const stbi_uc * data, int nx, int ny, clip_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    memcpy(img->buf.data(), data, img->buf.size());
}

bool clip_image_load_from_file(const char * fname, clip_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load(fname, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to load image '%s'\n", __func__, fname);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

// Bilinear resize function
static void bilinear_resize(const clip_image_u8& src, clip_image_u8& dst, int target_width, int target_height) {
    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    const int src_width = src.nx;
    const int src_height = src.ny;

    constexpr double bilinear_filter_support = 1.0;

    static constexpr uint32_t precision_bits = 32 - 8 - 2;

    const double scale_x = static_cast<double>(src_width) / target_width;
    const double scale_y = static_cast<double>(src_height) / target_height;

    std::vector<int32_t> bounds_horiz;
    std::vector<double> kk_horiz;
    int32_t ksize_horiz = 0;

    double filterscale_x = scale_x;
    if (filterscale_x < 1.0) {
        filterscale_x = 1.0;
    }

    const double support_x = bilinear_filter_support * filterscale_x;

    ksize_horiz = static_cast<int32_t>(ceil(support_x)) * 2 + 1;

    kk_horiz.resize(target_width * ksize_horiz);

    bounds_horiz.resize(target_width * 2);

    constexpr double half_pixel = 0.5;
    for (int32_t xx = 0; xx < target_width; ++xx) {
        double center = (xx + half_pixel) * scale_x - half_pixel;
        double ww = 0.0;
        double ss = 1.0 / filterscale_x;

        auto xmin = static_cast<int32_t>(center - support_x + half_pixel);
        if (xmin < 0) {
            xmin = 0;
        }

        auto xmax = static_cast<int32_t>(center + support_x + half_pixel);
        if (xmax > src_width) {
            xmax = src_width;
        }
        xmax -= xmin;

        double* k = &kk_horiz[xx * ksize_horiz];
        for (int32_t x = 0; x < xmax; ++x) {
            // Bilinear filter: if |x| < 1.0, return 1.0 - |x|, else return 0.0
            double filter_x = (x + xmin - center + half_pixel) * ss;
            if (filter_x < 0.0) {
                filter_x = -filter_x;
            }
            double w = (filter_x < 1.0) ? (1.0 - filter_x) : 0.0;
            k[x] = w;
            ww += w;
        }

        for (int32_t x = 0; x < xmax; ++x) {
            if (ww != 0.0) {
                k[x] /= ww;
            }
        }

        for (int32_t x = xmax; x < ksize_horiz; ++x) {
            k[x] = 0;
        }

        bounds_horiz[xx * 2 + 0] = xmin;
        bounds_horiz[xx * 2 + 1] = xmax;
    }
    

    std::vector<int32_t> bounds_vert;
    std::vector<double> kk_vert;
    int32_t ksize_vert = 0;

    double filterscale_y = scale_y;
    if (filterscale_y < 1.0) {
        filterscale_y = 1.0;
    }

    const double support_y = bilinear_filter_support * filterscale_y;

    ksize_vert = static_cast<int32_t>(ceil(support_y)) * 2 + 1;

    kk_vert.resize(target_height * ksize_vert);

    bounds_vert.resize(target_height * 2);

    for (int32_t yy = 0; yy < target_height; ++yy) {
        double center = (yy + half_pixel) * scale_y - half_pixel;
        double ww = 0.0;
        double ss = 1.0 / filterscale_y;

        auto ymin = static_cast<int32_t>(center - support_y + half_pixel);
        if (ymin < 0) {
            ymin = 0;
        }

        auto ymax = static_cast<int32_t>(center + support_y + half_pixel);
        if (ymax > src_height) {
            ymax = src_height;
        }
        ymax -= ymin;

        double* k = &kk_vert[yy * ksize_vert];
        for (int32_t y = 0; y < ymax; ++y) {
            // Bilinear filter: if |x| < 1.0, return 1.0 - |x|, else return 0.0
            double filter_y = (y + ymin - center + half_pixel) * ss;
            if (filter_y < 0.0) {
                filter_y = -filter_y;
            }
            double w = (filter_y < 1.0) ? (1.0 - filter_y) : 0.0;
            k[y] = w;
            ww += w;
        }

        for (int32_t y = 0; y < ymax; ++y) {
            if (ww != 0.0) {
                k[y] /= ww;
            }
        }

        for (int32_t y = ymax; y < ksize_vert; ++y) {
            k[y] = 0;
        }

        bounds_vert[yy * 2 + 0] = ymin;
        bounds_vert[yy * 2 + 1] = ymax;
    }

    const int32_t ybox_first = bounds_vert[0];
    const int32_t ybox_last = bounds_vert[target_height * 2 - 2] + bounds_vert[target_height * 2 - 1];
    std::vector<uint8_t> temp_buffer(3 * (ybox_last - ybox_first) * target_width);

    std::vector<double> kk_horiz_norm(kk_horiz.begin(), kk_horiz.end());
    constexpr auto shifted_coeff = static_cast<double>(1U << precision_bits);
    for (auto& k : kk_horiz_norm) {
        if (k < 0) {
            k = trunc(-half_pixel + k * shifted_coeff);
        } else {
            k = trunc(half_pixel + k * shifted_coeff);
        }
    }

    for (int32_t yy = 0; yy < ybox_last - ybox_first; ++yy) {
        for (int32_t xx = 0; xx < target_width; ++xx) {
            const int32_t xmin = bounds_horiz[xx * 2 + 0];
            const int32_t xmax = bounds_horiz[xx * 2 + 1];
            const double* k = &kk_horiz_norm[xx * ksize_horiz];

            for (int32_t c = 0; c < 3; ++c) {
                double ss = static_cast<double>(1U << (precision_bits - 1U)); // init_buffer
                for (int32_t x = 0; x < xmax; ++x) {
                    ss += static_cast<double>(src.buf[3 * ((yy + ybox_first) * src_width + (x + xmin)) + c]) * k[x];
                }

                auto saturate_val = static_cast<intmax_t>(ss) >> precision_bits;
                if (saturate_val < 0) {
                    temp_buffer[3 * (yy * target_width + xx) + c] = 0;
                } else if (saturate_val > 255) {
                    temp_buffer[3 * (yy * target_width + xx) + c] = 255;
                } else {
                    temp_buffer[3 * (yy * target_width + xx) + c] = static_cast<uint8_t>(saturate_val);
                }
            }
        }
    }

    for (int32_t i = 0; i < target_height; ++i) {
        bounds_vert[i * 2] -= ybox_first;
    }

    std::vector<double> kk_vert_norm(kk_vert.begin(), kk_vert.end());
    for (auto& k : kk_vert_norm) {
        if (k < 0) {
            k = trunc(-half_pixel + k * shifted_coeff);
        } else {
            k = trunc(half_pixel + k * shifted_coeff);
        }
    }

    for (int32_t yy = 0; yy < target_height; ++yy) {
        for (int32_t xx = 0; xx < target_width; ++xx) {
            const int32_t ymin = bounds_vert[yy * 2 + 0];
            const int32_t ymax = bounds_vert[yy * 2 + 1];
            const double* k = &kk_vert_norm[yy * ksize_vert];

            for (int32_t c = 0; c < 3; ++c) {
                double ss = static_cast<double>(1U << (precision_bits - 1U)); // init_buffer
                for (int32_t y = 0; y < ymax; ++y) {
                    ss += static_cast<double>(temp_buffer[3 * ((y + ymin) * target_width + xx) + c]) * k[y];
                }

                auto saturate_val = static_cast<intmax_t>(ss) >> precision_bits;
                if (saturate_val < 0) {
                    dst.buf[3 * (yy * target_width + xx) + c] = 0;
                } else if (saturate_val > 255) {
                    dst.buf[3 * (yy * target_width + xx) + c] = 255;
                } else {
                    dst.buf[3 * (yy * target_width + xx) + c] = static_cast<uint8_t>(saturate_val);
                }
            }
        }
    }
}

// Normalize image to float32 - careful with pytorch .to(model.device, dtype=torch.float16) - this sometimes reduces precision (32>16>32), sometimes not
static void normalize_image_u8_to_f32(const clip_image_u8* src, clip_image_f32* dst, const float mean[3], const float std[3]) {
    dst->nx = src->nx;
    dst->ny = src->ny;
    dst->buf.resize(src->buf.size());

    for (size_t i = 0; i < src->buf.size(); ++i) {
        int c = i % 3; // rgb
        dst->buf[i] = (static_cast<float>(src->buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

static void normalize_image_f32(const clip_image_f32* src, clip_image_f32* dst, const float mean[3], const float std[3]) {
    dst->nx = src->nx;
    dst->ny = src->ny;
    dst->buf.resize(src->buf.size());

    for (size_t i = 0; i < src->buf.size(); ++i) {
        int c = i % 3; // rgb
        dst->buf[i] = (src->buf[i] - mean[c]) / std[c];
    }
}

static void rescale_image_u8_to_f32(const clip_image_u8* src, clip_image_f32* dst, const double scale) {
    dst->nx = src->nx;
    dst->ny = src->ny;
    dst->buf.resize(src->buf.size());

    for (size_t i = 0; i < src->buf.size(); ++i) {
        dst->buf[i] = static_cast<float>(src->buf[i]) * scale;
    }
}

inline int clip(int x, int lower, int upper) {
    return std::max(lower, std::min(x, upper));
}

static bool bicubic_resize(const clip_image_u8 &img, clip_image_u8 &dst, int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)target_width;
    ty = (float)ny / (float)target_height;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (i = 0; i < target_height; i++) {
        for (j = 0; j < target_width; j++) {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            for (k = 0; k < 3; k++) {
                for (jj = 0; jj <= 3; jj++) {
                    d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                }
            }
        }
    }

    return true;
}

// llava-1.6 type of resize_and_pad (black)
static void resize_and_pad_image(const clip_image_u8& image, clip_image_u8 &image_output, const std::pair<int, int>& target_resolution) {
    int target_width = target_resolution.first;
    int target_height = target_resolution.second;

    float scale_w = static_cast<float>(target_width) / image.nx;
    float scale_h = static_cast<float>(target_height) / image.ny;

    int new_width, new_height;

    if (scale_w < scale_h) {
        new_width = target_width;
        new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
    } else {
        new_height = target_height;
        new_width = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
    }

    clip_image_u8 resized_image;
    // bilinear_resize(image, resized_image, new_width, new_height);
    bicubic_resize(image, resized_image, new_width, new_height);

    clip_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.resize(3 * target_width * target_height, 0); // Initialize with black

    // Calculate padding offsets
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;

    // Copy the resized image into the center of the padded buffer
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] = resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    image_output = std::move(padded_image);
}

/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static std::pair<int, int> select_best_resolution(const std::pair<int, int> & original_size, const std::vector<std::pair<int, int>> & possible_resolutions) {
    int original_width = original_size.first;
    int original_height = original_size.second;
    std::pair<int, int> best_fit;
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width = resolution.first;
        int height = resolution.second;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // LOG_INF("resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

static std::vector<clip_image_u8*> divide_to_patches_u8(const clip_image_u8 & image, int patch_size) {
    std::vector<clip_image_u8*> patches;
    int width = image.nx;
    int height = image.ny;
    for (int i = 0; i < height; i += patch_size) {
        for (int j = 0; j < width; j += patch_size) {
            clip_image_u8 *patch = clip_image_u8_init();
            patch->nx = std::min(patch_size, width - j);
            patch->ny = std::min(patch_size, height - i);
            patch->buf.resize(3 * patch->nx * patch->ny);
            for (int y = 0; y < patch->ny; ++y) {
                for (int x = 0; x < patch->nx; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        patch->buf[3 * (y * patch->nx + x) + c] = image.buf[3 * ((i + y) * width + (j + x)) + c];
                    }
                }
            }
            patches.push_back(patch);
        }
    }
    return patches;
}

static int ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

static std::pair<int, int> uhd_find_best_resize(std::pair<int, int> original_size, int scale_resolution, int patch_size, bool allow_upscale = false) {
    int width = original_size.first;
    int height = original_size.second;
    if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
        float r = static_cast<float>(width) / height;
        height = static_cast<int>(scale_resolution / std::sqrt(r));
        width = static_cast<int>(height * r);
    }
    int best_width = ensure_divide(width, patch_size);
    int best_height = ensure_divide(height, patch_size);
    return std::make_pair(best_width, best_height);
}

static std::pair<int, int> uhd_get_refine_size(std::pair<int, int> original_size, std::pair<int, int> grid, int scale_resolution, int patch_size, bool allow_upscale = false) {
    int width, height;
    std::tie(width, height) = original_size;
    int grid_x, grid_y;
    std::tie(grid_x, grid_y) = grid;

    int refine_width = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);

    int grid_width = refine_width / grid_x;
    int grid_height = refine_height / grid_y;

   // auto best_grid_size = find_best_resize(std::make_tuple(grid_width, grid_height), scale_resolution, patch_size, allow_upscale); (old line)
    auto best_grid_size = uhd_find_best_resize(std::make_pair(grid_width, grid_height), scale_resolution, patch_size, allow_upscale); // (new line) => fixes conversion for make_tuple to make_pair
    int best_grid_width, best_grid_height;
    std::tie(best_grid_width, best_grid_height) = best_grid_size;

  //  std::pair<int, int> refine_size = std::make_tuple(best_grid_width * grid_x, best_grid_height * grid_y); (old line)
    std::pair<int, int> refine_size = std::make_pair(best_grid_width * grid_x, best_grid_height * grid_y); // (new line)
    return refine_size;
}

static std::pair<int, int> uhd_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
    std::vector<int> candidate_split_grids_nums;
    for (int i : {multiple - 1, multiple, multiple + 1}) {
        if (i == 1 || i > max_slice_nums) {
            continue;
        }
        candidate_split_grids_nums.push_back(i);
    }

    std::vector<std::pair<int, int>> candidate_grids;
    for (int split_grids_nums : candidate_split_grids_nums) {
        int m = 1;
        while (m <= split_grids_nums) {
            if (split_grids_nums % m == 0) {
                candidate_grids.emplace_back(m, split_grids_nums / m);
            }
            ++m;
        }
    }

    std::pair<int, int> best_grid{1, 1};
    float min_error = std::numeric_limits<float>::infinity();
    for (const auto& grid : candidate_grids) {
        float error = std::abs(log_ratio - std::log(1.0 * grid.first / grid.second));
        if (error < min_error) {
            best_grid = grid;
            min_error = error;
        }
    }
    return best_grid;
}

// inspired from LLaVA-UHD:
//    -> https://arxiv.org/pdf/2403.11703
//    -> https://github.com/thunlp/LLaVA-UHD
//    -> https://github.com/thunlp/LLaVA-UHD/blob/302301bc2175f7e717fb8548516188e89f649753/llava_uhd/train/llava-uhd/slice_logic.py#L118
static std::vector<std::vector<clip_image_u8 *>> uhd_slice_image(const clip_image_u8 * img, const int max_slice_nums=9, const int scale_resolution=448, const int patch_size=14) {
    const std::pair<int, int> original_size={img->nx,img->ny};
    const int original_width = img->nx;
    const int original_height = img->ny;
    const float log_ratio = log(1.0*original_width/original_height);
    const float ratio = 1.0 * original_width * original_height/ (scale_resolution * scale_resolution);
    const int multiple = fmin(ceil(ratio), max_slice_nums);

    std::vector<std::vector<clip_image_u8 *>> images;
    LOG_INF("%s: multiple %d\n", __func__, multiple);
    images.push_back(std::vector<clip_image_u8 *>());

    if (multiple <= 1) {
        auto best_size = uhd_find_best_resize(original_size, scale_resolution, patch_size, true);
        clip_image_u8 * source_image = clip_image_u8_init();
        bicubic_resize(*img, *source_image, best_size.first, best_size.second);
        // source_image = image.resize(best_size, Image.Resampling.BICUBIC)
        images[images.size()-1].push_back(source_image);
    }
    else if (multiple > 1) {
        auto best_size = uhd_find_best_resize(original_size, scale_resolution, patch_size);
        clip_image_u8 * source_image = clip_image_u8_init();
        bicubic_resize(*img, *source_image, best_size.first, best_size.second);
        // source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        LOG_INF("%s: image_size: %d %d; source_image size: %d %d\n", __func__, img->nx, img->ny, best_size.first, best_size.second);
        images[images.size()-1].push_back(source_image);

        std::pair<int, int> best_grid = uhd_best_grid(max_slice_nums, multiple, log_ratio);
        LOG_INF("%s: image_size: %d %d; best_grid: %d %d\n", __func__, img->nx, img->ny, best_grid.first, best_grid.second);

        auto refine_size = uhd_get_refine_size(original_size, best_grid, scale_resolution, patch_size, true);
        clip_image_u8 * refine_image = clip_image_u8_init();
        bicubic_resize(*img, *refine_image, refine_size.first, refine_size.second);

        LOG_INF("%s: refine_image_size: %d %d; refine_size: %d %d\n", __func__, refine_image->nx, refine_image->ny, refine_size.first, refine_size.second);

        // split_to_patches
        int width = refine_image->nx;
        int height = refine_image->ny;
        int grid_x = int(width / best_grid.first);
        int grid_y = int(height / best_grid.second);
        for (int patches_i = 0, ic = 0; patches_i < height && ic < best_grid.second; patches_i += grid_y, ic += 1){
            images.push_back(std::vector<clip_image_u8 *>());
            for(int patches_j = 0, jc = 0; patches_j < width && jc < best_grid.first; patches_j += grid_x, jc += 1){
                clip_image_u8 * patch = clip_image_u8_init();
                patch->nx = grid_x;
                patch->ny = grid_y;
                patch->buf.resize(3 * patch->nx * patch->ny);
                for (int y = patches_i; y < patches_i + grid_y; ++y) {
                    for (int x = patches_j; x < patches_j + grid_x; ++x) {
                        const int i = 3 * (y * refine_image->nx + x);
                        const int j = 3 * ((y-patches_i) * patch->nx + (x-patches_j));
                        patch->buf[j]   = refine_image->buf[i];
                        patch->buf[j+1] = refine_image->buf[i+1];
                        patch->buf[j+2] = refine_image->buf[i+2];
                    }
                }
                images[images.size()-1].push_back(patch);
            }
        }
        clip_image_u8_free(refine_image);
    }
    return images;
}

int clip_uhd_num_image_embeds_col(struct clip_ctx * ctx_clip) {
    const int max_slice_nums=9;
    const int scale_resolution=448;
    const int original_width = ctx_clip->load_image_size->width;
    const int original_height = ctx_clip->load_image_size->height;
    const float log_ratio = log(1.0*original_width/original_height);
    const float ratio = 1.0 * original_width * original_height/ (scale_resolution * scale_resolution);
    const int multiple = fmin(ceil(ratio), max_slice_nums);
    std::pair<int, int> best_grid = uhd_best_grid(max_slice_nums, multiple, log_ratio);
    return best_grid.first;
}

// returns the normalized float tensor for llava-1.5, for spatial_unpad with anyres processing for llava-1.6 it returns the normalized image patch tensors as a vector
// res_imgs memory is being allocated here, previous allocations will be freed if found
bool clip_image_preprocess(struct clip_ctx * ctx, const clip_image_u8 * img, clip_image_f32_batch * res_imgs) {

    {
        res_imgs->size = 1;
        res_imgs->data = new clip_image_f32[res_imgs->size];
        clip_image_u8 resized_image;
        int32_t sz=ctx->vision_model.hparams.image_size;
        bilinear_resize(*img, resized_image,sz,sz);
        clip_image_f32 * res = clip_image_f32_init();
        //clip_image_save_to_bmp(resized_image, "resized.bmp");
        // normalize_image_u8_to_f32(&resized_image, res, ctx->image_mean, ctx->image_std);
        rescale_image_u8_to_f32(&resized_image, res, 0.00392156862745098);
        normalize_image_f32(res, res, ctx->image_mean, ctx->image_std);
        res_imgs->data[0] = *res;
        clip_image_f32_free(res);
        return true;
    }
}

ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx) {
    return ctx->vision_model.image_newline;
}

void clip_free(clip_ctx * ctx) {
    ggml_free(ctx->ctx_data);
    gguf_free(ctx->ctx_gguf);

    ggml_backend_buffer_free(ctx->params_buffer);
    ggml_backend_free(ctx->backend);
    ggml_gallocr_free(ctx->compute_alloc);
    delete ctx;
}

size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    int extra_tokens = 0;
    return (clip_n_patches(ctx) + extra_tokens) * clip_n_mmproj_embd(ctx) * sizeof(float);
}

size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_h, int img_w) {
    clip_image_f32 img;
    img.nx = img_w;
    img.ny = img_h;
    return clip_n_patches_by_img(ctx, &img) * clip_n_mmproj_embd(ctx) * sizeof(float);
}

int32_t clip_image_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.image_size;
}

int32_t clip_patch_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.patch_size;
}

int32_t clip_hidden_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.hidden_size;
}

const char * clip_patch_merge_type(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.mm_patch_merge_type;
}

const int32_t * clip_image_grid(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.image_grid_pinpoints;
}

int clip_n_patches(const struct clip_ctx * ctx) {
    clip_image_f32 img;
    img.nx = ctx->vision_model.hparams.image_size;
    img.ny = ctx->vision_model.hparams.image_size;
    return clip_n_patches_by_img(ctx, &img);
}

int clip_n_patches_by_img(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->vision_model.hparams;

    int n_patches = (params.image_size / params.patch_size) * (params.image_size / params.patch_size);

    return n_patches;
}

static std::vector<std::vector<std::vector<float>>> get_1d_sincos_pos_embed_from_grid_new(int embed_dim, const std::vector<std::vector<float>> & pos) {
    assert(embed_dim % 2 == 0);
    int H = pos.size();
    int W = pos[0].size();

    std::vector<float> omega(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; ++i) {
        omega[i] = 1.0 / pow(10000.0, static_cast<float>(i) / (embed_dim / 2));
    }

    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                float out_value = pos[h][w] * omega[d];
                emb[h][w][d] = sin(out_value);
                emb[h][w][d + embed_dim / 2] = cos(out_value);
            }
        }
    }

    return emb;
}

static std::vector<std::vector<std::vector<float>>> get_2d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<std::vector<std::vector<float>>> & grid) {
    assert(embed_dim % 2 == 0);
    std::vector<std::vector<std::vector<float>>> emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[0]); // (H, W, D/2)
    std::vector<std::vector<std::vector<float>>> emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[1]); // (H, W, D/2)

    int H = emb_h.size();
    int W = emb_h[0].size();
    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                emb[h][w][d] = emb_h[h][w][d];
                emb[h][w][d + embed_dim / 2] = emb_w[h][w][d];
            }
        }
    }
    return emb;
}

static std::vector<std::vector<float>> get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int> image_size) {
    int grid_h_size = image_size.first;
    int grid_w_size = image_size.second;

    std::vector<float> grid_h(grid_h_size);
    std::vector<float> grid_w(grid_w_size);

    for (int i = 0; i < grid_h_size; ++i) {
        grid_h[i] = static_cast<float>(i);
    }
    for (int i = 0; i < grid_w_size; ++i) {
        grid_w[i] = static_cast<float>(i);
    }

    std::vector<std::vector<float>> grid(grid_h_size, std::vector<float>(grid_w_size));
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid[h][w] = grid_w[w];
        }
    }
    std::vector<std::vector<std::vector<float>>> grid_2d = {grid, grid};
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid_2d[0][h][w] = grid_h[h];
            grid_2d[1][h][w] = grid_w[w];
        }
    }

    std::vector<std::vector<std::vector<float>>> pos_embed_3d = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_2d);

    int H = image_size.first;
    int W = image_size.second;
    std::vector<std::vector<float>> pos_embed_2d(H * W, std::vector<float>(embed_dim));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            pos_embed_2d[w * H + h] = pos_embed_3d[h][w];
        }
    }

    return pos_embed_2d;
}

bool clip_image_encode(struct clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    if (!ctx->has_vision_encoder) {
        LOG_ERR("This gguf file seems to have no vision encoder\n");
        return false;
    }

    clip_image_f32_batch imgs{};
    imgs.size = 1;
    imgs.data = img;
    return clip_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool clip_image_batch_encode(clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs, float * vec) {
    if (!ctx->has_vision_encoder) {
        LOG_ERR("This gguf file seems to have no vision encoder\n");
        return false;
    }

    int batch_size = imgs->size;
    GGML_ASSERT(batch_size == 1); // TODO: support multiple images

    // build the inference graph
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs, ctx->load_image_size, true);
    ggml_gallocr_alloc_graph(ctx->compute_alloc, gf);

    // set inputs
    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size = hparams.image_size;
    int image_size_width  = image_size;
    int image_size_height = image_size;

    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int num_positions = num_patches + (ctx->has_class_embedding ? 1 : 0);

    {
        struct ggml_tensor * inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        float * data = (float *)malloc(ggml_nbytes(inp_raw));

        for (size_t i = 0; i < imgs->size; i++) {
            const int nx = imgs->data[i].nx;
            const int ny = imgs->data[i].ny;

            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < 3; k++) {
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            data[(b * 3 * n) + k * n + y * nx + x] = imgs->data[b].buf[3 * (y * nx + x) + k];
                        }
                    }
                }
            }
        }
        ggml_backend_tensor_set(inp_raw, data, 0, ggml_nbytes(inp_raw));
        free(data);
    }

    {
        {
            if (ctx->has_class_embedding) {
                struct ggml_tensor * embeddings = ggml_graph_get_tensor(gf, "embeddings");

                void* zero_mem = malloc(ggml_nbytes(embeddings));
                memset(zero_mem, 0, ggml_nbytes(embeddings));
                ggml_backend_tensor_set(embeddings, zero_mem, 0, ggml_nbytes(embeddings));
                free(zero_mem);
            }
        }

        {
            struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "positions");

            int* positions_data = (int*)malloc(ggml_nbytes(positions));
            for (int i = 0; i < num_positions; i++) {
                positions_data[i] = i;
            }
            ggml_backend_tensor_set(positions, positions_data, 0, ggml_nbytes(positions));
            free(positions_data);
        }
    }

    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }

    ggml_backend_graph_compute(ctx->backend, gf);

    // the last node is the embedding tensor
    struct ggml_tensor * embeddings = ggml_graph_node(gf, -1);

    // copy the embeddings to the location passed by the user
    ggml_backend_tensor_get(embeddings, vec, 0, ggml_nbytes(embeddings));

    return true;
}

bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype) {
    assert(itype < GGML_TYPE_COUNT);
    ggml_type type = static_cast<ggml_type>(itype);

    auto * ctx_clip = clip_model_load(fname_inp, NULL, 2);

    const auto & ctx_src = ctx_clip->ctx_gguf;
    const auto & ctx_data = ctx_clip->ctx_data;

    auto * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_src);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", itype);

    auto fout = std::ofstream(fname_out, std::ios::binary);

    const int n_tensors = gguf_get_n_tensors(ctx_src);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
        gguf_add_tensor(ctx_out, cur);
    }

    const size_t meta_size = gguf_get_meta_size(ctx_out);
    for (size_t i = 0; i < meta_size; ++i) {
        fout.put(0);
    }

    // regexes of tensor names to be quantized
    const std::vector<std::string> k_names = {
        ".*weight",
    };

    std::vector<uint8_t> work(512);
    std::vector<float> conv_buf(512);
    size_t total_size_org = 0;
    size_t total_size_new = 0;

    for (int i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name.c_str());

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;

        bool quantize = false;
        for (const auto & s : k_names) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // quantize only 2D tensors and bigger than block size
        quantize &= (ggml_n_dims(cur) == 2) && cur->ne[0] > ggml_blck_size(type);

        if (quantize) {
            new_type = type;
            if (new_type >= GGML_TYPE_Q2_K && name.find("embd") != std::string::npos) {
                new_type = GGML_TYPE_Q8_0; // ggml_get_rows needs non K type
                // LOG_ERR("%s: quantizing %s to %s\n", __func__, name.c_str(), ggml_type_name(new_type));
            }
            const size_t n_elms = ggml_nelements(cur);
            float * f32_data;

            switch (cur->type) {
            case GGML_TYPE_F32:
                f32_data = (float *)cur->data;
                break;
            case GGML_TYPE_F16:
                if (conv_buf.size() < n_elms) {
                    conv_buf.resize(n_elms);
                }
                for (size_t j = 0; j < n_elms; ++j) {
                    conv_buf[j] = ggml_fp16_to_fp32(((ggml_fp16_t *)cur->data)[j]);
                }
                f32_data = (float *)conv_buf.data();
                break;
            default:
                LOG_ERR("Please use an input file in f32 or f16\n");
                gguf_free(ctx_out);
                return false;
            }

            if (work.size() < n_elms * 4) {
                work.resize(n_elms * 4);
            }
            new_data = work.data();

            new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, n_elms/cur->ne[0], cur->ne[0], nullptr);
        } else {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }
        const size_t orig_size = ggml_nbytes(cur);
        total_size_org += orig_size;
        total_size_new += new_size;
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        GGML_ASSERT(gguf_get_tensor_size(ctx_out, gguf_find_tensor(ctx_out, name.c_str())) == new_size);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data);
        fout.write((const char *)new_data, new_size);
        size_t pad = GGML_PAD(new_size, gguf_get_alignment(ctx_out)) - new_size;
        for (size_t j = 0; j < pad; ++j) {
            fout.put(0);
        }

        LOG_INF("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), ggml_n_dims(cur), quantize,
               orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
    }

    // go back to beginning of file and write the updated metadata
    fout.seekp(0, std::ios::beg);
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(ctx_out, meta.data());
    fout.write((const char *)meta.data(), meta_size);

    fout.close();

    clip_free(ctx_clip);
    gguf_free(ctx_out);

    {
        LOG_INF("%s: original  size = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        LOG_INF("%s: quantized size = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);
    }

    return true;
}

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    return ctx->vision_model.mm_2_b->ne[0];
}

bool clip_encode_float_image (struct clip_ctx * ctx, int n_threads, float * img, int h, int w, float * vec) {
    clip_image_f32 clip_img;
    clip_img.buf.resize(h * w * 3);
    for (int i = 0; i < h*w*3; i++)
    {
        clip_img.buf[i] = img[i];
    }
    clip_img.nx = w;
    clip_img.ny = h;
    clip_image_encode(ctx, n_threads, &clip_img, vec);
    return true;
}
