#include <fstream>
#include <filesystem>
#include <thread>

#include "backend.h"
#include "llama_cpp_backend.h"
#include "llama.h"
#include "llama-model.h"
#include "commondef.h"
#include "logger.h"

#if ENABLE_VISION
#include "llava.h"
#endif

namespace rwkvmobile {

#ifdef ENABLE_LLAMACPP
int llama_cpp_backend::init(void * extra) {

    return RWKV_SUCCESS;
}

int llama_cpp_backend::load_model(std::string model_path) {
    llama_model_params model_params = llama_model_default_params();

#if defined(__APPLE__) || defined(__MACH__)
    model_params.n_gpu_layers = 99;
#else
    model_params.n_gpu_layers = 0;
#endif

    LOGI("n_gpu_layers: %d", model_params.n_gpu_layers);
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1048576;
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

// #ifdef __ANDROID__
//     // TODO: set according to the number of prime cores on the device
//     llama_set_n_threads(ctx, 2, 2);
// #endif

    vocab_size = model->vocab.n_tokens();
    hidden_size = llama_model_n_embd(model);
    num_heads = hidden_size / 64;
    n_layers = llama_model_n_layer(model);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::eval(int id, float *& logits) {
    llama_batch batch = llama_batch_get_one(&id, 1);
    llama_decode(ctx, batch);

    float * logits_out = llama_get_logits_ith(ctx, -1);
    if (!logits_out) {
        return RWKV_ERROR_EVAL;
    }
    logits = logits_out;

    return RWKV_SUCCESS;
}

int llama_cpp_backend::eval(std::vector<int> ids, float *& logits, bool skip_logits_copy) {
    llama_batch batch = llama_batch_get_one(ids.data(), ids.size());
    llama_decode(ctx, batch);
    float * logits_out = llama_get_logits_ith(ctx, -1);
    if (!logits_out) {
        return RWKV_ERROR_EVAL;
    }
    logits = logits_out;

    return RWKV_SUCCESS;
}

int llama_cpp_backend::eval_with_embeddings(const float *embeddings, int n_tokens, float *& logits) {
    int n_embd = llama_model_n_embd(model);
  
    // llava_embd_batch llava_batch = llava_embd_batch(embd, n_eval, n_past, 0);
    llama_batch batch = {
        /*n_tokens       =*/ n_tokens,
        /*tokens         =*/ nullptr,
        /*embd           =*/ (float *)embeddings,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
    };
    llama_decode(ctx, batch);
    float * logits_out = llama_get_logits_ith(ctx, -1);
    if (!logits_out) {
        return RWKV_ERROR_EVAL;
    }
    logits = logits_out;

    return RWKV_SUCCESS;
}

bool llama_cpp_backend::is_available() {
    return true;
}

int llama_cpp_backend::clear_state() {
    llama_memory_clear(llama_get_memory(ctx), true);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::get_state(std::any &state) {
    std::vector<uint8_t> state_mem(llama_state_get_size(ctx));
    llama_state_get_data(ctx, state_mem.data(), state_mem.size());
    state = std::move(state_mem);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::set_state(std::any state) {
    try {
        std::vector<uint8_t> state_mem = std::any_cast<std::vector<uint8_t>>(state);
        llama_state_set_data(ctx, state_mem.data(), state_mem.size());
    } catch (const std::bad_any_cast &e) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
}

int llama_cpp_backend::free_state(std::any state) {
    try {
        std::vector<uint8_t> state_mem = std::any_cast<std::vector<uint8_t>>(state);
        state_mem.clear();
    } catch (const std::bad_any_cast &e) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
}

int llama_cpp_backend::release_model() {
    if (ctx)
        llama_free(ctx);
    if (model)
        llama_model_free(model);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::release() {
    return RWKV_SUCCESS;
}

#else

int llama_cpp_backend::init(void * extra) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::load_model(std::string model_path) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::eval(int id, float *& logits) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::eval(std::vector<int> ids, float *& logits, bool skip_logits_copy) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::clear_state() {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::get_state(std::any &state) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::set_state(std::any state) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::free_state(std::any state) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::release_model() {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::release() {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}


bool llama_cpp_backend::is_available() {
    return false;
}

#endif

} // namespace rwkvmobile