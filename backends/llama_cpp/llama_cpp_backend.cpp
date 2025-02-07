#include <fstream>
#include <filesystem>

#include "backend.h"
#include "llama_cpp_backend.h"
#include "llama.h"
#include "llama-model.h"
#include "commondef.h"

namespace rwkvmobile {

#ifdef ENABLE_LLAMACPP
int llama_cpp_backend::init(void * extra) {

    return RWKV_SUCCESS;
}

int llama_cpp_backend::load_model(std::string model_path) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1048576;
    ctx_params.n_batch = 1048576;
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    vocab_size = model->vocab.n_tokens();
    hidden_size = llama_model_n_embd(model);
    num_heads = hidden_size / 64;
    n_layers = llama_model_n_layer(model);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::eval(int id, std::vector<float> &logits) {
    llama_batch batch = llama_batch_get_one(&id, 1);
    llama_decode(ctx, batch);

    float * logits_out = llama_get_logits_ith(ctx, -1);
    if (!logits_out) {
        return RWKV_ERROR_EVAL;
    }
    logits.assign(logits_out, logits_out + vocab_size);

    return RWKV_SUCCESS;
}

int llama_cpp_backend::eval(std::vector<int> ids, std::vector<float> &logits) {
    llama_batch batch = llama_batch_get_one(ids.data(), ids.size());
    llama_decode(ctx, batch);
    float * logits_out = llama_get_logits_ith(ctx, -1);
    if (!logits_out) {
        return RWKV_ERROR_EVAL;
    }
    logits.assign(logits_out, logits_out + vocab_size);

    return RWKV_SUCCESS;
}

bool llama_cpp_backend::is_available() {
    return true;
}

int llama_cpp_backend::clear_state() {
    llama_kv_cache_clear(ctx);
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

#else

int llama_cpp_backend::init(void * extra) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::load_model(std::string model_path) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::eval(int id, std::vector<float> &logits) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int llama_cpp_backend::eval(std::vector<int> ids, std::vector<float> &logits) {
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

bool llama_cpp_backend::is_available() {
    return false;
}

#endif

} // namespace rwkvmobile