#include "runtime.h"
#include "commondef.h"
#include "c_api.h"
#include <cstring>

namespace rwkvmobile {

extern "C" {

rwkvmobile_runtime_t rwkvmobile_runtime_init_with_name(const char * backend_name) {
    runtime * rt = new runtime();
    if (rt == nullptr) {
        return nullptr;
    }
    rt->init(backend_name);
    return rt;
}

int rwkvmobile_runtime_load_model(rwkvmobile_runtime_t handle, const char * model_path) {
    if (handle == nullptr || model_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->load_model(model_path);
}

int rwkvmobile_runtime_release(rwkvmobile_runtime_t handle) {
    if (handle == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->release();
}

int rwkvmobile_runtime_load_tokenizer(rwkvmobile_runtime_t handle, const char * vocab_file) {
    if (handle == nullptr || vocab_file == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->load_tokenizer(vocab_file);
}

int rwkvmobile_runtime_eval_logits(rwkvmobile_runtime_t handle, const int * ids, int ids_len, float * logits, int logits_len) {
    if (handle == nullptr || ids == nullptr || logits == nullptr || ids_len <= 0 || logits_len <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    std::vector<int> ids_vec(ids, ids + ids_len);
    std::vector<float> logits_vec(logits, logits + logits_len);
    return rt->eval_logits(ids_vec, logits_vec);
}

int rwkvmobile_runtime_eval_chat(
    rwkvmobile_runtime_t handle,
    const char * input,
    char * response,
    const int max_length,
    void (*callback)(const char *)) {
    if (handle == nullptr || input == nullptr || response == nullptr || max_length <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class runtime *>(handle);
    std::string response_str;
    int ret = rt->chat(
        std::string(input),
        response_str,
        max_length,
        callback);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    strncpy(response, response_str.c_str(), response_str.size());
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_eval_chat_with_history(
    rwkvmobile_runtime_t handle,
    const char ** inputs,
    const int num_inputs,
    char * response,
    const int max_length,
    void (*callback)(const char *)) {
    if (handle == nullptr || inputs == nullptr || num_inputs == 0 || response == nullptr || max_length <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class runtime *>(handle);
    std::vector<std::string> inputs_vec;
    for (int i = 0; i < num_inputs; i++) {
        inputs_vec.push_back(std::string(inputs[i]));
    }
    std::string response_str;
    int ret = rt->chat(
        inputs_vec,
        response_str,
        max_length,
        callback);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    strncpy(response, response_str.c_str(), response_str.size());
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_gen_completion(
    rwkvmobile_runtime_t handle,
    const char * prompt,
    char * completion,
    const int length) {
    if (handle == nullptr || prompt == nullptr || completion == nullptr || length <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class runtime *>(handle);
    std::string completion_str;
    int ret = rt->gen_completion(
        std::string(prompt),
        completion_str,
        length);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    strncpy(completion, completion_str.c_str(), completion_str.size());
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_clear_state(rwkvmobile_runtime_t handle) {
    if (handle == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->clear_state();
}

int rwkvmobile_runtime_get_available_backend_names(rwkvmobile_runtime_t handle, char * backend_names_buffer, int buffer_size) {
    if (handle == nullptr || backend_names_buffer == nullptr || buffer_size <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    auto backend_names = rt->get_available_backends_str();
    if (backend_names.size() >= buffer_size) {
        return RWKV_ERROR_ALLOC;
    }
    strncpy(backend_names_buffer, backend_names.c_str(), buffer_size);
    return RWKV_SUCCESS;
}

} // extern "C"
} // namespace rwkvmobile
