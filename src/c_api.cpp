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
    int ret = rt->release();
    delete rt;
    return ret;
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
    const int max_length,
    const int stop_code,
    void (*callback)(const char *, const int)) {
    if (handle == nullptr || prompt == nullptr || completion == nullptr || max_length <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class runtime *>(handle);
    std::string completion_str;
    int ret = rt->gen_completion(
        std::string(prompt),
        completion_str,
        max_length,
        stop_code,
        callback);
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

int rwkvmobile_runtime_get_available_backend_names(char * backend_names_buffer, int buffer_size) {
    if (backend_names_buffer == nullptr || buffer_size <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    runtime * rt = new runtime();
    if (rt == nullptr) {
        return RWKV_ERROR_ALLOC;
    }
    auto backend_names = rt->get_available_backends_str();
    if (backend_names.size() >= buffer_size) {
        return RWKV_ERROR_ALLOC;
    }
    strncpy(backend_names_buffer, backend_names.c_str(), buffer_size);
    delete rt;
    return RWKV_SUCCESS;
}

struct sampler_params rwkvmobile_runtime_get_sampler_params(rwkvmobile_runtime_t runtime) {
    struct sampler_params params;
    params.temperature = 0;
    params.top_k = 0;
    params.top_p = 0;
    if (runtime == nullptr) {
        return params;
    }
    auto rt = static_cast<class runtime *>(runtime);
    params.temperature = rt->get_temperature();
    params.top_k = rt->get_top_k();
    params.top_p = rt->get_top_p();
    return params;
}

void rwkvmobile_runtime_set_sampler_params(rwkvmobile_runtime_t runtime, struct sampler_params params) {
    if (runtime == nullptr) {
        return;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_sampler_params(params.temperature, params.top_k, params.top_p);
}

struct penalty_params rwkvmobile_runtime_get_penalty_params(rwkvmobile_runtime_t runtime) {
    struct penalty_params params;
    params.presence_penalty = 0;
    params.frequency_penalty = 0;
    params.penalty_decay = 0;
    if (runtime == nullptr) {
        return params;
    }
    auto rt = static_cast<class runtime *>(runtime);
    params.presence_penalty = rt->get_presence_penalty();
    params.frequency_penalty = rt->get_frequency_penalty();
    params.penalty_decay = rt->get_penalty_decay();
    return params;
}

void rwkvmobile_runtime_set_penalty_params(rwkvmobile_runtime_t runtime, struct penalty_params params) {
    if (runtime == nullptr) {
        return;
    }
    auto rt = static_cast<class runtime *>(runtime);
    rt->set_penalty_params(params.presence_penalty, params.frequency_penalty, params.penalty_decay);
}

int rwkvmobile_runtime_set_prompt(rwkvmobile_runtime_t runtime, const char * prompt) {
    if (runtime == nullptr || prompt == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    return rt->set_prompt(prompt);
}

int rwkvmobile_runtime_get_prompt(rwkvmobile_runtime_t runtime, char * prompt, const int buf_len) {
    if (runtime == nullptr || prompt == nullptr || buf_len <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(runtime);
    std::string prompt_str = rt->get_prompt();
    if (prompt_str.size() >= buf_len) {
        return RWKV_ERROR_ALLOC;
    }
    strncpy(prompt, prompt_str.c_str(), buf_len);
    return RWKV_SUCCESS;
}

} // extern "C"
} // namespace rwkvmobile
