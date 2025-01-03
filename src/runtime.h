#ifndef RUNTIME_H
#define RUNTIME_H

#include <string>
#include <map>
#include <memory>
#include "backend.h"
#include "tokenizer.h"
#include "sampler.h"

namespace rwkvmobile {

class runtime {
public:
    runtime() {};
    ~runtime() {};
    int init(std::string backend_name);
    int init(int backend_id);
    int load_model(std::string model_path);
    int load_tokenizer(std::string vocab_file);
    int eval_logits(int id, std::vector<float> &logits);
    int eval_logits(std::vector<int> ids, std::vector<float> &logits);

    // without history
    int chat(std::string input, std::string &response, const int max_length, void (*callback)(const char *) = nullptr);

    // with history
    int chat(std::vector<std::string> inputs, std::string &response, const int max_length, void (*callback)(const char *) = nullptr);
    int gen_completion(std::string prompt, std::string &completion, int length);

    int clear_state() {
        if (_backend == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        _occurences.clear();
        auto ptr = _state_head->next;
        while (ptr) {
            auto tmp = ptr;
            ptr = ptr->next;
            _backend->free_state(tmp->state);
            delete tmp;
        }
        return _backend->clear_state();
    }

    int release() {
        if (_backend == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        clear_state();
        int ret = _backend->release_model();
        if (ret != RWKV_SUCCESS) {
            return ret;
        }
        return _backend->release();
    }

    inline int set_seed(int64_t seed) {
        if (_sampler == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        _sampler->set_seed(seed);
        _seed = seed;
        return 0;
    }

    inline int64_t get_seed() { return _seed; }

    inline void set_user_role(std::string role) { user_role = role; }
    inline void set_response_role(std::string role) { response_role = role; }
    std::string get_user_role() { return user_role; }
    std::string get_response_role() { return response_role; }

    inline std::vector<std::string> get_stop_codes() { return _stop_codes; }
    inline void set_stop_codes(std::vector<std::string> stop_codes) { _stop_codes = stop_codes; }

    inline void set_sampler_params(float temperature, int top_k, float top_p) {
        _temperature = temperature;
        _top_k = top_k;
        _top_p = top_p;
    }

    inline void set_penalty_params(float presence_penalty, float frequency_penalty, float penalty_decay) {
        _presence_penalty = presence_penalty;
        _frequency_penalty = frequency_penalty;
        _penalty_decay = penalty_decay;
    }

    inline float get_temperature() { return _temperature; }
    inline int get_top_k() { return _top_k; }
    inline float get_top_p() { return _top_p; }
    inline float get_presence_penalty() { return _presence_penalty; }
    inline float get_frequency_penalty() { return _frequency_penalty; }
    inline float get_penalty_decay() { return _penalty_decay; }

    std::string get_available_backends_str();
    int get_available_backend_ids(std::vector<int> &backend_ids);
    std::string backend_id_to_str(int backend_id) {
        return backend_enum_to_str(backend_id);
    }
    int backend_str_to_id(std::string backend_str) {
        return backend_str_to_enum(backend_str);
    }

    std::vector<int> tokenizer_encode(std::string text) {
        if (_tokenizer == nullptr) {
            return {};
        }
        return _tokenizer->encode(text);
    }

    std::string tokenizer_decode(std::vector<int> ids) {
        if (_tokenizer == nullptr) {
            return "";
        }
        return _tokenizer->decode(ids);
    }

    std::string tokenizer_decode(int id) {
        if (_tokenizer == nullptr) {
            return "";
        }
        return _tokenizer->decode(id);
    }

    int sampler_sample(std::vector<float> logits) {
        if (_sampler == nullptr) {
            return -1;
        }
        return _sampler->sample(logits.data(), logits.size(), _temperature, _top_k, _top_p);
    }

    struct state_node {
        std::any state;
        unsigned long long hash = 0;
        struct state_node * next = nullptr;
    } * _state_head = nullptr;

private:
    std::unique_ptr<execution_provider> _backend;
    std::unique_ptr<tokenizer_base> _tokenizer;
    std::unique_ptr<sampler> _sampler;

    int _vocab_size = 65536;

    float _temperature = 1.0;
    int _top_k = 128;
    float _top_p = 0.3;
    float _presence_penalty = 0.0;
    float _frequency_penalty = 1.0;
    float _penalty_decay = 0.996;
    int64_t _seed = 42;
    std::string user_role = "User";
    std::string response_role = "Assistant";

    std::vector<std::string> _stop_codes = {"\n\n", "Assistant:", "User:"};

    std::map<int, float> _occurences;

    unsigned long long hash_string(std::string str) {
        unsigned long long hash = 0, p = 13131;
        for (auto c : str) {
            hash = hash * p + c;
        }
        return hash;
    }
};

}

#endif
