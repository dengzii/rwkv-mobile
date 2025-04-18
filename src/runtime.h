#ifndef RUNTIME_H
#define RUNTIME_H

#include <string>
#include <map>
#include <memory>
#include <functional>
#include "backend.h"
#include "tokenizer.h"
#include "sampler.h"
#include "soc_detect.h"

#include "logger.h"

#ifdef ENABLE_VISION
#include "clip.h"
#endif

#ifdef ENABLE_WHISPER
#include "whisper.h"
#endif

#ifdef ENABLE_TTS
#include "cosyvoice.h"
#endif

namespace rwkvmobile {

class runtime {
public:
    runtime() {
        _soc_detect.detect_platform();
    };
    ~runtime() {};
    int init(std::string backend_name);
    int init(int backend_id);
    int init(std::string backend_name, void * extra);
    int init(int backend_id, void * extra);
    int load_model(std::string model_path);
    int load_tokenizer(std::string vocab_file);
    int load_vision_encoder(std::string model_path);
    int load_whisper_encoder(std::string model_path);
    int eval_logits(int id, float *& logits);
    int eval_logits(std::vector<int> ids, float *& logits);
    int eval_logits_with_embeddings(const float *embeddings, int n_tokens, float *& logits);
    void free_logits_if_allocated(float *& logits) {
        if (_backend != nullptr) {
            _backend->free_logits_if_allocated(logits);
        }
    }

    // without history
    int chat(std::string input, const int max_length, void (*callback)(const char *, const int) = nullptr, bool enable_reasoning = false);

    // with history
    int chat(std::vector<std::string> inputs, const int max_length, void (*callback)(const char *, const int) = nullptr, bool enable_reasoning = false);
    int gen_completion(std::string prompt, int max_length, int stop_code, void (*callback)(const char *, const int));

    int set_prompt(std::string prompt);
    std::string get_prompt();

    std::string get_response_buffer_content() { return _response_buffer; }
    const std::vector<int32_t> get_response_buffer_ids() { return _response_buffer_ids; }
#ifdef ENABLE_VISION
    int set_image_prompt(std::string path);
#endif

#ifdef ENABLE_WHISPER
    int set_audio_prompt(std::string path);
#endif

#ifdef ENABLE_TTS
    int cosyvoice_load_models(
        std::string speech_tokenizer_path,
        std::string campplus_path,
        std::string flow_encoder_path,
        std::string flow_decoder_estimator_path,
        std::string hift_generator_path,
        std::string tts_tokenizer_path,
        std::string spk_info_path
    );

    int cosyvoice_release_models();
    int run_tts_internal(std::string tts_text, std::string instruction_text,
        std::vector<int> speech_tokens, std::vector<std::vector<float>> speech_features, std::vector<float> speech_embedding,
        std::vector<float> &output_samples);
    int run_tts(std::string tts_text, std::string instruction_text, std::string prompt_wav_path, std::string output_wav_path);
    int run_tts_with_predefined_spks(std::string tts_text, std::string instruction_text, std::string spks_name, std::string output_wav_path);

    std::string cosyvoice_get_spk_names();
#endif

    int clear_state() {
        if (_backend == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        _occurences.clear();
        auto ptr = _state_head->next;
        while (ptr) {
            auto tmp = ptr;
            ptr = ptr->next;
            if (tmp->state.has_value()) {
                _backend->free_state(tmp->state);
            }
            delete tmp;
        }
        _state_head->next = nullptr;
        return _backend->clear_state();
    }

    int release() {
        if (_backend == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        clear_state();
        if (_state_head != nullptr && _state_head->state.has_value()) {
            _backend->free_state(_state_head->state);
            delete _state_head;
            _state_head = nullptr;
        }
        int ret = _backend->release_model();
        if (ret != RWKV_SUCCESS) {
            return ret;
        }
        _tokenizer = nullptr;
        _sampler = nullptr;
        return _backend->release();
    }

    int release_vision_encoder();
    int release_whisper_encoder();

    inline int set_seed(int64_t seed) {
        if (_sampler == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        _sampler->set_seed(seed);
        _seed = seed;
        return 0;
    }

    inline int64_t get_seed() { return _seed; }

    inline void set_user_role(std::string role) { _user_role = role; }
    inline void set_response_role(std::string role) { _response_role = role; }
    inline void set_bos_token(std::string token) { _bos_token = token; }
    inline void set_eos_token(std::string token) {
        _eos_token = token;
        for (auto &stop_code : _stop_codes) {
            if (stop_code == _eos_token) {
                return;
            }
        }
        _stop_codes.push_back(_eos_token);
    }
    std::string get_user_role() { return _user_role; }
    std::string get_response_role() { return _response_role; }
    std::string get_bos_token() { return _bos_token; }
    std::string get_eos_token() { return _eos_token; }

    int get_vocab_size() { return _vocab_size; }

    inline std::vector<std::string> get_stop_codes() { return _stop_codes; }
    inline void set_stop_codes(std::vector<std::string> stop_codes) { _stop_codes = stop_codes; }
    inline std::vector<int> get_token_banned() { return _token_banned; }
    inline void set_token_banned(std::vector<int> token_banned) { _token_banned = token_banned; }
    inline std::string get_thinking_token() { return _thinking_token; }
    inline void set_thinking_token(std::string thinking_token) { _thinking_token = thinking_token; }

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

    inline bool is_generating() { return _is_generating; }
    inline void set_is_generating(bool is_generating) { _is_generating = is_generating; }

    inline bool get_stop_signal() { return _stop_signal; }
    inline void set_stop_signal(bool stop_signal) { _stop_signal = stop_signal; }

    std::string get_available_backends_str();
    int get_available_backend_ids(std::vector<int> &backend_ids);

    double get_avg_decode_speed();
    double get_avg_prefill_speed();

    // for state management
    struct state_node {
        std::any state;
        unsigned long long hash = 0;
        struct state_node * next = nullptr;
    } * _state_head = nullptr;

    // platform info
    const char * get_platform_name() {
        return _soc_detect.get_platform_name();
    }

    const char * get_soc_name() {
        return _soc_detect.get_soc_name();
    }

    const char * get_soc_partname() {
        return _soc_detect.get_soc_partname();
    }

    // backend
    std::string backend_id_to_str(int backend_id) {
        return backend_enum_to_str(backend_id);
    }
    int backend_str_to_id(std::string backend_str) {
        return backend_str_to_enum(backend_str);
    }

    // tokenizer
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

    // sampler
    int sampler_sample(std::vector<float> logits) {
        if (_sampler == nullptr) {
            return -1;
        }
        return _sampler->sample(logits.data(), logits.size(), _temperature, _top_k, _top_p);
    }
private:
    std::unique_ptr<execution_provider, std::function<void(execution_provider*)>> _backend;
    std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>> _tokenizer;
    std::unique_ptr<sampler> _sampler;

    soc_detect _soc_detect;

    int _vocab_size = 0;

    float _temperature = 2.0;
    int _top_k = 128;
    float _top_p = 0.5;
    float _presence_penalty = 0.5;
    float _frequency_penalty = 0.5;
    float _penalty_decay = 0.996;
    int64_t _seed = 42;
    std::string _user_role = "User";
    std::string _response_role = "Assistant";
    std::string _prompt;
    std::string _thinking_token = "<think";

    bool _is_generating = false;
    bool _stop_signal = false;

    std::vector<std::string> _stop_codes = {"\n\n", "\nUser:", "User:"};
    std::vector<int> _token_banned = {};
    std::string _bos_token = "";
    std::string _eos_token = "\n\n";

    std::map<int, float> _occurences;

    std::string _response_buffer;
    std::vector<int32_t> _response_buffer_ids;

    void apply_logits_penalties(float * logits, int vocab_size, float presence_penalty, float frequency_penalty, float penalty_decay);

    unsigned long long hash_string(std::string str) {
        unsigned long long hash = 0, p = 13131;
        for (size_t i = 0; i < str.size(); i++) {
            hash = hash * p + str[i];
        }
        return hash;
    }

    const int _decode_duration_window = 30;
    const int _prefill_duration_window = 10;

    std::vector<double> _decode_durations_ms;
    std::vector<double> _prefill_durations_ms;

#ifdef ENABLE_VISION
    std::unique_ptr<clip_ctx, std::function<void(clip_ctx*)>> _vision_encoder;
#endif

#ifdef ENABLE_WHISPER
    std::unique_ptr<whisper_context, std::function<void(whisper_context*)>> _whisper_encoder;
#endif

#ifdef ENABLE_TTS
    std::unique_ptr<cosyvoice> _cosyvoice;
#endif
};

}

#endif
