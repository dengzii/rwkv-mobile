#include "runtime.h"
#include "backend.h"
#include "logger.h"
#include <functional>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <thread>
#ifdef ENABLE_WEBRWKV
#include "web_rwkv_backend.h"
#endif

#ifdef ENABLE_NCNN
#include "ncnn_rwkv_backend.h"
#endif

#ifdef ENABLE_LLAMACPP
#include "llama_cpp_backend.h"
#endif

#ifdef ENABLE_QNN
#include "qnn_backend.h"
#endif

#ifdef ENABLE_VISION
#include "llava.h"
#include "clip.h"
#endif

#ifdef ENABLE_WHISPER
#include "whisper.h"
#endif

#if defined(ENABLE_TTS) || defined(ENABLE_WHISPER)
#include "audio.h"
#endif

namespace rwkvmobile {

std::string backend_enum_to_str(int backend) {
    switch (backend) {
        case RWKV_BACKEND_WEBRWKV:
            return "web-rwkv";
        case RWKV_BACKEND_NCNN:
            return "ncnn";
        case RWKV_BACKEND_LLAMACPP:
            return "llama.cpp";
        case RWKV_BACKEND_QNN:
            return "qnn";
        default:
            return "unknown";
    }
}

int backend_str_to_enum(std::string backend) {
    if (backend == "web-rwkv") {
        return RWKV_BACKEND_WEBRWKV;
    } else if (backend == "ncnn") {
        return RWKV_BACKEND_NCNN;
    } else if (backend == "llama.cpp") {
        return RWKV_BACKEND_LLAMACPP;
    } else if (backend == "qnn") {
        return RWKV_BACKEND_QNN;
    }
    return -1;
}

void runtime::apply_logits_penalties(float * logits, int vocab_size, float presence_penalty, float frequency_penalty, float penalty_decay) {
    if (!logits) {
        return;
    }
    for (auto &[id, occurence] : _occurences) {
        if (id >= vocab_size) {
            continue;
        }
        logits[id] -=
            _frequency_penalty * occurence + _presence_penalty;
        _occurences[id] *= _penalty_decay;
    }

    for (auto &token_banned : _token_banned) {
        if (token_banned >= vocab_size) {
            continue;
        }
        logits[token_banned] = -INFINITY;
    }
}

int runtime::init(std::string backend_name) {
    return init(backend_name, nullptr);
}

int runtime::init(std::string backend_name, void * extra) {
    int backend_id = backend_str_to_enum(backend_name);
    if (backend_id < 0) {
        return RWKV_ERROR_BACKEND;
    }
    int ret = init(backend_id, extra);
    if (!ret) {
        LOGI("Initialized runtime with backend: %s\n", backend_name.c_str());
    } else {
        LOGE("Failed to initialize runtime with backend: %s, errno = %d\n", backend_name.c_str(), ret);
    }
    return ret;
}

int runtime::init(int backend_id) {
    return init(backend_id, nullptr);
}

int runtime::init(int backend_id, void * extra) {
    _sampler = std::unique_ptr<sampler>(new sampler);
    if (_sampler == nullptr) {
        return RWKV_ERROR_SAMPLER;
    }

    if (backend_id == RWKV_BACKEND_WEBRWKV) {
#ifdef ENABLE_WEBRWKV
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new web_rwkv_backend,
            [](execution_provider *p) {
                delete (web_rwkv_backend*)p;
            });
#else
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_NCNN) {
#ifdef ENABLE_NCNN
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new ncnn_rwkv_backend,
            [](execution_provider *p) {
                delete (ncnn_rwkv_backend*)p;
            });
#else
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_LLAMACPP) {
#ifdef ENABLE_LLAMACPP
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new llama_cpp_backend,
            [](execution_provider *p) {
                delete (llama_cpp_backend*)p;
            });
#else
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_QNN) {
#ifdef ENABLE_QNN
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new qnn_backend,
            [](execution_provider *p) {
                delete (qnn_backend*)p;
            });
#else
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
    }
    return _backend->init(extra);
}

int runtime::load_model(std::string model_path) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    int ret =  _backend->load_model(model_path);
    if (!ret) {
        LOGI("Loaded model from: %s\n", model_path.c_str());
        LOGI("Model num_layers: %d, num_heads: %d, hidden_size: %d, vocab_size: %d\n",
             _backend->n_layers, _backend->num_heads, _backend->hidden_size, _backend->vocab_size);
    } else {
        LOGE("Failed to load model from: %s, errno = %d\n", model_path.c_str(), ret);
    }

    // Initialize state
    _state_head = new state_node;
    if (_state_head == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
    }
    _backend->clear_state();
    _backend->get_state(_state_head->state);

    _vocab_size = _backend->get_num_vocab();
    return ret;
}

int runtime::load_tokenizer(std::string vocab_file) {
    if (_tokenizer != nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    _tokenizer = std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>>(new trie_tokenizer,
        [](tokenizer_base *p) {
            delete (trie_tokenizer*)p;
        });
    if (_tokenizer == nullptr) {
        return RWKV_ERROR_TOKENIZER;
    }
    return _tokenizer->load(vocab_file);
}

int runtime::load_vision_encoder(std::string model_path) {
#ifdef ENABLE_VISION
    _vision_encoder = std::unique_ptr<clip_ctx, std::function<void(clip_ctx*)>>(clip_model_load(model_path.c_str(), 0),
        [](clip_ctx *p) {
            clip_free(p);
        });
    if (_vision_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
#endif
}

int runtime::load_whisper_encoder(std::string model_path) {
#ifdef ENABLE_WHISPER
    whisper_context_params cparams = whisper_context_default_params();
    _whisper_encoder = std::unique_ptr<whisper_context, std::function<void(whisper_context*)>>(whisper_init_from_file_with_params(model_path.c_str(), cparams),
        [](whisper_context *p) {
            whisper_free(p);
        });
    if (_whisper_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    whisper_init_state(_whisper_encoder.get());
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
#endif
}

int runtime::release_vision_encoder() {
#ifdef ENABLE_VISION
    _vision_encoder = nullptr;
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
#endif
}

int runtime::release_whisper_encoder() {
#ifdef ENABLE_WHISPER
    _whisper_encoder = nullptr;
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
#endif
}

int runtime::get_available_backend_ids(std::vector<int> &backend_ids) {
    backend_ids = std::vector<int>();

#ifdef ENABLE_WEBRWKV
    // Snapdragon platform doesn't support WebRWKV backend yet
    if (_soc_detect.get_platform_type() != PLATFORM_SNAPDRAGON) {
        backend_ids.push_back(RWKV_BACKEND_WEBRWKV);
    }
#endif

#ifdef ENABLE_NCNN
    backend_ids.push_back(RWKV_BACKEND_NCNN);
#endif

#ifdef ENABLE_LLAMACPP
    backend_ids.push_back(RWKV_BACKEND_LLAMACPP);
#endif

#ifdef ENABLE_QNN
    if (_soc_detect.get_platform_type() == PLATFORM_SNAPDRAGON) {
        auto supported_soc_names = std::vector<std::string>{"SM8650", "SM8635", "SM8550", "SM8475"};
        if (std::find(supported_soc_names.begin(), supported_soc_names.end(), _soc_detect.get_soc_partname()) != supported_soc_names.end()) {
            backend_ids.push_back(RWKV_BACKEND_QNN);
        }
    }
#endif

    return RWKV_SUCCESS;
}

std::string runtime::get_available_backends_str() {
    std::vector<int> backend_ids;
    get_available_backend_ids(backend_ids);
    std::string ret = "";
    for (auto id : backend_ids) {
        ret += backend_enum_to_str(id) + ",";
    }
    return ret;
}

int runtime::eval_logits(int id, float *& logits) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    int ret = _backend->eval(id, logits);
    auto end = std::chrono::high_resolution_clock::now();
    _decode_durations_ms.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    if (_decode_durations_ms.size() > _decode_duration_window) {
        _decode_durations_ms.erase(_decode_durations_ms.begin());
    }
    return ret;
}

int runtime::eval_logits(std::vector<int> ids, float *& logits) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    int ret = _backend->eval(ids, logits);
    auto end = std::chrono::high_resolution_clock::now();
    _prefill_durations_ms.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)ids.size());
    if (_prefill_durations_ms.size() > _prefill_duration_window) {
        _prefill_durations_ms.erase(_prefill_durations_ms.begin());
    }
    return ret;
}

int runtime::eval_logits_with_embeddings(const float *embeddings, int n_tokens, float *& logits) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return _backend->eval_with_embeddings(embeddings, n_tokens, logits);
}

int runtime::chat(std::string input, const int max_length, void (*callback)(const char *, const int), bool enable_reasoning) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    set_is_generating(true);
    _stop_signal = false;
    std::string prompt = input + _eos_token + _response_role + ":";
    if (!_user_role.empty()) {
        prompt = _bos_token + _user_role + ": " + prompt;
    }
    std::vector<int> ids = _tokenizer->encode(prompt);
    float *logits = nullptr;

    _response_buffer = "";
    _response_buffer_ids.clear();
    int ret = eval_logits(ids, logits);
    if (ret) {
        return ret;
    }

    for (int i = 0; i < max_length; i++) {
        apply_logits_penalties(logits, _vocab_size, _presence_penalty, _frequency_penalty, _penalty_decay);

        int idx = _sampler->sample(logits, _vocab_size, _temperature, _top_k, _top_p);
        _backend->free_logits_if_allocated(logits);
        if (idx == 0) {
            break;
        }
        _occurences[idx]++;

        _response_buffer += _tokenizer->decode(idx);
        _response_buffer_ids.push_back(idx);
        if (callback) {
            callback(_response_buffer.c_str(), idx);
        }

        bool stopping = false;
        for (auto &stop_code : _stop_codes) {
            if (_response_buffer.size() >= stop_code.size() &&
                _response_buffer.compare(_response_buffer.size() - stop_code.size(), stop_code.size(), stop_code) == 0) {
                stopping = true;
                break;
            }
        }

        ret = eval_logits(idx, logits);
        if (ret) return ret;
        if (stopping) break;
        if (_stop_signal) break;
    }

    if (callback) {
        callback(_response_buffer.c_str(), 0);
    }

    set_is_generating(false);
    _stop_signal = false;
    return RWKV_SUCCESS;
}

int runtime::chat(std::vector<std::string> inputs, const int max_length, void (*callback)(const char *, const int), bool enable_reasoning) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    set_is_generating(true);
    _stop_signal = false;

    struct state_node *node = _state_head;
    int start_idx = 0;
    bool edited = false;
    while (node && node->next && (start_idx < (int)inputs.size())) {
        LOGD("Comparing state node %i hash %llu with %llu\n", start_idx, node->next->hash, hash_string(inputs[start_idx]));
        unsigned long long input_hash = hash_string(inputs[start_idx]);
        if (node->next->hash != input_hash) {
            edited = true;
            struct state_node *ptr = node;
            while(ptr->next) {
                struct state_node *tmp = ptr->next;
                ptr->next = ptr->next->next;
                _backend->free_state(tmp->state);
                delete tmp;
            }
            node->next = nullptr;
            break;
        }
        start_idx++;
        node = node->next;
    }

    if (edited == false && start_idx % 2 == 1) {
        node = _state_head;
        start_idx--;
        for (int i = 0; i < start_idx; i++) {
            node = node->next;
        }
        while(node->next) {
            struct state_node *tmp = node->next;
            node->next = node->next->next;
            _backend->free_state(tmp->state);
            delete tmp;
        }
        edited = true;
    }

    LOGI("Loading state node %i hash %llu\n", start_idx-1, node->hash);
    _backend->set_state(node->state);

    float *logits = nullptr;
    _response_buffer = "";
    _response_buffer_ids.clear();
    int ret;
    for (int i = start_idx; i < (int)inputs.size(); i++) {
        std::string prompt;
        if (i % 2 == 0) {
            if (!_user_role.empty()) {
                prompt = _bos_token + _user_role + ": " + inputs[i] + _eos_token;
            } else {
                prompt = inputs[i] + _eos_token;
            }
        } else {
            if (i == inputs.size() - 1) {
                prompt = _bos_token + _response_role + ": " + inputs[i];
            } else {
                prompt = _bos_token + _response_role + ": " + inputs[i] + _eos_token;
            }
        }
        LOGD("Processing history %i: \"%s\"\n", i, prompt.c_str());
        if (i == inputs.size() - 1) {
            if (i % 2 == 0) {
                if (enable_reasoning) {
                    prompt += _response_role + ": " + _thinking_token;
                    _response_buffer += " " + _thinking_token;
                    auto tmp_ids = _tokenizer->encode(" " + _thinking_token);
                    _response_buffer_ids.insert(_response_buffer_ids.end(), tmp_ids.begin(), tmp_ids.end());
                } else {
                    prompt += _response_role + ":";
                }
            } else {
                _response_buffer = inputs[i];
                _response_buffer_ids = _tokenizer->encode(inputs[i]);
            }
        }
        std::vector<int> ids = _tokenizer->encode(prompt);
        ret = eval_logits(ids, logits);
        if (ret) return ret;
        node->next = new state_node;
        if (node->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
        node = node->next;
        node->hash = hash_string(inputs[i]);
        _backend->get_state(node->state);
        LOGD("New state node %i hash %llu\n", i, node->hash);
    }

    if (edited || start_idx == 0) {
        _occurences.clear();
        for (int i = 1; i < inputs.size(); i += 2) {
            std::vector<int> ids = _tokenizer->encode(" " + inputs[i]);
            for (auto id: ids) {
                for (auto &[_id, occurence] : _occurences) {
                    _occurences[_id] *= _penalty_decay;
                }
                _occurences[id]++;
            }
        }
    }

    bool resuming = false;
    if (!edited && inputs.size() % 2 == 0) {
        // case that resume generation from the last assistant message
        _response_buffer = inputs[inputs.size() - 1];
        _response_buffer_ids = _tokenizer->encode(inputs[inputs.size() - 1]);
        auto ret = eval_logits(_response_buffer_ids.back(), logits);
        if (ret) return ret;
        resuming = true;
    }

    for (int i = 0; i < max_length; i++) {
        apply_logits_penalties(logits, _vocab_size, _presence_penalty, _frequency_penalty, _penalty_decay);

        int idx = _sampler->sample(logits, _vocab_size, _temperature, _top_k, _top_p);
        _backend->free_logits_if_allocated(logits);
        if (idx == 0) {
            break;
        }

        std::string decoded = _tokenizer->decode(idx);
        std::string tmp = _response_buffer + decoded;
        bool stopping = false;
        for (auto &stop_code : _stop_codes) {
            if (tmp.size() >= stop_code.size() &&
                tmp.compare(tmp.size() - stop_code.size(), stop_code.size(), stop_code) == 0) {
                stopping = true;
                break;
            }
        }

        if (stopping || _stop_signal) {
            break;
        }

        _response_buffer += decoded;
        _response_buffer_ids.emplace_back(idx);
        if (i == 0 && _response_buffer[0] == ' ') {
            _response_buffer = _response_buffer.substr(1);
        }

        _occurences[idx]++;
        if (callback) {
            callback(_response_buffer.c_str(), idx);
        }

        ret = eval_logits(idx, logits);
        if (ret) return ret;
    }

    // only eval stop code if generation is not forced to stop
    if (!_stop_signal) {
        ret = eval_logits(_tokenizer->encode(_stop_codes[0]), logits);
        if (ret) return ret;
    }

    LOGD("Response: \"%s\"\n", _response_buffer.c_str());

    if (!resuming) {
        node->next = new state_node;
        if (node->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
        node = node->next;
    }
    node->hash = hash_string(_response_buffer);
    _backend->get_state(node->state);
    start_idx = -1;
    node = _state_head;
    while(node->next) {
        start_idx++;
        node = node->next;
    }
    LOGD("New state node %i hash %llu\n", start_idx, node->hash);
    if (callback) {
        callback(_response_buffer.c_str(), 0);
    }

    set_is_generating(false);
    _stop_signal = false;
    return RWKV_SUCCESS;
}

int runtime::set_prompt(std::string prompt) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    unsigned long long hash;
    if (prompt.empty()) {
        hash = 0;
    } else {
        hash = hash_string(prompt);
    }
    if (_state_head->hash == hash) {
        return RWKV_SUCCESS;
    }
    _prompt = prompt;
    clear_state();
    _state_head->hash = hash;

    if (prompt.empty()) {
        return RWKV_SUCCESS;
    }
    if (_state_head->state.has_value()) {
        _backend->free_state(_state_head->state);
    }
    float *logits = nullptr;
    std::vector<int> ids = _tokenizer->encode(prompt);
    int ret = eval_logits(ids, logits);
    if (ret) {
        return ret;
    }
    _backend->get_state(_state_head->state);
    _backend->free_logits_if_allocated(logits);
    return RWKV_SUCCESS;
}

std::string runtime::get_prompt() {
    return _prompt;
}

#ifdef ENABLE_VISION
int runtime::set_image_prompt(std::string path) {
    if (_backend == nullptr || _tokenizer == nullptr || _vision_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    unsigned long long hash;
    std::string prompt = "<img src=\"" + path + "\">";
    hash = hash_string(prompt);
    if (_state_head->hash == hash) {
        return RWKV_SUCCESS;
    }
    _prompt = prompt;
    clear_state();
    _state_head->hash = hash;

    if (prompt.empty()) {
        return RWKV_SUCCESS;
    }
    if (_state_head->state.has_value()) {
        _backend->free_state(_state_head->state);
    }

    auto embd = llava_image_embed_make_with_filename(_vision_encoder.get(), 4, path.c_str());
    if (embd == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    float *logits = nullptr;

    int ret = eval_logits_with_embeddings(embd->embed, embd->n_image_pos, logits);
    if (ret) {
        return ret;
    }
    _backend->get_state(_state_head->state);
    llava_image_embed_free(embd);
    _backend->free_logits_if_allocated(logits);
    return RWKV_SUCCESS;
}
#endif

#ifdef ENABLE_WHISPER
int runtime::set_audio_prompt(std::string path) {
    if (_backend == nullptr || _tokenizer == nullptr || _whisper_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    unsigned long long hash;
    std::string prompt = "<audio src=\"" + path + "\">";
    hash = hash_string(prompt);
    if (_state_head->hash == hash) {
        return RWKV_SUCCESS;
    }
    _prompt = prompt;
    clear_state();
    _state_head->hash = hash;

    if (prompt.empty()) {
        return RWKV_SUCCESS;
    }
    if (_state_head->state.has_value()) {
        _backend->free_state(_state_head->state);
    }

    wav_file wav;
    wav.load(path);

    auto start = std::chrono::high_resolution_clock::now();
    whisper_pcm_to_mel(_whisper_encoder.get(), wav.samples.data(), wav.samples.size(), 4);
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("whisper_pcm_to_mel time: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    start = std::chrono::high_resolution_clock::now();
    whisper_encode(_whisper_encoder.get(), 0, 4);
    end = std::chrono::high_resolution_clock::now();
    LOGI("whisper_encode time: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    float *logits = nullptr;

    auto embd = whisper_get_adapter_output_tensor(_whisper_encoder.get());

    int ret = eval_logits_with_embeddings((const float *)embd->data, embd->ne[1], logits);
    if (ret) {
        return ret;
    }
    _backend->get_state(_state_head->state);
    _backend->free_logits_if_allocated(logits);
    return RWKV_SUCCESS;
}
#endif

#ifdef ENABLE_TTS
static void save_samples_to_wav(std::vector<float> samples, std::string path) {
    wav_file wav_file;
    wav_file.sample_rate = 24000;
    wav_file.num_channels = 1;
    wav_file.num_samples = samples.size();
    wav_file.bit_depth = 16;
    wav_file.audio_format = 1;
    wav_file.byte_rate = 24000 * 16 / 8;
    wav_file.block_align = 2;
    wav_file.samples = samples;
    wav_file.save(path);
}

int runtime::cosyvoice_load_models(
    std::string speech_tokenizer_path,
    std::string campplus_path,
    std::string flow_encoder_path,
    std::string flow_decoder_estimator_path,
    std::string hift_generator_path,
    std::string tts_tokenizer_path,
    std::string spk_info_path
) {
    _cosyvoice = std::make_unique<cosyvoice>();
    _cosyvoice->load_speech_tokenizer(speech_tokenizer_path);
    _cosyvoice->load_campplus(campplus_path);
    _cosyvoice->load_flow_encoder(flow_encoder_path);
    _cosyvoice->load_flow_decoder_estimator(flow_decoder_estimator_path);
    _cosyvoice->load_hift_generator(hift_generator_path);
    _cosyvoice->load_spk_info(spk_info_path);

    _tokenizer = std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>>(new trie_tokenizer,
        [](tokenizer_base *p) {
            delete (trie_tokenizer*)p;
        });
    if (_tokenizer == nullptr) {
        return RWKV_ERROR_TOKENIZER;
    }
    return _tokenizer->load(tts_tokenizer_path);
}

int runtime::cosyvoice_release_models() {
    _cosyvoice = nullptr;
    return RWKV_SUCCESS;
}

int runtime::run_tts_internal(std::string tts_text, std::string instruction_text,
    const std::string prompt_wav_path, const std::string spk_name,
    std::vector<float> &output_samples) {
#ifdef __ANDROID__
    setenv("KMP_DUPLICATE_LIB_OK", "1", 1);
#endif
    if (_cosyvoice == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    if (_cosyvoice == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // prepare input tokens for llm
    tts_text = _cosyvoice->normalize_text(tts_text);
    std::vector<int> tts_tokens = _tokenizer->encode(tts_text);
    std::vector<int> prompt_tokens;
    if (!instruction_text.empty()) {
        prompt_tokens = _tokenizer->encode(instruction_text + "<|endofprompt|>");
    }
    int min_len, max_len;
    std::vector<int> llm_tokens = _cosyvoice->get_llm_tokens(tts_tokens, prompt_tokens, min_len, max_len);

    std::vector<int> decoded_tokens;
    bool is_llm_decoding = true;
    std::thread llm_thread([&]() {
        auto start = std::chrono::high_resolution_clock::now();
        clear_state();
        float *logits = nullptr;
        eval_logits(llm_tokens, logits);
        const int speech_vocab_size = 6562;
        const int speech_vocab_offset = 65548;
        for (int i = 0; i < max_len; i++) {
            int token_id = _cosyvoice->speech_token_sampler(logits, speech_vocab_size, decoded_tokens, (i < min_len));
            free_logits_if_allocated(logits);
            if (token_id == speech_vocab_size - 1) {
                break;
            }
            decoded_tokens.push_back(token_id);
            eval_logits(token_id + speech_vocab_offset, logits);
        }
        auto end = std::chrono::high_resolution_clock::now();
        LOGI("[TTS] llm decode time: %f ms", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
        is_llm_decoding = false;
    });
    llm_thread.detach();

    std::vector<int> speech_tokens;
    std::vector<std::vector<float>> speech_features(80);
    std::vector<float> speech_embedding;
    if (!prompt_wav_path.empty()) {
        if (_cosyvoice->process_zeroshot(prompt_wav_path, speech_tokens, speech_features, speech_embedding, 24000) != true) {
            LOGE("[TTS] Failed to process prompt audio");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
    } else if (!spk_name.empty()) {
        speech_embedding = _cosyvoice->get_spk_embedding(spk_name);
        if (speech_embedding.empty()) {
            LOGE("[TTS] Failed to get spk embedding");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
    }

    while (is_llm_decoding) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    decoded_tokens.insert(decoded_tokens.begin(), speech_tokens.begin(), speech_tokens.end());

    _cosyvoice->speech_token_to_wav(decoded_tokens, speech_features, speech_embedding, output_samples);
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] total time: %f ms", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
    LOGI("[TTS] output samples length: %f", output_samples.size() / 24000.0);
    LOGI("[TTS] rtf: %f", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6f * 24000.0 / output_samples.size());
    return RWKV_SUCCESS;
}

int runtime::run_tts(std::string tts_text, std::string instruction_text, std::string prompt_wav_path, std::string output_wav_path) {
    if (_cosyvoice == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<float> output_samples;
    run_tts_internal(tts_text, instruction_text, prompt_wav_path, "", output_samples);

    if (!output_wav_path.empty()) {
        save_samples_to_wav(output_samples, output_wav_path);
    }

    return RWKV_SUCCESS;
}

int runtime::run_tts_with_predefined_spks(std::string tts_text, std::string instruction_text, std::string spks_name, std::string output_wav_path) {
    if (_cosyvoice == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<float> speech_embedding = _cosyvoice->get_spk_embedding(spks_name);
    
    std::vector<float> output_samples;
    run_tts_internal(tts_text, instruction_text, "", spks_name, output_samples);

    if (!output_wav_path.empty()) {
        save_samples_to_wav(output_samples, output_wav_path);
    }

    return RWKV_SUCCESS;
}

std::string runtime::cosyvoice_get_spk_names() {
    if (_cosyvoice == nullptr) {
        return "";
    }
    return _cosyvoice->get_spk_names();
}
#endif

int runtime::gen_completion(std::string prompt, int max_length, int stop_code, void (*callback)(const char *, const int)) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    set_is_generating(true);
    _stop_signal = false;

    std::vector<int> ids = _tokenizer->encode(prompt);
    float *logits = nullptr;
    int ret = eval_logits(ids, logits);
    if (ret || !logits) {
        set_is_generating(false);
        return ret;
    }

    _response_buffer = prompt;
    _response_buffer_ids = ids;
    for (int i = 0; i < max_length; i++) {
        apply_logits_penalties(logits, _vocab_size, _presence_penalty, _frequency_penalty, _penalty_decay);

        int idx = _sampler->sample(logits, _vocab_size, _temperature, _top_k, _top_p);
        _backend->free_logits_if_allocated(logits);
        bool stopping = (idx == stop_code);

        std::string next = _tokenizer->decode(idx);
        _response_buffer += next;
        _response_buffer_ids.push_back(idx);
        ret = eval_logits(idx, logits);
        if (callback) {
            callback(_response_buffer.c_str(), idx);
        }

        if (stopping || _stop_signal) {
            break;
        }

        _occurences[idx]++;
    }

    if (callback) {
        callback(_response_buffer.c_str(), 0);
    }
    set_is_generating(false);
    _stop_signal = false;
    return RWKV_SUCCESS;
}

double runtime::get_avg_decode_speed() {
    if (_decode_durations_ms.size() == 0) {
        return 0.0;
    } else {
        double avg_time = 0.0;
        for (auto duration : _decode_durations_ms) {
            avg_time += duration;
        }
        avg_time /= _decode_durations_ms.size();
        return 1000.0 / avg_time;
    }
}

double runtime::get_avg_prefill_speed() {
    if (_prefill_durations_ms.size() == 0) {
        return 0.0;
    } else {
        double avg_time = 0.0;
        for (auto duration : _prefill_durations_ms) {
            avg_time += duration;
        }
        avg_time /= _prefill_durations_ms.size();
        return 1000.0 / avg_time;
    }
}

} // namespace rwkvmobile

