#include "runtime.h"
#include "backend.h"
#include "logger.h"
#include <functional>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <cstring>
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

#ifdef ENABLE_MNN
#include "mnn_rwkv_backend.h"
#endif

#ifdef ENABLE_COREML
#include "coreml_rwkv_backend.h"
#endif

#ifdef ENABLE_VISION
#include "llava.h"
#include "clip.h"
#endif

#ifdef ENABLE_WHISPER
#include "whisper.h"
#endif

#ifdef ENABLE_TTS
#include "frontend_utils.h"
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
        case RWKV_BACKEND_MNN:
            return "mnn";
        case RWKV_BACKEND_COREML:
            return "coreml";
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
    } else if (backend == "mnn") {
        return RWKV_BACKEND_MNN;
    } else if (backend == "coreml") {
        return RWKV_BACKEND_COREML;
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
    } else if (backend_id == RWKV_BACKEND_MNN) {
#ifdef ENABLE_MNN
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new mnn_rwkv_backend,
            [](execution_provider *p) {
                delete (mnn_rwkv_backend*)p;
            });
#else
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
#endif
    } else if (backend_id == RWKV_BACKEND_COREML) {
#ifdef ENABLE_COREML
        _backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new coreml_rwkv_backend,
            [](execution_provider *p) {
                delete (coreml_rwkv_backend*)p;
            });
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

int runtime::load_vision_encoder(std::string model_path, std::string adapter_path) {
#ifdef ENABLE_VISION
    auto adapter_path_cstr = adapter_path.empty() ? NULL : adapter_path.c_str();
    _vision_encoder = std::unique_ptr<clip_ctx, std::function<void(clip_ctx*)>>(clip_model_load(model_path.c_str(), adapter_path_cstr, 0),
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

#ifdef ENABLE_COREML
    backend_ids.push_back(RWKV_BACKEND_COREML);
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
    _decode_speed = 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return ret;
}

int runtime::eval_logits(std::vector<int> ids, float *& logits) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int i = 0;
    int ret;
    for (; i + _prefill_chunk_size <= ids.size(); i += _prefill_chunk_size) {
        auto ids_chunk = std::vector<int>(ids.begin() + i, ids.begin() + i + _prefill_chunk_size);
        ret = _backend->eval(ids_chunk, logits);
        if (ret != RWKV_SUCCESS) return ret;
        if (_current_prefill_total_tokens > 0) {
            _current_prefill_finished_tokens += _prefill_chunk_size;
            _prefill_progress = (double)_current_prefill_finished_tokens / _current_prefill_total_tokens;
            LOGD("Update prefill_progress = %f", _prefill_progress);
        }
    }
    if (i < ids.size()) {
        auto ids_left = std::vector<int>(ids.begin() + i, ids.end());
        ret = _backend->eval(ids_left, logits);
        if (_current_prefill_total_tokens > 0) {
            _current_prefill_finished_tokens += ids_left.size();
            _prefill_progress = (double)_current_prefill_finished_tokens / _current_prefill_total_tokens;
            LOGD("Update prefill_progress = %f", _prefill_progress);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    _prefill_speed = ids.size() * 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return ret;
}

int runtime::eval_logits_with_embeddings(const float *embeddings, int n_tokens, float *& logits) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto ret = _backend->eval_with_embeddings(embeddings, n_tokens, logits);
    auto end = std::chrono::high_resolution_clock::now();
    if (n_tokens > 1) {
        _prefill_speed = n_tokens * 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    } else {
        _decode_speed = 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    return ret;
}

int runtime::chat(std::string input, const int max_length, void (*callback)(const char *, const int, const char *), bool enable_reasoning) {
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
    _response_buffer_eos_found = false;
    _prefill_progress_start(ids.size());
    int ret = eval_logits(ids, logits);
    if (ret) {
        return ret;
    }
    _prefill_progress_finish();

    for (int i = 0; i < max_length; i++) {
        apply_logits_penalties(logits, _vocab_size, _presence_penalty, _frequency_penalty, _penalty_decay);

        int idx = _sampler->sample(logits, _vocab_size, _temperature, _top_k, _top_p);
        _backend->free_logits_if_allocated(logits);
        if (idx == 0) {
            break;
        }
        _occurences[idx]++;

        std::string next = _tokenizer->decode(idx);
        _response_buffer += next;
        _response_buffer_ids.push_back(idx);
        if (callback) {
            callback(_response_buffer.c_str(), idx, next.c_str());
        }

        for (auto &stop_code : _stop_codes) {
            if (_response_buffer.size() >= stop_code.size() &&
                _response_buffer.compare(_response_buffer.size() - stop_code.size(), stop_code.size(), stop_code) == 0) {
                _response_buffer_eos_found = true;
                break;
            }
        }

        ret = eval_logits(idx, logits);
        if (ret) return ret;
        if (_response_buffer_eos_found) break;
        if (_stop_signal) break;
    }

    set_is_generating(false);
    _stop_signal = false;
    return RWKV_SUCCESS;
}

std::string runtime::apply_chat_template(std::vector<std::string> inputs, bool enable_reasoning) {
    static auto replace_text = [](const std::string& text, const std::string& old_str, const std::string& new_str) -> std::string {
        std::string result = text;
        size_t pos = 0;
        while ((pos = result.find(old_str, pos)) != std::string::npos) {
            result.replace(pos, old_str.length(), new_str);
            pos += new_str.length();
        }
        return result;
    };

    std::string text = _prompt;
    for (int i = 0; i < inputs.size(); i++) {
        if (i % 2 == 0) {
            auto user_text = inputs[i];
            user_text = replace_text(user_text, "\r\n", "\n");
            user_text = replace_text(user_text, "\n\n", "\n");

            if (!_user_role.empty()) {
                text += _bos_token + _user_role + ": " + inputs[i] + _eos_token;
            } else {
                text += inputs[i] + _eos_token;
            }
        } else {
            if (i == inputs.size() - 1) {
                text += _response_role + ": " + inputs[i];
            } else {
                text += _response_role + ": " + inputs[i] + _eos_token;
            }
        }
    }

    if (inputs.size() % 2 != 0) {
        text +=  _response_role + ":";
        if (enable_reasoning) {
            text += " " + _thinking_token;
        }
    }
    return text;
}

runtime::state_node* runtime::match_and_load_state(const std::vector<int> &ids, std::vector<int> &new_ids_to_prefill) {
    auto node = _state_head;
    size_t compare_pos = 0;

    // find the last node that matches the input text
    while (node->next) {

        if (compare_pos + node->next->ids.size() > ids.size() || !std::equal(ids.begin() + compare_pos, ids.begin() + compare_pos + node->next->ids.size(), node->next->ids.begin())) {
            // the text will diverge at next node
            break;
        }
        std::string debug_msg = "matched tokens:";
        for (auto id : node->next->ids) {
            debug_msg += std::to_string(id) + " ";
        }
        LOGI("%s\n", debug_msg.c_str());
        compare_pos += node->next->ids.size();
        node = node->next;
    }

    _backend->set_state(node->state);
    while (node->next) {
        auto tmp = node->next->next;
        _backend->free_state(node->next->state);
        delete node->next;
        node->next = tmp;
    }

    new_ids_to_prefill = std::vector<int>(ids.begin() + compare_pos, ids.end());
    std::string debug_msg = "new tokens to prefill: ";
    for (auto id : new_ids_to_prefill) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());
    return node;
}

int runtime::register_state_checkpoint(state_node* &node, const std::vector<int> &ids, const float *logits) {
    node->next = new state_node;
    if (node->next == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
    }
    node = node->next;
    node->ids = std::vector<int>(ids);
    _backend->get_state(node->state);
    node->last_logits.resize(_vocab_size);
    memcpy(node->last_logits.data(), logits, _vocab_size * sizeof(float));
    return RWKV_SUCCESS;
}

int runtime::chat(std::vector<std::string> inputs, const int max_length, void (*callback)(const char *, const int, const char *), bool enable_reasoning) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    set_is_generating(true);
    _stop_signal = false;
    _response_buffer.clear();
    _response_buffer_ids.clear();
    _response_buffer_eos_found = false;

    if (_prefilling_thread.joinable() && _prefilling_thread.get_id() != std::this_thread::get_id()) {
        LOGD("Found prefilling thread, joining\n");
        _prefilling_thread.join();
        LOGD("_prefilling_thread finished.\n");
    }

    auto input_text = apply_chat_template(inputs, enable_reasoning);
    LOGD("Applied chat template: \"%s\"\n", input_text.c_str());
    std::vector<int> text_ids = _tokenizer->encode(input_text);
    std::string debug_msg = "text_ids: ";
    for (auto id : text_ids) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());

    float *logits = nullptr;
    std::vector<int> tokens_to_prefill;
    auto node = match_and_load_state(text_ids, tokens_to_prefill);

    if (tokens_to_prefill.size() > 0) {
        _prefill_progress_start(tokens_to_prefill.size());
        std::string debug_msg = "new tokens to prefill: ";
        for (auto id : tokens_to_prefill) {
            debug_msg += std::to_string(id) + " ";
        }
        LOGD("%s\n", debug_msg.c_str());

        // save a state checkpoint every about 256 tokens
        int checkpoint_interval = 256;
        for (int i = 0; i < tokens_to_prefill.size(); i += checkpoint_interval) {
            std::vector<int> tokens_to_prefill_chunk = std::vector<int>(tokens_to_prefill.begin() + i, tokens_to_prefill.begin() + std::min(i + checkpoint_interval, (int)tokens_to_prefill.size()));
            eval_logits(tokens_to_prefill_chunk, logits);
            int ret = register_state_checkpoint(node, tokens_to_prefill_chunk, logits);
            if (ret) return ret;
            _backend->free_logits_if_allocated(logits);
        }
    }
    _prefill_progress_finish();

    _response_buffer = input_text.substr(input_text.rfind(_response_role + ":") + (_response_role + ":").size());
    std::vector<int> response_ids_raw;
    _response_buffer_ids = _tokenizer->encode(_response_buffer);
    int ret;

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

    if (logits == nullptr) {
        if (node->last_logits.size() == _vocab_size) {
            logits = node->last_logits.data();
        } else {
            LOGE("no logits found, neither from saved state nor from new tokens to prefill\n");
            ret = eval_logits(text_ids.back(), logits);
            if (ret) return ret;
            response_ids_raw.emplace_back(text_ids.back());
        }
    }

    int decoded_idx = 0;
    bool thinking_end_tag_found = false;
    bool is_pseudo_thinking = enable_reasoning && _response_buffer.find("</think>") != std::string::npos;
    for (int i = 0; i < max_length; i++) {
        apply_logits_penalties(logits, _vocab_size, _presence_penalty, _frequency_penalty, _penalty_decay);

        if (is_pseudo_thinking && i == 0) {
            // token 61 is '<', 261 is '\n\n'
            logits[61] = -1e9f;
            logits[261] = -1e9f;
        } else if (is_pseudo_thinking && i == 1 && decoded_idx == 11) {
            logits[61] = -1e9f;
        }

        decoded_idx = _sampler->sample(logits, _vocab_size, _temperature, _top_k, _top_p);
        if (decoded_idx == 0) {
            break;
        }

        std::string decoded = _tokenizer->decode(decoded_idx);
        std::string tmp = _response_buffer + decoded;
        for (auto &stop_code : _stop_codes) {
            if (enable_reasoning && !thinking_end_tag_found && stop_code == "\n\n") {
                continue;
            }
            if (tmp.size() >= stop_code.size() &&
                tmp.compare(tmp.size() - stop_code.size(), stop_code.size(), stop_code) == 0) {
                LOGD("stop code found: %s\n", stop_code.c_str());
                _response_buffer_eos_found = true;
                break;
            }
        }

        if (enable_reasoning && !thinking_end_tag_found) {
            if (tmp.find("</think>") != std::string::npos) {
                thinking_end_tag_found = true;
            }
        }

        if (_response_buffer_eos_found || _stop_signal) {
            LOGD("stopping generation, eos_found: %d, stop_signal: %d\n", _response_buffer_eos_found, _stop_signal);
            break;
        }

        if (i != 0 || logits != node->last_logits.data()) {
            _backend->free_logits_if_allocated(logits);
        }
        ret = eval_logits(decoded_idx, logits);
        if (ret) return ret;

        response_ids_raw.emplace_back(decoded_idx);
        _response_buffer += decoded;
        _response_buffer_ids.emplace_back(decoded_idx);
        if (i == 0 && _response_buffer[0] == ' ') {
            _response_buffer = _response_buffer.substr(1);
        }

        _occurences[decoded_idx]++;
        if (callback) {
            callback(_response_buffer.c_str(), decoded_idx, decoded.c_str());
        }
    }

    if (response_ids_raw.size() > 0) {
        int ret = register_state_checkpoint(node, response_ids_raw, logits);
        if (ret) return ret;
    }

    if (logits != node->last_logits.data()) {
        _backend->free_logits_if_allocated(logits);
    }

    set_is_generating(false);
    _stop_signal = false;
    return RWKV_SUCCESS;
}

int runtime::set_prompt(std::string prompt) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    LOGD("Setting and processing prompt: \"%s\"\n", prompt.c_str());
    std::vector<int> ids = _tokenizer->encode(prompt);
    if (_state_head->next == nullptr) {
        _state_head->next = new state_node;
        if (_state_head->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
    }

    if (_state_head->next->ids == ids) {
        return RWKV_SUCCESS;
    }
    _prompt = prompt;
    _backend->clear_state();
    _state_head->next->ids = ids;

    if (ids.empty()) {
        return RWKV_SUCCESS;
    }
    if (_state_head->next->state.has_value()) {
        _backend->free_state(_state_head->next->state);
    }
    float *logits = nullptr;
    int ret = eval_logits(ids, logits);
    if (ret) {
        return ret;
    }
    _backend->get_state(_state_head->next->state);
    _state_head->next->last_logits.resize(_vocab_size);
    memcpy(_state_head->next->last_logits.data(), logits, _vocab_size * sizeof(float));
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
    std::string prompt = "<img src=\"" + path + "\">";
    std::vector<int> ids = _tokenizer->encode(prompt);

    if (_state_head->next == nullptr) {
        _state_head->next = new state_node;
        if (_state_head->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
    }

    if (_state_head->next->ids == ids) {
        return RWKV_SUCCESS;
    }
    _prompt = prompt;
    _backend->set_state(_state_head->state);
    _state_head->next->ids = ids;

    if (ids.empty()) {
        return RWKV_SUCCESS;
    }
    if (_state_head->next->state.has_value()) {
        _backend->free_state(_state_head->next->state);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto embd = llava_image_embed_make_with_filename(_vision_encoder.get(), 4, path.c_str());
    if (embd == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("siglip duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    float *logits = nullptr;

    start = std::chrono::high_resolution_clock::now();
    int ret = eval_logits_with_embeddings(embd->embed, embd->n_image_pos, logits);
    if (ret) {
        return ret;
    }
    end = std::chrono::high_resolution_clock::now();
    LOGI("eval_logits_with_embeddings duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    _backend->get_state(_state_head->next->state);
    _state_head->next->last_logits.resize(_vocab_size);
    memcpy(_state_head->next->last_logits.data(), logits, _vocab_size * sizeof(float));
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
    std::string prompt = "<audio src=\"" + path + "\">";
    std::vector<int> ids = _tokenizer->encode(prompt);

    if (_state_head->next == nullptr) {
        _state_head->next = new state_node;
        if (_state_head->next == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
        }
    }

    if (_state_head->next->ids == ids) {
        return RWKV_SUCCESS;
    }
    _prompt = prompt;
    _backend->clear_state();
    _state_head->next->ids = ids;

    if (ids.empty()) {
        return RWKV_SUCCESS;
    }
    if (_state_head->next->state.has_value()) {
        _backend->free_state(_state_head->next->state);
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
    _backend->get_state(_state_head->next->state);
    _state_head->next->last_logits.resize(_vocab_size);
    memcpy(_state_head->next->last_logits.data(), logits, _vocab_size * sizeof(float));
    _backend->free_logits_if_allocated(logits);
    return RWKV_SUCCESS;
}
#endif

#ifdef ENABLE_TTS
static void save_samples_to_wav(std::vector<float> samples, std::string path, int sample_rate = 24000) {
    wav_file wav_file;
    wav_file.sample_rate = sample_rate;
    wav_file.num_channels = 1;
    wav_file.num_samples = samples.size();
    wav_file.bit_depth = 16;
    wav_file.audio_format = 1;
    wav_file.byte_rate = sample_rate * 16 / 8;
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
    std::string tts_tokenizer_path
) {
    _cosyvoice = std::make_unique<cosyvoice>();
    _cosyvoice->load_speech_tokenizer(speech_tokenizer_path);
    _cosyvoice->load_campplus(campplus_path);
    _cosyvoice->load_flow_encoder(flow_encoder_path);
    _cosyvoice->load_flow_decoder_estimator(flow_decoder_estimator_path);
    _cosyvoice->load_hift_generator(hift_generator_path);

    _tokenizer = std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>>(new trie_tokenizer,
        [](tokenizer_base *p) {
            delete (trie_tokenizer*)p;
        });
    if (_tokenizer == nullptr) {
        return RWKV_ERROR_TOKENIZER;
    }
    return _tokenizer->load(tts_tokenizer_path);
}

int runtime::sparktts_load_models(
    std::string wav2vec2_path,
    std::string bicodec_tokenizer_path,
    std::string bicodec_detokenizer_path
) {
    _sparktts = std::make_unique<sparktts>();
    if (!_sparktts->load_models(wav2vec2_path, bicodec_tokenizer_path, bicodec_detokenizer_path)) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
}

int runtime::sparktts_release_models() {
    _sparktts = nullptr;
    return RWKV_SUCCESS;
}

int runtime::cosyvoice_release_models() {
    _cosyvoice = nullptr;
    return RWKV_SUCCESS;
}

int runtime::run_tts_internal(std::string tts_text, std::string instruction_text,
    const std::string prompt_wav_path, const std::string prompt_speech_text,
    std::vector<float> &output_samples) {
    if (_cosyvoice == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    if (_cosyvoice == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    _tts_generation_progress = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    // prepare input tokens for llm
    std::vector<int> llm_tokens_without_tts, tts_tokens;
    std::string input_text = "";
    if (!prompt_speech_text.empty()) {
        input_text = prompt_speech_text;
    }
    if (!instruction_text.empty()) {
        input_text = instruction_text + "<|endofprompt|>" + input_text;
    }

    // input_text += tts_text;
    llm_tokens_without_tts = _tokenizer->encode(input_text);
    tts_tokens = _tokenizer->encode(tts_text);
    LOGD("[TTS] pre-tts input text: %s", input_text.c_str());
    LOGD("[TTS] tts text: %s", tts_text.c_str());

    std::vector<int> llm_tokens = std::vector<int>(llm_tokens_without_tts);
    for (auto token : tts_tokens) {
        llm_tokens.push_back(token);
    }
    int content_length = llm_tokens.size();
    auto it = std::find(llm_tokens.begin(), llm_tokens.end(), 65531);
    if (it != llm_tokens.end()) {
        content_length = std::distance(llm_tokens.begin(), it);
    }

    const float max_token_text_ratio = 20;
    const float min_token_text_ratio = 2;
    const int min_len = content_length * min_token_text_ratio;
    const int max_len = content_length * max_token_text_ratio;
    LOGI("[TTS] min_len: %d, max_len: %d", min_len, max_len);

    const int sos_eos_token = 72110;
    const int task_token = 72111;
    const int speech_vocab_offset = 65548;
    llm_tokens_without_tts.insert(llm_tokens_without_tts.begin(), sos_eos_token);
    tts_tokens.push_back(task_token);

    std::vector<int> speech_tokens;
    std::vector<std::vector<float>> speech_features(80);
    std::vector<float> speech_embedding;

    static auto calc_checksum = [](const std::string &path) -> unsigned int {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            LOGE("[TTS] Failed to open prompt wav file: %s", path.c_str());
            return 0;
        }

        uint32_t checksum = 0;
        char buffer[4096];
        while (file.read(buffer, sizeof(buffer))) {
            for (size_t i = 0; i < file.gcount(); i++) {
                checksum = ((checksum << 5) + checksum) + buffer[i];
            }
        }
        if (file.gcount() > 0) {
            for (size_t i = 0; i < file.gcount(); i++) {
                checksum = ((checksum << 5) + checksum) + buffer[i];
            }
        }
        file.close();

        return checksum;
    };

    bool read_from_cache = false;
    if (!_cache_dir.empty() && !prompt_speech_text.empty()) {
        uint32_t checksum = calc_checksum(prompt_wav_path);
        if (checksum == 0) {
            LOGE("[TTS] Failed to calculate checksum of prompt wav file: %s", prompt_wav_path.c_str());
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }

        std::string cache_dir = _cache_dir + "/tts_cache/";
        if (!std::filesystem::exists(cache_dir)) {
            std::filesystem::create_directory(cache_dir);
        }

        std::string cache_file = cache_dir + std::to_string(checksum) + ".cache";

        if (std::filesystem::exists(cache_file)) {
            std::ifstream cache(cache_file, std::ios::binary);
            if (cache) {
                LOGI("[TTS] Loading cached speech tokens/features/embedding");
                
                size_t tokens_size;
                cache.read(reinterpret_cast<char*>(&tokens_size), sizeof(size_t));
                speech_tokens.resize(tokens_size);
                cache.read(reinterpret_cast<char*>(speech_tokens.data()), tokens_size * sizeof(int));

                size_t features_size;
                cache.read(reinterpret_cast<char*>(&features_size), sizeof(size_t));
                for (size_t i = 0; i < features_size; i++) {
                    size_t feature_len;
                    cache.read(reinterpret_cast<char*>(&feature_len), sizeof(size_t));
                    speech_features[i].resize(feature_len);
                    cache.read(reinterpret_cast<char*>(speech_features[i].data()), feature_len * sizeof(float));
                }

                size_t embedding_size; 
                cache.read(reinterpret_cast<char*>(&embedding_size), sizeof(size_t));
                speech_embedding.resize(embedding_size);
                cache.read(reinterpret_cast<char*>(speech_embedding.data()), embedding_size * sizeof(float));

                cache.close();
                read_from_cache = true;
                LOGI("[TTS] Loaded speech tokens/features/embedding from cache file: %s", cache_file.c_str());
            }
        }
    }

    if (!read_from_cache && !prompt_wav_path.empty()) {
        if (_cosyvoice->process_zeroshot(prompt_wav_path, speech_tokens, speech_features, speech_embedding, 24000) != true) {
            LOGE("[TTS] Failed to process prompt audio");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }

        // save to cache if the prompt wav has corresponding prompt speech text
        if (!_cache_dir.empty() && !prompt_speech_text.empty()) {
            uint32_t checksum = calc_checksum(prompt_wav_path);
            if (checksum == 0) {
                LOGE("[TTS] Failed to calculate checksum of prompt wav file: %s", prompt_wav_path.c_str());
                return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
            }

            std::string cache_file = _cache_dir + "/tts_cache/" + std::to_string(checksum) + ".cache";
            std::ofstream cache(cache_file, std::ios::binary);
            if (cache) {
                size_t tokens_size = speech_tokens.size();
                cache.write(reinterpret_cast<const char*>(&tokens_size), sizeof(size_t));
                cache.write(reinterpret_cast<const char*>(speech_tokens.data()), speech_tokens.size() * sizeof(int));

                size_t features_size = speech_features.size();
                cache.write(reinterpret_cast<const char*>(&features_size), sizeof(size_t));
                for (size_t i = 0; i < features_size; i++) {
                    size_t feature_len = speech_features[i].size();
                    cache.write(reinterpret_cast<const char*>(&feature_len), sizeof(size_t));
                    cache.write(reinterpret_cast<const char*>(speech_features[i].data()), feature_len * sizeof(float));
                }

                size_t embedding_size = speech_embedding.size();
                cache.write(reinterpret_cast<const char*>(&embedding_size), sizeof(size_t));
                cache.write(reinterpret_cast<const char*>(speech_embedding.data()), speech_embedding.size() * sizeof(float));
                cache.close();
                LOGI("[TTS] Saved speech tokens/features/embedding to cache file: %s", cache_file.c_str());
            }
        }
    } else if (prompt_wav_path.empty()) {
        LOGE("[TTS] No prompt wav path provided");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    if (!prompt_speech_text.empty()) {
        for (auto token : speech_tokens) {
            tts_tokens.push_back(token + speech_vocab_offset);
        }
    }

    _tts_generation_progress = 0.2;
    // std::string debug_msg = "tokens: [";
    // for (int i = 0; i < llm_tokens.size(); i++) {
    //     debug_msg += std::to_string(llm_tokens[i]) + ", ";
    // }
    // LOGD("[TTS] %s]", debug_msg.c_str());

    std::vector<int> decoded_tokens;
    // bool is_llm_decoding = true;
    // std::thread llm_thread([&]() {
    {
        auto start = std::chrono::high_resolution_clock::now();
        // _backend->clear_state();
        float *logits = nullptr;
        // eval_logits(llm_tokens, logits);
        std::vector<int> new_ids_to_prefill;
        auto node = match_and_load_state(llm_tokens_without_tts, new_ids_to_prefill);
        if (!new_ids_to_prefill.empty()) {
            eval_logits(new_ids_to_prefill, logits);
            register_state_checkpoint(node, new_ids_to_prefill, logits);
        }
        eval_logits(tts_tokens, logits);
        register_state_checkpoint(node, tts_tokens, logits);

        const int speech_vocab_size = 6562;
        for (int i = 0; i < max_len; i++) {
            int token_id = _cosyvoice->speech_token_sampler(logits, speech_vocab_size, decoded_tokens, (i < min_len));
            free_logits_if_allocated(logits);
            if (token_id == speech_vocab_size - 1) {
                break;
            }
            decoded_tokens.push_back(token_id);
            eval_logits(token_id + speech_vocab_offset, logits);
            _tts_generation_progress = 0.2 + 0.3 * (float)i / max_len;
        }
        auto end = std::chrono::high_resolution_clock::now();
        LOGI("[TTS] llm decode time: %f ms", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
        // is_llm_decoding = false;
    }
    // });
    // llm_thread.detach();

    // while (is_llm_decoding) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // }

    decoded_tokens.insert(decoded_tokens.begin(), speech_tokens.begin(), speech_tokens.end());
    _tts_generation_progress = 0.5;

    _cosyvoice->speech_token_to_wav(decoded_tokens, speech_features, speech_embedding, output_samples, 
        [this](float progress) {
            _tts_generation_progress = 0.5 + progress * 0.5;
        }
    );
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] total time: %f ms", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
    LOGI("[TTS] output samples length: %f", output_samples.size() / 24000.0);
    LOGI("[TTS] rtf: %f", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6f * 24000.0 / output_samples.size());
    _tts_generation_progress = 1.0;
    return RWKV_SUCCESS;
}

int runtime::run_tts(std::string tts_text, std::string instruction_text, std::string prompt_speech_text, std::string prompt_wav_path, std::string output_wav_path) {
    if (_cosyvoice == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    _tts_last_output_files.clear();

    auto texts = tts_frontend_utils::process_text(tts_text,
        [this](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(text);
        },
        _tn_list
    );

    _tts_total_num_output_wavs = texts.size();

    if (output_wav_path.find(".wav") != std::string::npos) {
        output_wav_path = output_wav_path.substr(0, output_wav_path.find(".wav"));
    }

    for (int i = 0; i < texts.size(); i++) {
        LOGI("[TTS] Split text %i: %s\n", i, texts[i].c_str());
        std::vector<float> output_samples;
        run_tts_internal(texts[i], instruction_text, prompt_wav_path, prompt_speech_text, output_samples);

        auto output_file = output_wav_path + "." + std::to_string(i) + ".wav";
        save_samples_to_wav(output_samples, output_file);
        LOGI("[TTS] Saved file %s\n", output_file.c_str());
        _tts_last_output_files.push_back(output_file);
    }

    set_is_generating(false);
    return RWKV_SUCCESS;
}

int runtime::run_spark_tts(std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    static const int tts_tag_token_offset = 8193;
    static const int global_token_offset = 8196;
    // static const int text_token_offset = 12292;

    _tts_last_output_files.clear();

    wav_file wav;
    wav.load(prompt_audio_path);
    wav.resample(16000);

    std::vector<int> global_tokens;
    std::vector<int> semantic_tokens;
    _sparktts->tokenize_audio(wav.samples, global_tokens, semantic_tokens);
    if (prompt_audio_text.empty()) {
        semantic_tokens.clear();
    }

    std::string full_text = prompt_audio_text + tts_text;
    auto text_tokens = tokenizer_encode(full_text);
    if (text_tokens.empty()) {
        LOGE("[TTS] Text tokenizer encode failed");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
    for (int i = 0; i < text_tokens.size(); i++) {
        input_tokens.push_back(text_tokens[i]);
    }
    input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
    for (int i = 0; i < global_tokens.size(); i++) {
        input_tokens.push_back(global_tokens[i] + global_token_offset);
    }
    input_tokens.push_back(tts_tag_token_offset + 1); // tag_1
    for (int i = 0; i < semantic_tokens.size(); i++) {
        input_tokens.push_back(semantic_tokens[i]);
    }

    std::vector<int> output_tokens;

    static const int tts_max_length = 3000;
    static const int tts_top_k = 50;
    static const float tts_top_p = 0.95;
    static const float tts_temperature = 1.0;
    static const int tts_eos_token = 8192;
    clear_state();
    float *logits = nullptr;
    int ret = eval_logits(input_tokens, logits);
    if (ret || !logits) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    for (int i = 0; i < tts_max_length; i++) {
        int idx = _sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
        _backend->free_logits_if_allocated(logits);
        if (idx == tts_eos_token) {
            break;
        }

        output_tokens.push_back(idx);
        ret = eval_logits(idx, logits);
        if (ret || !logits) {
            LOGE("[TTS] Error evaluating logits");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
    }

    // output_tokens = {5668, 8187, 2800, 3139, 3711, 5930, 2169, 4, 2244, 3112, 1771, 7273, 5087, 8098, 7713, 2148, 479, 5486, 5736, 4782, 5099, 3142, 1952, 4380, 7484, 4682, 3156, 2476, 6077, 6009, 102, 5190, 4413, 1033, 1876, 6098, 2630, 7350, 2017, 5635, 8056, 207, 1338, 740, 7700, 7730, 6683, 7887, 2525, 7954, 2756, 2013, 442, 6737, 2157, 5776, 4385, 7390, 3995, 3360, 7318, 7412, 3584, 3006, 1824, 4922, 5581, 4846, 7856, 4080, 4268, 3130, 6167, 5184, 5908, 6928, 5308, 5921, 6980, 4369, 6225, 2716, 7419, 5427, 7079, 5905, 6008, 2134, 6320, 4544, 1294, 6298, 7346, 5565, 2737, 5618, 1618, 4990, 5792, 7330, 7620, 2655, 7886, 7666, 5086, 2910, 7904, 7663, 2910, 6627, 3953, 7645, 2910, 4085, 3953, 4051, 4612, 7251, 2305, 3650, 4523, 4383, 2134, 5949, 178, 1108, 4171, 4265, 3861, 1123, 4797, 1144, 2385, 4893, 6953, 1457, 7166, 1312, 5296, 4889, 4302, 2321, 6066, 4066, 3514, 7127, 7286, 2098, 2090, 557, 2974, 7793, 902, 956, 3602, 494, 6888, 6470, 7005, 3263, 7678, 3540, 4479, 4491, 5712, 6985, 7452, 5781, 7355, 933, 5931, 187, 5871, 5778, 4427, 5193, 2531, 6835, 2947, 1749, 615, 4230, 5641, 3828, 6129, 7378, 6263, 1016, 2197, 3812, 6795, 5468, 1490, 1055, 2349, 3660, 3483, 707, 7579, 388, 1982, 7213, 4159, 4095, 404, 3343, 8073, 1660, 5160, 7729, 2518, 5333, 958, 6192, 977, 6204, 3473, 6497, 3390, 3055, 6850, 7415, 7487, 4140, 7425, 949, 1194, 6164, 2973, 347, 4051, 4085, 2910, 347, 347, 4554, 2027, 4951, 4358, 6615, 4092, 6447, 4962, 6059, 5688, 5317, 578, 7422, 5630, 6927, 3807, 6159, 4119, 434, 1663, 3109, 2945, 6106, 3503, 4689, 4194, 7610, 3728, 5800, 561, 4037, 3581, 6643, 7617, 7470, 4545, 5949, 3856, 4056, 1067, 2342, 3995, 2987, 2978, 5965, 2367, 5806, 2488, 4785, 7793, 4728, 6830, 4282, 7545, 2028, 4823, 4517, 3940, 5001, 3112, 3936, 2312, 3122, 6632, 6059, 7363, 6541, 5044, 7130, 1146, 7016, 4082, 3016, 98, 2973, 3638, 2125, 2035, 347, 4085, 4702, 2164, 7118, 1020, 4445, 3105, 5450, 4258, 3261, 7249, 4487, 271, 7531, 1023, 2994, 6817, 6632, 6865, 5143, 134, 375, 7987, 7268, 754, 7018, 6597, 6698, 2826, 662, 5959, 7751, 1716, 2554, 3554, 957, 8168, 2577, 6512, 1860, 4519, 2484, 5074, 3646, 972, 7078, 3990, 521, 6845, 5443, 1589, 6973, 7136, 7101, 2968, 2452, 6496, 4321, 4092, 2829, 1029, 7878, 6415, 3977, 3787, 4848, 5736, 3653, 8101, 4369, 5303, 4579, 3787, 2458, 2923, 4906, 5944, 7182, 375, 5379, 7075, 2785, 4966, 2974, 2445, 141, 3079, 868, 3493, 2534, 6018, 6694, 654, 2101, 2182, 1209, 4520, 173, 6316, 2948, 5799, 5537, 6526, 3759, 145, 7110, 1026, 6383, 5158, 4170, 1170, 3868, 407, 4140, 3915, 7370, 6901, 7927, 2314, 2508, 285, 4280, 7084, 6239, 6065, 4248, 5696, 2226, 2659, 4655, 6726, 3990, 569, 4329, 2142, 1287, 8062, 6396, 1063, 7974, 4365, 6472, 2973, 2561, 3694, 3953, 4085, 3140, 5772, 347, 3140, 3953, 347, 3140, 3953, 2910, 7904, 1972, 2338, 7580, 3059, 4882, 3168, 3774, 3082, 6422, 3040, 3819, 1550, 6790, 4192, 1450, 4588, 1977, 623, 2897, 259, 2596, 938, 3672, 243, 5001, 3837, 1922, 5532, 5861, 2187, 1240, 5718, 4429, 4511, 3156, 296, 4002, 3004, 6497, 6680, 830, 290, 2127, 3404, 5799, 4025, 6253, 6643, 1221, 4809, 1294, 7548, 1474, 6366, 59, 7117, 4090, 1652, 5400, 6287, 4533, 2532, 504, 3814, 5813, 972, 1013, 1755, 2572, 7229, 1112, 2093, 5729, 6864, 3318, 638, 1042, 3376, 2628, 3768, 1456, 34, 1231, 4029, 7987, 82, 6877, 7258, 7390, 2644, 2836, 3295, 6177, 1545, 2166, 4447, 2142, 1259, 4468, 5800, 2849, 2625, 816, 2644, 3230, 6248, 3815, 4993, 6695, 365, 6019, 6644, 5576, 4300, 839, 972, 6222, 5769, 2724, 5712, 4664, 2840, 4725, 899, 5990, 3541, 2871, 4551, 2139, 2323, 7020, 8012, 7220, 2027, 1296, 661, 3923, 5839, 3869, 6209, 4922, 2116, 1934, 2340, 3554, 6264, 5585, 2862, 6210, 4914, 5594, 966, 6744, 6320, 2672, 3841, 4931, 3995, 7904, 1876, 6618, 6977, 6968, 478, 304, 3374, 3344, 4603, 7098, 6611, 2223, 6523, 2232, 6510, 7101, 2896, 5029, 3085, 2973, 3546, 3694, 5194, 2415, 6627, 6627, 3953, 6627, 5194, 4085, 6386, 6386, 4785, 8092, 4212, 992, 4236, 1473, 1814, 8017, 1993, 4770, 1754, 707, 6436, 4038, 1195, 7121, 521, 451, 7482, 6055, 7897, 5006, 1977, 4107, 2164, 1660, 5174, 8167, 4746, 7230, 4151, 4428, 1675, 4686, 5850, 876, 5876, 2925, 588, 6803, 231, 543, 7321, 308, 2780, 7609, 7431, 3450, 2073, 819, 6298, 5284, 6503, 7161, 612, 3760, 1697, 3841, 2991, 1819, 2380, 7524, 7825, 397, 647, 2836, 411, 7132, 6567, 4857, 5778, 6824, 1843, 1402, 1339, 5462, 1045, 994, 2692, 482, 7180, 6408, 7116, 1998, 7258, 2687, 2079, 8101, 8188, 7641, 5768, 1609, 1636, 1734, 3612, 3405, 2348, 2551, 2021, 2125, 4263, 7238, 5249, 1745, 1601, 7900, 3986, 7076, 3322, 7045, 6192, 174, 6649, 1696, 3980, 7630, 7427, 4383, 1214, 725, 876, 4128, 3093, 2738, 5471, 6983, 5314, 5737, 2819, 1442, 2696, 1983, 2455, 5166, 3637, 3056, 7632, 1653, 6756, 444, 7655, 3239, 2231, 6968, 7176, 4193, 7012, 3802, 7758, 1231, 4885, 4853, 1664, 3912, 5772, 4085, 3140, 3953, 347, 7645, 347, 347, 6223, 205, 903, 5795, 1311, 7678, 7061, 5738, 8057, 3295, 3151, 4213, 7439, 5234, 7767, 1719, 7777, 6244, 4197, 6066, 4803, 2254, 3486, 5903, 603, 5389, 4573, 6629, 4080, 3059, 4846, 1209, 7912, 2790, 6981, 284, 6489, 6921, 6336, 6536, 7700, 1847, 3730, 1402, 6842, 4974, 6045, 3105, 4862, 90, 1793, 7407, 6477, 5648, 2961, 2730, 7118, 7258, 7231, 4414, 1345, 3339, 1355, 5222, 2220, 1823, 2474, 5424, 6374, 6936, 1562, 5579, 1924, 1056, 4924, 98, 14, 2118, 2428, 5857, 7954, 5846, 383, 7952, 398, 905, 7586, 3198, 3903, 4412, 2142, 3733, 868, 7532, 1655, 2685, 2458, 7714, 4176, 8092, 7448, 5413, 826, 7771, 448, 5279, 4501, 2845, 7251, 733, 693, 4372, 2196, 7705, 2058, 5649, 2611, 2715, 4820};

    std::vector<float> output_samples = _sparktts->detokenize_audio(global_tokens, output_tokens);
    save_samples_to_wav(output_samples, output_wav_path, 16000);

    set_is_generating(false);
    return RWKV_SUCCESS;
}
#endif

int runtime::gen_completion(std::string prompt, int max_length, int stop_code, void (*callback)(const char *, const int, const char *)) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    _response_buffer = "";
    _response_buffer_ids.clear();
    _response_buffer_eos_found = false;
    set_is_generating(true);
    _stop_signal = false;

    std::vector<int> ids = _tokenizer->encode(prompt);
    _prefill_progress_start(ids.size());

    float *logits = nullptr;
    int ret = eval_logits(ids, logits);
    if (ret || !logits) {
        set_is_generating(false);
        return ret;
    }
    _prefill_progress_finish();

    _response_buffer = prompt;
    _response_buffer_ids = ids;
    static int idx = 0;
    bool apply_penalties = _presence_penalty > 0.0f && _frequency_penalty > 0.0f && _penalty_decay > 0.0f;
    for (int i = 0; i < max_length; i++) {
        if (apply_penalties) {
            apply_logits_penalties(logits, _vocab_size, _presence_penalty, _frequency_penalty, _penalty_decay);
        }

        idx = _sampler->sample(logits, _vocab_size, _temperature, _top_k, _top_p);
        _backend->free_logits_if_allocated(logits);
        _response_buffer_eos_found = (idx == stop_code);

        std::string next = _tokenizer->decode(idx);
        _response_buffer += next;
        _response_buffer_ids.push_back(idx);
        ret = eval_logits(idx, logits);
        if (callback) {
            callback(_response_buffer.c_str(), idx, next.c_str());
        }

        if (_response_buffer_eos_found || _stop_signal) {
            break;
        }

        if (apply_penalties) {
            _occurences[idx]++;
        }
    }

    set_is_generating(false);
    _stop_signal = false;
    return RWKV_SUCCESS;
}

double runtime::get_avg_decode_speed() {
    double speed_from_backend = _backend->get_decode_speed();
    if (speed_from_backend > 0) {
        return speed_from_backend;
    }

    if (_decode_speed < 0) {
        return 0.0;
    } else {
        return _decode_speed;
    }
}

double runtime::get_avg_prefill_speed() {
    double speed_from_backend = _backend->get_prefill_speed();
    if (speed_from_backend > 0) {
        return speed_from_backend;
    }

    if (_prefill_speed < 0) {
        return 0.0;
    } else {
        return _prefill_speed;
    }
}

} // namespace rwkvmobile

