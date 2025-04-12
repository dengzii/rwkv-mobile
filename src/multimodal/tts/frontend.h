#pragma once

#include <string>
#include "sampler.h"
#include "onnxruntime_cxx_api.h"

namespace rwkvmobile {

class frontend {
public:
    frontend() {};

    ~frontend() {};

    bool load_speech_tokenizer(const std::string model_path);

    bool load_campplus(const std::string model_path);

    bool load_flow_encoder(const std::string model_path);

    bool load_flow_decoder_estimator(const std::string model_path);

    bool process_zeroshot(const std::string prompt_audio_path, std::vector<int> &speech_tokens, std::vector<std::vector<float>> &speech_features, std::vector<float> &speech_embedding, const int resample_rate = 24000);

    bool speech_token_to_wav(const std::vector<int> tokens, const std::vector<std::vector<float>> speech_features, const std::vector<float> speech_embedding);

    std::vector<int> extract_speech_tokens(std::vector<float> audio_samples, int sample_rate);

    std::vector<float> extract_speech_embedding(std::vector<float> audio_samples, int sample_rate);

    std::vector<int> get_llm_tokens(const std::vector<int> tts_tokens, const std::vector<int> prompt_tokens, int &min_len, int &max_len);

    std::string normalize_text(std::string text);

    int speech_token_sampler(float *logits, size_t size, std::vector<int> decoded_tokens, bool ignore_eos = false);

private:
    Ort::Env *env = nullptr;
    Ort::Session *speech_tokenizer_session = nullptr;
    Ort::Session *campplus_session = nullptr;
    Ort::Session *flow_encoder_session = nullptr;
    Ort::Session *flow_decoder_estimator_session = nullptr;

    std::vector<float> random_noise;
    std::vector<float> t_span;

    sampler _sampler;
};

}
