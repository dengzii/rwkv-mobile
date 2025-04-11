#pragma once

#include <string>
#include "onnxruntime_cxx_api.h"

namespace rwkvmobile {

class frontend {
public:
    frontend() {};
    ~frontend() {};

    bool load_speech_tokenizer(const std::string model_path);

    bool load_campplus(const std::string model_path);

    bool process_zeroshot(const std::vector<int> tts_tokens, const std::vector<int> prompt_tokens, const std::string prompt_audio_path, const int resample_rate = 24000);

    std::vector<int> extract_speech_tokens(std::vector<float> audio_samples, int sample_rate);

    std::vector<float> extract_speech_embedding(std::vector<float> audio_samples, int sample_rate);

    std::string normalize_text(std::string text);

private:
    Ort::Env *env = nullptr;
    Ort::Session *speech_tokenizer_session = nullptr;
    Ort::Session *campplus_session = nullptr;
};

}
