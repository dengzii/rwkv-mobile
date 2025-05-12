#pragma once

#include <string>
#include <map>
#include "sampler.h"
#include "tokenizer.h"

#include "MNN/Interpreter.hpp"
#include <MNN/expr/Module.hpp>

namespace rwkvmobile {

class cosyvoice {
public:
    cosyvoice() {
        MNN::ScheduleConfig config;
        mnn_runtime = MNN::Interpreter::createRuntime({config});
    };

    ~cosyvoice() {
        if (hift_generator_interpretor) {
            delete hift_generator_interpretor;
        }
        if (speech_tokenizer_module) {
            delete speech_tokenizer_module;
        }
        if (campplus_interpretor) {
            delete campplus_interpretor;
        }
        if (flow_encoder_interpretor) {
            delete flow_encoder_interpretor;
        }
        if (flow_decoder_interpretor) {
            delete flow_decoder_interpretor;
        }
    };

    bool load_speech_tokenizer(const std::string model_path);

    bool load_campplus(const std::string model_path);

    bool load_flow_encoder(const std::string model_path);

    bool load_flow_decoder_estimator(const std::string model_path);

    bool load_hift_generator(const std::string model_path);

    bool process_zeroshot(const std::string prompt_audio_path, std::vector<int> &speech_tokens, std::vector<std::vector<float>> &speech_features, std::vector<float> &speech_embedding, const int resample_rate = 24000);

    bool speech_token_to_wav(const std::vector<int> tokens, const std::vector<std::vector<float>> speech_features, const std::vector<float> speech_embedding, std::vector<float> &output_samples, std::function<void(float)> progress_callback = nullptr);

    std::vector<int> extract_speech_tokens(std::vector<float> audio_samples, int sample_rate);

    std::vector<float> extract_speech_embedding(std::vector<float> audio_samples, int sample_rate);

    int speech_token_sampler(float *logits, size_t size, std::vector<int> decoded_tokens, bool ignore_eos = false);

    int set_cfm_steps(int cfm_steps) {
        if (cfm_steps < 1 || cfm_steps > 10) {
            return RWKV_ERROR_INVALID_PARAMETERS;
        }
        this->cfm_steps = cfm_steps;
        return RWKV_SUCCESS;
    }

private:
    MNN::Interpreter *hift_generator_interpretor = nullptr;
    MNN::Session *hift_generator_mnn_session = nullptr;

    MNN::Express::Module *speech_tokenizer_module = nullptr;

    MNN::Interpreter *campplus_interpretor = nullptr;
    MNN::Session *campplus_mnn_session = nullptr;

    MNN::Interpreter *flow_encoder_interpretor = nullptr;
    MNN::Session *flow_encoder_mnn_session = nullptr;

    MNN::Interpreter *flow_decoder_interpretor = nullptr;
    MNN::Session *flow_decoder_mnn_session = nullptr;

    MNN::RuntimeInfo mnn_runtime;

    std::vector<float> random_noise;
    std::vector<float> t_span;
    int cfm_steps = 5;

    sampler _sampler;
    std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>> _tokenizer;
};

}
