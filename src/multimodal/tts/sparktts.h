#pragma once

#include <string>
#include <map>
#include <vector>

#include "MNN/Interpreter.hpp"
#include <MNN/expr/Module.hpp>

// #include "net.h"
// #include "mat.h"

namespace rwkvmobile {

class sparktts {
public:
    sparktts() {
        MNN::ScheduleConfig config;
        mnn_runtime = MNN::Interpreter::createRuntime({config});
    };

    ~sparktts() {
        if (wav2vec2_mnn_interpretor) {
            delete wav2vec2_mnn_interpretor;
        }
        if (bicodec_tokenizer_mnn_interpretor) {
            delete bicodec_tokenizer_mnn_interpretor;
        }
        if (bicodec_detokenizer_mnn_interpretor) {
            delete bicodec_detokenizer_mnn_interpretor;
        }
    };

    bool load_models(std::string wav2vec2_model_path, std::string bicodec_tokenizer_path, std::string bicodec_detokenizer_path);

    std::vector<float> get_ref_clip(std::vector<float> audio);

    std::vector<float> extract_wav2vec2_features(std::vector<float> audio);

    void zero_mean_unit_variance_normalize(std::vector<float>& input_values);

    bool tokenize_audio(std::vector<float> audio, std::vector<int> &global_tokens, std::vector<int> &semantic_tokens);

    std::vector<float> detokenize_audio(std::vector<int> global_tokens, std::vector<int> semantic_tokens);

private:
    // some configs
    const int sample_rate = 16000;
    const int ref_segment_duration = 6;
    const int latent_hop_length = 320;

    MNN::Interpreter *wav2vec2_mnn_interpretor = nullptr;
    MNN::Session *wav2vec2_mnn_session = nullptr;

    MNN::Interpreter *bicodec_tokenizer_mnn_interpretor = nullptr;
    MNN::Session *bicodec_tokenizer_mnn_session = nullptr;
    MNN::Interpreter *bicodec_detokenizer_mnn_interpretor = nullptr;
    MNN::Session *bicodec_detokenizer_mnn_session = nullptr;

    int current_semantic_tokens_size = 0;

    MNN::RuntimeInfo mnn_runtime;

    // ncnn::Net bicodec_detokenizer_ncnn_net;
};

}
