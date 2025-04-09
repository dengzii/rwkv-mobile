#include "frontend.h"
#include "audio.h"
#include "logger.h"
#include "librosa.h"
#include <chrono>
#include "onnxruntime_cxx_api.h"

namespace rwkvmobile {

bool frontend::load_speech_tokenizer(const std::string model_path) {
    if (env == nullptr) {
        env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "rwkv_mobile");
    }
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    speech_tokenizer_session = new Ort::Session(*env, model_path.c_str(), session_options);
    return true;
}

bool frontend::load_campplus(const std::string model_path) {
    if (env == nullptr) {
        env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "rwkv_mobile");
    }
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    campplus_session = new Ort::Session(*env, model_path.c_str(), session_options);
    return true;
}

std::vector<int> frontend::extract_speech_tokens(std::vector<float> audio_samples, int sample_rate) {
    auto start = std::chrono::high_resolution_clock::now();
    int fmin = 0;
    int fmax = sample_rate / 2;
    int n_fft = 400;
    int n_hop = 160;
    int n_mel = 128;
    std::vector<std::vector<float>> mels = logMelSpectrogram(audio_samples, sample_rate, n_fft, n_hop, n_mel, fmin, fmax, 2.0, true, false);
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] extract_speech_tokens Log-Melspectrogram time: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    LOGI("[TTS] mels.size(): %dx%d", mels.size(), mels[0].size());

#if 0
    float mean = 0.0f;
    float std = 0.0f;
    for (int i = 0; i < mels.size(); i++) {
        for (int j = 0; j < mels[i].size(); j++) {
            mean += mels[i][j];
        }
    }
    mean /= mels.size() * mels[0].size();
    for (int i = 0; i < mels.size(); i++) {
        for (int j = 0; j < mels[i].size(); j++) {
            std += (mels[i][j] - mean) * (mels[i][j] - mean);
        }
    }
    std = std::sqrt(std / (mels.size() * mels[0].size()));
    LOGI("[TTS] Log-Melspectrogram Mean: %f, Std: %f", mean, std);
#endif
    Ort::RunOptions run_options;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(mels.size()), static_cast<int64_t>(mels[0].size())};
    std::vector<int64_t> feat_len_shape = {1};
    int32_t feat_len = mels[0].size();
    Ort::Value feat_input = Ort::Value::CreateTensor<float>(allocator, input_shape.data(), input_shape.size());
    for (int i = 0; i < mels.size(); i++) {
        memcpy(feat_input.GetTensorMutableData<float>() + i * mels[i].size(), mels[i].data(), mels[i].size() * sizeof(float));
    }
    Ort::Value feat_len_input = Ort::Value::CreateTensor<int32_t>(memory_info, &feat_len, 1, feat_len_shape.data(), feat_len_shape.size());
    std::vector<const char*> input_names = {"feats", "feats_length"};
    std::vector<const char*> output_names = {"indices"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(feat_input));
    inputs.push_back(std::move(feat_len_input));
    auto output_tensor = speech_tokenizer_session->Run(run_options, input_names.data(), inputs.data(), 2, output_names.data(), 1);
    auto output = output_tensor[0].GetTensorMutableData<int32_t>();
    int64_t output_size = output_tensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<int> output_vector(output_size);
    std::memcpy(output_vector.data(), output, output_size * sizeof(int32_t));

    return output_vector;
}

bool frontend::process_zeroshot(const std::string tts_text, const std::string prompt_text, const std::string prompt_audio_path, const int resample_rate) {
    if (speech_tokenizer_session == nullptr) {
        LOGE("[TTS] Speech tokenizer is not loaded");
        return false;
    }

    wav_file prompt_audio;
    prompt_audio.load(prompt_audio_path);
    if (prompt_audio.samples.size() / prompt_audio.sample_rate > 30) {
        LOGE("[TTS] Prompt audio is too long: should be less than 30 seconds");
        return false;
    }

    if (prompt_audio.sample_rate != 16000) {
        LOGI("[TTS] Resampling prompt audio to 16000 Hz");
        prompt_audio.resample(16000);
        if (prompt_audio.sample_rate != resample_rate) {
            LOGE("[TTS] Resample to %d Hz failed", resample_rate);
            return false;
        }
    }

    auto samples_16k = prompt_audio.samples;
    auto start = std::chrono::high_resolution_clock::now();
    auto speech_tokens = extract_speech_tokens(samples_16k, 16000);
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] extract_speech_tokens time: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    prompt_audio.resample(resample_rate);
    if (prompt_audio.sample_rate != resample_rate) {
        LOGE("[TTS] Resample to %d Hz failed", resample_rate);
        return false;
    }

    int fmin = 0;
    int fmax = 8000;
    int n_fft = 1920;
    int n_hop = 480;
    int n_mel = 80;
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> features = melSpectrogram(prompt_audio.samples, resample_rate, n_fft, n_hop, n_mel, fmin, fmax, 1.0, true, true);
    dynamic_range_compression(features);
    end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] feat_extractor Melspectrogram time: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

#if 0
    LOGI("[TTS] features.size(): %dx%d", features.size(), features[0].size());
    float mean = 0.0f;
    float std = 0.0f;
    for (int i = 0; i < features.size(); i++) {
        for (int j = 0; j < features[i].size(); j++) {
            mean += features[i][j];
        }
    }
    mean /= features.size() * features[0].size();
    for (int i = 0; i < features.size(); i++) {
        for (int j = 0; j < features[i].size(); j++) {
            std += (features[i][j] - mean) * (features[i][j] - mean);
        }
    }
    std = std::sqrt(std / (features.size() * features[0].size()));
    LOGI("[TTS] melspectrogram Mean: %f, Std: %f", mean, std);
#endif

    if (resample_rate == 24000) {
        int token_length = std::min(features[0].size() / 2, speech_tokens.size());
        for (int i = 0; i < features.size(); i++) {
            features[i].resize(token_length * 2);
        }
        speech_tokens.resize(token_length);
    }
    LOGI("[TTS] features.size(): %dx%d", features.size(), features[0].size());
    LOGI("[TTS] speech_tokens.size(): %d", speech_tokens.size());
    return true;
}

}
