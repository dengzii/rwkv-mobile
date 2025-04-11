#include "frontend.h"
#include "audio.h"
#include "logger.h"
#include "librosa.h"
#include <chrono>
#include "onnxruntime_cxx_api.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

#define PRINT_FEATURE_INFO 0

static void debug_print_mean_std(std::vector<float> feat, std::string name) {
#if PRINT_FEATURE_INFO
    LOGI("[TTS] %s.size(): %d", name.c_str(), feat.size());
    float mean = 0.0f;
    float std = 0.0f;
    for (int i = 0; i < feat.size(); i++) {
        mean += feat[i];
    }
    mean /= feat.size();
    for (int i = 0; i < feat.size(); i++) {
        std += (feat[i] - mean) * (feat[i] - mean);
    }
    std = std::sqrt(std / (feat.size()));
    LOGI("[TTS] %s Mean: %f, Std: %f", name.c_str(), mean, std);
#endif
}

static void debug_print_mean_std_2d(std::vector<std::vector<float>> feat, std::string name) {
#if PRINT_FEATURE_INFO
    LOGI("[TTS] %s.size(): %dx%d", name.c_str(), feat.size(), feat[0].size());
    float mean = 0.0f;
    float std = 0.0f;
    for (int i = 0; i < feat.size(); i++) {
        for (int j = 0; j < feat[i].size(); j++) {
            mean += feat[i][j];
        }
    }
    mean /= feat.size() * feat[0].size();
    for (int i = 0; i < feat.size(); i++) {
        for (int j = 0; j < feat[i].size(); j++) {
            std += (feat[i][j] - mean) * (feat[i][j] - mean);
        }
    }
    std = std::sqrt(std / (feat.size() * feat[0].size()));
    LOGI("[TTS] %s Mean: %f, Std: %f", name.c_str(), mean, std);
#endif
}

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
    if (speech_tokenizer_session == nullptr) {
        LOGE("[TTS] speech_tokenizer model not loaded.")
        return std::vector<int>();
    }
    auto start = std::chrono::high_resolution_clock::now();
    int fmin = 0;
    int fmax = sample_rate / 2;
    int n_fft = 400;
    int n_hop = 160;
    int n_mel = 128;
    std::vector<std::vector<float>> mels = logMelSpectrogram(audio_samples, sample_rate, n_fft, n_hop, n_mel, fmin, fmax, 2.0, true, false);
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] extract_speech_tokens Log-Melspectrogram duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    LOGD("[TTS] mels.size(): %dx%d", mels.size(), mels[0].size());

    debug_print_mean_std_2d(mels, "24000Hz mel");

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

std::vector<float> frontend::extract_speech_embedding(std::vector<float> audio_samples, int sample_rate) {
    if (campplus_session == nullptr) {
        LOGE("[TTS] speech_tokenizer model not loaded.")
        return std::vector<float>();
    }

    knf::FbankOptions opts;
    opts.frame_opts.dither = 0;
    opts.frame_opts.samp_freq = sample_rate;
    opts.mel_opts.num_bins = 80;
    knf::OnlineFbank fbank(opts);
    fbank.AcceptWaveform(sample_rate, audio_samples.data(), audio_samples.size());
    int32_t n = fbank.NumFramesReady();
    std::vector<std::vector<float>> feat_kaldi;
    for (int i = 0; i < n; i++) {
        feat_kaldi.emplace_back(std::move(std::vector<float>(fbank.GetFrame(i), fbank.GetFrame(i) + 80)));
    }

    debug_print_mean_std_2d(feat_kaldi, "feat_kaldi");

    std::vector<float> mean(80, 0.f);
    for (int i = 0; i < feat_kaldi.size(); i++) {
        for (int j = 0; j < feat_kaldi[i].size(); j++) {
            mean[j] += feat_kaldi[i][j];
        }
    }
    for (int j = 0; j < 80; j++) {
        mean[j] /= feat_kaldi.size();
    }
    for (int i = 0; i < feat_kaldi.size(); i++) {
        for (int j = 0; j < feat_kaldi[i].size(); j++) {
            feat_kaldi[i][j] -= mean[j];
        }
    }

    Ort::RunOptions run_options;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(feat_kaldi.size()), static_cast<int64_t>(feat_kaldi[0].size())};
    Ort::Value feat_input = Ort::Value::CreateTensor<float>(allocator, input_shape.data(), input_shape.size());
    for (int i = 0; i < feat_kaldi.size(); i++) {
        memcpy(feat_input.GetTensorMutableData<float>() + i * feat_kaldi[i].size(), feat_kaldi[i].data(), feat_kaldi[i].size() * sizeof(float));
    }

    std::vector<const char*> input_names = {"input"};
    std::vector<const char*> output_names = {"output"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(feat_input));

    auto output_tensor = campplus_session->Run(run_options, input_names.data(), inputs.data(), 1, output_names.data(), 1);
    auto output = output_tensor[0].GetTensorMutableData<float>();
    int64_t output_size = output_tensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output_vector(output_size);
    std::memcpy(output_vector.data(), output, output_size * sizeof(float));
    LOGD("[TTS] speech embedding size: %d", output_vector.size());
    debug_print_mean_std(output_vector, "speech_embedding");

    return output_vector;
}

bool frontend::process_zeroshot(const std::vector<int> tts_tokens, const std::vector<int> prompt_tokens, const std::string prompt_audio_path, const int resample_rate) {
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
    LOGI("[TTS] extract_speech_tokens duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

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
    LOGI("[TTS] feat_extractor duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    debug_print_mean_std_2d(features, "features");

    if (resample_rate == 24000) {
        int token_length = std::min(features[0].size() / 2, speech_tokens.size());
        for (int i = 0; i < features.size(); i++) {
            features[i].resize(token_length * 2);
        }
        speech_tokens.resize(token_length);
    }
    LOGD("[TTS] features.size(): %dx%d", features.size(), features[0].size());
    LOGD("[TTS] speech_tokens.size(): %d", speech_tokens.size());

    start = std::chrono::high_resolution_clock::now();
    auto speech_embedding = extract_speech_embedding(samples_16k, 16000);
    end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] extract_speech_embedding duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    std::vector<int> tokens(tts_tokens.size() + prompt_tokens.size());
    std::copy(prompt_tokens.begin(), prompt_tokens.end(), tokens.begin());
    std::copy(tts_tokens.begin(), tts_tokens.end(), tokens.begin() + prompt_tokens.size());

    std::string debug_msg = "tokens: [";
    for (int i = 0; i < tokens.size(); i++) {
        debug_msg += std::to_string(tokens[i]) + ", ";
    }
    LOGI("[TTS] %s]", debug_msg.c_str());

    int content_length = tts_tokens.size();
    for (int i = 0; i < tokens.size(); i++) {
        if (tokens[i] == 65531) {
            // end_of_prompt_index = i;
            content_length = content_length - (i + 1);
            break;
        }
    }
    LOGI("[TTS] content_length: %d", content_length);

    float max_token_text_ratio = 20;
    float min_token_text_ratio = 2;
    int min_len = content_length * min_token_text_ratio;
    int max_len = content_length * max_token_text_ratio;
    LOGI("[TTS] min_len: %d, max_len: %d", min_len, max_len);

    return true;
}

std::string frontend::normalize_text(std::string text) {
    auto replace = [](std::string &text, const std::string &from, const std::string &to) {
        while (text.find(from) != std::string::npos) {
            text.replace(text.find(from), from.length(), to);
        }
    };

    replace(text, "\n", "");

    // remove blank between chinese characters
    // TODO

    replace(text, "²", "平方");
    replace(text, "³", "立方");
    replace(text, "（", "");
    replace(text, "）", "");
    replace(text, "【", "");
    replace(text, "】", "");
    replace(text, "`", "");
    replace(text, "”", "");
    replace(text, "——", " ");

    return text;
}

}
