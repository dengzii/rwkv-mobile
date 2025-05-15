#include <chrono>
#include <random>
#include <fstream>

#include "cosyvoice.h"
#include "audio.h"
#include "logger.h"
#include "librosa.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "kaldi-native-fbank/csrc/istft.h"

#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>

#define PRINT_FEATURE_INFO 0
#define ORT_LOGGING_LEVEL ORT_LOGGING_LEVEL_WARNING

static void debug_print_mean_std(std::vector<float> feat, std::string name) {
#if PRINT_FEATURE_INFO
    rwkvmobile::LOGI("[TTS] %s.size(): %d", name.c_str(), feat.size());
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
    rwkvmobile::LOGI("[TTS] %s Mean: %f, Std: %f", name.c_str(), mean, std);
#endif
}

static void debug_print_mean_std_2d(std::vector<std::vector<float>> feat, std::string name) {
#if PRINT_FEATURE_INFO
    rwkvmobile::LOGI("[TTS] %s.size(): %dx%d", name.c_str(), feat.size(), feat[0].size());
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
    rwkvmobile::LOGI("[TTS] %s Mean: %f, Std: %f", name.c_str(), mean, std);
#endif
}

namespace rwkvmobile {

bool cosyvoice::load_speech_tokenizer(const std::string model_path) {
    MNN::Express::Module::Config mdconfig;
    speech_tokenizer_module = MNN::Express::Module::load({"feats", "feats_length"}, {"indices"}, model_path.c_str(), &mdconfig);

    return true;
}

bool cosyvoice::load_campplus(const std::string model_path) {
    campplus_interpretor = MNN::Interpreter::createFromFile(model_path.c_str());
    MNN::ScheduleConfig conf;
    campplus_mnn_session = campplus_interpretor->createSession(conf, mnn_runtime);

    return true;
}

bool cosyvoice::load_flow_encoder(const std::string model_path) {
    flow_encoder_interpretor = MNN::Interpreter::createFromFile(model_path.c_str());
    MNN::ScheduleConfig conf;
    flow_encoder_mnn_session = flow_encoder_interpretor->createSession(conf, mnn_runtime);

    return true;
}

bool cosyvoice::load_flow_decoder_estimator(const std::string model_path) {
    flow_decoder_interpretor = MNN::Interpreter::createFromFile(model_path.c_str());
    MNN::ScheduleConfig conf;
    flow_decoder_mnn_session = flow_decoder_interpretor->createSession(conf, mnn_runtime);

    return true;
}

bool cosyvoice::load_hift_generator(const std::string model_path) {
    hift_generator_interpretor = MNN::Interpreter::createFromFile(model_path.c_str());
    MNN::ScheduleConfig conf;
    hift_generator_mnn_session = hift_generator_interpretor->createSession(conf, mnn_runtime);

    return true;
}

std::vector<int> cosyvoice::extract_speech_tokens(std::vector<float> audio_samples, int sample_rate) {
    if (speech_tokenizer_module == nullptr) {
        LOGE("[TTS] speech_tokenizer model not loaded.");
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

    std::vector<MNN::Express::VARP> inputs(2);
    inputs[0] = MNN::Express::_Input({1, static_cast<int>(mels.size()), static_cast<int>(mels[0].size())}, MNN::Express::NCHW, halide_type_of<float>());
    inputs[1] = MNN::Express::_Input({1}, MNN::Express::NCHW, halide_type_of<int>());

    float *feat_input_pointer = inputs[0]->writeMap<float>();
    int *feat_len_pointer = inputs[1]->writeMap<int>();

    for (int i = 0; i < mels.size(); i++) {
        memcpy(feat_input_pointer + i * mels[i].size(), mels[i].data(), mels[i].size() * sizeof(float));
    }
    *feat_len_pointer = mels[0].size();

    std::vector<MNN::Express::VARP> outputs = speech_tokenizer_module->onForward(inputs);
    auto output_info = outputs[0]->getInfo();
    auto output_ptr = outputs[0]->readMap<int>();
    int output_size = output_info->size;
    std::vector<int> output_vector((int*)output_ptr, (int*)output_ptr + output_size);
    outputs[0]->unMap();

    return output_vector;
}

std::vector<float> cosyvoice::extract_speech_embedding(std::vector<float> audio_samples, int sample_rate) {
    if (campplus_mnn_session == nullptr) {
        LOGE("[TTS] campplus model not loaded.");
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

    auto input_tensors = campplus_interpretor->getSessionInputAll(campplus_mnn_session);
    std::vector<int> input_shape = {1, static_cast<int>(feat_kaldi.size()), static_cast<int>(feat_kaldi[0].size())};
    campplus_interpretor->resizeTensor(input_tensors["input"], input_shape);
    campplus_interpretor->resizeSession(campplus_mnn_session);

    auto nchw_tensor = new MNN::Tensor(input_tensors["input"], MNN::Tensor::CAFFE);
    for (int i = 0; i < feat_kaldi.size(); i++) {
        memcpy((float*)nchw_tensor->host<float>() + i * feat_kaldi[i].size(), feat_kaldi[i].data(), feat_kaldi[i].size() * sizeof(float));
    }
    input_tensors["input"]->copyFromHostTensor(nchw_tensor);
    delete nchw_tensor;

    campplus_interpretor->runSession(campplus_mnn_session);

    auto output_tensors = campplus_interpretor->getSessionOutputAll(campplus_mnn_session);
    void *output_ptr = output_tensors["output"]->map(MNN::Tensor::MAP_TENSOR_READ, output_tensors["output"]->getDimensionType());
    int output_size = output_tensors["output"]->elementSize();
    std::vector<float> output_vector((float*)output_ptr, (float*)output_ptr + output_size);
    output_tensors["output"]->unmap(MNN::Tensor::MAP_TENSOR_READ, output_tensors["output"]->getDimensionType(), output_ptr);

    LOGD("[TTS] speech embedding size: %d", output_vector.size());
    debug_print_mean_std(output_vector, "speech_embedding");

    return output_vector;
}

bool cosyvoice::process_zeroshot(const std::string prompt_audio_path, std::vector<int> &speech_tokens, std::vector<std::vector<float>> &speech_features, std::vector<float> &speech_embedding, const int resample_rate) {
    if (speech_tokenizer_module == nullptr) {
        LOGE("[TTS] Speech tokenizer is not loaded");
        return false;
    }

    wav_file prompt_audio;
    prompt_audio.load(prompt_audio_path);
    if (prompt_audio.samples.size() / prompt_audio.sample_rate > 30) {
        LOGE("[TTS] Prompt audio is too long: should be less than 30 seconds");
        return false;
    }
    auto original_sample_rate = prompt_audio.sample_rate;
    std::vector<float> original_samples(prompt_audio.samples);
    auto original_num_samples = prompt_audio.num_samples;

    if (prompt_audio.sample_rate != 16000) {
        LOGI("[TTS] Resampling prompt audio to 16000 Hz");
        prompt_audio.resample(16000);
        if (prompt_audio.sample_rate != 16000) {
            LOGE("[TTS] Resample to %d Hz failed", 16000);
            return false;
        }
    }
    prompt_audio.bit_depth = 16;
    prompt_audio.num_channels = 1;
    prompt_audio.byte_rate = 16000 * 16 / 8;
    prompt_audio.block_align = 2;
    prompt_audio.audio_format = 1;
    prompt_audio.num_samples = prompt_audio.samples.size();
    prompt_audio.save("prompt_audio_16k.wav");

    std::vector<float> samples_16k(prompt_audio.samples);
    auto start = std::chrono::high_resolution_clock::now();
    speech_tokens = extract_speech_tokens(samples_16k, 16000);
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] extract_speech_tokens duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    prompt_audio.samples.assign(original_samples.begin(), original_samples.end());
    prompt_audio.sample_rate = original_sample_rate;
    prompt_audio.num_samples = original_num_samples;
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
    speech_features = melSpectrogram(prompt_audio.samples, resample_rate, n_fft, n_hop, n_mel, fmin, fmax, 1.0, true, true);
    dynamic_range_compression(speech_features);
    end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] feat_extractor duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    debug_print_mean_std_2d(speech_features, "speech_features");

    if (resample_rate == 24000) {
        int token_length = std::min(speech_features[0].size() / 2, speech_tokens.size());
        for (int i = 0; i < speech_features.size(); i++) {
            speech_features[i].resize(token_length * 2);
        }
        speech_tokens.resize(token_length);
    }
    LOGD("[TTS] speech_features.size(): %dx%d", speech_features.size(), speech_features[0].size());
    LOGD("[TTS] speech_tokens.size(): %d", speech_tokens.size());

    start = std::chrono::high_resolution_clock::now();
    speech_embedding = extract_speech_embedding(samples_16k, 16000);
    end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] extract_speech_embedding duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    return true;
}

bool cosyvoice::speech_token_to_wav(const std::vector<int> tokens, const std::vector<std::vector<float>> speech_features, const std::vector<float> speech_embedding, std::vector<float> &output_samples, std::function<void(float)> progress_callback) {
    if (flow_encoder_mnn_session == nullptr) {
        LOGE("[TTS] Flow encoder is not loaded");
        return false;
    }

    if (hift_generator_mnn_session == nullptr) {
        LOGE("[TTS] Hift generator is not loaded");
        return false;
    }

    LOGD("[TTS] tokens.size(): %d", tokens.size());
    std::string debug_msg = "tokens: [";
    for (int i = 0; i < tokens.size(); i++) {
        debug_msg += std::to_string(tokens[i]) + ", ";
    }
    LOGI("[TTS] %s]", debug_msg.c_str());

    LOGD("[TTS] speech_features.size(): %dx%d", speech_features.size(), speech_features[0].size());
    LOGD("[TTS] speech_embedding.size(): %d", speech_embedding.size());

    // Flow encoder

    auto start = std::chrono::high_resolution_clock::now();

    auto encoder_inputs = flow_encoder_interpretor->getSessionInputAll(flow_encoder_mnn_session);
    flow_encoder_interpretor->resizeTensor(encoder_inputs["token"], {1, static_cast<int>(tokens.size())});
    flow_encoder_interpretor->resizeTensor(encoder_inputs["prompt_feat"], {1, static_cast<int>(speech_features.size()), static_cast<int>(speech_features[0].size())});
    flow_encoder_interpretor->resizeTensor(encoder_inputs["embedding"], {1, static_cast<int>(speech_embedding.size())});
    flow_encoder_interpretor->resizeSession(flow_encoder_mnn_session);

    auto token_input_tensor = new MNN::Tensor(encoder_inputs["token"], MNN::Tensor::CAFFE);
    for (int i = 0; i < tokens.size(); i++) {
        memcpy((int*)token_input_tensor->host<int>() + i, &tokens[i], sizeof(int));
    }
    encoder_inputs["token"]->copyFromHostTensor(token_input_tensor);
    delete token_input_tensor;

    auto feature_input_tensor = new MNN::Tensor(encoder_inputs["prompt_feat"], MNN::Tensor::CAFFE);
    for (int i = 0; i < speech_features.size(); i++) {
        memcpy((float*)feature_input_tensor->host<float>() + i * speech_features[i].size(), speech_features[i].data(), speech_features[i].size() * sizeof(float));
    }
    encoder_inputs["prompt_feat"]->copyFromHostTensor(feature_input_tensor);
    delete feature_input_tensor;

    auto embd_input_tensor = new MNN::Tensor(encoder_inputs["embedding"], MNN::Tensor::CAFFE);
    memcpy((float*)embd_input_tensor->host<float>(), speech_embedding.data(), speech_embedding.size() * sizeof(float));
    encoder_inputs["embedding"]->copyFromHostTensor(embd_input_tensor);
    delete embd_input_tensor;

    flow_encoder_interpretor->runSession(flow_encoder_mnn_session);
    progress_callback(0.3f);

    auto encoder_outputs = flow_encoder_interpretor->getSessionOutputAll(flow_encoder_mnn_session);

    void *conds_output_host = encoder_outputs["conds"]->map(MNN::Tensor::MAP_TENSOR_READ, encoder_outputs["conds"]->getDimensionType());
    void *embd_output_host = encoder_outputs["embedding_out"]->map(MNN::Tensor::MAP_TENSOR_READ, encoder_outputs["embedding_out"]->getDimensionType());
    void *mu_output_host = encoder_outputs["mu"]->map(MNN::Tensor::MAP_TENSOR_READ, encoder_outputs["mu"]->getDimensionType());

    LOGD("[TTS] mu size: %dx%dx%d", encoder_outputs["mu"]->shape()[0], encoder_outputs["mu"]->shape()[1], encoder_outputs["mu"]->shape()[2]);
    LOGD("[TTS] embedding_out size: %dx%d", encoder_outputs["embedding_out"]->shape()[0], encoder_outputs["embedding_out"]->shape()[1]);
    LOGD("[TTS] conds size: %dx%dx%d", encoder_outputs["conds"]->shape()[0], encoder_outputs["conds"]->shape()[1], encoder_outputs["conds"]->shape()[2]);

    int feat_len = encoder_outputs["mu"]->shape()[2];
    int mel_len1 = speech_features[0].size();
    int mel_len2 = feat_len - mel_len1;
    LOGD("[TTS] mel_len1: %d, mel_len2: %d", mel_len1, mel_len2);
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] flow_encoder duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // Flow decoder
    const int n_timesteps = cfm_steps;
    int len_mu = encoder_outputs["mu"]->shape()[0] * encoder_outputs["mu"]->shape()[1] * encoder_outputs["mu"]->shape()[2];
    if (len_mu != 80 * feat_len) {
        LOGE("[TTS] size mismatch: len_mu: %d, feat_len: %d", len_mu, feat_len);
        return false;
    }

    std::mt19937 generator(time(nullptr));
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    if (random_noise.size() < len_mu) {
        random_noise.resize(len_mu);
        std::generate(random_noise.begin(), random_noise.end(), [&]() { return distribution(generator); });
    }
    if (t_span.empty() || t_span.size() != n_timesteps + 1) {
        t_span.resize(n_timesteps + 1);
        for (int i = 0; i < n_timesteps + 1; i++) {
            t_span[i] = i * 1.0f / n_timesteps;
            // cosine schedule
            t_span[i] = 1 - cos(t_span[i] * 0.5 * M_PI);
        }
    }

    float dt = t_span[1] - t_span[0];
    float t = t_span[0];

    std::vector<float> x_vector(320 * feat_len);
    std::vector<float> x_cfg_vector(320 * feat_len, 0.0f);

    const float inference_cfg_rate = 0.7;

    start = std::chrono::high_resolution_clock::now();
    memcpy(x_vector.data(), random_noise.data(), len_mu * sizeof(float));
    memcpy(x_vector.data() + len_mu, mu_output_host, len_mu * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < 80; i++) {
        for (int j = 0; j < feat_len; j++) {
            x_vector[2 * len_mu + i * feat_len + j] = ((float*)embd_output_host)[i];
        }
    }
    memcpy(x_vector.data() + 3 * len_mu, conds_output_host, len_mu * sizeof(float));

    memcpy(x_cfg_vector.data(), random_noise.data(), len_mu * sizeof(float));

    auto decoder_inputs = flow_decoder_interpretor->getSessionInputAll(flow_decoder_mnn_session);
    flow_decoder_interpretor->resizeTensor(decoder_inputs["x"], {2, 320, feat_len});
    flow_decoder_interpretor->resizeTensor(decoder_inputs["mask"], {2, 1, feat_len});
    flow_decoder_interpretor->resizeTensor(decoder_inputs["t"], {2});
    flow_decoder_interpretor->resizeSession(flow_decoder_mnn_session);

    // unmap encoder output tensors
    encoder_outputs["conds"]->unmap(MNN::Tensor::MAP_TENSOR_READ, encoder_outputs["conds"]->getDimensionType(), conds_output_host);
    encoder_outputs["embedding_out"]->unmap(MNN::Tensor::MAP_TENSOR_READ, encoder_outputs["embedding_out"]->getDimensionType(), embd_output_host);
    encoder_outputs["mu"]->unmap(MNN::Tensor::MAP_TENSOR_READ, encoder_outputs["mu"]->getDimensionType(), mu_output_host);

    progress_callback(0.4f);
    for (int i = 1; i <= n_timesteps; i++) {
        auto x_input_tensor = new MNN::Tensor(decoder_inputs["x"], MNN::Tensor::CAFFE);
        memcpy((float*)x_input_tensor->host<float>(), x_vector.data(),  320 * feat_len * sizeof(float));
        memcpy((float*)x_input_tensor->host<float>() + 320 * feat_len, x_cfg_vector.data(), 320 * feat_len * sizeof(float));
        decoder_inputs["x"]->copyFromHostTensor(x_input_tensor);
        delete x_input_tensor;

        auto mask_input_tensor = new MNN::Tensor(decoder_inputs["mask"], MNN::Tensor::CAFFE);
        for (int j = 0; j < 2 * feat_len; j++) {
            ((float*)mask_input_tensor->host<float>())[j] = 1.0f;
        }
        decoder_inputs["mask"]->copyFromHostTensor(mask_input_tensor);
        delete mask_input_tensor;

        auto t_input_tensor = new MNN::Tensor(decoder_inputs["t"], MNN::Tensor::CAFFE);
        ((float*)t_input_tensor->host<float>())[0] = t;
        ((float*)t_input_tensor->host<float>())[1] = t;
        decoder_inputs["t"]->copyFromHostTensor(t_input_tensor);
        delete t_input_tensor;

        progress_callback(0.4f + 0.4f * i / n_timesteps);
        flow_decoder_interpretor->runSession(flow_decoder_mnn_session);

        auto decoder_outputs = flow_decoder_interpretor->getSessionOutputAll(flow_decoder_mnn_session);
        void *dphi_dt_output_host = decoder_outputs["output"]->map(MNN::Tensor::MAP_TENSOR_READ, decoder_outputs["output"]->getDimensionType());

        #pragma omp parallel for
        for (int j = 0; j < len_mu; j++) {
            float dphi_dt_val = (1.0 + inference_cfg_rate) * ((float*)dphi_dt_output_host)[j] - inference_cfg_rate * ((float*)dphi_dt_output_host)[j + len_mu];
            x_vector[j] += dphi_dt_val * dt;
            x_cfg_vector[j] += dphi_dt_val * dt;
        }

        decoder_outputs["output"]->unmap(MNN::Tensor::MAP_TENSOR_READ, decoder_outputs["output"]->getDimensionType(), dphi_dt_output_host);

        if (i != n_timesteps) {
            t += dt;
            dt = t_span[i + 1] - t;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] flow_decoder_estimator diffusion duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // Hift generator
    start = std::chrono::high_resolution_clock::now();
    auto hift_inputs = hift_generator_interpretor->getSessionInputAll(hift_generator_mnn_session);
    // Resize input tensor
    hift_generator_interpretor->resizeTensor(hift_inputs["speech_feat"], {1, 80, mel_len2});
    hift_generator_interpretor->resizeSession(hift_generator_mnn_session);

    // Map input tensor and copy data
    void *speech_feat_input_host = hift_inputs["speech_feat"]->map(MNN::Tensor::MAP_TENSOR_WRITE, hift_inputs["speech_feat"]->getDimensionType());
    #pragma omp parallel for
    for (int i = 0; i < 80; i++) {
        memcpy(((float*)speech_feat_input_host) + i * mel_len2, x_vector.data() + i * (mel_len1 + mel_len2) + mel_len1, mel_len2 * sizeof(float));
    }
    hift_inputs["speech_feat"]->unmap(MNN::Tensor::MAP_TENSOR_WRITE, hift_inputs["speech_feat"]->getDimensionType(), speech_feat_input_host);

    progress_callback(0.8f);
    // Run inference
    hift_generator_interpretor->runSession(hift_generator_mnn_session);
    progress_callback(0.9f);
    // Get output tensors
    auto hift_outputs = hift_generator_interpretor->getSessionOutputAll(hift_generator_mnn_session);
    void *real_output_host = hift_outputs["real"]->map(MNN::Tensor::MAP_TENSOR_READ, hift_outputs["real"]->getDimensionType());
    void *imag_output_host = hift_outputs["img"]->map(MNN::Tensor::MAP_TENSOR_READ, hift_outputs["img"]->getDimensionType());

    auto real_shape = hift_outputs["real"]->shape();
    auto imag_shape = hift_outputs["img"]->shape();
    LOGD("[TTS] real size: %dx%dx%d", real_shape[0], real_shape[1], real_shape[2]);
    LOGD("[TTS] img size: %dx%dx%d", imag_shape[0], imag_shape[1], imag_shape[2]);

    // Copy output data
    int real_size = real_shape[0] * real_shape[1] * real_shape[2];
    int imag_size = imag_shape[0] * imag_shape[1] * imag_shape[2];
    std::vector<float> real_vector((float*)real_output_host, (float*)real_output_host + real_size);
    std::vector<float> imag_vector((float*)imag_output_host, (float*)imag_output_host + imag_size);

    // Unmap output tensors
    hift_outputs["real"]->unmap(MNN::Tensor::MAP_TENSOR_READ, hift_outputs["real"]->getDimensionType(), real_output_host);
    hift_outputs["img"]->unmap(MNN::Tensor::MAP_TENSOR_READ, hift_outputs["img"]->getDimensionType(), imag_output_host);
    debug_print_mean_std(real_vector, "real_vector");
    debug_print_mean_std(imag_vector, "imag_vector");
    knf::StftResult stft_result;
    stft_result.real = std::move(real_vector);
    stft_result.imag = std::move(imag_vector);
    stft_result.num_frames = static_cast<int32_t>(hift_outputs["real"]->shape()[1]);

    int istft_n_fft = 16;
    int istft_hop_length = 4;

    end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] hift duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    start = std::chrono::high_resolution_clock::now();
    // istft
    knf::StftConfig stft_config;
    stft_config.n_fft = istft_n_fft;
    stft_config.hop_length = istft_hop_length;
    stft_config.window_type = "hann";
    stft_config.win_length = istft_n_fft;
    knf::IStft istft(stft_config);
    std::vector<float> speech_output_istft = istft.Compute(stft_result);
    float max_val = 0.0f;
    for (int i = 0; i < speech_output_istft.size(); i++) {
        max_val = std::max(max_val, std::abs(speech_output_istft[i]));
    }
    LOGI("[TTS] speech_output_istft abs max_val: %f", max_val);
    for (int i = 0; i < speech_output_istft.size(); i++) {
        if (max_val > 1.0f) {
            speech_output_istft[i] = speech_output_istft[i] / max_val;
        }
        speech_output_istft[i] = std::max(std::min(speech_output_istft[i], 0.99f), -0.99f);
        speech_output_istft[i] = speech_output_istft[i] * 0.90;
    }
    end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] istft duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    output_samples = std::move(speech_output_istft);
    progress_callback(1.0f);

    return true;
}

int cosyvoice::speech_token_sampler(float *logits, size_t size, std::vector<int> decoded_tokens, bool ignore_eos) {
    if (logits == nullptr) {
        return 0;
    }

    int num_trials = 0, max_trials = 100;
    const int eos_token = 6562;
    int token_id = eos_token;
    int top_k = 25;
    float top_p = 0.8;
    float tau_r = 0.1;
    int win_size = 10;
    while (num_trials < max_trials) {
        token_id = _sampler.sample(logits, size, 1.0, top_k, top_p);
        int win_size_actual = std::min(win_size, (int)decoded_tokens.size());
        std::map<int, int> rep_count;
        for (int i = 0; i < win_size_actual; i++) {
            rep_count[decoded_tokens[decoded_tokens.size() - win_size_actual + i]]++;
        }

        for (auto &[token, count] : rep_count) {
            logits[token] -= count * 0.5 + 0.2;
        }

        if (rep_count[token_id] >= win_size * tau_r) {
            token_id = _sampler.sample(logits, size, 1.0, 35, top_p);
        }

        if (!ignore_eos || token_id != eos_token) {
            break;
        }
        num_trials++;
    }
    return token_id;
}

}
