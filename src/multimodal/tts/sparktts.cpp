#include "sparktts.h"
#include <numeric>
#include <cmath>
#include <filesystem>
#include <cstring>
#include "logger.h"
#include "audio.h"

namespace rwkvmobile {

std::vector<float> sparktts::get_ref_clip(std::vector<float> audio) {
    // return audio;
    int ref_segment_length = ref_segment_duration * sample_rate / latent_hop_length;
    ref_segment_length *= latent_hop_length;

    if (audio.size() < ref_segment_length) {
        // wav = np.tile(wav, ref_segment_length // wav_length + 1)
        int repeat_times = ref_segment_length / audio.size() + 1;
        std::vector<float> ref_clip(repeat_times * audio.size());
        for (int i = 0; i < repeat_times; i++) {
            std::copy(audio.begin(), audio.end(), ref_clip.begin() + i * audio.size());
        }
        return std::vector<float>(ref_clip.begin(), ref_clip.begin() + ref_segment_length);
    } else {
        return std::vector<float>(audio.begin(), audio.begin() + ref_segment_length);
    }
}

std::vector<float> sparktts::extract_wav2vec2_features(const std::vector<float> audio) {
    // preprocess audio
    std::vector<float> audio_values(audio);
    zero_mean_unit_variance_normalize(audio_values);

    auto input_tensors = wav2vec2_mnn_interpretor->getSessionInputAll(wav2vec2_mnn_session);
    std::vector<int> input_shape = {1, static_cast<int>(audio_values.size())};
    wav2vec2_mnn_interpretor->resizeTensor(input_tensors["input"], input_shape);
    wav2vec2_mnn_interpretor->resizeSession(wav2vec2_mnn_session);

    auto nchw_tensor = new MNN::Tensor(input_tensors["input"], MNN::Tensor::CAFFE);
    memcpy(nchw_tensor->host<float>(), audio_values.data(), audio_values.size() * sizeof(float));
    input_tensors["input"]->copyFromHostTensor(nchw_tensor);
    delete nchw_tensor;

    wav2vec2_mnn_interpretor->runSession(wav2vec2_mnn_session);

    auto output_tensors = wav2vec2_mnn_interpretor->getSessionOutputAll(wav2vec2_mnn_session);
    void *output_ptr = output_tensors["output"]->map(MNN::Tensor::MAP_TENSOR_READ, output_tensors["output"]->getDimensionType());
    int output_size = output_tensors["output"]->elementSize();
    std::vector<float> output_values((float*)output_ptr, (float*)output_ptr + output_size);
    output_tensors["output"]->unmap(MNN::Tensor::MAP_TENSOR_READ, output_tensors["output"]->getDimensionType(), output_ptr);

    return output_values;
}

void sparktts::zero_mean_unit_variance_normalize(std::vector<float>& input_values) {
    float mean = std::accumulate(input_values.begin(), input_values.end(), 0.0f) / input_values.size();
    float std = std::sqrt(std::accumulate(input_values.begin(), input_values.end(), 0.0f, [mean](float a, float b) {
        return a + (b - mean) * (b - mean);
    }) / input_values.size() + 1e-7f);
    for (int i = 0; i < input_values.size(); i++) {
        input_values[i] = (input_values[i] - mean) / std;
    }
}

bool sparktts::load_models(std::string wav2vec2_model_path, std::string bicodec_tokenizer_path, std::string bicodec_detokenizer_path) {
    try {
        if (!std::filesystem::exists(wav2vec2_model_path)) {
            LOGE("[TTS] Wav2Vec2 model file not found: %s", wav2vec2_model_path.c_str());
            return false;
        }

        MNN::ScheduleConfig conf;
        conf.type = MNN_FORWARD_CPU;
        conf.numThread = 2;
        // conf.mode = 1;
        MNN::BackendConfig backendConfig;
        backendConfig.memory = MNN::BackendConfig::Memory_Low;
        backendConfig.power = MNN::BackendConfig::Power_High;
        backendConfig.precision = MNN::BackendConfig::Precision_Low;
        conf.backendConfig = &backendConfig;
        wav2vec2_mnn_interpretor = MNN::Interpreter::createFromFile(wav2vec2_model_path.c_str());
        wav2vec2_mnn_session = wav2vec2_mnn_interpretor->createSession(conf, mnn_runtime);

        if (!std::filesystem::exists(bicodec_tokenizer_path)) {
            LOGE("[TTS] Bicoder Tokenizer model file not found: %s", bicodec_tokenizer_path.c_str());
            return false;
        }
        bicodec_tokenizer_mnn_interpretor = MNN::Interpreter::createFromFile(bicodec_tokenizer_path.c_str());
        bicodec_tokenizer_mnn_session = bicodec_tokenizer_mnn_interpretor->createSession(conf, mnn_runtime);

        if (!std::filesystem::exists(bicodec_detokenizer_path)) {
            LOGE("[TTS] Bicoder Detokenizer model file not found: %s", bicodec_detokenizer_path.c_str());
            return false;
        }
        bicodec_detokenizer_mnn_interpretor = MNN::Interpreter::createFromFile(bicodec_detokenizer_path.c_str());
        bicodec_detokenizer_mnn_session = bicodec_detokenizer_mnn_interpretor->createSession(conf, mnn_runtime);

        // auto remove_extension = [](std::string path) {
        //     size_t lastindex = path.find_last_of(".");
        //     return path.substr(0, lastindex);
        // };
        // ncnn::set_cpu_powersave(2);
        // std::string param_path = remove_extension(bicodec_detokenizer_path) + ".param";
        // std::string bin_path = remove_extension(bicodec_detokenizer_path) + ".bin";
        // bicodec_detokenizer_ncnn_net.opt.use_vulkan_compute = 1;
        // bicodec_detokenizer_ncnn_net.opt.use_fp16_packed = false;
        // bicodec_detokenizer_ncnn_net.opt.use_fp16_storage = false;
        // bicodec_detokenizer_ncnn_net.opt.use_fp16_arithmetic = false;

        // int ret = 0;
        // ret = bicodec_detokenizer_ncnn_net.load_param(param_path.c_str());
        // if (ret == -1) {
        //     LOGE("[TTS] Error loading Bicoder Detokenizer param: %s", param_path.c_str());
        //     return false;
        // }
        // ret = bicodec_detokenizer_ncnn_net.load_model(bin_path.c_str());
        // if (ret == -1) {
        //     LOGE("[TTS] Error loading Bicoder Detokenizer model: %s", bin_path.c_str());
        //     return false;
        // }
    } catch (const std::exception &e) {
        LOGE("[TTS] Error loading models: %s", e.what());
        return false;
    }
    if (wav2vec2_mnn_interpretor == nullptr || bicodec_tokenizer_mnn_interpretor == nullptr || bicodec_detokenizer_mnn_interpretor == nullptr) {
        LOGE("[TTS] Error loading models");
        return false;
    }
    LOGI("[TTS] SparkTTS models loaded successfully");
    return true;
}

bool sparktts::tokenize_audio(std::vector<float> audio, std::vector<int> &global_tokens, std::vector<int> &semantic_tokens) {
    if (audio.size() == 0) {
        LOGE("[TTS] Audio is empty");
        return false;
    }

    if (wav2vec2_mnn_interpretor == nullptr) {
        LOGE("[TTS] Wav2Vec2 model not loaded");
        return false;
    }

    if (bicodec_tokenizer_mnn_interpretor == nullptr) {
        LOGE("[TTS] Bicoder Tokenizer model not loaded");
        return false;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> wav2vec2_features = extract_wav2vec2_features(audio);
    std::vector<float> ref_wav_samples = get_ref_clip(audio);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOGI("[TTS] Extract wav2vec2 features time: %f ms", duration);

    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> ref_wav_mel = melSpectrogram(ref_wav_samples, 16000, 1024, 320, 128, 10, 8000, 1.0f, true, false);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOGI("[TTS] Mel spectrogram time: %f ms", duration);

    start = std::chrono::high_resolution_clock::now();
    auto input_tensors = bicodec_tokenizer_mnn_interpretor->getSessionInputAll(bicodec_tokenizer_mnn_session);
    std::vector<int> input_shape_feat = {1, static_cast<int>(wav2vec2_features.size() / 1024), 1024};
    bicodec_tokenizer_mnn_interpretor->resizeTensor(input_tensors["feat"], input_shape_feat);
    bicodec_tokenizer_mnn_interpretor->resizeSession(bicodec_tokenizer_mnn_session);

    auto nchw_tensor_mel = new MNN::Tensor(input_tensors["ref_wav_mel"], MNN::Tensor::CAFFE);
    for (int i = 0; i < ref_wav_mel.size(); i++) {
        memcpy(nchw_tensor_mel->host<float>() + i * 301, ref_wav_mel[i].data(), ref_wav_mel[i].size() * sizeof(float));
    }
    input_tensors["ref_wav_mel"]->copyFromHostTensor(nchw_tensor_mel);
    delete nchw_tensor_mel;

    auto nchw_tensor_feat = new MNN::Tensor(input_tensors["feat"], MNN::Tensor::CAFFE);
    memcpy(nchw_tensor_feat->host<float>(), wav2vec2_features.data(), wav2vec2_features.size() * sizeof(float));
    input_tensors["feat"]->copyFromHostTensor(nchw_tensor_feat);
    delete nchw_tensor_feat;

    bicodec_tokenizer_mnn_interpretor->runSession(bicodec_tokenizer_mnn_session);

    auto output_tensors = bicodec_tokenizer_mnn_interpretor->getSessionOutputAll(bicodec_tokenizer_mnn_session);
    void *output_ptr_semantic_tokens = output_tensors["semantic_tokens"]->map(MNN::Tensor::MAP_TENSOR_READ, output_tensors["semantic_tokens"]->getDimensionType());
    int output_size_semantic_tokens = output_tensors["semantic_tokens"]->elementSize();
    semantic_tokens = std::vector<int>((int*)output_ptr_semantic_tokens, (int*)output_ptr_semantic_tokens + output_size_semantic_tokens);
    output_tensors["semantic_tokens"]->unmap(MNN::Tensor::MAP_TENSOR_READ, output_tensors["semantic_tokens"]->getDimensionType(), output_ptr_semantic_tokens);

    void *output_ptr_global_tokens = output_tensors["global_tokens"]->map(MNN::Tensor::MAP_TENSOR_READ, output_tensors["global_tokens"]->getDimensionType());
    int output_size_global_tokens = output_tensors["global_tokens"]->elementSize();
    global_tokens = std::vector<int>((int*)output_ptr_global_tokens, (int*)output_ptr_global_tokens + output_size_global_tokens);
    output_tensors["global_tokens"]->unmap(MNN::Tensor::MAP_TENSOR_READ, output_tensors["global_tokens"]->getDimensionType(), output_ptr_global_tokens);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOGI("[TTS] Tokenize audio time: %f ms", duration);
    return true;
}

std::vector<float> sparktts::detokenize_audio(std::vector<int> global_tokens, std::vector<int> semantic_tokens) {
    if (global_tokens.size() == 0) {
        LOGE("[TTS] Global tokens are empty");
        return std::vector<float>();
    }

    if (semantic_tokens.size() == 0) {
        LOGE("[TTS] Semantic tokens are empty");
        return std::vector<float>();
    }

    if (bicodec_detokenizer_mnn_interpretor == nullptr) {
        LOGE("[TTS] Bicoder Detokenizer model not loaded");
        return std::vector<float>();
    }

    LOGD("[TTS] semantic_tokens size: %d", semantic_tokens.size());

        auto start = std::chrono::high_resolution_clock::now();

    auto input_tensors = bicodec_detokenizer_mnn_interpretor->getSessionInputAll(bicodec_detokenizer_mnn_session);
    std::vector<int> input_shape_semantic_tokens = {1, static_cast<int>(semantic_tokens.size())};
    bicodec_detokenizer_mnn_interpretor->resizeTensor(input_tensors["semantic_tokens"], input_shape_semantic_tokens);
    bicodec_detokenizer_mnn_interpretor->resizeSession(bicodec_detokenizer_mnn_session);

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        LOGI("[TTS] Resize tensors time: %f ms", duration);

        start = std::chrono::high_resolution_clock::now();

    auto nchw_tensor_semantic_tokens = new MNN::Tensor(input_tensors["semantic_tokens"], MNN::Tensor::CAFFE);
    memcpy(nchw_tensor_semantic_tokens->host<int>(), semantic_tokens.data(), semantic_tokens.size() * sizeof(int));
    input_tensors["semantic_tokens"]->copyFromHostTensor(nchw_tensor_semantic_tokens);
    delete nchw_tensor_semantic_tokens;

    auto nchw_tensor_global_tokens = new MNN::Tensor(input_tensors["global_tokens"], MNN::Tensor::CAFFE);
    memcpy(nchw_tensor_global_tokens->host<int>(), global_tokens.data(), global_tokens.size() * sizeof(int));
    input_tensors["global_tokens"]->copyFromHostTensor(nchw_tensor_global_tokens);
    delete nchw_tensor_global_tokens;

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        LOGI("[TTS] Prepare tensors time: %f ms", duration);

        start = std::chrono::high_resolution_clock::now();
    bicodec_detokenizer_mnn_interpretor->runSession(bicodec_detokenizer_mnn_session);

    auto output_tensors = bicodec_detokenizer_mnn_interpretor->getSessionOutputAll(bicodec_detokenizer_mnn_session);
    void *output_ptr_audio = output_tensors["wav_rec"]->map(MNN::Tensor::MAP_TENSOR_READ, output_tensors["wav_rec"]->getDimensionType());
    int output_size_audio = output_tensors["wav_rec"]->elementSize();
    std::vector<float> output_values((float*)output_ptr_audio, (float*)output_ptr_audio + output_size_audio);
    output_tensors["wav_rec"]->unmap(MNN::Tensor::MAP_TENSOR_READ, output_tensors["wav_rec"]->getDimensionType(), output_ptr_audio);

    // ncnn::Extractor ex = bicodec_detokenizer_ncnn_net.create_extractor();
    // ncnn::Mat semantic_tokens_mat(semantic_tokens.size(), (void*)semantic_tokens.data());
    // ncnn::Mat global_tokens_mat(global_tokens.size(), (void*)global_tokens.data());
    // ncnn::Mat output_mat;
    // ex.input("in0", semantic_tokens_mat);
    // ex.input("in1", global_tokens_mat);
    // ex.extract("out0", output_mat);
    // std::vector<float> output_values((float*)output_mat.data, (float*)output_mat.data + output_mat.total());

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOGI("[TTS] Detokenize audio time: %f ms", duration);
    return output_values;
}

}