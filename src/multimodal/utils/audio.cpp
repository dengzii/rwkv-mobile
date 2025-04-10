#include "audio.h"
#include "logger.h"
#include <fstream>
#include <cstdint>
#include "soxr.h"
#include "librosa.h"
#include <cmath>

namespace rwkvmobile {

static int16_t twoBytesToInt16(const char* bytes) {
    if (bytes == nullptr) {
        return 0;
    }
    return reinterpret_cast<const int16_t*>(bytes)[0];
}

static int32_t fourBytesToInt32(const char* bytes) {
    if (bytes == nullptr) {
        return 0;
    }
    return reinterpret_cast<const int32_t*>(bytes)[0];
}

bool wav_file::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char info[5] = {0};
    file.read(info, 4);
    if (info[0] != 'R' || info[1] != 'I' || info[2] != 'F' || info[3] != 'F') {
        return false;
    }

    file.read(info, 4); // chunk size
    file.read(info, 4); // WAVE
    if (info[0] != 'W' || info[1] != 'A' || info[2] != 'V' || info[3] != 'E') {
        return false;
    }
    file.read(info, 4); // fmt
    file.read(info, 4); // fmt size
    file.read(info, 2); // audio format
    audio_format = twoBytesToInt16(info);
    LOGI("[WAV] audio_format: %d", audio_format);
    file.read(info, 2); // num channels
    num_channels = twoBytesToInt16(info);
    LOGI("[WAV] num_channels: %d", num_channels);

    file.read(info, 4); // sample rate
    sample_rate = fourBytesToInt32(info);
    LOGI("[WAV] sample_rate: %d", sample_rate);

    file.read(info, 4); // byte rate
    byte_rate = fourBytesToInt32(info);
    LOGI("[WAV] byte_rate: %d", byte_rate);

    file.read(info, 2); // block align
    block_align = twoBytesToInt16(info);
    LOGI("[WAV] block_align: %d", block_align);

    file.read(info, 2); // bit depth
    bit_depth = twoBytesToInt16(info);
    LOGI("[WAV] bit_depth: %d", bit_depth);

    file.read(info, 4); // chunk name
    std::string chunk_name(info, 4);
    file.read(info, 4); // chunk size
    int32_t chunk_size = fourBytesToInt32(info);
    while (chunk_name != "data") {
        for (int32_t i = 0; i < chunk_size / 2; i++) {
            file.read(info, 2);
        }
        file.read(info, 4); // data format
        chunk_name = std::string(info, 4);
    }
    file.read(info, 4); // data size
    num_samples = fourBytesToInt32(info) / (bit_depth / 8);
    LOGI("[WAV] num_samples: %d", num_samples);

    if (bit_depth == 16) {
        std::vector<int16_t> samples_int16(num_samples);
        file.read(reinterpret_cast<char*>(samples_int16.data()), num_samples * sizeof(int16_t));
        samples.resize(num_samples);
        for (int i = 0; i < num_samples; i++) {
            samples[i] = static_cast<float>(samples_int16[i]) / 32768.0f;
        }
    } else if (bit_depth == 8) {
        std::vector<int8_t> samples_int8(num_samples);
        file.read(reinterpret_cast<char*>(samples_int8.data()), num_samples * sizeof(int8_t));
        samples.resize(num_samples);
        for (int i = 0; i < num_samples; i++) {
            samples[i] = static_cast<float>(samples_int8[i]) / 128.0f;
        }
    } else {
        LOGE("[WAV] Unsupported bit depth yet: %d", bit_depth);
        return false;
    }
    file.close();

    return true;
}

bool wav_file::save(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write("RIFF", 4);
    int32_t chunk_size = num_samples * (bit_depth / 8) + 36;
    file.write(reinterpret_cast<const char*>(&chunk_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    int32_t fmt_size = 16;
    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bit_depth), 2);
    file.write("data", 4);
    int32_t data_size = num_samples * (bit_depth / 8);
    file.write(reinterpret_cast<const char*>(&data_size), 4);
    for (int i = 0; i < num_samples; i++) {
        if (bit_depth == 16) {
            int16_t sample = static_cast<int16_t>(samples[i] * 32768.0f);
            file.write(reinterpret_cast<const char*>(&sample), 2);
        } else if (bit_depth == 8) {
            int8_t sample = static_cast<int8_t>(samples[i] * 128.0f);
            file.write(reinterpret_cast<const char*>(&sample), 1);
        }
    }
    file.close();
    return true;
}

void wav_file::resample(int new_sample_rate) {
    if (samples.empty()) {
        LOGE("[WAV] samples is empty");
        return;
    }
    if (sample_rate == new_sample_rate) {
        return;
    }
    LOGI("[WAV] resampling from %d to %d", sample_rate, new_sample_rate);
    LOGD("[WAV] origin num_samples: %d", num_samples);
    LOGD("[WAV] new num_samples: %d", num_samples / sample_rate * new_sample_rate);

    std::vector<float> resampled_samples(num_samples / sample_rate * new_sample_rate);
    auto soxr_ret = soxr_oneshot(sample_rate, new_sample_rate, num_channels, samples.data(), samples.size(), NULL, resampled_samples.data(), resampled_samples.size(), NULL, NULL, NULL, NULL);
    if (soxr_ret != 0) {
        LOGE("[WAV] soxr_oneshot failed");
        return;
    }
    samples = resampled_samples;
    sample_rate = new_sample_rate;
    num_samples = resampled_samples.size();
}

std::vector<std::vector<float>> melSpectrogram(std::vector<float>& audio, int sample_rate, int n_fft, int n_hop, int n_mel, int fmin, int fmax, float power, bool center, bool return_magnitude) {
    return librosa::Feature::melspectrogram(audio, sample_rate, n_fft, n_hop, "hann", center, "reflect", power, n_mel, fmin, fmax, return_magnitude);
}

std::vector<std::vector<float>> logMelSpectrogram(std::vector<float>& audio, int sample_rate, int n_fft, int n_hop, int n_mel, int fmin, int fmax, float power, bool center, bool return_magnitude) {
    std::vector<std::vector<float>> mels = melSpectrogram(audio, sample_rate, n_fft, n_hop, n_mel, fmin, fmax, power, center, return_magnitude);

    float max_val = -1e20;
    for (int i = 0; i < mels.size(); i++) {
        for (int j = 0; j < mels[i].size(); j++) {
            mels[i][j] = log10f(std::max(mels[i][j], 1e-10f));
            max_val = std::max(max_val, mels[i][j]);
        }
    }

    for (int i = 0; i < mels.size(); i++) {
        for (int j = 0; j < mels[i].size(); j++) {
            mels[i][j] = std::max(mels[i][j], max_val - 8.0f);
            mels[i][j] = (mels[i][j] + 4.0f) / 4.0f;
        }
    }
    return mels;
}

void dynamic_range_compression(std::vector<std::vector<float>>& features) {
    for (int i = 0; i < features.size(); i++) {
        for (int j = 0; j < features[i].size(); j++) {
            features[i][j] = log(std::max(1e-5f, features[i][j]));
        }
    }
}

}
