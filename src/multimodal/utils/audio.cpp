#include "audio.h"
#include "logger.h"
#include <fstream>
#include <cstdint>
// #include "librosa.h"
namespace rwkvmobile {

static int16_t twoBytesToInt16(const char* bytes) {
    if (bytes == nullptr) {
        return 0;
    }
    return reinterpret_cast<const int16_t*>(bytes)[0];
}

static uint16_t twoBytesToUint16(const char* bytes) {
    if (bytes == nullptr) {
        return 0;
    }
    return reinterpret_cast<const uint16_t*>(bytes)[0];
}

static uint32_t fourBytesToUint32(const char* bytes) {
    if (bytes == nullptr) {
        return 0;
    }
    return reinterpret_cast<const uint32_t*>(bytes)[0];
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
    file.read(info, 4); // fmt
    file.read(info, 4); // fmt size
    file.read(info, 2); // audio format
    file.read(info, 2); // num channels
    num_channels = twoBytesToUint16(info);
    LOGI("num_channels: %d", num_channels);

    file.read(info, 4); // sample rate
    sample_rate = fourBytesToUint32(info);
    LOGI("sample_rate: %d", sample_rate);

    file.read(info, 4); // byte rate
    file.read(info, 2); // block align
    file.read(info, 2); // bit depth
    bit_depth = twoBytesToUint16(info);
    LOGI("bit_depth: %d", bit_depth);

    file.read(info, 4); // chunk name
    std::string chunk_name(info, 4);
    file.read(info, 4); // chunk size
    uint32_t chunk_size = fourBytesToUint32(info);
    while (chunk_name != "data") {
        for (uint32_t i = 0; i < chunk_size / 2; i++) {
            file.read(info, 2);
        }
        file.read(info, 4); // data format
        chunk_name = std::string(info, 4);
    }
    file.read(info, 4); // data size
    num_samples = fourBytesToUint32(info) / (bit_depth / 8);
    LOGI("num_samples: %d", num_samples);

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
        LOGE("Unsupported bit depth yet: %d", bit_depth);
        return false;
    }
    file.close();

    return true;
}

}
