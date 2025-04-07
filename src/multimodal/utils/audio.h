#pragma once

#include <vector>
#include <string>
#include <cstdint>
namespace rwkvmobile {

class wav_file {
public:
    wav_file() {
        sample_rate = 0;
        num_channels = 0;
        bit_depth = 0;
        num_samples = 0;
    };
    ~wav_file() = default;
    bool load(const std::string& path);
    bool save(const std::string& path);

    void resample(int new_sample_rate);

    std::vector<float> samples;

    int16_t audio_format;
    int16_t num_channels;
    int32_t sample_rate;
    int32_t byte_rate;
    int16_t block_align;
    int16_t bit_depth;
    int32_t num_samples;
};

}