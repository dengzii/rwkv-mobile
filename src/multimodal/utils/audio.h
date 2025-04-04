#pragma once

#include <vector>
#include <string>
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

    std::vector<float> samples;
    int sample_rate;
    int num_channels;
    int bit_depth;
    int num_samples;
};

}