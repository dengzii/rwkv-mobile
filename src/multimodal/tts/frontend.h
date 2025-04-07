#pragma once

#include <string>

namespace rwkvmobile {

class frontend {
public:
    frontend() = default;
    ~frontend() = default;

    void process_zeroshot(const std::string tts_text, const std::string prompt_text, const std::string prompt_audio_path, const int resample_rate = 24000);
};

}
