#include "frontend.h"
#include "audio.h"

namespace rwkvmobile {

void frontend::process_zeroshot(const std::string tts_text, const std::string prompt_text, const std::string prompt_audio_path, const int resample_rate) {
    wav_file prompt_audio;
    prompt_audio.load(prompt_audio_path);
    prompt_audio.resample(resample_rate);
    // TODO
}

}
