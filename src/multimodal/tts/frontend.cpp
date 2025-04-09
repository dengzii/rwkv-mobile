#include "frontend.h"
#include "audio.h"
#include "logger.h"
#include "librosa.h"
#include <chrono>

#define N_FFT 400
#define N_HOP 160
#define N_MEL 128

namespace rwkvmobile {

bool frontend::process_zeroshot(const std::string tts_text, const std::string prompt_text, const std::string prompt_audio_path, const int resample_rate) {
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

    auto start = std::chrono::high_resolution_clock::now();
    int fmin = 0;
    int fmax = prompt_audio.sample_rate / 2;
    std::vector<std::vector<float>> mels = logMelSpectrogram(prompt_audio.samples, prompt_audio.sample_rate, N_FFT, N_HOP, N_MEL, fmin, fmax);
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("[TTS] Log-Melspectrogram time: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

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

    prompt_audio.resample(resample_rate);
    if (prompt_audio.sample_rate != resample_rate) {
        LOGE("[TTS] Resample to %d Hz failed", resample_rate);
        return false;
    }
    // TODO
    return true;
}

}
